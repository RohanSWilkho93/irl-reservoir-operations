"""
deepmaxent/tuning.py
====================
Optuna hyperparameter search for Deep MaxEnt IRL, then retrain + save the single
best model.  Single-stage (no warm-start): the objective trains a full MaxEnt
model per trial and returns the **validation unified score** (maximised).

A grid guard prunes trials whose discretization is too large
(`n_states = n_storage * n_inflow > guards.max_states`); skipped trials are
logged so coverage is never silently truncated.

Saves into results/<reservoir>/deepmaxent/<run_id>/ :
    best_config.json   reward_net.pt (weights + feature stats + config)
    policy_Pi.npy  reward_table_R.npy  s_space.npy  r_space.npy  i_space.npy
    metrics.json  run_args.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import optuna
import torch
import yaml

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from deepmaxent.config import DMConfig
from deepmaxent.data import load_raw_reservoir_data
from deepmaxent.mdp import (create_spaces, create_trajectories, build_inflow_transitions,
                            build_transition_matrix, grid_sizes)
from deepmaxent.trainer import MaxEntTrainer
from iqlearn.utils.runs import _resolve_device, _writeback_yaml, _resolve_run_id
from utils.optuna_dist import build_study, run_optimize, n_completed


# =============================================================================
# MDP assembly (shared by tuning, retrain, and results)
# =============================================================================

def build_mdp(cfg: DMConfig, data):
    """Build spaces, per-split trajectories, inflow transitions and the matrix P."""
    s_space, r_space, i_space = create_spaces(data.full, cfg.storage_step, cfg.release_step, cfg.inflow_step)
    tr, tr_raw, s_map, r_map, i_map = create_trajectories(
        data.train, s_space, r_space, i_space, cfg.storage_step, cfg.release_step, cfg.inflow_step)
    va, va_raw, *_ = create_trajectories(
        data.val, s_space, r_space, i_space, cfg.storage_step, cfg.release_step, cfg.inflow_step)
    te, te_raw, *_ = create_trajectories(
        data.test, s_space, r_space, i_space, cfg.storage_step, cfg.release_step, cfg.inflow_step)
    inflow_trans = build_inflow_transitions(data.train, i_map, cfg.inflow_step)
    P, n_s_bins = build_transition_matrix(s_space, r_space, i_space, inflow_trans, cfg.flow_to_volume_factor)
    return dict(s_space=s_space, r_space=r_space, i_space=i_space, s_map=s_map, r_map=r_map, i_map=i_map,
                train=(tr, tr_raw), val=(va, va_raw), test=(te, te_raw), P=P, n_s_bins=n_s_bins)


# =============================================================================
# Search space -> DMConfig
# =============================================================================

def _sample_config(trial, ss: dict, fvf: float, n_mc: int) -> DMConfig:
    def f(name, spec):
        return trial.suggest_float(name, spec["low"], spec["high"],
                                   log=spec.get("log", False), step=spec.get("step"))
    return DMConfig(
        seed=trial.suggest_categorical("seed", ss["seed"]),
        storage_step=float(trial.suggest_categorical("storage_step", ss["storage_step"])),
        release_step=float(trial.suggest_categorical("release_step", ss["release_step"])),
        inflow_step=float(trial.suggest_categorical("inflow_step", ss["inflow_step"])),
        gamma=trial.suggest_categorical("gamma", ss["gamma"]),
        tau=f("tau", ss["tau"]),
        hidden_dim1=trial.suggest_categorical("hidden_dim1", ss["hidden_dim1"]),
        hidden_dim2=trial.suggest_categorical("hidden_dim2", ss["hidden_dim2"]),
        dropout=f("dropout", ss["dropout"]),
        lr=f("lr", ss["lr"]),
        n_iterations=trial.suggest_categorical("n_iterations", ss["n_iterations"]),
        batch_size=trial.suggest_categorical("batch_size", ss["batch_size"]),
        val_early_stop_patience=trial.suggest_categorical("val_early_stop_patience", ss["val_early_stop_patience"]),
        n_mc_simulations=n_mc, flow_to_volume_factor=fvf,
    )


def _config_from_params(p: dict, fvf: float, n_mc: int) -> DMConfig:
    return DMConfig(
        seed=p["seed"], storage_step=float(p["storage_step"]), release_step=float(p["release_step"]),
        inflow_step=float(p["inflow_step"]), gamma=p["gamma"], tau=p["tau"],
        hidden_dim1=p["hidden_dim1"], hidden_dim2=p["hidden_dim2"], dropout=p["dropout"],
        lr=p["lr"], n_iterations=p["n_iterations"], batch_size=p["batch_size"],
        val_early_stop_patience=p["val_early_stop_patience"], n_mc_simulations=n_mc,
        flow_to_volume_factor=fvf,
    )


# =============================================================================
# Objective
# =============================================================================

def _make_objective(algo_cfg, data, device, max_states, max_p_elems, tuning_n_mc):
    ss = algo_cfg["deepmaxent"]["search_space"]
    fvf = data.flow_to_volume_factor

    def objective(trial):
        cfg = _sample_config(trial, ss, fvf, tuning_n_mc)
        n_s, n_r, n_i, n_states = grid_sizes(data.full, cfg.storage_step, cfg.release_step, cfg.inflow_step)
        if n_states > max_states:
            print(f"  trial {trial.number}: n_states={n_states} > {max_states} — pruned", file=sys.stderr)
            raise optuna.TrialPruned()
        p_elems = n_states * 12 * n_r * n_states          # dense transition matrix size
        if p_elems > max_p_elems:
            print(f"  trial {trial.number}: transition matrix {p_elems:.2e} elems "
                  f"({p_elems*4/1e9:.1f} GB) > {max_p_elems:.2e} — pruned", file=sys.stderr)
            raise optuna.TrialPruned()
        try:
            np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
            mdp = build_mdp(cfg, data)
            tr = MaxEntTrainer(cfg, mdp["P"], mdp["train"][0], mdp["s_space"], mdp["r_space"], mdp["i_space"],
                               mdp["s_map"], mdp["r_map"], mdp["i_map"], mdp["n_s_bins"], device, verbose=False)
            _, best_Pi, _, _, _ = tr.train(mdp["val"][0])
            val_svf, _ = tr.evaluate_svf(mdp["val"][0], best_Pi)
            val = tr.evaluate_full(*mdp["val"], svf_diff=val_svf, Pi=best_Pi)
            return float(val["unified_score"])
        except optuna.TrialPruned:
            raise
        except Exception as exc:                       # noqa: BLE001 — robustness over purity
            print(f"  trial {trial.number} failed ({type(exc).__name__}: {exc}); pruned", file=sys.stderr)
            raise optuna.TrialPruned()
    return objective


# =============================================================================
# Orchestration
# =============================================================================

def _save_run_args(folder, reservoir, run_id, cli_args, n_trials):
    g = lambda n: getattr(cli_args, n, None) if cli_args else None
    payload = {"deepmaxent_tune": {
        "reservoir": reservoir, "run_id": run_id, "folder": folder.name, "n_trials": n_trials,
        "device": g("device"), "num_workers": g("num_workers"),
        "timestamp": datetime.now().isoformat(timespec="seconds")}}
    with open(folder / "run_args.json", "w") as f:
        json.dump(payload, f, indent=2)


def run_deepmaxent_tuning(*, reservoir, res_cfg, res_cfg_path, algo_cfg, data,
                          device_str, run_id=None, cli_args=None) -> dict:
    dm = algo_cfg["deepmaxent"]
    opt = dm.get("optuna", {}) or {}
    rt = dm.get("runtime", {}) or {}
    n_trials = int(getattr(cli_args, "n_trials", None) or opt.get("n_trials", 500))
    n_jobs = int(rt["num_workers"]) if rt.get("num_workers") is not None else int(opt.get("n_jobs", 1))
    max_states = int((dm.get("guards", {}) or {}).get("max_states", 5000))
    max_p_elems = float((dm.get("guards", {}) or {}).get("max_transition_elems", 2.0e8))
    tuning_n_mc = int((dm.get("tuning", {}) or {}).get("n_mc_simulations", 30))
    final_n_mc = int((dm.get("tuning", {}) or {}).get("final_n_mc_simulations", 50))

    # ---- distributed (local shared-journal) options ----
    storage = getattr(cli_args, "storage", None)
    study_name = getattr(cli_args, "study_name", None) or f"{reservoir}_deepmaxent"
    role = getattr(cli_args, "role", None) or "full"
    if storage is not None and run_id is None:
        sys.exit("ERROR: --storage requires an explicit --run_id so every worker + the "
                 "finalize step target the same run folder.")

    base_dir = _ROOT / "results" / reservoir / "deepmaxent"
    run_id, folder = _resolve_run_id(base_dir, run_id)
    folder.mkdir(parents=True, exist_ok=True)
    ns0 = grid_sizes(data.full, DMConfig().storage_step, DMConfig().release_step, DMConfig().inflow_step)[3]
    print(f"  reservoir={reservoir}  run_id={run_id}  device={device_str}  "
          f"years: train={len(data.train_years)}/val={len(data.val_years)}/test={len(data.test_years)}")
    print(f"  grid guard: max_states={max_states}  max_transition_elems={max_p_elems:.1e}   "
          f"trials={n_trials} jobs={n_jobs}")

    study, shared = build_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=2048),
                                storage=storage, study_name=study_name)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    if role != "finalize":
        print(f"  optimize: role={role}  shared={shared}  n_jobs={n_jobs}  n_trials(cap)={n_trials}")
        run_optimize(study, _make_objective(algo_cfg, data, device_str, max_states, max_p_elems, tuning_n_mc),
                     n_trials=n_trials, n_jobs=n_jobs, shared=shared)
    if role == "worker":
        print(f"  [worker] exiting — {n_completed(study)} completed trials in the shared study (no save).")
        return {"run_folder": folder, "run_id": run_id, "role": "worker", "n_completed": n_completed(study)}

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        sys.exit("All Deep MaxEnt trials failed/pruned. Loosen the grid guard or widen the search space.")
    best = study.best_trial
    print(f"\n  Best trial #{best.number}  val unified score = {best.value:.4f}")

    # ---- retrain the winner (full MC) and save everything ----
    cfg = _config_from_params(best.params, data.flow_to_volume_factor, final_n_mc)
    np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
    mdp = build_mdp(cfg, data)
    trainer = MaxEntTrainer(cfg, mdp["P"], mdp["train"][0], mdp["s_space"], mdp["r_space"], mdp["i_space"],
                            mdp["s_map"], mdp["r_map"], mdp["i_map"], mdp["n_s_bins"], device_str, verbose=True)
    best_R, best_Pi, best_epoch, history, best_state = trainer.train(mdp["val"][0])

    # metrics on all splits with the saved policy
    val_svf, _ = trainer.evaluate_svf(mdp["val"][0], best_Pi)
    test_svf, _ = trainer.evaluate_svf(mdp["test"][0], best_Pi)
    val_m = trainer.evaluate_full(*mdp["val"], svf_diff=val_svf, Pi=best_Pi)
    test_m = trainer.evaluate_full(*mdp["test"], svf_diff=test_svf, Pi=best_Pi)

    np.save(folder / "s_space.npy", mdp["s_space"]); np.save(folder / "r_space.npy", mdp["r_space"])
    np.save(folder / "i_space.npy", mdp["i_space"])
    np.save(folder / "policy_Pi.npy", best_Pi); np.save(folder / "reward_table_R.npy", best_R)
    torch.save({"state_dict": {k: v.cpu() for k, v in trainer.r_net.state_dict().items()},
                "stats": trainer.r_net.stats, "config": cfg.to_dict()}, folder / "reward_net.pt")
    with open(folder / "best_config.json", "w") as f:
        json.dump({"reservoir": reservoir, "best_score": float(best.value), "best_epoch": best_epoch,
                   "config": cfg.to_dict(),
                   "grid": dict(zip(("n_storage", "n_release", "n_inflow", "n_states"),
                                    grid_sizes(data.full, cfg.storage_step, cfg.release_step, cfg.inflow_step)))},
                  f, indent=2)
    with open(folder / "metrics.json", "w") as f:
        json.dump({"best_epoch": best_epoch,
                   "val": {"svf_diff": float(val_svf), "release_corr": float(val_m["release_corr"]),
                           "storage_corr": float(val_m["storage_corr"]), "release_nrmse": float(val_m["release_nrmse"]),
                           "storage_nrmse": float(val_m["storage_nrmse"]), "unified_score": float(val_m["unified_score"])},
                   "test": {"svf_diff": float(test_svf), "release_corr": float(test_m["release_corr"]),
                            "storage_corr": float(test_m["storage_corr"]), "release_nrmse": float(test_m["release_nrmse"]),
                            "storage_nrmse": float(test_m["storage_nrmse"]), "unified_score": float(test_m["unified_score"])}},
                  f, indent=2)
    _save_run_args(folder, reservoir, run_id, cli_args, n_trials)

    print(f"  Saved reward_net.pt + spaces + policy + metrics → {folder}")
    print(f"  val unified={val_m['unified_score']:.4f}  test unified={test_m['unified_score']:.4f}")
    return {"run_folder": folder, "run_id": run_id, "best_score": float(best.value),
            "config": cfg, "val_metrics": val_m, "test_metrics": test_m}


# =============================================================================
# CLI / standalone
# =============================================================================

def _parse_args():
    p = argparse.ArgumentParser(description="Deep MaxEnt IRL hyperparameter tuning (Optuna).")
    p.add_argument("--reservoir", required=True)
    p.add_argument("--data_path", default=None)
    p.add_argument("--split_train", type=int, default=None)
    p.add_argument("--split_val", type=int, default=None)
    p.add_argument("--split_test", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--n_trials", type=int, default=None)
    p.add_argument("--run_id", type=int, default=None)
    # distributed local-journal tuning (multiple worker processes, any scheduler; no internet)
    p.add_argument("--storage", default=None,
                   help="Local JournalFileStorage path for shared-journal distributed tuning. "
                        "Omit for in-memory. Requires --run_id.")
    p.add_argument("--study_name", default=None, help="Shared study name (default: <reservoir>_deepmaxent).")
    p.add_argument("--role", choices=["full", "worker", "finalize"], default="full",
                   help="worker = add trials only; finalize = retrain+save best; full = tune then save.")
    p.add_argument("--save-config", dest="save_config", action="store_true",
                   help="Persist CLI overrides back into the YAML config files "
                        "(default: overrides apply to this run only).")
    return p.parse_args()


def _apply_cli_overrides(args, res_cfg, algo_cfg, res_cfg_path, algo_cfg_path,
                         save_config: bool = False) -> None:
    res_u, algo_u = {}, {}
    if args.data_path is not None:
        res_cfg["data_path"] = args.data_path; res_u["data_path"] = args.data_path
    for k, key in (("split_train", "train"), ("split_val", "val"), ("split_test", "test")):
        v = getattr(args, k)
        if v is not None:
            res_cfg.setdefault("split", {})[key] = v; res_u.setdefault("split", {})[key] = v
    if args.device is not None:
        algo_cfg["deepmaxent"]["runtime"]["device"] = args.device
        algo_u.setdefault("deepmaxent", {}).setdefault("runtime", {})["device"] = args.device
    if args.num_workers is not None:
        algo_cfg["deepmaxent"]["runtime"]["num_workers"] = args.num_workers
        algo_u.setdefault("deepmaxent", {}).setdefault("runtime", {})["num_workers"] = args.num_workers
    if args.n_trials is not None:
        algo_cfg["deepmaxent"]["optuna"]["n_trials"] = args.n_trials
        algo_u.setdefault("deepmaxent", {}).setdefault("optuna", {})["n_trials"] = args.n_trials
    if save_config:
        if res_u: _writeback_yaml(res_cfg_path, res_u)
        if algo_u: _writeback_yaml(algo_cfg_path, algo_u)


def main():
    args = _parse_args()
    res_cfg_path = _ROOT / "configs" / "reservoirs" / f"{args.reservoir}.yaml"
    algo_cfg_path = _ROOT / "configs" / "algorithms" / "deepmaxent.yaml"
    if not res_cfg_path.exists(): sys.exit(f"Reservoir config not found: {res_cfg_path}")
    res_cfg = yaml.safe_load(open(res_cfg_path)); algo_cfg = yaml.safe_load(open(algo_cfg_path))
    _apply_cli_overrides(args, res_cfg, algo_cfg, res_cfg_path, algo_cfg_path,
                         save_config=getattr(args, "save_config", False))
    device_str = _resolve_device(algo_cfg["deepmaxent"]["runtime"]["device"])
    print(f"\nLoading data for '{args.reservoir}' …")
    data = load_raw_reservoir_data(res_cfg, res_cfg_path)
    run_deepmaxent_tuning(reservoir=args.reservoir, res_cfg=res_cfg, res_cfg_path=res_cfg_path,
                          algo_cfg=algo_cfg, data=data, device_str=device_str,
                          run_id=args.run_id, cli_args=args)


if __name__ == "__main__":
    main()
