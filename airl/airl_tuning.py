"""
airl/airl_tuning.py
===================
Stage-2 AIRL tuning, warm-started from bc_policy.pt.  Each Optuna trial:
  build agent (BC-warm-started actor + fresh critic/discriminator) ->
  discriminator warm-up (policy frozen) -> joint PPO + adversarial training ->
  score = best validation composite (maximised).

After the search, the winner is retrained and saved as airl_agent.pt (policy +
critic + reward_net + shaping_net + configs).  The actor family is inherited from
the BC checkpoint, so this same code tunes every family.
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

from utils.data import load_reservoir_data
from utils.optuna_dist import build_study, run_optimize, n_completed
from iqlearn.utils.runs import _resolve_device, _writeback_yaml, _find_run_folder
from iqlearn.iq_tuning import _resolve_mass_balance, _col_bounds
from iqlearn.environment import ReservoirRollout
from airl.config import AIRLConfig
from airl.agent import AIRLAgent
from airl.environment import AIRLEnv, expert_transitions
from airl.scoring import rollout_fidelity, composite_score

POLICY_TYPE = "airl"


# =============================================================================
# Search space -> AIRLConfig
# =============================================================================

def _sample_config(trial, ss, state_dim, seed, device) -> AIRLConfig:
    def f(name, spec):
        return trial.suggest_float(name, spec["low"], spec["high"], log=spec.get("log", False))
    return AIRLConfig(
        state_dim=state_dim, action_dim=1,
        critic_hidden_dim=trial.suggest_categorical("critic_hidden_dim", ss["critic_hidden_dim"]),
        critic_n_hidden_layers=trial.suggest_categorical("critic_n_hidden_layers", ss["critic_n_hidden_layers"]),
        disc_hidden_dim=trial.suggest_categorical("disc_hidden_dim", ss["disc_hidden_dim"]),
        disc_n_hidden_layers=trial.suggest_categorical("disc_n_hidden_layers", ss["disc_n_hidden_layers"]),
        lr_policy=f("lr_policy", ss["lr_policy"]), lr_critic=f("lr_critic", ss["lr_critic"]),
        lr_discriminator=f("lr_discriminator", ss["lr_discriminator"]),
        disc_updates=trial.suggest_categorical("disc_updates", ss["disc_updates"]),
        warmup_disc_updates=trial.suggest_categorical("warmup_disc_updates", ss["warmup_disc_updates"]),
        gradient_penalty_coef=f("gradient_penalty_coef", ss["gradient_penalty_coef"]),
        label_smoothing_epsilon=f("label_smoothing_epsilon", ss["label_smoothing_epsilon"]),
        gamma=trial.suggest_categorical("gamma", ss["gamma"]),
        gae_lambda=f("gae_lambda", ss["gae_lambda"]),
        clip_epsilon=f("clip_epsilon", ss["clip_epsilon"]),
        entropy_coef=f("entropy_coef", ss["entropy_coef"]),
        ppo_epochs=trial.suggest_categorical("ppo_epochs", ss["ppo_epochs"]),
        kl_regularization_coef=f("kl_regularization_coef", ss["kl_regularization_coef"]),
        warmup_iterations=trial.suggest_categorical("warmup_iterations", ss["warmup_iterations"]),
        num_iterations=trial.suggest_categorical("num_iterations", ss["num_iterations"]),
        steps_per_iteration=trial.suggest_categorical("steps_per_iteration", ss["steps_per_iteration"]),
        batch_size=trial.suggest_categorical("batch_size", ss["batch_size"]),
        early_stopping_patience=trial.suggest_categorical("early_stopping_patience", ss["early_stopping_patience"]),
        seed=seed, device=device)


def _config_from_params(p, state_dim, seed, device) -> AIRLConfig:
    return AIRLConfig(
        state_dim=state_dim, action_dim=1,
        critic_hidden_dim=p["critic_hidden_dim"], critic_n_hidden_layers=p["critic_n_hidden_layers"],
        disc_hidden_dim=p["disc_hidden_dim"], disc_n_hidden_layers=p["disc_n_hidden_layers"],
        lr_policy=p["lr_policy"], lr_critic=p["lr_critic"], lr_discriminator=p["lr_discriminator"],
        disc_updates=p["disc_updates"], warmup_disc_updates=p["warmup_disc_updates"],
        gradient_penalty_coef=p["gradient_penalty_coef"], label_smoothing_epsilon=p["label_smoothing_epsilon"],
        gamma=p["gamma"], gae_lambda=p["gae_lambda"], clip_epsilon=p["clip_epsilon"],
        entropy_coef=p["entropy_coef"], ppo_epochs=p["ppo_epochs"],
        kl_regularization_coef=p["kl_regularization_coef"], warmup_iterations=p["warmup_iterations"],
        num_iterations=p["num_iterations"], steps_per_iteration=p["steps_per_iteration"],
        batch_size=p["batch_size"], early_stopping_patience=p["early_stopping_patience"],
        seed=seed, device=device)


# =============================================================================
# Orchestration
# =============================================================================

def _build_agent(cfg, bc_ckpt, data, mb, norm_bounds, device, train_env=None, val_rollout=None):
    agent = AIRLAgent(cfg, bc_ckpt, device)
    agent.add_expert_data(*expert_transitions(data.train))
    if train_env is None:
        train_env = AIRLEnv(data.train, data.state_cols, mb, norm_bounds, device)
    if val_rollout is None:
        val_rollout = ReservoirRollout(data.val, data.state_cols, mb, norm_bounds, device)
    return agent, train_env, val_rollout


def run_airl_tuning(*, reservoir, res_cfg, algo_cfg, data, device_str, run_id=None, cli_args=None) -> dict:
    airl = algo_cfg["airl_tuning"]
    ss = airl["search_space"]
    opt = airl.get("optuna", {}) or {}
    rt = airl.get("runtime", {}) or {}
    n_trials = int(getattr(cli_args, "n_trials", None) or opt.get("n_trials", 100))
    n_jobs = int(rt["num_workers"]) if rt.get("num_workers") is not None else int(opt.get("n_jobs", 1))

    # ---- distributed (local shared-journal) options ----
    storage = getattr(cli_args, "storage", None)
    study_name = getattr(cli_args, "study_name", None) or f"{reservoir}_airl"
    role = getattr(cli_args, "role", None) or "full"
    if storage is not None and run_id is None:
        sys.exit("ERROR: --storage requires an explicit --run_id so every worker + the "
                 "finalize step target the same run folder.")

    base_dir = _ROOT / "results" / reservoir / "airl"
    if run_id is None:
        ids = ([int(d.name) for d in base_dir.iterdir() if d.is_dir() and d.name.isdigit()]
               if base_dir.exists() else [])
        if not ids:
            sys.exit(f"ERROR: no run folder under {base_dir}. Run AIRL BC tuning first, or pass --run_id.")
        run_id = max(ids)
    folder = _find_run_folder(base_dir, run_id)
    bc_path = folder / "bc_policy.pt"
    if not bc_path.exists():
        sys.exit(f"ERROR: bc_policy.pt not found in {folder}. Run AIRL BC tuning for run_id={run_id} first.")

    bc_ckpt = torch.load(bc_path, map_location="cpu", weights_only=False)
    bc_config = bc_ckpt["config"]
    seed = int(bc_config["seed"]); policy_family = bc_config.get("policy_family", "beta")
    if data.state_dim != bc_config["state_dim"]:
        sys.exit(f"ERROR: state_dim mismatch — data={data.state_dim} vs bc_policy={bc_config['state_dim']}.")

    mb = _resolve_mass_balance(res_cfg, data, cli_args)
    norm_bounds = {mb.storage_col: _col_bounds(data, mb.storage_col),
                   mb.inflow_col: _col_bounds(data, mb.inflow_col),
                   mb.action_col: _col_bounds(data, mb.action_col)}

    print(f"\nAIRL tuning  |  reservoir={reservoir}  run_id={run_id}  family={policy_family}  "
          f"seed={seed}  trials={n_trials}  device={device_str}")
    print(f"  warm-start: {bc_path.name}   state_dim={data.state_dim}")

    def objective(trial):
        cfg = _sample_config(trial, ss, data.state_dim, seed, device_str)
        try:
            np.random.seed(seed); torch.manual_seed(seed)
            agent, train_env, val_rollout = _build_agent(cfg, bc_ckpt, data, mb, norm_bounds, device_str)
            agent.warmup_discriminator(train_env, cfg.warmup_iterations)
            return float(agent.train_with_validation(train_env, val_rollout, trial))
        except optuna.TrialPruned:
            raise
        except Exception as exc:                       # noqa: BLE001
            print(f"  trial {trial.number} failed ({type(exc).__name__}: {exc})", file=sys.stderr)
            return 0.0

    study, shared = build_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed),
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2),
                                storage=storage, study_name=study_name)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    if role != "finalize":
        print(f"  optimize: role={role}  shared={shared}  n_jobs={n_jobs}  n_trials(cap)={n_trials}")
        run_optimize(study, objective, n_trials=n_trials, n_jobs=n_jobs, shared=shared)
    if role == "worker":
        print(f"  [worker] exiting — {n_completed(study)} completed trials in the shared study (no save).")
        return {"run_folder": folder, "run_id": run_id, "role": "worker", "n_completed": n_completed(study)}

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        sys.exit("All AIRL trials failed or pruned.")
    best = study.best_trial
    print(f"\n  Best trial #{best.number}  val composite = {best.value:.4f}")

    # retrain winner + save
    cfg = _config_from_params(best.params, data.state_dim, seed, device_str)
    np.random.seed(seed); torch.manual_seed(seed)
    agent, train_env, val_rollout = _build_agent(cfg, bc_ckpt, data, mb, norm_bounds, device_str)
    agent.warmup_discriminator(train_env, cfg.warmup_iterations)
    best_score = agent.train_with_validation(train_env, val_rollout, trial=None)

    agent.policy.eval()
    val_fid = rollout_fidelity(val_rollout.rollout(agent, deterministic=True))
    test_roll = ReservoirRollout(data.test, data.state_cols, mb, norm_bounds, device_str)
    test_fid = rollout_fidelity(test_roll.rollout(agent, deterministic=True))

    agent_path = folder / "airl_agent.pt"
    agent.save(agent_path)
    with open(folder / "airl_best_config.json", "w") as f:
        json.dump({"best_params": best.params, "search_best_score": float(best.value),
                   "retrained_best_score": float(best_score), "seed": seed,
                   "policy_family": policy_family, "mass_balance": _mb_dict(mb),
                   "airl_config": cfg.to_dict()}, f, indent=2)
    with open(folder / "metrics.json", "w") as f:
        json.dump({"val": val_fid, "test": test_fid,
                   "val_composite": float(best_score)}, f, indent=2)
    _save_run_args(folder, reservoir, run_id, cli_args, mb)

    print(f"  Saved airl_agent.pt → {folder}")
    print(f"  val: rel_corr={val_fid['release_corr']:.3f} stor_corr={val_fid['storage_corr']:.3f} | "
          f"test: rel_corr={test_fid['release_corr']:.3f} stor_corr={test_fid['storage_corr']:.3f}")
    return {"run_folder": folder, "run_id": run_id, "best_score": best_score,
            "val_metrics": val_fid, "test_metrics": test_fid}


def _mb_dict(mb):
    from dataclasses import asdict as _asdict
    return _asdict(mb)


def _save_run_args(folder, reservoir, run_id, cli_args, mb):
    g = lambda n: getattr(cli_args, n, None) if cli_args else None
    payload = {"airl_tune": {"reservoir": reservoir, "run_id": run_id, "folder": folder.name,
               "device": g("device"), "n_trials": g("n_trials"), "num_workers": g("num_workers"),
               "mass_balance": _mb_dict(mb), "timestamp": datetime.now().isoformat(timespec="seconds")}}
    with open(folder / "airl_run_args.json", "w") as f:
        json.dump(payload, f, indent=2)


# =============================================================================
# CLI
# =============================================================================

def _parse_args():
    p = argparse.ArgumentParser(description="AIRL Stage-2 tuning (warm-started from bc_policy.pt).")
    p.add_argument("--reservoir", required=True)
    p.add_argument("--run_id", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--n_trials", type=int, default=None)
    # distributed local-journal tuning (multiple worker processes, any scheduler; no internet)
    p.add_argument("--storage", default=None,
                   help="Local JournalFileStorage path for shared-journal distributed tuning. "
                        "Omit for in-memory. Requires --run_id.")
    p.add_argument("--study_name", default=None, help="Shared study name (default: <reservoir>_airl).")
    p.add_argument("--role", choices=["full", "worker", "finalize"], default="full",
                   help="worker = add trials only; finalize = retrain+save best; full = tune then save.")
    p.add_argument("--storage_variable", default=None)
    p.add_argument("--inflow_variable", default=None)
    p.add_argument("--max_storage", type=float, default=None)
    p.add_argument("--min_storage", type=float, default=None)
    p.add_argument("--max_release", type=float, default=None)
    p.add_argument("--min_release", type=float, default=None)
    p.add_argument("--seconds_per_day", type=float, default=None)
    p.add_argument("--volume_factor", type=float, default=None)
    p.add_argument("--save-config", dest="save_config", action="store_true",
                   help="Persist CLI overrides back into the YAML config files "
                        "(default: overrides apply to this run only).")
    return p.parse_args()


def _apply_cli_overrides(args, res_cfg, algo_cfg, res_cfg_path, algo_cfg_path,
                         save_config: bool = False) -> None:
    g = lambda n: getattr(args, n, None)
    res_u = {}
    cols_u = {k: g(v) for k, v in (("storage", "storage_variable"), ("inflow", "inflow_variable")) if g(v) is not None}
    if cols_u:
        res_cfg.setdefault("columns", {}).update(cols_u); res_u["columns"] = cols_u
    mb_u = {k: g(k) for k in ("max_storage", "min_storage", "max_release", "min_release",
                              "seconds_per_day", "volume_factor") if g(k) is not None}
    if mb_u:
        res_cfg.setdefault("reservoir", {}).setdefault("mass_balance", {}).update(mb_u)
        res_u["reservoir"] = {"mass_balance": mb_u}
    airl = algo_cfg.setdefault("airl_tuning", {})
    rt_u = {k: g(v) for k, v in (("device", "device"), ("num_workers", "num_workers")) if g(v) is not None}
    op_u = {"n_trials": g("n_trials")} if g("n_trials") is not None else {}
    if rt_u: airl.setdefault("runtime", {}).update(rt_u)
    if op_u: airl.setdefault("optuna", {}).update(op_u)
    algo_sub = {}
    if rt_u: algo_sub["runtime"] = rt_u
    if op_u: algo_sub["optuna"] = op_u
    if save_config:
        if res_u: _writeback_yaml(res_cfg_path, res_u)
        if algo_sub: _writeback_yaml(algo_cfg_path, {"airl_tuning": algo_sub})


def main():
    args = _parse_args()
    res_cfg_path = _ROOT / "configs" / "reservoirs" / f"{args.reservoir}.yaml"
    algo_cfg_path = _ROOT / "configs" / "algorithms" / "airl.yaml"
    res_cfg = yaml.safe_load(open(res_cfg_path)); algo_cfg = yaml.safe_load(open(algo_cfg_path))
    _apply_cli_overrides(args, res_cfg, algo_cfg, res_cfg_path, algo_cfg_path,
                         save_config=getattr(args, "save_config", False))
    device_str = _resolve_device(algo_cfg["airl_tuning"]["runtime"]["device"])
    data = load_reservoir_data(res_cfg, res_cfg_path)
    run_airl_tuning(reservoir=args.reservoir, res_cfg=res_cfg, algo_cfg=algo_cfg,
                    data=data, device_str=device_str, run_id=args.run_id, cli_args=args)


if __name__ == "__main__":
    main()
