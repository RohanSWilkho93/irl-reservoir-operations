"""
iqlearn/bc_tuning.py
====================
Optuna hyperparameter search for Behavioral Cloning, with DATA-DRIVEN policy
family selection.

The user never chooses a distribution.  `run_bc_tuning` inspects the expert
release record and picks the candidate pair (see iqlearn.distributions.
detect_family_pair):

    release has zero days  ->  HardGating, SoftGating
    release is continuous  ->  Beta, LogNormal

It then runs a full Optuna search for BOTH families in the pair, retrains each
winner, and keeps ONLY the better one as the canonical policy.  The winning
family + its hyperparameters are recorded in bc_best_config.json and the weights
in bc_policy.pt — the warm-start that IQ-Learn then refines.

Validation metric (maximised), per trial:
    score = (release_pearson_r + (1 - release_rmse)) / 2
on the policy's deterministic action (the family mean) over normalised [0, 1]
release.  Both terms are in [0, 1]; RMSE on normalised data is scale-free.

Search space (configs/algorithms/iqlearn.yaml -> bc_tuning):
    shared : backbone (architecture, optimizer, scheduler, early stopping)
    <family> : the family-specific block (e.g. beta.alpha_max, lognormal.sigma_min,
               hardgating.zero_threshold, softgating.zero_threshold).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass, asdict, field
from datetime import datetime
from pathlib import Path

import numpy as np
import optuna
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import yaml

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils.data import load_reservoir_data
from utils.metrics import rmse, safe_pearsonr
from utils.optuna_dist import build_study, run_optimize, n_completed
from iqlearn.networks.policy import build_policy_network
from iqlearn.distributions import detect_family_pair
from iqlearn.utils.runs import _resolve_device, _writeback_yaml, _resolve_run_id


POLICY_TYPE = "parametric"


# =============================================================================
# BCConfig
# =============================================================================

@dataclass
class BCConfig:
    """All hyperparameters to build + train one parametric BC policy."""
    state_dim:  int
    action_dim: int = 1

    # shared backbone
    hidden_dim:              int   = 128
    n_hidden_layers:         int   = 3
    dropout:                 float = 0.1
    lr:                      float = 1e-3
    epochs:                  int   = 300
    batch_size:              int   = 128
    scheduler_type:          str   = "cosine"
    early_stopping_patience: int   = 30
    seed:                    int   = 42
    device:                  str   = "cpu"

    # distribution family + its hyperparameters (serialised; rebuilds the policy)
    policy_family: str  = "beta"
    dist_params:   dict = field(default_factory=dict)


# =============================================================================
# Train + validate one configuration
# =============================================================================

def _build_scheduler(optimizer, scheduler_type: str):
    if scheduler_type == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=1)
    if scheduler_type == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=5, factor=0.5)
    return None


def _train_and_validate(config: BCConfig, data, trial: optuna.Trial | None = None,
                        return_model: bool = False):
    """Train the parametric policy and return the validation score (and model)."""
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    device = torch.device(config.device)
    model = build_policy_network(config).to(device)
    dist  = model.distribution
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = _build_scheduler(optimizer, config.scheduler_type)

    train_states  = torch.tensor(data.train.states,  dtype=torch.float32)
    train_actions = torch.tensor(data.train.actions, dtype=torch.float32)
    train_dl = DataLoader(TensorDataset(train_states, train_actions),
                          batch_size=config.batch_size, shuffle=True, drop_last=False)

    val_states_t   = torch.tensor(data.val.states, dtype=torch.float32).to(device)
    val_actions_np = data.val.actions

    def _val_score() -> float:
        model.eval()
        with torch.no_grad():
            pred = dist.mean_action(model(val_states_t)).cpu().numpy()
        corr, _ = safe_pearsonr(val_actions_np, pred)
        return (corr + (1.0 - rmse(val_actions_np, pred))) / 2.0

    best_score, patience_ctr, best_weights = -float("inf"), 0, None
    for epoch in range(config.epochs):
        model.train()
        for states_b, actions_b in train_dl:
            states_b = states_b.to(device)
            actions_b = actions_b.to(device)
            optimizer.zero_grad()
            loss = dist.nll(model(states_b), actions_b)
            if not torch.isfinite(loss):
                return (-float("inf"), None) if return_model else -float("inf")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        if config.scheduler_type == "cosine":
            scheduler.step()

        score = _val_score()
        if config.scheduler_type == "plateau":
            scheduler.step(score)

        if score > best_score + 1e-6:
            best_score, patience_ctr = score, 0
            best_weights = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_ctr += 1
            if patience_ctr >= config.early_stopping_patience:
                break

        if trial is not None:
            trial.report(score, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

    if best_weights is not None:
        model.load_state_dict(best_weights)
    final = _val_score()
    return (final, model) if return_model else final


# =============================================================================
# Optuna objective (per family)
# =============================================================================

def _sample_dist_params(trial, family: str, fam_block: dict) -> dict:
    """Sample the family-specific distribution hyperparameters."""
    def f(name, spec):
        return trial.suggest_float(name, spec["low"], spec["high"], log=spec.get("log", False))

    if family == "beta":
        return {"alpha_min": 1.0, "beta_min": 1.0,
                "alpha_max": float(trial.suggest_categorical("alpha_max", fam_block["alpha_max"])),
                "beta_max":  float(trial.suggest_categorical("beta_max",  fam_block["beta_max"]))}
    if family == "lognormal":
        return {"sigma_min":   f("sigma_min",   fam_block["sigma_min"]),
                "log_epsilon": f("log_epsilon", fam_block["log_epsilon"])}
    if family in ("hardgating", "softgating"):
        return {"zero_threshold": float(trial.suggest_categorical("zero_threshold",
                                                                  fam_block["zero_threshold"]))}
    raise ValueError(f"Unknown family '{family}'.")


def _make_objective(algo_cfg: dict, data, device: str, family: str):
    shared = algo_cfg["bc_tuning"]["shared"]
    fam_block = algo_cfg["bc_tuning"].get(family, {})

    def objective(trial: optuna.Trial) -> float:
        lr_sp = shared["lr"]
        config = BCConfig(
            state_dim               = data.state_dim,
            action_dim              = 1,
            hidden_dim              = trial.suggest_categorical("hidden_dim", shared["hidden_dim"]),
            n_hidden_layers         = trial.suggest_categorical("n_hidden_layers", shared["n_hidden_layers"]),
            dropout                 = trial.suggest_categorical("dropout", shared["dropout"]),
            lr                      = trial.suggest_float("lr", lr_sp["low"], lr_sp["high"],
                                                          log=lr_sp.get("log", False)),
            epochs                  = trial.suggest_categorical("epochs", shared["epochs"]),
            batch_size              = trial.suggest_categorical("batch_size", shared["batch_size"]),
            scheduler_type          = trial.suggest_categorical("scheduler_type", shared["scheduler_type"]),
            early_stopping_patience = trial.suggest_categorical("early_stopping_patience",
                                                                shared["early_stopping_patience"]),
            seed                    = trial.suggest_categorical("seed", shared["seed"]),
            device                  = device,
            policy_family           = family,
            dist_params             = _sample_dist_params(trial, family, fam_block),
        )
        return _train_and_validate(config, data, trial=trial)

    return objective


def _config_from_best(best, data, device: str, family: str) -> BCConfig:
    """Rebuild a BCConfig from a study's best trial params."""
    p = best.params
    fam_keys = {
        "beta":       ("alpha_max", "beta_max"),
        "lognormal":  ("sigma_min", "log_epsilon"),
        "hardgating": ("zero_threshold",),
        "softgating": ("zero_threshold",),
    }[family]
    if family == "beta":
        dist_params = {"alpha_min": 1.0, "beta_min": 1.0,
                       "alpha_max": float(p["alpha_max"]), "beta_max": float(p["beta_max"])}
    else:
        dist_params = {k: float(p[k]) for k in fam_keys}
    return BCConfig(
        state_dim=data.state_dim, action_dim=1,
        hidden_dim=p["hidden_dim"], n_hidden_layers=p["n_hidden_layers"], dropout=p["dropout"],
        lr=p["lr"], epochs=p["epochs"], batch_size=p["batch_size"],
        scheduler_type=p["scheduler_type"], early_stopping_patience=p["early_stopping_patience"],
        seed=p["seed"], device=device, policy_family=family, dist_params=dist_params,
    )


def _tune_one_family(family: str, algo_cfg: dict, data, device: str,
                     n_trials: int, n_jobs: int, reservoir: str,
                     *, storage=None, study_name=None, role="full"):
    """
    Run one family's Optuna study; return (score, BCConfig, model_state_dict).

    role='worker'   -> contribute trials to the shared study, return (None, None, None).
    role='finalize' -> skip optimize; rebuild + retrain the best from the completed study.
    role='full'     -> optimize then retrain the best (default).
    """
    print(f"\n  ── Tuning family: {family}  ({n_trials} trials, {n_jobs} job(s), role={role}) ──")
    study, shared = build_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=20),
        storage=storage, study_name=(study_name or f"{reservoir}_bc_{family}"),
    )
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    if role != "finalize":
        run_optimize(study, _make_objective(algo_cfg, data, device, family),
                     n_trials=n_trials, n_jobs=n_jobs, shared=shared)
    if role == "worker":
        print(f"     {family}: [worker] {n_completed(study)} completed trials (no retrain).")
        return None, None, None

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        print(f"     {family}: all trials failed/pruned — skipping.")
        return -float("inf"), None, None

    best = study.best_trial
    cfg = _config_from_best(best, data, device, family)
    score, model = _train_and_validate(cfg, data, trial=None, return_model=True)
    print(f"     {family}: best search score={best.value:.4f}  retrained={score:.4f}")
    return score, cfg, {k: v.cpu() for k, v in model.state_dict().items()}


# =============================================================================
# Orchestration
# =============================================================================

def _save_run_args(results_dir, reservoir, run_id, cli_args, families, winner) -> None:
    g = lambda name: getattr(cli_args, name, None) if cli_args else None
    run_args = {"bc_tune": {
        "reservoir": reservoir, "run_id": run_id, "folder": results_dir.name,
        "candidate_families": list(families), "winning_family": winner,
        "data_path": g("data_path"), "date_column": g("date_column"),
        "state_variables": g("state_variables"), "use_month_encoding": g("use_month_encoding"),
        "split_train": g("split_train"), "split_val": g("split_val"), "split_test": g("split_test"),
        "device": g("device"), "n_trials": g("n_trials"), "num_workers": g("num_workers"),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }}
    with open(results_dir / "run_args.json", "w") as f:
        json.dump(run_args, f, indent=2)


def run_bc_tuning(*, reservoir, res_cfg, res_cfg_path, algo_cfg,
                  data, device_str, run_id=None, cli_args=None) -> dict:
    """
    Detect the candidate family pair from the data, tune BOTH, keep the winner.

    Saves the winner's bc_policy.pt + bc_best_config.json (with policy_family +
    dist_params) into results/<reservoir>/iqlearn/<run_id>/ and returns handles.
    """
    bc = algo_cfg["bc_tuning"]
    n_jobs = (bc["runtime"]["num_workers"] if bc["runtime"]["num_workers"] is not None
              else bc["optuna"]["n_jobs"])
    n_trials = bc["optuna"]["n_trials"]

    # ---- distributed (local shared-journal) options ----
    storage = getattr(cli_args, "storage", None)
    role = getattr(cli_args, "role", None) or "full"
    sname_prefix = getattr(cli_args, "study_name", None)
    if storage is not None and run_id is None:
        sys.exit("ERROR: --storage requires an explicit --run_id so every worker + the "
                 "finalize step target the same run folder.")

    # ---- data-driven family selection (enforces the Paper-1 pairing) ----
    sel = bc.get("selection", {}) or {}
    families = detect_family_pair(
        data.train.raw_actions,
        zero_frac_threshold=float(sel.get("zero_frac_threshold", 0.01)),
        zero_release_eps=sel.get("zero_release_eps", None),
    )
    _eps = sel.get("zero_release_eps")
    if _eps is None:
        _eps = max(1e-6, 1e-4 * float(np.max(data.train.raw_actions)))
    zero_frac = float(np.mean(np.asarray(data.train.raw_actions) <= _eps))
    print(f"  state_dim={data.state_dim}  train={len(data.train.states)}  "
          f"val={len(data.val.states)}  test={len(data.test.states)}  device={device_str}")
    print(f"  zero-release fraction (train) = {zero_frac:.3%}  ->  candidate families: {families}")

    bc_base_dir = _ROOT / "results" / reservoir / "iqlearn"
    run_id, results_dir = _resolve_run_id(bc_base_dir, run_id)
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nRun folder : {results_dir.name}  (run_id={run_id})")

    # ---- tune both candidate families, keep the better ----
    best_score, best_cfg, best_state, best_family = -float("inf"), None, None, None
    for family in families:
        sname = f"{sname_prefix}_{family}" if sname_prefix else f"{reservoir}_bc_{family}"
        score, cfg, state = _tune_one_family(family, algo_cfg, data, device_str,
                                             n_trials, n_jobs, reservoir,
                                             storage=storage, study_name=sname, role=role)
        if state is not None and score > best_score:
            best_score, best_cfg, best_state, best_family = score, cfg, state, family

    if role == "worker":
        print(f"\n  [worker] BC trials contributed for {families}; exiting (no policy saved).")
        return {"run_folder": results_dir, "run_id": run_id, "role": "worker",
                "candidate_families": list(families)}

    if best_cfg is None:
        sys.exit("All BC trials failed for every candidate family. Check data + search space.")

    print(f"\n  ✓ Winning family: {best_family}  (val score {best_score:.4f})")

    # ---- save the winner (only) ----
    best_config_path = results_dir / "bc_best_config.json"
    with open(best_config_path, "w") as f:
        json.dump({"reservoir": reservoir, "policy_type": POLICY_TYPE,
                   "policy_family": best_family, "candidate_families": list(families),
                   "best_score": best_score, "config": asdict(best_cfg)}, f, indent=2)

    policy_path = results_dir / "bc_policy.pt"
    torch.save({"state_dict": best_state, "config": asdict(best_cfg),
                "policy_type": POLICY_TYPE, "policy_family": best_family}, policy_path)
    print(f"Best config saved → {best_config_path}")
    print(f"Pretrained policy saved → {policy_path}\n")

    _save_run_args(results_dir, reservoir, run_id, cli_args, families, best_family)

    return {"run_folder": results_dir, "best_config": best_cfg,
            "best_config_path": best_config_path, "best_score": best_score,
            "policy_path": policy_path, "policy_family": best_family,
            "candidate_families": list(families)}


# =============================================================================
# CLI / standalone
# =============================================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Behavioral Cloning tuning (data-driven family selection) with Optuna.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--reservoir", required=True,
                   help="Reservoir name — must match configs/reservoirs/<name>.yaml.")
    p.add_argument("--data_path", default=None)
    p.add_argument("--date_column", default=None)
    p.add_argument("--state_variables", nargs="+", default=None)
    p.add_argument("--use_month_encoding", type=lambda s: s.lower() == "true", default=None)
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
    p.add_argument("--study_name", default=None, help="Shared study-name prefix (default: <reservoir>_bc).")
    p.add_argument("--role", choices=["full", "worker", "finalize"], default="full",
                   help="worker = add trials only; finalize = pick+retrain+save the winner; "
                        "full = tune then save (default).")
    p.add_argument("--save-config", dest="save_config", action="store_true",
                   help="Persist CLI overrides back into the YAML config files "
                        "(default: overrides apply to this run only).")
    return p.parse_args()


def _apply_cli_overrides(args, res_cfg, algo_cfg, res_cfg_path, algo_cfg_path,
                         save_config: bool = False) -> None:
    """Apply CLI overrides in-memory; only persist to the YAML files when
    `save_config` is True (reservoir keys at top level; algorithm keys under the
    bc_tuning block).  Default is ephemeral: overrides affect this run only."""
    res_updates, algo_updates = {}, {}

    if args.data_path is not None:
        res_cfg["data_path"] = args.data_path
        res_updates["data_path"] = args.data_path
    if args.date_column is not None:
        res_cfg["columns"]["date"] = args.date_column
        res_updates.setdefault("columns", {})["date"] = args.date_column
    if args.state_variables is not None:
        res_cfg["columns"]["state"] = args.state_variables
        res_updates.setdefault("columns", {})["state"] = args.state_variables
    if args.use_month_encoding is not None:
        res_cfg["columns"]["use_month_encoding"] = args.use_month_encoding
        res_updates.setdefault("columns", {})["use_month_encoding"] = args.use_month_encoding
    if args.split_train is not None:
        res_cfg.setdefault("split", {})["train"] = args.split_train
        res_updates.setdefault("split", {})["train"] = args.split_train
    if args.split_val is not None:
        res_cfg.setdefault("split", {})["val"] = args.split_val
        res_updates.setdefault("split", {})["val"] = args.split_val
    if args.split_test is not None:
        res_cfg.setdefault("split", {})["test"] = args.split_test
        res_updates.setdefault("split", {})["test"] = args.split_test

    if args.device is not None:
        algo_cfg["bc_tuning"]["runtime"]["device"] = args.device
        algo_updates.setdefault("bc_tuning", {}).setdefault("runtime", {})["device"] = args.device
    if args.num_workers is not None:
        algo_cfg["bc_tuning"]["runtime"]["num_workers"] = args.num_workers
        algo_updates.setdefault("bc_tuning", {}).setdefault("runtime", {})["num_workers"] = args.num_workers
    if args.n_trials is not None:
        algo_cfg["bc_tuning"]["optuna"]["n_trials"] = args.n_trials
        algo_updates.setdefault("bc_tuning", {}).setdefault("optuna", {})["n_trials"] = args.n_trials

    if save_config:
        if res_updates:
            _writeback_yaml(res_cfg_path, res_updates)
        if algo_updates:
            _writeback_yaml(algo_cfg_path, algo_updates)


def main():
    args = _parse_args()
    res_cfg_path  = _ROOT / "configs" / "reservoirs" / f"{args.reservoir}.yaml"
    algo_cfg_path = _ROOT / "configs" / "algorithms" / "iqlearn.yaml"
    if not res_cfg_path.exists():
        sys.exit(f"Reservoir config not found: {res_cfg_path}")
    res_cfg  = yaml.safe_load(open(res_cfg_path))
    algo_cfg = yaml.safe_load(open(algo_cfg_path))

    _apply_cli_overrides(args, res_cfg, algo_cfg, res_cfg_path, algo_cfg_path,
                         save_config=args.save_config)
    device_str = _resolve_device(algo_cfg["bc_tuning"]["runtime"]["device"])

    print(f"\nLoading data for reservoir '{args.reservoir}' …")
    data = load_reservoir_data(res_cfg, res_cfg_path)

    run_bc_tuning(reservoir=args.reservoir, res_cfg=res_cfg, res_cfg_path=res_cfg_path,
                  algo_cfg=algo_cfg, data=data, device_str=device_str,
                  run_id=args.run_id, cli_args=args)


if __name__ == "__main__":
    main()
