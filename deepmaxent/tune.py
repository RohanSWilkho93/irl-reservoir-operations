"""
deepmaxent/tune.py
==================
Optuna hyperparameter search for Deep Maximum Entropy IRL.

Usage
-----
# Minimal — reservoir config supplies everything else
    python deepmaxent/tune.py --reservoir conchas

# Full override on first run (values are written back to configs for
# reproducibility; subsequent runs need only --reservoir)
    python deepmaxent/tune.py --reservoir conchas \\
        --data_path data/conchas.csv \\
        --date_column date \\
        --state_variables storage net_inflow \\
        --use_month_encoding false \\
        --reward_features \\
        --split_train 14 --split_val 1 --split_test 3 \\
        --device cpu --num_workers 4 --n_trials 2000 \\
        --run_id 1

CLI arguments
-------------
Required:
    --reservoir NAME        Reservoir name, e.g. conchas.
                            Resolved to configs/reservoirs/<name>.yaml.

Optional — reservoir config overrides (written back to YAML on first use):
    --data_path     PATH    Path to reservoir CSV.
    --date_column   STR     Name of the date column in the CSV.
    --state_variables COL [COL ...]
                            State variable column names (space-separated).
                            Must be exactly two: storage and inflow column.
    --use_month_encoding true|false
                            Include month in MDP and reward network features.
    --reward_features [COL ...]
                            Extra CSV columns conditioning the reward network
                            (conditional IRL).  Pass no values to set empty.
    --split_train   INT     Training years.
    --split_val     INT     Validation years.
    --split_test    INT     Test years.

Optional — algorithm config overrides (written back to deepmaxent.yaml):
    --device        STR     Compute device: auto | cpu | cuda | cuda:N | mps.
    --num_workers   INT     Parallel Optuna workers.
    --n_trials      INT     Total Optuna trials.

Optional — run control:
    --run_id        INT     Run identifier.  Auto-incremented if omitted.

What this script does
---------------------
1.  Loads reservoir YAML and deepmaxent.yaml.
2.  Loads and splits reservoir data once — DataFrames are shared across
    all trials.
3.  Creates an Optuna study backed by SQLite for crash recovery and
    parallel-worker coordination.
4.  Runs n_trials Optuna trials.  Each trial:
    a.  Suggests hyperparameters including step sizes (from reservoir YAML
        deepmaxent section).
    b.  Prunes immediately if n_storage_bins × n_inflow_bins > max_states.
    c.  Looks up or builds the MDP cache for this step-size combination.
    d.  Builds a DeepMaxEntConfig and MaxEntTrainer.
    e.  Calls train_fast() — no Monte-Carlo, early-stop on val SAVF diff.
    f.  Returns the best validation SAVF diff across all epochs.
5.  Saves best_config.json and tune_summary.json to the run folder.

MDP caching
-----------
Step sizes determine the state-action grid and the transition tensor P.
Building P is the most expensive operation.  Since the reservoir YAML
defines a finite list of candidates per step, and 2000 trials will revisit
most combinations many times, this script caches (spaces, trajectories, P)
keyed by (storage_step, release_step, inflow_step).  The cache is
per-process; parallel workers each maintain their own independent cache.
A threading lock prevents duplicate builds when n_jobs > 1.

Run folder
----------
results/<reservoir>/deepmaxent/<run_id>/
    optuna_study.db   — SQLite Optuna storage (shared across workers,
                        supports resuming after interruption).
    best_config.json  — DeepMaxEntConfig of the best trial.
    tune_summary.json — Best trial statistics and run metadata.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import optuna
import torch
import yaml

# ---------------------------------------------------------------------------
# Repo root — allows running from any working directory
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from deepmaxent.core import (
    DeepMaxEntConfig,
    MaxEntTrainer,
    build_inflow_transitions,
    build_transition_matrix,
    create_spaces,
    create_trajectories,
    load_and_split_data,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Deep MaxEnt IRL — Optuna hyperparameter search",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- Required ----
    p.add_argument(
        "--reservoir", required=True,
        help="Reservoir name (e.g. conchas).  "
             "Resolved to configs/reservoirs/<name>.yaml.",
    )

    # ---- Reservoir config overrides (written back to YAML) ----
    p.add_argument(
        "--data_path", default=None,
        help="Override data_path in the reservoir config.",
    )
    p.add_argument(
        "--date_column", default=None,
        help="Override columns.date in the reservoir config.",
    )
    p.add_argument(
        "--state_variables", nargs="+", default=None,
        metavar="COL",
        help="Override columns.state in the reservoir config.  "
             "Pass exactly two column names: storage then inflow.",
    )
    p.add_argument(
        "--use_month_encoding",
        type=lambda x: x.lower() in ("true", "1", "yes"),
        default=None,
        metavar="true|false",
        help="Override columns.use_month_encoding (true|false).",
    )
    p.add_argument(
        "--reward_features", nargs="*", default=None,
        metavar="COL",
        help="Override deepmaxent.reward_features.  Extra CSV columns that "
             "condition the reward network (conditional IRL).  "
             "Pass no values to set an empty list.",
    )
    p.add_argument(
        "--split_train", type=int, default=None,
        help="Override split.train in the reservoir config.",
    )
    p.add_argument(
        "--split_val", type=int, default=None,
        help="Override split.val in the reservoir config.",
    )
    p.add_argument(
        "--split_test", type=int, default=None,
        help="Override split.test in the reservoir config.",
    )

    # ---- Algorithm config overrides (written back to deepmaxent.yaml) ----
    p.add_argument(
        "--device", default=None,
        help="Override runtime.device: auto | cpu | cuda | cuda:N | mps.",
    )
    p.add_argument(
        "--num_workers", type=int, default=None,
        help="Override runtime.num_workers (parallel Optuna workers).",
    )
    p.add_argument(
        "--n_trials", type=int, default=None,
        help="Override optuna.n_trials in the algorithm config.",
    )

    # ---- Run control ----
    p.add_argument(
        "--run_id", type=int, default=None,
        help="Integer run identifier.  Auto-incremented from existing "
             "runs if omitted.",
    )

    return p.parse_args()


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _deep_update(target: dict, source: dict) -> None:
    """
    Recursively update *target* with values from *source*.

    Descends into nested dicts rather than replacing them wholesale.
    Compatible with ruamel.yaml CommentedMap so YAML comments survive
    the write-back.
    """
    for key, val in source.items():
        if key in target and hasattr(target[key], "items") and hasattr(val, "items"):
            _deep_update(target[key], val)
        else:
            target[key] = val


def _writeback_yaml(path: Path, updates: dict) -> None:
    """
    Merge *updates* into an existing YAML file and write it back.

    Uses ruamel.yaml (comment-preserving) if installed; falls back to plain
    PyYAML with a warning.

    Parameters
    ----------
    path    : Path to the YAML file to update.
    updates : Dict of key → value pairs to merge (may be nested).
    """
    try:
        from ruamel.yaml import YAML
        ryaml = YAML()
        ryaml.preserve_quotes = True
        ryaml.best_width = 4096
        with open(path, "r") as f:
            doc = ryaml.load(f)
        _deep_update(doc, updates)
        with open(path, "w") as f:
            ryaml.dump(doc, f)
    except ImportError:
        import warnings
        warnings.warn(
            "ruamel.yaml not installed — writing YAML with plain PyYAML. "
            "YAML comments will be lost.  Install ruamel.yaml to preserve them.",
            UserWarning,
            stacklevel=2,
        )
        with open(path, "r") as f:
            doc = yaml.safe_load(f)
        _deep_update(doc, updates)
        with open(path, "w") as f:
            yaml.dump(doc, f, default_flow_style=False, sort_keys=False)


def _validate_required_config(res_cfg: dict, algo_cfg: dict, res_path: Path) -> None:
    """
    Verify that all keys required to run tune.py are present after CLI overrides.

    Raises SystemExit with a clear message if anything is missing.
    """
    errors: List[str] = []

    # Reservoir YAML
    if not res_cfg.get("data_path"):
        errors.append(
            f"  reservoir config: 'data_path' is missing.\n"
            f"    Fix: add 'data_path: <path>' to {res_path}, or pass --data_path."
        )
    cols = res_cfg.get("columns", {})
    if not cols.get("date"):
        errors.append(
            f"  reservoir config: 'columns.date' is missing.\n"
            f"    Fix: add it to {res_path}, or pass --date_column."
        )
    state_vars = cols.get("state")
    if not state_vars or len(state_vars) < 2:
        errors.append(
            f"  reservoir config: 'columns.state' must have at least two entries "
            f"(storage, inflow).\n"
            f"    Fix: set it in {res_path}, or pass --state_variables col1 col2."
        )
    if not cols.get("action"):
        errors.append(
            f"  reservoir config: 'columns.action' is missing.\n"
            f"    Fix: add it to {res_path}."
        )
    split = res_cfg.get("split", {})
    for key in ("train", "val", "test"):
        if split.get(key) is None:
            errors.append(
                f"  reservoir config: 'split.{key}' is missing.\n"
                f"    Fix: add it to {res_path}, or pass --split_{key}."
            )

    # deepmaxent section — step-size search space must exist
    dm = res_cfg.get("deepmaxent", {})
    for key in ("storage_step", "release_step", "inflow_step"):
        if not dm.get(key):
            errors.append(
                f"  reservoir config: 'deepmaxent.{key}' is missing.\n"
                f"    Fix: add a list of candidate values to the 'deepmaxent:' "
                f"section of {res_path}.\n"
                f"    Example:  {key}: [5.0, 10.0, 15.0]"
            )

    # Algorithm YAML
    if algo_cfg.get("optuna", {}).get("n_trials") is None:
        errors.append(
            "  algorithm config: 'optuna.n_trials' is missing.\n"
            "    Fix: add it to configs/algorithms/deepmaxent.yaml, "
            "or pass --n_trials."
        )
    if algo_cfg.get("state_space", {}).get("max_states") is None:
        errors.append(
            "  algorithm config: 'state_space.max_states' is missing.\n"
            "    Fix: add it to configs/algorithms/deepmaxent.yaml."
        )

    if errors:
        sys.exit(
            f"\nERROR: Required configuration is missing:\n\n"
            + "\n".join(errors)
            + "\n"
        )


def _load_configs(reservoir: str) -> Tuple[dict, dict]:
    res_path  = _REPO_ROOT / "configs" / "reservoirs" / f"{reservoir}.yaml"
    algo_path = _REPO_ROOT / "configs" / "algorithms" / "deepmaxent.yaml"

    if not res_path.exists():
        available = ", ".join(
            p.stem for p in res_path.parent.glob("*.yaml")
        )
        sys.exit(
            f"\nERROR: Reservoir config not found: {res_path}\n"
            f"  Available reservoirs: {available}\n"
        )

    if not algo_path.exists():
        sys.exit(
            f"\nERROR: Algorithm config not found: {algo_path}\n"
            f"  Expected: configs/algorithms/deepmaxent.yaml\n"
        )

    with open(res_path)  as f:
        res_cfg  = yaml.safe_load(f)
    with open(algo_path) as f:
        algo_cfg = yaml.safe_load(f)

    return res_cfg, algo_cfg


def _resolve_device(args: argparse.Namespace, algo_cfg: dict) -> torch.device:
    raw = args.device or algo_cfg.get("runtime", {}).get("device") or "auto"
    if raw == "auto":
        raw = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(raw)


def _resolve_n_jobs(args: argparse.Namespace, algo_cfg: dict) -> int:
    return (
        args.num_workers
        or algo_cfg.get("runtime", {}).get("num_workers")
        or algo_cfg.get("optuna", {}).get("n_jobs")
        or 1
    )


def _resolve_data_path(args: argparse.Namespace, res_cfg: dict) -> Path:
    raw = args.data_path or res_cfg["data_path"]
    path = Path(str(raw).replace("\\", "/"))
    if not path.is_absolute():
        path = _REPO_ROOT / path
    return path


# ---------------------------------------------------------------------------
# Run folder
# ---------------------------------------------------------------------------

def _make_run_folder(
    reservoir: str,
    run_id: Optional[int],
) -> Tuple[Path, int]:
    """
    Create (or locate) the run folder under results/<reservoir>/deepmaxent/.

    If run_id is None, auto-increment from the highest existing integer
    folder.  The first run on a reservoir gets ID 1.
    """
    base = _REPO_ROOT / "results" / reservoir / "deepmaxent"
    base.mkdir(parents=True, exist_ok=True)

    if run_id is None:
        existing = sorted(
            int(d.name) for d in base.iterdir()
            if d.is_dir() and d.name.isdigit()
        )
        run_id = (existing[-1] + 1) if existing else 1

    run_dir = base / str(run_id)
    if run_dir.exists():
        print(f"  Resuming existing run folder: {run_dir}")
    else:
        run_dir.mkdir()
        print(f"  Created run folder: {run_dir}")

    return run_dir, run_id


# ---------------------------------------------------------------------------
# Search space helper
# ---------------------------------------------------------------------------

def _suggest(trial: optuna.Trial, name: str, spec: Any) -> Any:
    """
    Translate a deepmaxent.yaml search_space entry into an Optuna call.

    list              → suggest_categorical
    {low, high}       → suggest_float (uniform)
    {low, high, log}  → suggest_float (log-uniform)
    {low, high, step} → suggest_float (stepped)
    """
    if isinstance(spec, list):
        return trial.suggest_categorical(name, spec)
    low  = spec["low"]
    high = spec["high"]
    log  = spec.get("log",  False)
    step = spec.get("step", None)
    if step is not None:
        # step and log are mutually exclusive in Optuna — step takes priority
        return trial.suggest_float(name, low, high, step=step)
    return trial.suggest_float(name, low, high, log=log)  # log=False if absent


# ---------------------------------------------------------------------------
# MDP cache  (per-process, thread-safe)
# ---------------------------------------------------------------------------

_MDP_CACHE: Dict[Tuple[float, float, float], Dict] = {}
_MDP_LOCK  = threading.Lock()


def _get_or_build_mdp(
    storage_step:       float,
    release_step:       float,
    inflow_step:        float,
    train_data,
    val_data,
    storage_col:        str,
    action_col:         str,
    inflow_col:         str,
    use_month_encoding: bool,
    reward_features:    List[str],
    max_states:         int,
) -> Optional[Dict]:
    """
    Return cached MDP entry for this step-size combination, building on
    first access.

    Returns None if the implied state space exceeds max_states — the
    caller should raise TrialPruned.

    Thread safety: uses a lock so parallel workers don't duplicate-build
    the same entry.
    """
    key = (storage_step, release_step, inflow_step)

    with _MDP_LOCK:
        if key in _MDP_CACHE:
            return _MDP_CACHE[key]

    # Temporary config — only step sizes and feature flags matter here
    tmp_cfg = DeepMaxEntConfig(
        storage_step       = storage_step,
        release_step       = release_step,
        inflow_step        = inflow_step,
        use_month_encoding = use_month_encoding,
        reward_features    = reward_features,
    )

    # Spaces (cheap — just np.arange)
    s_space, r_space, i_space = create_spaces(
        train_data, tmp_cfg, storage_col, action_col, inflow_col,
    )

    # State space guard — prune before building P
    n_states = len(s_space) * len(i_space)
    if n_states > max_states:
        return None

    # Trajectories — train and val splits
    trajs, trajs_raw, s_map, r_map, i_map = create_trajectories(
        train_data, s_space, r_space, i_space, tmp_cfg,
        storage_col, action_col, inflow_col,
    )
    val_trajs, val_trajs_raw, *_ = create_trajectories(
        val_data, s_space, r_space, i_space, tmp_cfg,
        storage_col, action_col, inflow_col,
    )

    # Inflow transition matrix (training data only)
    inflow_trans = build_inflow_transitions(
        train_data, i_space, i_map, tmp_cfg, inflow_col,
    )

    # Transition tensor P — the expensive step
    n_months = 12 if use_month_encoding else 1
    P, n_s_bins = build_transition_matrix(
        s_space, r_space, i_space, inflow_trans, n_months,
    )

    entry = {
        "s_space":       s_space,
        "r_space":       r_space,
        "i_space":       i_space,
        "trajs":         trajs,
        "trajs_raw":     trajs_raw,
        "val_trajs":     val_trajs,
        "val_trajs_raw": val_trajs_raw,
        "s_map":         s_map,
        "r_map":         r_map,
        "i_map":         i_map,
        "inflow_trans":  inflow_trans,
        "P":             P,
        "n_s_bins":      n_s_bins,
        "n_states":      n_states,
    }

    # Double-checked store — another thread may have built it while we worked
    with _MDP_LOCK:
        if key not in _MDP_CACHE:
            _MDP_CACHE[key] = entry

    return _MDP_CACHE[key]


# ---------------------------------------------------------------------------
# Optuna objective (closure)
# ---------------------------------------------------------------------------

def _make_objective(
    train_data,
    val_data,
    storage_col:        str,
    action_col:         str,
    inflow_col:         str,
    use_month_encoding: bool,
    reward_features:    List[str],
    algo_cfg:           dict,
    res_cfg:            dict,
    device:             torch.device,
) -> Any:
    """Return a closure over fixed data and config for Optuna to call."""

    ss         = algo_cfg["search_space"]
    dm_res     = res_cfg.get("deepmaxent", {})
    max_states = algo_cfg["state_space"]["max_states"]

    def objective(trial: optuna.Trial) -> float:

        # ---- Step sizes (reservoir-specific search space) ----
        storage_step = trial.suggest_categorical(
            "storage_step", dm_res["storage_step"],
        )
        release_step = trial.suggest_categorical(
            "release_step", dm_res["release_step"],
        )
        inflow_step = trial.suggest_categorical(
            "inflow_step", dm_res["inflow_step"],
        )

        # ---- MDP lookup / build (prune if state space too large) ----
        mdp = _get_or_build_mdp(
            storage_step, release_step, inflow_step,
            train_data, val_data,
            storage_col, action_col, inflow_col,
            use_month_encoding, reward_features,
            max_states,
        )
        if mdp is None:
            raise optuna.exceptions.TrialPruned()

        # ---- Remaining hyperparameters (from deepmaxent.yaml) ----
        seed                = _suggest(trial, "seed",                ss["seed"])
        gamma               = _suggest(trial, "gamma",               ss["gamma"])
        tau                 = _suggest(trial, "tau",                 ss["tau"])
        hidden_dim1         = _suggest(trial, "hidden_dim1",         ss["hidden_dim1"])
        hidden_dim2         = _suggest(trial, "hidden_dim2",         ss["hidden_dim2"])
        dropout             = _suggest(trial, "dropout",             ss["dropout"])
        lr                  = _suggest(trial, "lr",                  ss["lr"])
        n_iterations        = _suggest(trial, "n_iterations",        ss["n_iterations"])
        batch_size          = _suggest(trial, "batch_size",          ss["batch_size"])
        early_stop_patience = _suggest(trial, "early_stop_patience", ss["early_stop_patience"])

        # ---- Config ----
        cfg = DeepMaxEntConfig(
            seed                = int(seed),
            storage_step        = float(storage_step),
            release_step        = float(release_step),
            inflow_step         = float(inflow_step),
            gamma               = float(gamma),
            tau                 = float(tau),
            hidden_dim1         = int(hidden_dim1),
            hidden_dim2         = int(hidden_dim2),
            dropout             = float(dropout),
            lr                  = float(lr),
            n_iterations        = int(n_iterations),
            batch_size          = int(batch_size),
            early_stop_patience = int(early_stop_patience),
            use_month_encoding  = use_month_encoding,
            reward_features     = reward_features,
        )

        # ---- Reproducibility ----
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)

        # ---- Trainer ----
        trainer = MaxEntTrainer(
            cfg          = cfg,
            P            = mdp["P"],
            trajs        = mdp["trajs"],
            trajs_raw    = mdp["trajs_raw"],
            s_space      = mdp["s_space"],
            r_space      = mdp["r_space"],
            i_space      = mdp["i_space"],
            s_map        = mdp["s_map"],
            r_map        = mdp["r_map"],
            i_map        = mdp["i_map"],
            n_s_bins     = mdp["n_s_bins"],
            inflow_trans = mdp["inflow_trans"],
            device       = device,
            verbose      = False,
        )

        # ---- Train (no MC, early-stop on val SAVF diff) ----
        _, _, best_epoch, history, _ = trainer.train_fast(mdp["val_trajs"])

        return float(history[best_epoch]["val_savf_diff"])

    return objective


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

def _save_results(
    run_dir:            Path,
    study:              optuna.Study,
    reservoir:          str,
    run_id:             int,
    use_month_encoding: bool,
    reward_features:    List[str],
    elapsed_sec:        float,
) -> None:
    """Save best_config.json and tune_summary.json to the run folder."""

    best = study.best_trial
    p    = best.params

    cfg = DeepMaxEntConfig(
        seed                = int(p["seed"]),
        storage_step        = float(p["storage_step"]),
        release_step        = float(p["release_step"]),
        inflow_step         = float(p["inflow_step"]),
        gamma               = float(p["gamma"]),
        tau                 = float(p["tau"]),
        hidden_dim1         = int(p["hidden_dim1"]),
        hidden_dim2         = int(p["hidden_dim2"]),
        dropout             = float(p["dropout"]),
        lr                  = float(p["lr"]),
        n_iterations        = int(p["n_iterations"]),
        batch_size          = int(p["batch_size"]),
        early_stop_patience = int(p["early_stop_patience"]),
        use_month_encoding  = use_month_encoding,
        reward_features     = reward_features,
    )

    # Wrapped format (consistent with BC): metadata + config dict
    best_config_dict = {
        "reservoir":         reservoir,
        "run_id":            run_id,
        "best_val_savf_diff": float(best.value),
        "trial_number":      best.number,
        "use_month_encoding": use_month_encoding,
        "reward_features":   reward_features,
        "config":            cfg.to_dict(),
    }
    with open(run_dir / "best_config.json", "w") as f:
        json.dump(best_config_dict, f, indent=2)

    # Trial state counts
    states = [t.state for t in study.trials]
    n_complete = sum(1 for s in states if s == optuna.trial.TrialState.COMPLETE)
    n_pruned   = sum(1 for s in states if s == optuna.trial.TrialState.PRUNED)
    n_failed   = sum(1 for s in states if s == optuna.trial.TrialState.FAIL)

    summary = {
        "reservoir":               reservoir,
        "run_id":                  run_id,
        "best_trial":              best.number,
        "best_val_savf_diff":      best.value,
        "best_params":             p,
        "use_month_encoding":      use_month_encoding,
        "reward_features":         reward_features,
        "n_trials_complete":       n_complete,
        "n_trials_pruned":         n_pruned,
        "n_trials_failed":         n_failed,
        "elapsed_minutes":         round(elapsed_sec / 60, 2),
        "mdp_combinations_cached": len(_MDP_CACHE),
    }

    with open(run_dir / "tune_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n  Saved best_config.json and tune_summary.json → {run_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    print(f"\n{'='*62}")
    print(f"  Deep MaxEnt IRL — Hyperparameter Search")
    print(f"  Reservoir : {args.reservoir}")
    print(f"{'='*62}\n")

    # ---- Load configs ----
    res_cfg, algo_cfg = _load_configs(args.reservoir)
    res_cfg_path  = _REPO_ROOT / "configs" / "reservoirs" / f"{args.reservoir}.yaml"
    algo_cfg_path = _REPO_ROOT / "configs" / "algorithms" / "deepmaxent.yaml"

    # ---- Apply CLI overrides and collect changes for write-back ----
    res_updates:  dict = {}
    algo_updates: dict = {}

    if args.data_path is not None:
        res_cfg["data_path"] = args.data_path
        res_updates["data_path"] = args.data_path

    if args.date_column is not None:
        res_cfg.setdefault("columns", {})["date"] = args.date_column
        res_updates.setdefault("columns", {})["date"] = args.date_column

    if args.state_variables is not None:
        if len(args.state_variables) < 2:
            sys.exit(
                "\nERROR: --state_variables requires at least two column names "
                "(storage, inflow).\n"
                f"  Provided: {args.state_variables}\n"
            )
        res_cfg.setdefault("columns", {})["state"] = args.state_variables
        res_updates.setdefault("columns", {})["state"] = args.state_variables

    if args.use_month_encoding is not None:
        res_cfg.setdefault("columns", {})["use_month_encoding"] = args.use_month_encoding
        res_updates.setdefault("columns", {})["use_month_encoding"] = args.use_month_encoding

    if args.reward_features is not None:
        # nargs="*" with no values → empty list; with values → list of cols
        res_cfg.setdefault("deepmaxent", {})["reward_features"] = args.reward_features
        res_updates.setdefault("deepmaxent", {})["reward_features"] = args.reward_features

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
        algo_cfg.setdefault("runtime", {})["device"] = args.device
        algo_updates.setdefault("runtime", {})["device"] = args.device

    if args.num_workers is not None:
        algo_cfg.setdefault("runtime", {})["num_workers"] = args.num_workers
        algo_updates.setdefault("runtime", {})["num_workers"] = args.num_workers

    if args.n_trials is not None:
        algo_cfg.setdefault("optuna", {})["n_trials"] = args.n_trials
        algo_updates.setdefault("optuna", {})["n_trials"] = args.n_trials

    # ---- Write overrides back to YAML for reproducibility ----
    if res_updates:
        _writeback_yaml(res_cfg_path, res_updates)
    if algo_updates:
        _writeback_yaml(algo_cfg_path, algo_updates)

    # ---- Validate all required config keys are present ----
    _validate_required_config(res_cfg, algo_cfg, res_cfg_path)

    # ---- Resolve effective runtime settings ----
    use_month_encoding: bool = bool(
        res_cfg["columns"].get("use_month_encoding", True)
    )
    reward_features: List[str] = list(
        res_cfg.get("deepmaxent", {}).get("reward_features", [])
    )
    data_path = _resolve_data_path(args, res_cfg)
    device    = _resolve_device(args, algo_cfg)
    n_jobs    = _resolve_n_jobs(args, algo_cfg)
    n_trials  = int(algo_cfg["optuna"]["n_trials"])

    # ---- Run folder ----
    run_dir, run_id = _make_run_folder(args.reservoir, args.run_id)

    print(f"  Run ID        : {run_id}")
    print(f"  Device        : {device}")
    print(f"  Workers       : {n_jobs}")
    print(f"  Trials        : {n_trials}")
    print(f"  Month enc.    : {use_month_encoding}")
    print(f"  Reward feats  : {reward_features if reward_features else '(none)'}")
    print(f"  Data          : {data_path}")
    print()

    # ---- Column names from (possibly updated) reservoir config ----
    storage_col = str(res_cfg["columns"]["state"][0])
    action_col  = str(res_cfg["columns"]["action"])
    inflow_col  = str(res_cfg["columns"]["state"][1])
    date_col    = str(res_cfg["columns"]["date"])

    # ---- Load data once (shared across all trials) ----
    print("  Loading and splitting data...")
    _, train_data, val_data, _, train_years, val_years, _ = load_and_split_data(
        str(data_path),
        date_col = date_col,
        n_train  = int(res_cfg["split"]["train"]),
        n_val    = int(res_cfg["split"]["val"]),
        n_test   = int(res_cfg["split"]["test"]),
    )
    print(
        f"  Train: {len(train_years)} years | "
        f"Val: {len(val_years)} years\n"
    )

    # ---- Optuna study ----
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    storage_url = f"sqlite:///{run_dir}/optuna_study.db"
    study = optuna.create_study(
        study_name     = f"{args.reservoir}_deepmaxent_{run_id}",
        direction      = "minimize",
        storage        = storage_url,
        load_if_exists = True,   # enables resuming after interruption
    )

    # How many trials remain (handles resume correctly)
    already_done = sum(
        1 for t in study.trials
        if t.state in (
            optuna.trial.TrialState.COMPLETE,
            optuna.trial.TrialState.PRUNED,
        )
    )
    remaining = max(0, n_trials - already_done)

    if remaining == 0:
        print(f"  Study already complete ({already_done} trials).  "
              f"Proceeding to save results.\n")
    else:
        if already_done > 0:
            print(f"  Resuming: {already_done} trials done, "
                  f"{remaining} remaining.\n")
        else:
            print(f"  Starting {n_trials}-trial search.\n")

    # ---- Progress callback ----
    def _on_trial_end(
        study: optuna.Study, trial: optuna.trial.FrozenTrial
    ) -> None:
        if trial.state != optuna.trial.TrialState.COMPLETE:
            return
        is_best = trial.value == study.best_value
        print(
            f"  [{trial.number:4d}]  "
            f"val_savf={trial.value:.4f}  "
            f"best={study.best_value:.4f}"
            + ("  ← new best" if is_best else "")
        )

    # ---- Optimise ----
    objective = _make_objective(
        train_data, val_data,
        storage_col, action_col, inflow_col,
        use_month_encoding, reward_features,
        algo_cfg, res_cfg,
        device,
    )

    t_start = time.time()
    if remaining > 0:
        study.optimize(
            objective,
            n_trials          = remaining,
            n_jobs            = n_jobs,
            callbacks         = [_on_trial_end],
            show_progress_bar = False,
        )
    elapsed = time.time() - t_start

    # ---- Guard: ensure at least one trial completed ----
    completed = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
    ]
    if not completed:
        sys.exit(
            "\nERROR: No trials completed successfully.\n"
            "  All trials were pruned or failed.\n"
            "  Possible causes:\n"
            "    • max_states too low — all step-size combinations exceed the limit.\n"
            "      Increase state_space.max_states in configs/algorithms/deepmaxent.yaml,\n"
            "      or add larger step sizes to the reservoir deepmaxent section.\n"
            "    • Invalid search space values — check configs/algorithms/deepmaxent.yaml.\n"
        )

    # ---- Save results ----
    _save_results(
        run_dir, study,
        args.reservoir, run_id,
        use_month_encoding, reward_features,
        elapsed,
    )

    # ---- Save run_args.json ----
    run_args_path = run_dir / "run_args.json"
    run_args = {
        "tune": {
            "reservoir":          args.reservoir,
            "run_id":             run_id,
            "data_path":          args.data_path,
            "date_column":        args.date_column,
            "state_variables":    args.state_variables,
            "use_month_encoding": args.use_month_encoding,
            "reward_features":    args.reward_features,
            "split_train":        args.split_train,
            "split_val":          args.split_val,
            "split_test":         args.split_test,
            "device":             args.device,
            "num_workers":        args.num_workers,
            "n_trials":           args.n_trials,
            "timestamp":          datetime.now().isoformat(timespec="seconds"),
        }
    }
    with open(run_args_path, "w") as f:
        json.dump(run_args, f, indent=2)
    print(f"  Run args saved     → {run_args_path}")

    print(f"\n{'='*62}")
    print(f"  Best val SAVF diff : {study.best_value:.6f}")
    print(f"  Best trial         : #{study.best_trial.number}")
    print(f"  MDP combos cached  : {len(_MDP_CACHE)}")
    print(f"  Elapsed            : {elapsed / 60:.1f} min")
    print(f"  Run folder         : {run_dir}")
    print(f"{'='*62}\n")


if __name__ == "__main__":
    main()
