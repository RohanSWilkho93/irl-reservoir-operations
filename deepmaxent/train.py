"""
deepmaxent/train.py
===================
Final training run for Deep Maximum Entropy IRL.

Usage
-----
    python deepmaxent/train.py --reservoir cottage_grove [options]

CLI arguments
-------------
Required:
    --reservoir NAME        Reservoir name, e.g. cottage_grove.
                            Resolved to configs/reservoirs/<name>.yaml.

Optional:
    --run_id        INT     Run identifier (from tune.py).  Defaults to the
                            highest existing run folder (most recent tune run).
    --device        STR     Compute device: auto | cpu | cuda | cuda:N | mps.
                            Overrides deepmaxent.yaml runtime.device.
    --n_mc_per_epoch INT    MC rollouts per epoch for per-epoch S_DeepMaxEnt
                            tracking.  Default 5 (light; full evaluation at
                            the end uses cfg.n_mc_simulations = 50).
    --data_path     PATH    Path to reservoir CSV.
                            Overrides reservoir YAML data_path.

What this script does
---------------------
1.  Locates the run folder created by tune.py
    (results/<reservoir>/deepmaxent/<run_id>/).
2.  Loads best_config.json — the Optuna-selected hyperparameters, including
    use_month_encoding and reward_features.
3.  Loads and splits reservoir data into train / val / test.
4.  Builds the MDP: spaces, trajectories for all three splits, empirical
    inflow transition matrix, and transition tensor P.
5.  Calls train_full() — per-epoch S_DeepMaxEnt tracking (n=n_mc_per_epoch),
    early-stop on validation S_DeepMaxEnt (maximise).
6.  Runs full evaluation (cfg.n_mc_simulations rollouts = 50) on val and
    test splits using the best checkpoint.
7.  Saves outputs to the run folder.

Outputs
-------
results/<reservoir>/deepmaxent/<run_id>/
    model.pt           — reward-network checkpoint (weights + config + metadata).
    train_log.json     — per-epoch log (train/val SAVF, S score, lr).
    val_metrics.json   — full validation evaluation (all metrics).
    test_metrics.json  — full test evaluation (all metrics).
    train_summary.json — run-level summary statistics.
    run_args.json      — CLI arguments used for this run (tune + train sections).

Note on use_month_encoding and reward_features
----------------------------------------------
These are loaded from best_config.json (written by tune.py) — not from the
reservoir YAML or CLI.  They were baked into the Optuna search and cannot be
overridden here without invalidating the saved hyperparameters.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
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
        description="Deep MaxEnt IRL — Final training run",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--reservoir", required=True,
        help="Reservoir name (e.g. cottage_grove).  "
             "Resolved to configs/reservoirs/<name>.yaml.",
    )
    p.add_argument(
        "--run_id", type=int, default=None,
        help="Integer run identifier (from tune.py).  "
             "Defaults to the highest existing run ID.",
    )
    p.add_argument(
        "--device", default=None,
        help="Compute device: auto | cpu | cuda | cuda:N | mps.  "
             "Overrides deepmaxent.yaml runtime.device.",
    )
    p.add_argument(
        "--n_mc_per_epoch", type=int, default=5,
        help="MC rollouts per epoch for per-epoch S_DeepMaxEnt tracking.  "
             "Keep small (5) to control wall-clock time; full evaluation "
             "at the end uses cfg.n_mc_simulations (default 50).",
    )
    p.add_argument(
        "--data_path", default=None,
        help="Path to reservoir CSV.  Overrides reservoir YAML data_path.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

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

    with open(res_path)  as f:
        res_cfg  = yaml.safe_load(f)
    with open(algo_path) as f:
        algo_cfg = yaml.safe_load(f)

    return res_cfg, algo_cfg


def _resolve_device(args: argparse.Namespace, algo_cfg: dict) -> torch.device:
    raw = args.device or algo_cfg["runtime"].get("device") or "auto"
    if raw == "auto":
        raw = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(raw)


def _resolve_data_path(args: argparse.Namespace, res_cfg: dict) -> Path:
    raw = args.data_path or res_cfg["data_path"]
    path = Path(str(raw).replace("\\", "/"))
    if not path.is_absolute():
        path = _REPO_ROOT / path
    return path


# ---------------------------------------------------------------------------
# best_config.json loader with validation
# ---------------------------------------------------------------------------

def _load_best_config(run_dir: Path, reservoir: str) -> dict:
    """
    Load and validate best_config.json written by tune.py.

    Checks:
      1. File exists.
      2. File is valid JSON.
      3. Required top-level keys are present ("reservoir", "config").
      4. Saved reservoir name matches --reservoir argument.
      5. "config" sub-dict is non-empty.

    Returns the parsed dict on success; exits with a clear error otherwise.
    """
    path = run_dir / "best_config.json"

    if not path.exists():
        sys.exit(
            f"\nERROR: best_config.json not found in {run_dir}.\n"
            f"  Run deepmaxent/tune.py --reservoir {reservoir} first.\n"
        )

    try:
        with open(path) as f:
            saved = json.load(f)
    except json.JSONDecodeError as e:
        sys.exit(
            f"\nERROR: best_config.json is not valid JSON.\n"
            f"  Error: {e}\n"
            f"  File:  {path}\n"
            f"  Re-run tune.py to regenerate it.\n"
        )

    required_keys = {"reservoir", "config"}
    missing = required_keys - set(saved.keys())
    if missing:
        sys.exit(
            f"\nERROR: best_config.json is missing required keys: {sorted(missing)}\n"
            f"  File: {path}\n"
            f"  Re-run tune.py to regenerate it.\n"
        )

    if saved["reservoir"] != reservoir:
        sys.exit(
            f"\nERROR: best_config.json was tuned for reservoir "
            f"'{saved['reservoir']}', but --reservoir is '{reservoir}'.\n"
            f"  Run tune.py for this reservoir first:\n"
            f"    python deepmaxent/tune.py --reservoir {reservoir}\n"
        )

    if not saved.get("config"):
        sys.exit(
            f"\nERROR: best_config.json['config'] is empty or missing.\n"
            f"  File: {path}\n"
            f"  Re-run tune.py to regenerate it.\n"
        )

    return saved


# ---------------------------------------------------------------------------
# Run folder  (find existing — created by tune.py)
# ---------------------------------------------------------------------------

def _find_run_folder(reservoir: str, run_id: Optional[int]) -> Tuple[Path, int]:
    """
    Locate an existing run folder under results/<reservoir>/deepmaxent/.

    If run_id is None, picks the highest existing integer folder (i.e., the
    most recent tune.py run).  Exits with a clear error if the folder or
    best_config.json is missing.
    """
    base = _REPO_ROOT / "results" / reservoir / "deepmaxent"

    if run_id is None:
        existing: List[int] = sorted(
            int(d.name) for d in base.iterdir()
            if d.is_dir() and d.name.isdigit()
        ) if base.exists() else []
        if not existing:
            sys.exit(
                f"\nERROR: No tune.py run folders found under {base}.\n"
                f"  Run deepmaxent/tune.py --reservoir {reservoir} first.\n"
            )
        run_id = existing[-1]

    run_dir = base / str(run_id)
    if not run_dir.exists():
        sys.exit(
            f"\nERROR: Run folder not found: {run_dir}\n"
            f"  Run deepmaxent/tune.py --reservoir {reservoir} "
            f"--run_id {run_id} first.\n"
        )

    return run_dir, run_id


# ---------------------------------------------------------------------------
# JSON serialization helper
# ---------------------------------------------------------------------------

def _to_json_safe(obj: Any) -> Any:
    """Recursively convert numpy types, inf, and nan to JSON-safe types."""
    if isinstance(obj, dict):
        return {k: _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _to_json_safe(obj.tolist())
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        v = float(obj)
        return None if (np.isinf(v) or np.isnan(v)) else v
    if isinstance(obj, float):
        return None if (np.isinf(obj) or np.isnan(obj)) else obj
    return obj


# ---------------------------------------------------------------------------
# Save outputs
# ---------------------------------------------------------------------------

def _save_outputs(
    run_dir:          Path,
    best_model_state: Dict,
    cfg:              "DeepMaxEntConfig",
    history:          List[Dict],
    best_epoch:       int,
    val_metrics:      Dict,
    test_metrics:     Dict,
    reservoir:        str,
    run_id:           int,
    elapsed_sec:      float,
) -> None:
    """Save all train.py outputs to the run folder."""

    best_record = history[best_epoch] if history else {}

    # Reward-network checkpoint — rich format (consistent with BC)
    torch.save(
        {
            "model_state_dict":   best_model_state,
            "best_epoch":         best_epoch,
            "best_val_s_deepmaxent": best_record.get("val_s_deepmaxent"),
            "best_val_savf_diff": best_record.get("val_savf_diff"),
            "config":             cfg.to_dict(),
        },
        run_dir / "model.pt",
    )

    # Per-epoch training log — named train_log.json (consistent with BC)
    train_log = {
        "reservoir":             reservoir,
        "run_id":                run_id,
        "best_epoch":            best_epoch,
        "best_val_s_deepmaxent": best_record.get("val_s_deepmaxent"),
        "best_val_savf_diff":    best_record.get("val_savf_diff"),
        "total_epochs":          len(history),
        "epoch_logs":            history,
    }
    with open(run_dir / "train_log.json", "w") as f:
        json.dump(_to_json_safe(train_log), f, indent=2)

    # Validation metrics  (drop 'results' which contains raw numpy arrays)
    val_save = {k: v for k, v in val_metrics.items() if k != "results"}
    with open(run_dir / "val_metrics.json", "w") as f:
        json.dump(_to_json_safe(val_save), f, indent=2)

    # Test metrics
    test_save = {k: v for k, v in test_metrics.items() if k != "results"}
    with open(run_dir / "test_metrics.json", "w") as f:
        json.dump(_to_json_safe(test_save), f, indent=2)

    # Run-level summary
    summary = {
        "reservoir":              reservoir,
        "run_id":                 run_id,
        "best_epoch":             best_epoch,
        "best_val_s_deepmaxent":  best_record.get("val_s_deepmaxent"),
        "best_val_savf_diff":     best_record.get("val_savf_diff"),
        "val_s_deepmaxent":       val_metrics.get("s_deepmaxent"),
        "val_release_corr":       val_metrics.get("release_corr"),
        "val_storage_corr":       val_metrics.get("storage_corr"),
        "val_release_nrmse":      val_metrics.get("release_nrmse"),
        "val_storage_nrmse":      val_metrics.get("storage_nrmse"),
        "val_release_rmse":       val_metrics.get("release_rmse"),
        "val_storage_rmse":       val_metrics.get("storage_rmse"),
        "test_s_deepmaxent":      test_metrics.get("s_deepmaxent"),
        "test_release_corr":      test_metrics.get("release_corr"),
        "test_storage_corr":      test_metrics.get("storage_corr"),
        "test_release_nrmse":     test_metrics.get("release_nrmse"),
        "test_storage_nrmse":     test_metrics.get("storage_nrmse"),
        "test_release_rmse":      test_metrics.get("release_rmse"),
        "test_storage_rmse":      test_metrics.get("storage_rmse"),
        "elapsed_minutes":        round(elapsed_sec / 60, 2),
    }
    with open(run_dir / "train_summary.json", "w") as f:
        json.dump(_to_json_safe(summary), f, indent=2)

    print(
        f"\n  Saved: model.pt, train_log.json, "
        f"val_metrics.json, test_metrics.json, train_summary.json"
    )
    print(f"  → {run_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    print(f"\n{'='*62}")
    print(f"  Deep MaxEnt IRL — Final Training Run")
    print(f"  Reservoir : {args.reservoir}")
    print(f"{'='*62}\n")

    # ---- Load reservoir and algorithm configs ----
    res_cfg, algo_cfg = _load_configs(args.reservoir)

    # ---- Locate run folder; load and validate best hyperparameters ----
    run_dir, run_id = _find_run_folder(args.reservoir, args.run_id)
    saved = _load_best_config(run_dir, args.reservoir)
    cfg   = DeepMaxEntConfig.from_dict(saved["config"])

    device    = _resolve_device(args, algo_cfg)
    data_path = _resolve_data_path(args, res_cfg)

    print(f"  Run ID          : {run_id}")
    print(f"  Device          : {device}")
    print(f"  n_mc_per_epoch  : {args.n_mc_per_epoch}")
    print(f"  Month enc.      : {cfg.use_month_encoding}")
    print(
        f"  Reward feats    : "
        f"{cfg.reward_features if cfg.reward_features else '(none)'}"
    )
    print(f"  Data            : {data_path}")
    print(f"  Config          : {run_dir / 'best_config.json'}")
    print()

    # ---- Column names from reservoir YAML ----
    storage_col = res_cfg["columns"]["state"][0]
    action_col  = str(res_cfg["columns"]["action"])
    inflow_col  = res_cfg["columns"]["state"][1]
    date_col    = res_cfg["columns"]["date"]

    # ---- Load and split data ----
    print("  Loading and splitting data...")
    (_, train_data, val_data, test_data,
     train_years, val_years, test_years) = load_and_split_data(
        str(data_path),
        date_col = date_col,
        n_train  = int(res_cfg["split"]["train"]),
        n_val    = int(res_cfg["split"]["val"]),
        n_test   = int(res_cfg["split"]["test"]),
    )
    print(
        f"  Train: {len(train_years)} years | "
        f"Val: {len(val_years)} years | "
        f"Test: {len(test_years)} years\n"
    )

    # ---- Reproducibility ----
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    # ---- Build MDP ----
    print("  Building MDP...")

    s_space, r_space, i_space = create_spaces(
        train_data, cfg, storage_col, action_col, inflow_col,
    )
    n_states = len(s_space) * len(i_space)
    print(
        f"  State space : {len(s_space)} storage × {len(i_space)} inflow "
        f"= {n_states} states | {len(r_space)} actions"
    )

    # Trajectories for all three splits
    trajs, trajs_raw, s_map, r_map, i_map = create_trajectories(
        train_data, s_space, r_space, i_space, cfg,
        storage_col, action_col, inflow_col,
    )
    val_trajs, val_trajs_raw, *_ = create_trajectories(
        val_data, s_space, r_space, i_space, cfg,
        storage_col, action_col, inflow_col,
    )
    test_trajs, test_trajs_raw, *_ = create_trajectories(
        test_data, s_space, r_space, i_space, cfg,
        storage_col, action_col, inflow_col,
    )

    inflow_trans = build_inflow_transitions(
        train_data, i_space, i_map, cfg, inflow_col,
    )

    n_months = 12 if cfg.use_month_encoding else 1
    P, n_s_bins = build_transition_matrix(
        s_space, r_space, i_space, inflow_trans, n_months,
    )
    print(f"  P tensor shape  : {P.shape}\n")

    # ---- Trainer ----
    trainer = MaxEntTrainer(
        cfg          = cfg,
        P            = P,
        trajs        = trajs,
        trajs_raw    = trajs_raw,
        s_space      = s_space,
        r_space      = r_space,
        i_space      = i_space,
        s_map        = s_map,
        r_map        = r_map,
        i_map        = i_map,
        n_s_bins     = n_s_bins,
        inflow_trans = inflow_trans,
        device       = device,
        verbose      = True,
    )

    # ---- Train ----
    print(
        f"  Training (max {cfg.n_iterations} epochs, "
        f"patience={cfg.early_stop_patience})...\n"
    )
    t_start = time.time()

    _, best_Pi, best_epoch, history, best_model_state = trainer.train_full(
        val_trajs      = val_trajs,
        val_trajs_raw  = val_trajs_raw,
        n_mc_per_epoch = args.n_mc_per_epoch,
    )
    elapsed = time.time() - t_start

    best_record   = history[best_epoch]
    best_s_str    = f"{best_record['val_s_deepmaxent']:.4f}"    if best_record.get("val_s_deepmaxent")    is not None else "N/A"
    best_savf_str = f"{best_record['val_savf_diff']:.4f}"       if best_record.get("val_savf_diff")       is not None else "N/A"
    print(f"\n  Training complete.")
    print(f"  Best epoch         : {best_epoch}")
    print(f"  Best val S_DM      : {best_s_str}")
    print(f"  Best val SAVF diff : {best_savf_str}")
    print(f"  Elapsed            : {elapsed / 60:.1f} min\n")

    # ---- Full evaluation (val) ----
    # train_full restores r_net to best epoch weights; best_Pi matches those weights.
    # Pass best_Pi directly to avoid redundant reward/policy recomputation.
    print(f"  Full evaluation — val split ({cfg.n_mc_simulations} MC rollouts)...")
    val_savf_diff, _ = trainer.evaluate_savf(val_trajs, best_Pi)
    val_metrics = trainer.evaluate_full(
        val_trajs, val_trajs_raw,
        savf_diff = val_savf_diff,
        Pi        = best_Pi,
    )
    print(
        f"  Val   S_DM={val_metrics['s_deepmaxent']:.4f}  "
        f"r_corr={val_metrics['release_corr']:.3f}  "
        f"s_corr={val_metrics['storage_corr']:.3f}  "
        f"r_nRMSE={val_metrics['release_nrmse']:.3f}  "
        f"s_nRMSE={val_metrics['storage_nrmse']:.3f}"
    )

    # ---- Full evaluation (test) ----
    print(f"  Full evaluation — test split ({cfg.n_mc_simulations} MC rollouts)...")
    test_savf_diff, _ = trainer.evaluate_savf(test_trajs, best_Pi)
    test_metrics = trainer.evaluate_full(
        test_trajs, test_trajs_raw,
        savf_diff = test_savf_diff,
        Pi        = best_Pi,
    )
    print(
        f"  Test  S_DM={test_metrics['s_deepmaxent']:.4f}  "
        f"r_corr={test_metrics['release_corr']:.3f}  "
        f"s_corr={test_metrics['storage_corr']:.3f}  "
        f"r_nRMSE={test_metrics['release_nrmse']:.3f}  "
        f"s_nRMSE={test_metrics['storage_nrmse']:.3f}"
    )

    # ---- Save outputs ----
    _save_outputs(
        run_dir          = run_dir,
        best_model_state = best_model_state,
        cfg              = cfg,
        history          = history,
        best_epoch       = best_epoch,
        val_metrics      = val_metrics,
        test_metrics     = test_metrics,
        reservoir        = args.reservoir,
        run_id           = run_id,
        elapsed_sec      = elapsed,
    )

    # ---- Update run_args.json with train arguments ----
    run_args_path = run_dir / "run_args.json"
    run_args: dict = {}
    if run_args_path.exists():
        with open(run_args_path) as f:
            run_args = json.load(f)

    run_args["train"] = {
        "reservoir":      args.reservoir,
        "run_id":         run_id,
        "device_cli":     args.device,
        "device_used":    str(device),
        "n_mc_per_epoch": args.n_mc_per_epoch,
        "data_path_cli":  args.data_path,
        "timestamp":      datetime.now().isoformat(timespec="seconds"),
    }
    with open(run_args_path, "w") as f:
        json.dump(run_args, f, indent=2)
    print(f"  Run args updated   → {run_args_path}")

    val_s_str  = f"{val_metrics['s_deepmaxent']:.6f}"  if val_metrics.get("s_deepmaxent")  is not None else "N/A"
    test_s_str = f"{test_metrics['s_deepmaxent']:.6f}" if test_metrics.get("s_deepmaxent") is not None else "N/A"

    print(f"\n{'='*62}")
    print(f"  Val  S_DeepMaxEnt  : {val_s_str}")
    print(f"  Test S_DeepMaxEnt  : {test_s_str}")
    print(f"  Best epoch         : {best_epoch}")
    print(f"  Elapsed            : {elapsed / 60:.1f} min")
    print(f"  Run folder         : {run_dir}")
    print(f"{'='*62}\n")


if __name__ == "__main__":
    main()
