"""
behavioral_cloning/train.py
===========================
Final model training for Behavioral Cloning.

MUST be run AFTER behavioral_cloning/tune.py.  Reads all hyperparameters from
results/<reservoir>/behavioral_cloning/best_config.json and trains a single
model with those parameters.  If best_config.json is not found, the script
exits with a clear error and instructions.

Training strategy
-----------------
Trains on the TRAINING split.  Validates on the VAL split.  Early stopping
on the composite score (release_pearson_r + (1 − release_rmse)) / 2 — the
same metric used during tuning.  The TEST split is never touched here; it is
reserved exclusively for generate_results.py.

Outputs
-------
results/<reservoir>/behavioral_cloning/model.pt
    PyTorch checkpoint: model weights, config, best epoch, best val score.

results/<reservoir>/behavioral_cloning/train_log.json
    Per-epoch training history: train loss, val correlation, val RMSE,
    composite val score.  Useful for plotting learning curves.

Usage
-----
# Standard — reads everything from best_config.json
python behavioral_cloning/train.py --reservoir garrison

# Override device (e.g., tuned on GPU cluster, training on local CPU)
python behavioral_cloning/train.py --reservoir garrison --device cpu

# Override seed (run an ensemble member with a different initialisation)
python behavioral_cloning/train.py --reservoir garrison --seed 0
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import yaml

# ---------------------------------------------------------------------------
# Project root on sys.path so sibling packages resolve correctly.
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils.data    import load_reservoir_data
from utils.metrics import rmse, safe_pearsonr
from networks.policy import build_policy_network, VALID_POLICY_TYPES  # VALID_POLICY_TYPES used in _validate_consistency

# ---------------------------------------------------------------------------
# BCConfig and _compute_loss are imported from tune.py — single source of
# truth for the hyperparameter container and loss functions.  train.py is a
# consumer of tune.py's outputs and shares its core definitions.
# ---------------------------------------------------------------------------
from behavioral_cloning.tune import (
    BCConfig,
    _bc_default,
    _compute_loss,
    _resolve_device,
)
from utils.runs import _find_run_folder


# =============================================================================
# Safeguarded config loader
# =============================================================================

def _load_best_config(results_dir: Path, reservoir: str) -> dict:
    """
    Load and validate best_config.json produced by tune.py.

    Performs five checks before returning:
      1. File exists (tune.py was run).
      2. File is valid JSON (not corrupted).
      3. Required top-level keys are present.
      4. Required sub-keys inside "config" are present.
      5. Saved reservoir name matches the --reservoir argument.

    Issues a WARNING (does not exit) if the best_score from tuning was very
    low, suggesting the model did not learn.

    Parameters
    ----------
    results_dir : Path to results/<reservoir>/behavioral_cloning/
    reservoir   : Reservoir name from the CLI.

    Returns
    -------
    dict  Parsed contents of best_config.json.
    """
    path = results_dir / "best_config.json"

    # ------------------------------------------------------------------
    # 1. File existence — most common failure mode
    # ------------------------------------------------------------------
    if not path.exists():
        sys.exit(
            f"\nERROR: best_config.json not found.\n"
            f"  Expected: {path}\n\n"
            f"  tune.py must be run before train.py.  Run:\n"
            f"    python behavioral_cloning/tune.py --reservoir {reservoir}\n"
        )

    # ------------------------------------------------------------------
    # 2. Valid JSON
    # ------------------------------------------------------------------
    try:
        with open(path, "r") as f:
            saved = json.load(f)
    except json.JSONDecodeError as e:
        sys.exit(
            f"\nERROR: best_config.json is not valid JSON.\n"
            f"  Error: {e}\n"
            f"  File:  {path}\n\n"
            f"  The file may be corrupted.  Re-run tune.py to regenerate it.\n"
        )
    except OSError as e:
        sys.exit(f"\nERROR: Cannot read best_config.json.\n  {e}\n")

    # ------------------------------------------------------------------
    # 3. Required top-level keys
    # ------------------------------------------------------------------
    required_top = {"reservoir", "policy_type", "best_score", "config"}
    missing_top  = required_top - set(saved.keys())
    if missing_top:
        sys.exit(
            f"\nERROR: best_config.json is missing required top-level keys: "
            f"{sorted(missing_top)}\n"
            f"  File: {path}\n\n"
            f"  The file may be from an older version of tune.py.  "
            f"Re-run tune.py to regenerate it.\n"
        )

    # ------------------------------------------------------------------
    # 4. Required sub-keys inside "config"
    # ------------------------------------------------------------------
    required_cfg = {
        "state_dim", "action_dim",
        "hidden_dim", "n_hidden_layers", "dropout",
        "lr", "epochs", "batch_size",
        "scheduler_type", "early_stopping_patience",
        "seed", "device",
    }
    cfg_dict    = saved["config"]
    missing_cfg = required_cfg - set(cfg_dict.keys())
    if missing_cfg:
        sys.exit(
            f"\nERROR: best_config.json['config'] is missing keys: "
            f"{sorted(missing_cfg)}\n"
            f"  File: {path}\n\n"
            f"  The file may be from an older version of tune.py.  "
            f"Re-run tune.py to regenerate it.\n"
        )

    # ------------------------------------------------------------------
    # 5. Reservoir name match
    # ------------------------------------------------------------------
    if saved["reservoir"] != reservoir:
        sys.exit(
            f"\nERROR: best_config.json was tuned for reservoir "
            f"'{saved['reservoir']}', but you are training for '{reservoir}'.\n\n"
            f"  These must match.  Run tune.py for this reservoir first:\n"
            f"    python behavioral_cloning/tune.py --reservoir {reservoir}\n"
        )

    # ------------------------------------------------------------------
    # Warning: suspiciously low validation score
    # ------------------------------------------------------------------
    best_score = saved["best_score"]
    if best_score < 0.1:
        print(
            f"\nWARNING: Tuning achieved a very low validation score "
            f"({best_score:.4f}).\n"
            f"  The trained model may not generalise well.\n"
            f"  Consider re-running tune.py with a larger n_trials or a wider "
            f"search space.\n",
            file=sys.stderr,
        )

    return saved


# =============================================================================
# Consistency checks between JSON and current config / data
# =============================================================================

def _validate_consistency(
    saved:      dict,
    res_cfg:    dict,
    data,
    reservoir:  str,
) -> str:
    """
    Cross-check best_config.json against the current reservoir config and data.

    Checks:
      • policy_type matches reservoir config (error if not).
      • state_dim  matches loaded data     (error if not — state variables
        may have changed since tuning).

    Returns
    -------
    str  policy_type string (already validated).
    """
    saved_policy   = saved["policy_type"]
    current_policy = str(res_cfg.get("policy_network", "")).lower().strip()

    # Saved policy type is a known type
    if saved_policy not in VALID_POLICY_TYPES:
        sys.exit(
            f"\nERROR: Invalid policy_type '{saved_policy}' in best_config.json.\n"
            f"  Valid options: {list(VALID_POLICY_TYPES)}\n"
            f"  The file may be corrupted or from an incompatible version.\n"
            f"  Re-run tune.py to regenerate best_config.json:\n"
            f"    python behavioral_cloning/tune.py --reservoir {reservoir}\n"
        )

    # Policy type mismatch between JSON and reservoir config
    if current_policy and saved_policy != current_policy:
        sys.exit(
            f"\nERROR: Policy type mismatch.\n"
            f"  best_config.json : policy_type   = '{saved_policy}'\n"
            f"  Reservoir config : policy_network = '{current_policy}'\n\n"
            f"  These must match.  Either:\n"
            f"    • Re-run tune.py with "
            f"--policy_network {current_policy}\n"
            f"    • Or restore  policy_network: {saved_policy}  "
            f"in configs/reservoirs/{reservoir}.yaml\n"
        )

    # state_dim mismatch — happens if state variables changed after tuning
    saved_dim   = saved["config"]["state_dim"]
    current_dim = data.state_dim
    if saved_dim != current_dim:
        sys.exit(
            f"\nERROR: state_dim mismatch.\n"
            f"  best_config.json : state_dim = {saved_dim}\n"
            f"  Current data     : state_dim = {current_dim}\n\n"
            f"  The state variables or use_month_encoding may have changed "
            f"since tuning.\n"
            f"  Re-run tune.py to obtain a matching best_config.json:\n"
            f"    python behavioral_cloning/tune.py --reservoir {reservoir}\n"
        )

    return saved_policy


# =============================================================================
# Training loop
# =============================================================================

def _train(
    config:      BCConfig,
    data,
    policy_type: str,
) -> tuple:
    """
    Train the policy network with the given hyperparameters.

    Training split  → model updates.
    Validation split → early stopping on composite score.

    The test split is never accessed here.

    Returns
    -------
    best_weights : dict   {layer_name: tensor}  Best model state_dict.
    best_score   : float  Best composite val score achieved.
    best_epoch   : int    Epoch index (0-based) when best score was reached.
    epoch_logs   : list[dict]  Per-epoch metrics (train loss, val corr/rmse/score).
    """
    # ---- Reproducibility ----
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    device = torch.device(config.device)

    # ---- Network and optimiser ----
    model     = build_policy_network(policy_type, config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    if config.scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=50, T_mult=1
        )
    elif config.scheduler_type == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", patience=5, factor=0.5
        )
    else:
        scheduler = None

    # ---- DataLoaders ----
    train_states  = torch.tensor(data.train.states,  dtype=torch.float32)
    train_actions = torch.tensor(data.train.actions, dtype=torch.float32)
    train_dl = DataLoader(
        TensorDataset(train_states, train_actions),
        batch_size = config.batch_size,
        shuffle    = True,
        drop_last  = False,
    )

    val_states_t   = torch.tensor(data.val.states, dtype=torch.float32).to(device)
    val_actions_np = data.val.actions   # numpy (N,) — stays on CPU for metrics

    # ---- Training state ----
    best_score   = -float("inf")
    best_epoch   = 0
    patience_ctr = 0
    best_weights = None
    epoch_logs: list[dict] = []

    print(f"\n  {'Epoch':>5}  {'TrainLoss':>10}  "
          f"{'ValCorr':>8}  {'ValRMSE':>8}  {'Score':>8}")
    print(f"  {'-'*5}  {'-'*10}  {'-'*8}  {'-'*8}  {'-'*8}")

    for epoch in range(config.epochs):
        # ----------------------------------------------------------------
        # Training
        # ----------------------------------------------------------------
        model.train()
        batch_losses: list[float] = []

        for states_b, actions_b in train_dl:
            states_b  = states_b.to(device)
            actions_b = actions_b.to(device)

            optimizer.zero_grad()
            output = model(states_b, deterministic=False)
            loss   = _compute_loss(output, actions_b, policy_type, config)

            # Hard error in final training (tune.py silences these as -inf)
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"\nNaN/Inf loss at epoch {epoch + 1}.\n"
                    f"  This usually means the learning rate is too high or the\n"
                    f"  data contains extreme outliers.\n"
                    f"  Try re-running tune.py with a smaller lr search range."
                )

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            batch_losses.append(loss.item())

        train_loss_avg = float(np.mean(batch_losses))

        if scheduler is not None and config.scheduler_type == "cosine":
            scheduler.step()

        # ----------------------------------------------------------------
        # Validation
        # ----------------------------------------------------------------
        model.eval()
        with torch.no_grad():
            val_out  = model(val_states_t, deterministic=True)
            val_pred = val_out.action.squeeze(1).cpu().numpy()   # (N,)

        val_corr, _ = safe_pearsonr(val_actions_np, val_pred)
        val_rmse_v  = rmse(val_actions_np, val_pred)
        score       = (val_corr + (1.0 - val_rmse_v)) / 2.0

        if scheduler is not None and config.scheduler_type == "plateau":
            scheduler.step(score)

        # ----------------------------------------------------------------
        # Log this epoch
        # ----------------------------------------------------------------
        is_best = score > best_score + 1e-6
        epoch_logs.append({
            "epoch":      epoch,
            "train_loss": round(train_loss_avg, 6),
            "val_corr":   round(float(val_corr),    6),
            "val_rmse":   round(float(val_rmse_v),  6),
            "val_score":  round(float(score),        6),
        })

        # Print every 25 epochs, at epoch 0, and whenever a new best is found
        if is_best or (epoch + 1) % 25 == 0 or epoch == 0:
            marker = "  ◀ best" if is_best else ""
            print(
                f"  {epoch + 1:>5d}  {train_loss_avg:>10.4f}  "
                f"{val_corr:>8.4f}  {val_rmse_v:>8.4f}  {score:>8.4f}{marker}"
            )

        # ----------------------------------------------------------------
        # Early stopping
        # ----------------------------------------------------------------
        if is_best:
            best_score   = score
            best_epoch   = epoch
            patience_ctr = 0
            best_weights = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_ctr += 1
            if patience_ctr >= config.early_stopping_patience:
                print(
                    f"\n  Early stopping at epoch {epoch + 1} "
                    f"(no improvement for {config.early_stopping_patience} epochs)."
                )
                break

    print(
        f"\n  Training complete.  "
        f"Best val score = {best_score:.4f}  at epoch {best_epoch + 1}."
    )

    # Guard: if best_weights is None, no valid checkpoint was ever saved.
    # This can happen if every validation score was NaN (e.g. silent numerical
    # instability that didn't trigger the loss NaN check).
    if best_weights is None:
        raise RuntimeError(
            "\nTraining failed: no valid model checkpoint was saved.\n"
            "  All validation scores were NaN or the score never exceeded -inf.\n"
            "  This usually indicates silent numerical instability.\n"
            "  Try re-running tune.py with a smaller lr range and re-running train.py."
        )

    return best_weights, best_score, best_epoch, epoch_logs


# =============================================================================
# CLI
# =============================================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Train the best Behavioral Cloning policy. "
            "Requires tune.py to have been run first."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--reservoir", required=True,
        help="Reservoir name — must match configs/reservoirs/<name>.yaml.",
    )
    p.add_argument(
        "--device", default=None,
        help=(
            "Override the device stored in best_config.json.  "
            "Useful when moving from a GPU cluster to a local machine.  "
            "Options: auto | cpu | cuda | cuda:N | mps."
        ),
    )
    p.add_argument(
        "--seed", type=int, default=None,
        help=(
            "Override the seed stored in best_config.json.  "
            "Use to train an ensemble member with a different initialisation."
        ),
    )
    p.add_argument(
        "--run_id", type=int, default=None,
        help=(
            "Integer run identifier matching the folder created by tune.py "
            "(e.g. 1 for folder '1_beta').  Required."
        ),
    )
    return p.parse_args()


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args = _parse_args()

    if args.run_id is None:
        sys.exit(
            "\nERROR: --run_id is required for train.py.\n"
            "  Pass the integer run_id created by tune.py, e.g.:\n"
            "    python behavioral_cloning/train.py --reservoir <name> --run_id 1\n"
        )

    res_cfg_path = _ROOT / "configs" / "reservoirs" / f"{args.reservoir}.yaml"
    bc_base_dir  = _ROOT / "results" / args.reservoir / "behavioral_cloning"
    results_dir  = _find_run_folder(bc_base_dir, args.run_id)

    # ------------------------------------------------------------------
    # Reservoir config
    # ------------------------------------------------------------------
    if not res_cfg_path.exists():
        sys.exit(
            f"\nERROR: Reservoir config not found: {res_cfg_path}\n"
            f"  Available: configs/reservoirs/*.yaml\n"
        )

    with open(res_cfg_path, "r") as f:
        res_cfg = yaml.safe_load(f)

    # ------------------------------------------------------------------
    # Load and validate best_config.json
    # ------------------------------------------------------------------
    saved = _load_best_config(results_dir, args.reservoir)

    print(f"\nLoaded best_config.json")
    print(f"  Reservoir  : {saved['reservoir']}")
    print(f"  Policy     : {saved['policy_type']}")
    print(f"  Best score : {saved['best_score']:.4f}  "
          f"(trial #{saved.get('trial_number', '?')})")

    # ------------------------------------------------------------------
    # Load data (also writes normalisation bounds back to res_cfg_path)
    # ------------------------------------------------------------------
    print(f"\nLoading data …")
    data = load_reservoir_data(res_cfg, res_cfg_path)
    print(
        f"  state_dim  = {data.state_dim}\n"
        f"  train rows = {len(data.train.states)}\n"
        f"  val rows   = {len(data.val.states)}\n"
        f"  test rows  = {len(data.test.states)}"
    )

    # ------------------------------------------------------------------
    # Consistency checks (policy type, state_dim)
    # ------------------------------------------------------------------
    policy_type = _validate_consistency(saved, res_cfg, data, args.reservoir)

    # ------------------------------------------------------------------
    # Reconstruct BCConfig from JSON
    # ------------------------------------------------------------------
    cfg_dict = saved["config"]
    config = BCConfig(
        state_dim               = cfg_dict["state_dim"],
        action_dim              = cfg_dict.get("action_dim",              1),
        hidden_dim              = cfg_dict["hidden_dim"],
        n_hidden_layers         = cfg_dict["n_hidden_layers"],
        dropout                 = cfg_dict["dropout"],
        lr                      = cfg_dict["lr"],
        epochs                  = cfg_dict["epochs"],
        batch_size              = cfg_dict["batch_size"],
        scheduler_type          = cfg_dict["scheduler_type"],
        early_stopping_patience = cfg_dict["early_stopping_patience"],
        seed                    = cfg_dict["seed"],
        device                  = cfg_dict["device"],
        # Network-specific — fall back to BCConfig defaults if key absent
        alpha_min       = cfg_dict.get("alpha_min",      _bc_default("alpha_min")),
        alpha_max       = cfg_dict.get("alpha_max",      _bc_default("alpha_max")),
        beta_min        = cfg_dict.get("beta_min",       _bc_default("beta_min")),
        beta_max        = cfg_dict.get("beta_max",       _bc_default("beta_max")),
        sigma_min       = cfg_dict.get("sigma_min",      _bc_default("sigma_min")),
        log_epsilon     = cfg_dict.get("log_epsilon",    _bc_default("log_epsilon")),
        zero_threshold  = cfg_dict.get("zero_threshold", _bc_default("zero_threshold")),
        mse_weight      = cfg_dict.get("mse_weight",     _bc_default("mse_weight")),
        gate_weight     = cfg_dict.get("gate_weight",    _bc_default("gate_weight")),
    )

    # ------------------------------------------------------------------
    # Apply CLI overrides
    # ------------------------------------------------------------------
    if args.device is not None:
        raw_device = args.device
    else:
        raw_device = config.device

    # Resolve 'auto' and handle unavailable devices gracefully
    resolved = _resolve_device(raw_device)
    if resolved.startswith("cuda") and not torch.cuda.is_available():
        print(
            f"\nWARNING: Requested device '{resolved}' but CUDA is not available.  "
            f"Falling back to CPU.\n",
            file=sys.stderr,
        )
        resolved = "cpu"
    elif resolved == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        print(
            f"\nWARNING: Requested device 'mps' but MPS is not available.  "
            f"Falling back to CPU.\n",
            file=sys.stderr,
        )
        resolved = "cpu"

    config.device = resolved

    if args.seed is not None:
        config.seed = args.seed
        print(f"\nSeed overridden to {args.seed} (was {cfg_dict['seed']}).")

    print(f"\nDevice : {config.device}")
    print(f"Seed   : {config.seed}\n")

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    print("Starting training …")
    best_weights, best_score, best_epoch, epoch_logs = _train(
        config, data, policy_type
    )

    # ------------------------------------------------------------------
    # Save model checkpoint
    # ------------------------------------------------------------------
    results_dir.mkdir(parents=True, exist_ok=True)

    model_path = results_dir / "model.pt"
    torch.save(
        {
            "model_state_dict": best_weights,
            "policy_type":      policy_type,
            "best_val_score":   best_score,
            "best_epoch":       best_epoch,
            "config":           asdict(config),
        },
        model_path,
    )
    print(f"\nModel saved  → {model_path}")

    # ------------------------------------------------------------------
    # Save training log
    # ------------------------------------------------------------------
    train_log = {
        "reservoir":      args.reservoir,
        "policy_type":    policy_type,
        "device":         config.device,
        "seed":           config.seed,
        "best_epoch":     best_epoch,
        "best_val_score": round(best_score, 6),
        "total_epochs":   len(epoch_logs),
        "epoch_logs":     epoch_logs,
    }

    log_path = results_dir / "train_log.json"
    with open(log_path, "w") as f:
        json.dump(train_log, f, indent=2)
    print(f"Train log saved → {log_path}\n")

    # ---- Update run_args.json with train arguments ----
    run_args_path = results_dir / "run_args.json"
    run_args: dict = {}
    if run_args_path.exists():
        with open(run_args_path, "r") as f:
            run_args = json.load(f)

    run_args["train"] = {
        "reservoir":      args.reservoir,
        "run_id":         args.run_id,
        "device_cli":     args.device,
        "device_used":    config.device,
        "seed_cli":       args.seed,
        "seed_used":      config.seed,
        "timestamp":      datetime.now().isoformat(timespec="seconds"),
    }
    with open(run_args_path, "w") as f:
        json.dump(run_args, f, indent=2)
    print(f"Run args updated → {run_args_path}\n")


if __name__ == "__main__":
    main()