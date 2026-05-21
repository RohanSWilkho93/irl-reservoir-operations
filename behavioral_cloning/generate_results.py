"""
behavioral_cloning/generate_results.py
=======================================
Evaluate the trained Behavioral Cloning model on the held-out TEST split
and produce publication-quality figures.

MUST be run AFTER behavioral_cloning/train.py.  Loads model.pt from the
results directory.  If model.pt is not found the script exits with a clear
error and instructions.

Monte Carlo rollouts
--------------------
Because all four policy networks are stochastic distributions (Beta,
Lognormal, Hardgating, Softgating), a single deterministic prediction
understates the spread of the learned policy.  This script runs N_MC
independent stochastic forward passes (model.eval(), deterministic=False)
over the full test sequence.

  • Each rollout samples from the output distribution at every timestep.
  • Dropout is disabled (model.eval()) so uncertainty comes exclusively
    from the policy distribution — not from network-weight uncertainty.

Reported metrics
----------------
For each MC rollout the Pearson r and nRMSE are computed against the
observed raw-unit release.  The MEAN and STD across all rollouts are
saved to test_metrics.json and the mean values are shown in the figure
title.

nRMSE is range-based: sqrt(MSE) / (max_observed − min_observed).
It is computed on DENORMALIZED (original engineering-unit) values so the
number is directly interpretable and comparable across reservoirs.

Outputs
-------
results/<reservoir>/behavioral_cloning/release_test.png
    Time-series figure with observed (blue), median MC (red dashed),
    and 25th–75th percentile band (salmon shading).

results/<reservoir>/behavioral_cloning/test_metrics.json
    mean_corr, std_corr, mean_nrmse, std_nrmse across all MC rollouts.

Usage
-----
# Standard
python behavioral_cloning/generate_results.py --reservoir garrison

# Override device and number of MC rollouts
python behavioral_cloning/generate_results.py --reservoir garrison \\
    --device cpu --n_mc 500
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml

# ---------------------------------------------------------------------------
# Project root on sys.path so sibling packages resolve correctly.
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils.data    import load_reservoir_data
from utils.metrics import nrmse, safe_pearsonr
from networks.policy import build_policy_network, VALID_POLICY_TYPES

from behavioral_cloning.tune import (
    BCConfig,
    _bc_default,
    _resolve_device,
)
from utils.runs import _find_run_folder

# ---------------------------------------------------------------------------
# Plot colours (match the example figure)
# ---------------------------------------------------------------------------
_BAND_COLOR     = "#F4A582"   # warm salmon — IQR shading
_OBSERVED_COLOR = "#1565C0"   # navy blue  — observed line
_MEDIAN_COLOR   = "#C0392B"   # crimson    — median MC line


# =============================================================================
# Checkpoint loader
# =============================================================================

def _load_checkpoint(results_dir: Path, reservoir: str) -> dict:
    """
    Load and validate model.pt produced by train.py.

    Checks:
      1. File exists  (train.py was run).
      2. File is loadable by torch.load (not corrupted).
      3. Required keys are present in the checkpoint.
      4. policy_type is a known value.

    Parameters
    ----------
    results_dir : Path to results/<reservoir>/behavioral_cloning/
    reservoir   : Reservoir name from the CLI (for error messages).

    Returns
    -------
    dict  Checkpoint as loaded by torch.load (tensors on CPU).
    """
    path = results_dir / "model.pt"

    # ------------------------------------------------------------------
    # 1. File existence
    # ------------------------------------------------------------------
    if not path.exists():
        sys.exit(
            f"\nERROR: model.pt not found.\n"
            f"  Expected: {path}\n\n"
            f"  train.py must be run before generate_results.py.  Run:\n"
            f"    python behavioral_cloning/train.py --reservoir {reservoir}\n"
        )

    # ------------------------------------------------------------------
    # 2. Loadable
    # ------------------------------------------------------------------
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as e:
        sys.exit(
            f"\nERROR: Cannot load model.pt.\n"
            f"  Error: {e}\n"
            f"  File:  {path}\n\n"
            f"  The file may be corrupted.  Re-run train.py to regenerate it.\n"
        )

    # ------------------------------------------------------------------
    # 3. Required keys
    # ------------------------------------------------------------------
    required = {"model_state_dict", "policy_type", "config",
                "best_val_score", "best_epoch"}
    missing  = required - set(ckpt.keys())
    if missing:
        sys.exit(
            f"\nERROR: model.pt is missing required keys: {sorted(missing)}\n"
            f"  File: {path}\n\n"
            f"  The checkpoint may be from an older version of train.py.  "
            f"Re-run train.py.\n"
        )

    required_cfg = {"state_dim", "hidden_dim", "n_hidden_layers", "dropout",
                    "lr", "epochs", "batch_size", "scheduler_type",
                    "early_stopping_patience", "seed", "device"}
    cfg_dict     = ckpt["config"]
    missing_cfg  = required_cfg - set(cfg_dict.keys())
    if missing_cfg:
        sys.exit(
            f"\nERROR: model.pt['config'] is missing keys: {sorted(missing_cfg)}\n"
            f"  Re-run train.py to regenerate the checkpoint.\n"
        )

    # ------------------------------------------------------------------
    # 4. Known policy type
    # ------------------------------------------------------------------
    policy_type = str(ckpt["policy_type"]).lower().strip()
    if policy_type not in VALID_POLICY_TYPES:
        sys.exit(
            f"\nERROR: Invalid policy_type '{policy_type}' in model.pt.\n"
            f"  Valid options: {list(VALID_POLICY_TYPES)}\n"
            f"  The checkpoint may be corrupted.  Re-run train.py.\n"
        )

    return ckpt


# =============================================================================
# Consistency checks
# =============================================================================

def _validate_consistency(
    ckpt:       dict,
    res_cfg:    dict,
    data,
    reservoir:  str,
) -> str:
    """
    Cross-check model.pt against the current reservoir config and loaded data.

    Errors on:
      • policy_type in checkpoint does not match reservoir config.
      • state_dim in checkpoint does not match loaded data.

    Returns
    -------
    str  Validated policy_type string.
    """
    saved_policy   = str(ckpt["policy_type"]).lower().strip()
    current_policy = str(res_cfg.get("policy_network", "")).lower().strip()

    if current_policy and saved_policy != current_policy:
        sys.exit(
            f"\nERROR: Policy type mismatch.\n"
            f"  model.pt         : policy_type   = '{saved_policy}'\n"
            f"  Reservoir config : policy_network = '{current_policy}'\n\n"
            f"  Re-run train.py after resolving the mismatch:\n"
            f"    python behavioral_cloning/train.py --reservoir {reservoir}\n"
        )

    saved_dim   = ckpt["config"]["state_dim"]
    current_dim = data.state_dim
    if saved_dim != current_dim:
        sys.exit(
            f"\nERROR: state_dim mismatch.\n"
            f"  model.pt     : state_dim = {saved_dim}\n"
            f"  Current data : state_dim = {current_dim}\n\n"
            f"  State variables may have changed since training.  "
            f"Re-run train.py:\n"
            f"    python behavioral_cloning/train.py --reservoir {reservoir}\n"
        )

    return saved_policy


# =============================================================================
# Model builder
# =============================================================================

def _build_model(ckpt: dict, device: torch.device) -> tuple:
    """
    Reconstruct the policy network from a checkpoint and load saved weights.

    Parameters
    ----------
    ckpt   : Checkpoint dict from _load_checkpoint.
    device : Target torch.device.

    Returns
    -------
    model       : nn.Module  Loaded, eval-mode policy network on `device`.
    policy_type : str
    config      : BCConfig
    """
    cfg_dict    = ckpt["config"]
    policy_type = str(ckpt["policy_type"]).lower().strip()

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
        device                  = str(device),
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

    model = build_policy_network(policy_type, config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()   # Dropout off — stochasticity comes from the distribution

    return model, policy_type, config


# =============================================================================
# Monte Carlo rollouts
# =============================================================================

def _run_mc(
    model,
    test_states_t: torch.Tensor,
    n_mc:          int,
    device:        torch.device,
) -> np.ndarray:
    """
    Run n_mc independent stochastic forward passes over the test sequence.

    model.eval() is already set by _build_model.  deterministic=False causes
    each call to sample from the output distribution (Beta / Lognormal / gate
    × Beta), so consecutive rollouts differ even though network weights are
    fixed.

    Seeded with 42 for reproducibility.

    Parameters
    ----------
    model         : Trained policy network in eval mode.
    test_states_t : Normalised test states, shape (N, state_dim), on `device`.
    n_mc          : Number of MC rollouts (default 500).
    device        : torch.device.

    Returns
    -------
    np.ndarray  shape (n_mc, N)  — normalised [0, 1] predictions.
    """
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)   # seed CUDA RNG for reproducible sampling on GPU

    # Guard: NaN test states (missing values in raw data) propagate through the
    # network and crash the Beta distribution.  Warn and fill with 0.0 so the
    # rollout completes; affected timesteps will produce near-uniform predictions.
    nan_mask = torch.isnan(test_states_t).any(dim=1)
    if nan_mask.any():
        n_nan = int(nan_mask.sum().item())
        print(
            f"\nWARNING: {n_nan} test timestep(s) contain NaN state values "
            f"(likely missing data in the CSV).  Filling with 0.0 for inference.\n"
            f"  Affected timestep indices: "
            f"{torch.where(nan_mask)[0].tolist()[:10]}"
            f"{'...' if n_nan > 10 else ''}\n",
            file=__import__("sys").stderr,
        )
        test_states_t = torch.nan_to_num(test_states_t, nan=0.0)

    N    = test_states_t.shape[0]
    preds = np.empty((n_mc, N), dtype=np.float32)

    with torch.no_grad():
        for i in range(n_mc):
            out      = model(test_states_t, deterministic=False)
            preds[i] = out.action.squeeze(1).cpu().numpy()

    return preds   # (n_mc, N)  normalised


# =============================================================================
# Metrics
# =============================================================================

def _compute_metrics(
    mc_preds_raw:  np.ndarray,
    raw_observed:  np.ndarray,
) -> dict:
    """
    Compute Pearson r and nRMSE for each MC rollout, then aggregate.

    Both metrics are computed on DENORMALIZED (engineering-unit) values.
    nRMSE = sqrt(MSE) / (max_observed − min_observed).

    Parameters
    ----------
    mc_preds_raw : (n_mc, N)  Predictions in original engineering units.
    raw_observed : (N,)       Observed values in original engineering units.

    Returns
    -------
    dict with keys: mean_corr, std_corr, mean_nrmse, std_nrmse.
    """
    n_mc    = mc_preds_raw.shape[0]
    corrs   = np.empty(n_mc)
    nrmses  = np.empty(n_mc)

    for i in range(n_mc):
        corrs[i]  = safe_pearsonr(raw_observed, mc_preds_raw[i])[0]
        nrmses[i] = nrmse(raw_observed, mc_preds_raw[i])

    return {
        "mean_corr":  float(np.mean(corrs)),
        "std_corr":   float(np.std(corrs)),
        "mean_nrmse": float(np.mean(nrmses)),
        "std_nrmse":  float(np.std(nrmses)),
    }


# =============================================================================
# Plot
# =============================================================================

def _plot_and_save(
    raw_observed:  np.ndarray,
    mc_preds_raw:  np.ndarray,
    metrics:       dict,
    dates,
    action_col:    str,
    reservoir:     str,
    save_path:     Path,
) -> None:
    """
    Produce a time-series figure showing observed vs. MC rollout ensemble.

    Figure elements
    ---------------
    Blue solid line       : Observed release.
    Salmon shaded band    : 25th–75th percentile of MC rollouts (IQR).
    Red dashed line       : Median of MC rollouts.

    Title contains the reservoir name, mean Pearson r and mean nRMSE.

    Parameters
    ----------
    raw_observed : (N,)       Observed values in engineering units.
    mc_preds_raw : (n_mc, N)  MC rollout predictions in engineering units.
    metrics      : dict from _compute_metrics.
    dates        : pd.DatetimeIndex  Test period dates (for x-axis labels).
    action_col   : str  Name of the action column (e.g. "release").
    reservoir    : str  Reservoir name (used in title).
    save_path    : Path to save the PNG file.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")          # non-interactive backend — safe for HPC
        import matplotlib.pyplot as plt
    except ImportError:
        sys.exit(
            "\nERROR: matplotlib is required for plotting.\n"
            "  Install it with:  pip install matplotlib\n"
        )

    N = len(raw_observed)
    x = np.arange(N)

    # ---- Summary statistics across MC rollouts ----
    median_raw = np.median(mc_preds_raw, axis=0)   # (N,)
    q25        = np.percentile(mc_preds_raw, 25, axis=0)
    q75        = np.percentile(mc_preds_raw, 75, axis=0)

    # ---- Infer time-step unit for x-axis label ----
    if len(dates) > 1:
        delta_days = (dates[1] - dates[0]).days
        time_unit  = "Days" if delta_days <= 3 else "Months"
    else:
        time_unit = "Steps"
    x_label = f"Time Steps ({time_unit})"

    # ---- Y-axis label ----
    y_label = f"{action_col.capitalize()} (m³/s)"

    # ---- Title ----
    mean_corr  = metrics["mean_corr"]
    mean_nrmse = metrics["mean_nrmse"]
    title = (
        f"{action_col.capitalize()}\n"
        f"$r$ = {mean_corr:.3f},  nRMSE = {mean_nrmse:.3f}"
    )

    # ---- Figure ----
    fig, ax = plt.subplots(figsize=(14, 4))

    # IQR shading (drawn first, lowest z-order)
    ax.fill_between(
        x, q25, q75,
        color=_BAND_COLOR, alpha=0.6,
        label="25th–75th percentile (IQR)",
        zorder=1,
    )

    # Median MC line
    ax.plot(
        x, median_raw,
        color=_MEDIAN_COLOR, linestyle="--", linewidth=1.5,
        label="Median (MC rollouts)",
        zorder=2,
    )

    # Observed (top layer)
    ax.plot(
        x, raw_observed,
        color=_OBSERVED_COLOR, linestyle="-", linewidth=1.5,
        label="Observed",
        zorder=3,
    )

    # ---- Formatting ----
    ax.set_title(title, fontsize=16, fontweight="bold", pad=10)
    ax.set_xlabel(x_label, fontsize=14)
    ax.set_ylabel(y_label, fontsize=14)
    ax.tick_params(axis="both", labelsize=12)

    # Light grey grid
    ax.grid(True, color="grey", linewidth=0.4, alpha=0.35, zorder=0)
    ax.set_axisbelow(True)

    # Legend — order: Observed, IQR, Median (matches visual top-to-bottom)
    handles, labels = ax.get_legend_handles_labels()
    order   = [labels.index("Observed"),
               labels.index("25th–75th percentile (IQR)"),
               labels.index("Median (MC rollouts)")]
    ax.legend(
        [handles[i] for i in order],
        [labels[i]  for i in order],
        fontsize=12, loc="upper right",
        framealpha=0.9, edgecolor="grey",
    )

    ax.set_xlim(0, N - 1)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Figure saved  → {save_path}")


# =============================================================================
# CLI
# =============================================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Evaluate the trained BC model on the test split and produce "
            "figures.  Requires train.py to have been run first."
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
            "Compute device.  Defaults to the device stored in model.pt.  "
            "Options: auto | cpu | cuda | cuda:N | mps."
        ),
    )
    p.add_argument(
        "--n_mc", type=int, default=500,
        help="Number of Monte Carlo rollouts for the ensemble.",
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

    if args.n_mc < 1:
        sys.exit("\nERROR: --n_mc must be a positive integer.\n")

    if args.run_id is None:
        sys.exit(
            "\nERROR: --run_id is required for generate_results.py.\n"
            "  Pass the integer run_id created by tune.py, e.g.:\n"
            "    python behavioral_cloning/generate_results.py "
            "--reservoir <name> --run_id 1\n"
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

    action_col = str(res_cfg["columns"]["action"])

    # ------------------------------------------------------------------
    # Load and validate checkpoint
    # ------------------------------------------------------------------
    ckpt = _load_checkpoint(results_dir, args.reservoir)

    print(f"\nLoaded model.pt")
    print(f"  Reservoir      : {args.reservoir}")
    print(f"  Policy         : {ckpt['policy_type']}")
    print(f"  Best val score : {ckpt['best_val_score']:.4f}  "
          f"(epoch {ckpt['best_epoch'] + 1})")

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print(f"\nLoading data …")
    data = load_reservoir_data(res_cfg, res_cfg_path)
    print(
        f"  state_dim  = {data.state_dim}\n"
        f"  test rows  = {len(data.test.states)}"
    )

    # ------------------------------------------------------------------
    # Consistency checks
    # ------------------------------------------------------------------
    policy_type = _validate_consistency(ckpt, res_cfg, data, args.reservoir)

    # ------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------
    raw_device = args.device if args.device is not None else ckpt["config"]["device"]
    resolved   = _resolve_device(raw_device)

    if resolved.startswith("cuda") and not torch.cuda.is_available():
        print(
            f"\nWARNING: Requested device '{resolved}' but CUDA is not "
            f"available.  Falling back to CPU.\n",
            file=sys.stderr,
        )
        resolved = "cpu"
    elif resolved == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        print(
            "\nWARNING: Requested device 'mps' but MPS is not available.  "
            "Falling back to CPU.\n",
            file=sys.stderr,
        )
        resolved = "cpu"

    device = torch.device(resolved)
    print(f"\nDevice : {device}")

    # ------------------------------------------------------------------
    # Build model
    # ------------------------------------------------------------------
    model, policy_type, config = _build_model(ckpt, device)

    # ------------------------------------------------------------------
    # Prepare test tensors
    # ------------------------------------------------------------------
    test_states_t = torch.tensor(
        data.test.states, dtype=torch.float32
    ).to(device)

    raw_observed = data.test.raw_actions   # (N,) original engineering units

    # ------------------------------------------------------------------
    # Monte Carlo rollouts
    # ------------------------------------------------------------------
    print(f"\nRunning {args.n_mc} MC rollouts …", end="", flush=True)

    mc_preds_norm = _run_mc(model, test_states_t, args.n_mc, device)
    # Shape: (n_mc, N)  — normalised [0, 1]

    # Denormalize each rollout to original engineering units
    mc_preds_raw = np.stack([
        data.normalizer.denormalize(action_col, mc_preds_norm[i])
        for i in range(args.n_mc)
    ])   # (n_mc, N) in original units

    print(" done.")

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    print("Computing metrics …", end="", flush=True)
    metrics = _compute_metrics(mc_preds_raw, raw_observed)
    print(" done.")

    print(
        f"\n  mean r     = {metrics['mean_corr']:.4f}  "
        f"(±{metrics['std_corr']:.4f})\n"
        f"  mean nRMSE = {metrics['mean_nrmse']:.4f}  "
        f"(±{metrics['std_nrmse']:.4f})"
    )

    # ------------------------------------------------------------------
    # Save metrics JSON
    # ------------------------------------------------------------------
    results_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = results_dir / "test_metrics.json"
    metrics_out  = {
        "reservoir":  args.reservoir,
        "policy_type": policy_type,
        "n_mc":        args.n_mc,
        "metrics": {
            "release_corr_mean":  round(metrics["mean_corr"],  6),
            "release_corr_std":   round(metrics["std_corr"],   6),
            "release_nrmse_mean": round(metrics["mean_nrmse"], 6),
            "release_nrmse_std":  round(metrics["std_nrmse"],  6),
        },
    }
    with open(metrics_path, "w") as f:
        json.dump(metrics_out, f, indent=2)
    print(f"\nMetrics saved → {metrics_path}")

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    plot_path = results_dir / "release_test.png"
    _plot_and_save(
        raw_observed  = raw_observed,
        mc_preds_raw  = mc_preds_raw,
        metrics       = metrics,
        dates         = data.test.dates,
        action_col    = action_col,
        reservoir     = args.reservoir,
        save_path     = plot_path,
    )

    # ---- Update run_args.json with generate_results arguments ----
    run_args_path = results_dir / "run_args.json"
    run_args: dict = {}
    if run_args_path.exists():
        with open(run_args_path, "r") as f:
            run_args = json.load(f)

    run_args["generate_results"] = {
        "reservoir":    args.reservoir,
        "run_id":       args.run_id,
        "device_cli":   args.device,
        "device_used":  resolved,
        "n_mc":         args.n_mc,
        "timestamp":    datetime.now().isoformat(timespec="seconds"),
    }
    with open(run_args_path, "w") as f:
        json.dump(run_args, f, indent=2)
    print(f"Run args updated → {run_args_path}\n")


if __name__ == "__main__":
    main()
