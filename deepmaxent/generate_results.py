"""
deepmaxent/generate_results.py
================================
Evaluate the trained Deep MaxEnt IRL model on the held-out TEST split
and produce publication-quality figures.

MUST be run AFTER deepmaxent/train.py.  Loads model.pt from the run folder.
If model.pt is not found the script exits with a clear error and instructions.

Monte Carlo rollouts
--------------------
The learned softmax policy is stochastic: at each timestep the action is
sampled from Pi[state, month, :].  This script runs n_mc independent
trajectory simulations (n_sims=1 per call to monte_carlo_simulate) so that
the per-timestep spread can be visualised as an IQR band.

  • Each rollout samples actions stochastically from the policy at every step.
  • The inflow sequence follows the observed expert record (teacher-forcing).
  • Storage evolves via the water-balance equation.

Reported metrics
----------------
Metrics are computed against the mean MC trajectory (averaged over n_mc runs).
SAVF diff is computed once from the full Pi on the test split.

Outputs
-------
results/<reservoir>/deepmaxent/<run_id>/
    test_metrics.json   — release/storage corr, nRMSE, MAE; SAVF diff / overlap.
    release_test.png    — observed (blue) vs median MC (red) + IQR band (salmon).
    storage_test.png    — same style for reservoir storage.
    run_args.json       — appended generate_results section.

Usage
-----
# Standard
python deepmaxent/generate_results.py --reservoir conchas --run_id 1

# Override device and rollout count
python deepmaxent/generate_results.py --reservoir conchas --run_id 1 \\
    --device cpu --n_mc 50
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
import torch
import yaml

# ---------------------------------------------------------------------------
# Repo root on sys.path so sibling packages resolve correctly.
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from deepmaxent.core import (
    DeepMaxEntConfig,
    MaxEntTrainer,
    build_inflow_transitions,
    build_transition_matrix,
    create_spaces,
    create_trajectories,
    load_and_split_data,
)
from utils.metrics import nrmse, safe_pearsonr
from utils.runs import _find_run_folder

# ---------------------------------------------------------------------------
# Plot colours (consistent with other generate_results modules)
# ---------------------------------------------------------------------------
_BAND_COLOR     = "#F4A582"   # warm salmon — IQR shading
_OBSERVED_COLOR = "#1565C0"   # navy blue   — observed line
_MEDIAN_COLOR   = "#C0392B"   # crimson     — median MC line

# ---------------------------------------------------------------------------
# Default MC rollout count
# ---------------------------------------------------------------------------
_N_MC_DEFAULT = 50


# =============================================================================
# Device resolution
# =============================================================================

def _resolve_device(device_str: str) -> str:
    """
    Resolve a device string to a concrete torch device string.

    "auto" → "cuda" if available, else "cpu".
    All other values are passed through unchanged.
    """
    if device_str == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_str


# =============================================================================
# Checkpoint loader
# =============================================================================

def _load_checkpoint(results_dir: Path, reservoir: str) -> dict:
    """
    Load and validate model.pt produced by deepmaxent/train.py.

    Checks:
      1. File exists  (train.py was run).
      2. File is loadable by torch.load (not corrupted).
      3. Required keys are present.

    Parameters
    ----------
    results_dir : Path to results/<reservoir>/deepmaxent/<run_id>/
    reservoir   : Reservoir name (for error messages).

    Returns
    -------
    dict  Checkpoint as loaded by torch.load (tensors on CPU).
    """
    path = results_dir / "model.pt"

    # 1. File existence
    if not path.exists():
        sys.exit(
            f"\nERROR: model.pt not found.\n"
            f"  Expected: {path}\n\n"
            f"  train.py must be run before generate_results.py.  Run:\n"
            f"    python deepmaxent/train.py --reservoir {reservoir}\n"
        )

    # 2. Loadable
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:
        sys.exit(
            f"\nERROR: Cannot load model.pt.\n"
            f"  Error: {e}\n"
            f"  File:  {path}\n\n"
            f"  The file may be corrupted.  Re-run train.py to regenerate it.\n"
        )

    # 3. Required keys
    required = {
        "model_state_dict",
        "config",
        "best_epoch",
        "best_val_s_deepmaxent",
        "best_val_savf_diff",
    }
    missing = required - set(ckpt.keys())
    if missing:
        sys.exit(
            f"\nERROR: model.pt is missing required keys: {sorted(missing)}\n"
            f"  File: {path}\n\n"
            f"  The checkpoint may be from an older version of train.py.  "
            f"Re-run train.py.\n"
        )

    return ckpt


# =============================================================================
# Plot
# =============================================================================

def _plot_and_save(
    raw_observed:  np.ndarray,
    mc_preds_raw:  np.ndarray,
    corr:          float,
    nrmse_val:     float,
    col_label:     str,
    save_path:     Path,
) -> None:
    """
    Produce a time-series figure showing observed vs. MC rollout ensemble.

    Figure elements
    ---------------
    Blue solid line    : Observed series.
    Salmon shaded band : 25th–75th percentile of MC rollouts (IQR).
    Red dashed line    : Median of MC rollouts.

    Title contains Pearson r and nRMSE computed against the mean MC trajectory.

    Parameters
    ----------
    raw_observed  : (N,)       Observed values in engineering units.
    mc_preds_raw  : (n_mc, N)  MC rollout predictions in engineering units.
    corr          : Pearson r (mean MC vs. observed).
    nrmse_val     : range-normalised RMSE (mean MC vs. observed).
    col_label     : "release" or "storage" — used for axis labels and title.
    save_path     : Path to save the PNG file.
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

    median_raw = np.median(mc_preds_raw, axis=0)   # (N,)
    q25        = np.percentile(mc_preds_raw, 25, axis=0)
    q75        = np.percentile(mc_preds_raw, 75, axis=0)

    # Y-axis label: storage in Mm³, release in m³/s
    y_unit = "Mm³" if col_label == "storage" else "m³/s"

    title = (
        f"{col_label.capitalize()}\n"
        f"$r$ = {corr:.3f},  nRMSE = {nrmse_val:.3f}"
    )

    fig, ax = plt.subplots(figsize=(14, 4))

    # IQR shading (lowest z-order)
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

    # Formatting
    ax.set_title(title, fontsize=16, fontweight="bold", pad=10)
    ax.set_xlabel("Time Steps (Days)", fontsize=14)
    ax.set_ylabel(f"{col_label.capitalize()} ({y_unit})", fontsize=14)
    ax.tick_params(axis="both", labelsize=12)

    ax.grid(True, color="grey", linewidth=0.4, alpha=0.35, zorder=0)
    ax.set_axisbelow(True)

    # Legend — order: Observed, IQR, Median
    handles, labels = ax.get_legend_handles_labels()
    order = [
        labels.index("Observed"),
        labels.index("25th–75th percentile (IQR)"),
        labels.index("Median (MC rollouts)"),
    ]
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
            "Evaluate the trained Deep MaxEnt IRL model on the test split "
            "and produce figures.  Requires train.py to have been run first."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--reservoir", required=True,
        help="Reservoir name — must match configs/reservoirs/<name>.yaml.",
    )
    p.add_argument(
        "--run_id", type=int, required=True,
        help=(
            "Integer run identifier matching the folder created by tune.py "
            "(e.g. 1 for folder '1_softgating').  Required."
        ),
    )
    p.add_argument(
        "--device", default=None,
        help=(
            "Compute device.  Defaults to CPU.  "
            "Options: auto | cpu | cuda | cuda:N | mps."
        ),
    )
    p.add_argument(
        "--n_mc", type=int, default=_N_MC_DEFAULT,
        help="Number of independent MC trajectory rollouts for plots/metrics.",
    )
    p.add_argument(
        "--data_path", default=None,
        help="Override data_path from reservoir YAML.",
    )
    return p.parse_args()


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args = _parse_args()

    if args.n_mc < 1:
        sys.exit("\nERROR: --n_mc must be a positive integer.\n")

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------
    res_cfg_path = _ROOT / "configs" / "reservoirs" / f"{args.reservoir}.yaml"
    dm_base_dir  = _ROOT / "results" / args.reservoir / "deepmaxent"
    results_dir  = _find_run_folder(dm_base_dir, args.run_id)

    if not res_cfg_path.exists():
        sys.exit(
            f"\nERROR: Reservoir config not found: {res_cfg_path}\n"
            f"  Available: configs/reservoirs/*.yaml\n"
        )

    # ------------------------------------------------------------------
    # Reservoir config
    # ------------------------------------------------------------------
    with open(res_cfg_path, "r") as f:
        res_cfg = yaml.safe_load(f)

    date_col    = str(res_cfg["columns"]["date"])
    storage_col = str(res_cfg["columns"]["state"][0])
    inflow_col  = str(res_cfg["columns"]["state"][1])
    action_col  = str(res_cfg["columns"]["action"])
    n_train     = int(res_cfg["split"]["train"])
    n_val       = int(res_cfg["split"]["val"])
    n_test      = int(res_cfg["split"]["test"])

    # Resolve data path — handle Windows backslash in YAML on Linux
    if args.data_path is not None:
        data_path = args.data_path
    else:
        raw = str(res_cfg["data_path"]).replace("\\", "/")
        data_path = str(_ROOT / raw)

    # ------------------------------------------------------------------
    # Load checkpoint
    # ------------------------------------------------------------------
    print(f"\nLoading checkpoint …")
    ckpt = _load_checkpoint(results_dir, args.reservoir)
    cfg  = DeepMaxEntConfig.from_dict(ckpt["config"])

    print(f"  Reservoir         : {args.reservoir}")
    print(f"  Best epoch        : {ckpt['best_epoch']}")
    if ckpt.get("best_val_s_deepmaxent") is not None:
        print(f"  Best val S        : {ckpt['best_val_s_deepmaxent']:.4f}")
    if ckpt.get("best_val_savf_diff") is not None:
        print(f"  Best val SAVF diff: {ckpt['best_val_savf_diff']:.4f}")
    print(f"  use_month_encoding: {cfg.use_month_encoding}")
    print(f"  reward_features   : {cfg.reward_features}")

    # ------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------
    raw_device = args.device if args.device is not None else "cpu"
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
    # Load and split data
    # ------------------------------------------------------------------
    print(f"\nLoading data …")
    (_, train_data, _, test_data,
     train_years, _, test_years) = load_and_split_data(
        data_path, date_col, n_train, n_val, n_test,
    )
    print(
        f"  Train years : {train_years[0]}–{train_years[-1]}  "
        f"({len(train_data)} rows)"
    )
    print(
        f"  Test years  : {test_years[0]}–{test_years[-1]}  "
        f"({len(test_data)} rows)"
    )

    # ------------------------------------------------------------------
    # Build MDP structures from training data
    # ------------------------------------------------------------------
    print("\nBuilding MDP …", end="", flush=True)

    s_space, r_space, i_space = create_spaces(
        train_data, cfg, storage_col, action_col, inflow_col,
    )
    train_trajs_d, train_trajs_raw, s_map, r_map, i_map = create_trajectories(
        train_data, s_space, r_space, i_space, cfg,
        storage_col, action_col, inflow_col,
    )
    test_trajs_d, test_trajs_raw, _, _, _ = create_trajectories(
        test_data, s_space, r_space, i_space, cfg,
        storage_col, action_col, inflow_col,
    )

    inflow_trans = build_inflow_transitions(
        train_data, i_space, i_map, cfg, inflow_col,
    )
    n_months = 12 if cfg.use_month_encoding else 1
    P, n_s_bins = build_transition_matrix(
        s_space, r_space, i_space, inflow_trans, n_months=n_months,
    )
    print(" done.")
    print(
        f"  State space  : {len(s_space)} storage × {len(i_space)} inflow "
        f"= {len(s_space) * len(i_space)} states"
    )
    print(
        f"  Action space : {len(r_space)} release bins"
    )
    print(f"  Months       : {n_months}")

    # ------------------------------------------------------------------
    # Build trainer and load saved reward-network weights
    # ------------------------------------------------------------------
    print("\nBuilding MaxEntTrainer …", end="", flush=True)
    trainer = MaxEntTrainer(
        cfg          = cfg,
        P            = P,
        trajs        = train_trajs_d,
        trajs_raw    = train_trajs_raw,
        s_space      = s_space,
        r_space      = r_space,
        i_space      = i_space,
        s_map        = s_map,
        r_map        = r_map,
        i_map        = i_map,
        n_s_bins     = n_s_bins,
        inflow_trans = inflow_trans,
        device       = device,
        verbose      = False,
    )
    trainer.r_net.load_state_dict(ckpt["model_state_dict"])
    trainer.r_net.to(device)
    trainer.r_net.eval()
    print(" done.")

    # ------------------------------------------------------------------
    # Compute reward table and policy (once)
    # ------------------------------------------------------------------
    print("Computing reward table and policy …", end="", flush=True)
    R  = trainer._calc_rewards()
    Pi = trainer._solve_mdp(R)
    print(" done.")

    # ------------------------------------------------------------------
    # SAVF evaluation on test split
    # ------------------------------------------------------------------
    print("Evaluating test SAVF …", end="", flush=True)
    test_savf_diff, test_savf_overlap = trainer.evaluate_savf(test_trajs_d, Pi=Pi)
    print(
        f" done.  "
        f"(savf_diff={test_savf_diff:.4f}, overlap={test_savf_overlap:.2f}%)"
    )

    # ------------------------------------------------------------------
    # Monte Carlo rollouts — n_mc independent trajectories
    #
    # Each call to monte_carlo_simulate with n_sims=1 produces one
    # independent stochastic trajectory sampled from Pi.  Collecting
    # n_mc such trajectories gives the full ensemble for IQR plotting.
    # ------------------------------------------------------------------
    np.random.seed(42)
    print(f"\nRunning {args.n_mc} MC rollouts …", end="", flush=True)

    mc_release_list: List[np.ndarray] = []
    mc_storage_list: List[np.ndarray] = []

    for _ in range(args.n_mc):
        res = trainer.monte_carlo_simulate(
            test_trajs_d, test_trajs_raw, n_sims=1, Pi=Pi,
        )
        mc_release_list.append(res["sim_release"])
        mc_storage_list.append(res["sim_storage"])

    mc_release_mat = np.stack(mc_release_list)   # (n_mc, N)
    mc_storage_mat = np.stack(mc_storage_list)   # (n_mc, N)

    # Observed expert sequences from raw test trajectories
    # Raw row layout: [storage(0), month(1), release(2), net_inflow(3), ...]
    expert_release = np.concatenate(
        [[row[2] for row in traj] for traj in test_trajs_raw]
    )
    expert_storage = np.concatenate(
        [[row[0] for row in traj] for traj in test_trajs_raw]
    )
    print(" done.")

    # ------------------------------------------------------------------
    # Metrics (mean MC trajectory vs. observed)
    # ------------------------------------------------------------------
    print("Computing metrics …", end="", flush=True)

    mean_release = mc_release_mat.mean(axis=0)
    mean_storage = mc_storage_mat.mean(axis=0)

    release_corr,  _ = safe_pearsonr(expert_release, mean_release)
    storage_corr,  _ = safe_pearsonr(expert_storage, mean_storage)
    release_nrmse_val = float(nrmse(expert_release, mean_release))
    storage_nrmse_val = float(nrmse(expert_storage, mean_storage))
    release_mae       = float(np.mean(np.abs(expert_release - mean_release)))
    storage_mae       = float(np.mean(np.abs(expert_storage - mean_storage)))

    release_corr = float(release_corr)
    storage_corr = float(storage_corr)
    print(" done.")

    print(
        f"\n  release r          = {release_corr:.4f}\n"
        f"  release nRMSE      = {release_nrmse_val:.4f}\n"
        f"  release MAE        = {release_mae:.4f}\n"
        f"  storage r          = {storage_corr:.4f}\n"
        f"  storage nRMSE      = {storage_nrmse_val:.4f}\n"
        f"  storage MAE        = {storage_mae:.4f}\n"
        f"  test SAVF diff     = {test_savf_diff:.4f}\n"
        f"  test SAVF overlap  = {test_savf_overlap:.2f}%"
    )

    # ------------------------------------------------------------------
    # Save test_metrics.json
    # ------------------------------------------------------------------
    results_dir.mkdir(parents=True, exist_ok=True)

    metrics_out = {
        "reservoir": args.reservoir,
        "run_id":    args.run_id,
        "n_mc":      args.n_mc,
        "metrics": {
            "release_corr":           round(release_corr,       6),
            "release_nrmse":          round(release_nrmse_val,  6),
            "release_mae":            round(release_mae,        6),
            "storage_corr":           round(storage_corr,       6),
            "storage_nrmse":          round(storage_nrmse_val,  6),
            "storage_mae":            round(storage_mae,        6),
            "test_savf_diff":         round(float(test_savf_diff),    6),
            "test_savf_overlap_pct":  round(float(test_savf_overlap), 4),
        },
    }
    metrics_path = results_dir / "test_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_out, f, indent=2)
    print(f"\nMetrics saved → {metrics_path}")

    # ------------------------------------------------------------------
    # release_test.png
    # ------------------------------------------------------------------
    _plot_and_save(
        raw_observed = expert_release,
        mc_preds_raw = mc_release_mat,
        corr         = release_corr,
        nrmse_val    = release_nrmse_val,
        col_label    = "release",
        save_path    = results_dir / "release_test.png",
    )

    # ------------------------------------------------------------------
    # storage_test.png
    # ------------------------------------------------------------------
    _plot_and_save(
        raw_observed = expert_storage,
        mc_preds_raw = mc_storage_mat,
        corr         = storage_corr,
        nrmse_val    = storage_nrmse_val,
        col_label    = "storage",
        save_path    = results_dir / "storage_test.png",
    )

    # ------------------------------------------------------------------
    # Update run_args.json (append generate_results section)
    # ------------------------------------------------------------------
    run_args_path = results_dir / "run_args.json"
    run_args: dict = {}
    if run_args_path.exists():
        with open(run_args_path, "r") as f:
            run_args = json.load(f)

    run_args["generate_results"] = {
        "reservoir":   args.reservoir,
        "run_id":      args.run_id,
        "device_cli":  args.device,
        "device_used": resolved,
        "n_mc":        args.n_mc,
        "timestamp":   datetime.now().isoformat(timespec="seconds"),
    }
    with open(run_args_path, "w") as f:
        json.dump(run_args, f, indent=2)
    print(f"Run args updated → {run_args_path}\n")


if __name__ == "__main__":
    main()
