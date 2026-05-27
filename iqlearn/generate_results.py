"""
iqlearn/generate_results.py
============================
Evaluate the trained IQ-Learn agent on the held-out TEST split and produce
publication-quality figures.

MUST be run AFTER iqlearn/train.py.  Loads model.pt from the results directory.
If model.pt is not found the script exits with a clear error.

Expected checkpoint keys (written by iqlearn/train.py via IQLearnAgent.save)
-----------------------------------------------------------------------------
    actor           : policy network state_dict
    critic          : IQCriticNetwork state_dict
    critic_target   : IQCriticNetwork state_dict  (not used here)
    config          : dict  (from dataclasses.asdict(IQLearnConfig))
    policy_type     : str   (e.g. 'hardgating')
    best_epoch      : int
    best_val_score  : float
    reservoir       : str
    training_stats  : dict of lists

State convention
----------------
Month encoding is PART OF THE STATE -- same as BC and AIRL.
With use_month_encoding = True (the standard):
    state = [storage_norm, inflow_norm, sin(2*pi*month/12), cos(2*pi*month/12)]
    state_dim = 4

Architecture convention
-----------------------
Actor  input  : cat([storage_norm, inflow_norm, sin_month, cos_month]) -- 4D
Actor  output : PolicyOutput (zero-inflated Beta / lognormal / etc.)
Critic input  : cat([storage_norm, inflow_norm, sin_month, cos_month, release_norm]) -- 5D
Critic output : scalar Q-value  (twin Q: take min)

Monte Carlo rollouts
--------------------
Each rollout simulates a full trajectory through the reservoir water-balance
physics, starting from the first observed test state.  Observed inflow is
used at each step (weather is exogenous and cannot be predicted).  Release
is sampled stochastically from the actor at each step.

Outputs
-------
results/<reservoir>/iqlearn/<run_id>_<policy_type>/
    test_metrics.json                   -- release / storage Pearson r + nRMSE
    release_test.png                    -- observed vs. MC median + IQR band
    storage_test.png                    -- observed vs. MC median + IQR band
    scatter_release.png                 -- scatter + 1:1 line (release, median MC)
    scatter_storage.png                 -- scatter + 1:1 line (storage, median MC)
    training_curves.png                 -- critic loss, actor loss, val-score history
    reward_contours.png                 -- 3x4 monthly Q-function contour grid
    shap_policy_total.png               -- mean |SHAP| per feature (policy network)
    shap_policy_monthly.png             -- SHAP heatmap by month (policy, sin/cos excluded)
    shap_qnetwork_total.png             -- mean |SHAP| per feature (Q-network)
    shap_qnetwork_monthly.png           -- SHAP heatmap by month (Q-network, sin/cos excluded)
    run_args.json                       -- updated with generate_results arguments

Usage
-----
python iqlearn/generate_results.py --reservoir conchas --run_id 1
python iqlearn/generate_results.py --reservoir conchas --run_id 1 \\
    --n_mc 100 --device cpu --n_shap_bg 100 --n_shap_explain 50
python iqlearn/generate_results.py --reservoir conchas --run_id 1 --skip_shap
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------------
import argparse
import copy
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Third-party
# ---------------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")          # non-interactive backend -- safe for headless runs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.ndimage import gaussian_filter

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Project root on sys.path so sibling packages resolve correctly
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from behavioral_cloning.tune import _resolve_device
from utils.data    import load_reservoir_data
from utils.metrics import nrmse, safe_pearsonr
from utils.runs    import _find_run_folder

# Policy factory and IQ-Learn networks -- imported from canonical modules so
# state_dict keys are guaranteed to match what train.py saved.
from networks.policy  import build_policy_network
from networks.iqlearn import IQCriticNetwork

# IQLearnConfig from core -- same class used by train.py; ensures from_dict is
# always in sync with whatever fields IQLearnConfig carries.
from iqlearn.core import IQLearnConfig

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Plot colours -- match BC / AIRL generate_results style
# ---------------------------------------------------------------------------
_BAND_COLOR     = "#F4A582"   # warm salmon  -- IQR shading
_OBSERVED_COLOR = "#1565C0"   # navy blue    -- observed line
_MEDIAN_COLOR   = "#C0392B"   # crimson      -- median MC line
_SCATTER_COLOR  = "#2C3E50"   # dark slate   -- scatter points
_SHAP_COLOR     = "#2980B9"   # steel blue   -- SHAP bars

# ---------------------------------------------------------------------------
# Physical constant
# ---------------------------------------------------------------------------
_SPD = 86_400.0   # seconds per day (release m^3/s -> volume Mm^3 via * _SPD / 1e6)


# =============================================================================
# Utilities
# =============================================================================

def _month_to_sin_cos(months: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert integer months (1-12) to circular (sin, cos) encoding.

    Returns
    -------
    sin_m, cos_m : each shape (N,), dtype float32
    """
    m = np.asarray(months, dtype=np.float32)
    sin_m = np.sin(2.0 * np.pi * m / 12.0).astype(np.float32)
    cos_m = np.cos(2.0 * np.pi * m / 12.0).astype(np.float32)
    return sin_m, cos_m


# =============================================================================
# Checkpoint loader
# =============================================================================

def _load_checkpoint(results_dir: Path, reservoir: str, policy_type: str) -> dict:
    """
    Load and validate model.pt produced by iqlearn/train.py.

    Checks
    ------
    1. File exists.
    2. File is loadable by torch.load.
    3. Required keys are present.
    4. Reservoir name in checkpoint matches the CLI argument.
    5. policy_type in checkpoint matches the CLI argument.
    """
    path = results_dir / "model.pt"

    # 1. Existence
    if not path.exists():
        sys.exit(
            f"\nERROR: model.pt not found.\n"
            f"  Expected : {path}\n\n"
            f"  iqlearn/train.py must be run before generate_results.py.  Run:\n"
            f"    python iqlearn/train.py --reservoir {reservoir} "
            f"--policy_type {policy_type} --run_id <id>\n"
        )

    # 2. Loadable
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:
        sys.exit(
            f"\nERROR: Cannot load model.pt.\n"
            f"  Error : {e}\n"
            f"  File  : {path}\n\n"
            f"  The file may be corrupted.  Re-run iqlearn/train.py.\n"
        )

    # 3. Required keys
    required = {
        "actor", "critic", "critic_target", "config",
        "policy_type", "best_epoch", "best_val_score", "reservoir",
    }
    missing = required - set(ckpt.keys())
    if missing:
        sys.exit(
            f"\nERROR: model.pt is missing required keys: {sorted(missing)}\n"
            f"  File : {path}\n\n"
            f"  The checkpoint may be from an older version of train.py.  "
            f"Re-run iqlearn/train.py.\n"
        )

    # 4. Reservoir match
    saved_res = str(ckpt["reservoir"]).lower().strip()
    if saved_res != reservoir.lower().strip():
        sys.exit(
            f"\nERROR: Reservoir mismatch.\n"
            f"  model.pt was trained on '{ckpt['reservoir']}', "
            f"but --reservoir '{reservoir}' was requested.\n"
        )

    # 5. Policy type match
    saved_pt = str(ckpt["policy_type"]).lower().strip()
    if saved_pt != policy_type:
        sys.exit(
            f"\nERROR: model.pt has policy_type='{saved_pt}' but "
            f"--policy_type='{policy_type}' was requested.\n"
            f"  Pass --policy_type {saved_pt} or re-run train.py.\n"
        )

    return ckpt


# =============================================================================
# Agent builder
# =============================================================================

def _build_agent(
    ckpt:   dict,
    device: torch.device,
) -> Tuple[nn.Module, IQCriticNetwork, IQLearnConfig]:
    """
    Reconstruct actor + critic from checkpoint and move to device.

    Actor is built via build_policy_network (same factory used at training time)
    so state_dict keys are guaranteed to match the saved checkpoint.
    Critic is built via IQCriticNetwork (same class as networks/iqlearn.py).

    Both networks are put into eval mode.
    """
    cfg    = IQLearnConfig.from_dict(ckpt["config"])
    actor  = build_policy_network(ckpt["policy_type"], cfg)
    critic = IQCriticNetwork(cfg)

    actor.load_state_dict(ckpt["actor"])
    critic.load_state_dict(ckpt["critic"])

    actor.to(device).eval()
    critic.to(device).eval()

    return actor, critic, cfg


# =============================================================================
# Monte Carlo rollouts
# =============================================================================

def _mc_rollout(
    actor:       nn.Module,
    test_df:     pd.DataFrame,   # raw test DataFrame with physical-unit columns
    normalizer,                  # utils.data.Normalizer
    bounds:      dict,           # {col: {min, max}}
    n_mc:        int,
    device:      torch.device,
    storage_col: str = "storage",
    inflow_col:  str = "net_inflow",
    action_col:  str = "release",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run n_mc stochastic environment rollouts over the test period.

    State at each step is the full 4D vector:
        [storage_norm, inflow_norm, sin_month, cos_month]
    Month encoding is part of the state -- no separate context argument.

    At each step:
        1. Build full 4D state (storage_norm, inflow_norm, sin_month, cos_month).
        2. Sample release action from the actor (stochastic).
        3. Denormalize release -> physical units.
        4. Apply water-balance: storage_next = storage + inflow_vol - release_vol.
        5. Clip storage to [min, max] training bounds.
        6. Advance to next observed inflow from test data.

    Returns
    -------
    mc_release : (n_mc, T)  release trajectories  [m^3/s]
    mc_storage : (n_mc, T)  storage trajectories  [Mm^3]

    where T = len(test_df) - 1  (number of water-balance steps).
    """
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    df = test_df.reset_index(drop=True)
    N  = len(df)
    T  = N - 1   # number of transition steps

    if T < 1:
        sys.exit("\nERROR: Test split has fewer than 2 rows.  Cannot run rollouts.\n")

    # Physical clipping limits from training bounds
    s_min = bounds[storage_col]["min"]
    s_max = bounds[storage_col]["max"]
    r_min = bounds[action_col]["min"]
    r_max = bounds[action_col]["max"]

    mc_release = np.empty((n_mc, T), dtype=np.float32)
    mc_storage = np.empty((n_mc, T), dtype=np.float32)

    with torch.no_grad():
        for i in range(n_mc):
            # Start from first observed test state
            storage_cur = float(df.iloc[0][storage_col])
            inflow_cur  = float(df.iloc[0][inflow_col])

            for t in range(T):
                month_t = int(df.iloc[t]["month"])

                # Normalize current state components
                s_norm = float(
                    normalizer.normalize(storage_col, np.array([storage_cur]))[0]
                )
                q_norm = float(
                    normalizer.normalize(inflow_col, np.array([inflow_cur]))[0]
                )

                # Month encoding
                sin_m, cos_m = _month_to_sin_cos(np.array([month_t]))
                sin_val = float(sin_m[0])
                cos_val = float(cos_m[0])

                # Full 4D state: [storage_norm, inflow_norm, sin_month, cos_month]
                state_t = torch.tensor(
                    [[s_norm, q_norm, sin_val, cos_val]],
                    dtype=torch.float32, device=device,
                )   # (1, 4)

                # Sample stochastic action from actor
                out = actor(state_t, deterministic=False)
                action_norm_val = float(out.action.squeeze().cpu().item())

                # Denormalize release
                release = float(
                    normalizer.denormalize(action_col, np.array([action_norm_val]))[0]
                )
                release = float(np.clip(release, r_min, r_max))

                # Record CURRENT step values (storage before action, release at action)
                mc_release[i, t] = release
                mc_storage[i, t] = storage_cur

                # Water-balance physics: storage [Mm^3], flows [m^3/s] -> [Mm^3]
                inflow_vol   = inflow_cur * _SPD / 1.0e6
                release_vol  = release    * _SPD / 1.0e6
                storage_next = storage_cur + inflow_vol - release_vol

                # Clip (spill absorbed at max; floor at min)
                storage_next = float(np.clip(storage_next, s_min, s_max))

                # Advance to next observed state
                storage_cur = storage_next
                if t + 1 < N:
                    inflow_cur = float(df.iloc[t + 1][inflow_col])

    return mc_release, mc_storage   # (n_mc, T), (n_mc, T)


# =============================================================================
# Metrics
# =============================================================================

def _compute_metrics(
    mc_release:  np.ndarray,   # (n_mc, T) in engineering units
    mc_storage:  np.ndarray,   # (n_mc, T) in engineering units
    obs_release: np.ndarray,   # (T,) observed release  [m^3/s]
    obs_storage: np.ndarray,   # (T,) observed storage  [Mm^3]
) -> dict:
    """
    Compute Pearson r and nRMSE for release and storage across MC rollouts.

    All metrics are computed on DENORMALIZED (original engineering-unit) values.
    nRMSE = sqrt(MSE) / (max_observed - min_observed).

    Returns
    -------
    dict with keys:
        release_corr_mean, release_corr_std, release_nrmse_mean, release_nrmse_std
        storage_corr_mean, storage_corr_std, storage_nrmse_mean, storage_nrmse_std
    """
    n_mc = mc_release.shape[0]
    rel_corrs  = np.empty(n_mc, dtype=np.float64)
    rel_nrmses = np.empty(n_mc, dtype=np.float64)
    sto_corrs  = np.empty(n_mc, dtype=np.float64)
    sto_nrmses = np.empty(n_mc, dtype=np.float64)

    for i in range(n_mc):
        rel_corrs[i],  _ = safe_pearsonr(obs_release, mc_release[i])
        rel_nrmses[i]    = nrmse(obs_release, mc_release[i])
        sto_corrs[i],  _ = safe_pearsonr(obs_storage, mc_storage[i])
        sto_nrmses[i]    = nrmse(obs_storage, mc_storage[i])

    return {
        "release_corr_mean":  float(np.mean(rel_corrs)),
        "release_corr_std":   float(np.std(rel_corrs)),
        "release_nrmse_mean": float(np.mean(rel_nrmses)),
        "release_nrmse_std":  float(np.std(rel_nrmses)),
        "storage_corr_mean":  float(np.mean(sto_corrs)),
        "storage_corr_std":   float(np.std(sto_corrs)),
        "storage_nrmse_mean": float(np.mean(sto_nrmses)),
        "storage_nrmse_std":  float(np.std(sto_nrmses)),
    }


# =============================================================================
# Time-series plot (MC ensemble: observed, median, IQR band)
# =============================================================================

def _plot_time_series(
    obs:       np.ndarray,   # (T,) observed values
    mc_preds:  np.ndarray,   # (n_mc, T) simulated values
    ylabel:    str,
    title:     str,
    dates:     pd.DatetimeIndex,
    save_path: Path,
) -> None:
    """
    Time-series figure: observed (blue), median MC (red dashed), IQR band (salmon).

    Same visual style as behavioral_cloning/generate_results.py and AIRL.
    """
    T = len(obs)
    x = np.arange(T)

    median_pred = np.median(mc_preds, axis=0)       # (T,)
    q25         = np.percentile(mc_preds, 25, axis=0)
    q75         = np.percentile(mc_preds, 75, axis=0)

    # Infer time step unit for x-axis label
    if len(dates) > 1:
        delta_days = (dates[1] - dates[0]).days
        time_unit  = "Days" if delta_days <= 3 else "Months"
    else:
        time_unit = "Steps"

    fig, ax = plt.subplots(figsize=(14, 4))

    ax.fill_between(
        x, q25, q75,
        color=_BAND_COLOR, alpha=0.6,
        label="25th-75th percentile (IQR)",
        zorder=1,
    )
    ax.plot(
        x, median_pred,
        color=_MEDIAN_COLOR, linestyle="--", linewidth=1.5,
        label="Median (MC rollouts)",
        zorder=2,
    )
    ax.plot(
        x, obs,
        color=_OBSERVED_COLOR, linestyle="-", linewidth=1.5,
        label="Observed",
        zorder=3,
    )

    ax.set_title(title, fontsize=16, fontweight="bold", pad=10)
    ax.set_xlabel(f"Time Steps ({time_unit})", fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.tick_params(axis="both", labelsize=12)
    ax.grid(True, color="grey", linewidth=0.4, alpha=0.35, zorder=0)
    ax.set_axisbelow(True)

    handles, labels = ax.get_legend_handles_labels()
    order = [
        labels.index("Observed"),
        labels.index("25th-75th percentile (IQR)"),
        labels.index("Median (MC rollouts)"),
    ]
    ax.legend(
        [handles[i] for i in order],
        [labels[i]  for i in order],
        fontsize=12, loc="upper right",
        framealpha=0.9, edgecolor="grey",
    )
    ax.set_xlim(0, T - 1)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {save_path.name}")


# =============================================================================
# Scatter plot  (observed vs. simulated, 1:1 line)
# =============================================================================

def _plot_scatter(
    expert:    np.ndarray,
    simulated: np.ndarray,
    xlabel:    str,
    ylabel:    str,
    title:     str,
    save_path: Path,
) -> None:
    """Scatter of simulated vs. observed with a 1:1 reference line."""
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(expert, simulated, s=4, alpha=0.35, color=_SCATTER_COLOR, zorder=2)

    lo = min(expert.min(), simulated.min())
    hi = max(expert.max(), simulated.max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.0, label="1:1", zorder=3)

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.tick_params(labelsize=10)
    ax.legend(fontsize=10)
    ax.grid(True, color="grey", lw=0.4, alpha=0.35)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {save_path.name}")


# =============================================================================
# Training curves  (critic loss, actor loss, validation score)
# =============================================================================

def _plot_training_curves(train_log: dict, eval_interval: int, save_path: Path) -> None:
    """
    Two-panel training history:
      Panel 1 — critic_loss and actor_loss per update step.
      Panel 2 — val_score, val_release_corr, val_release_nrmse per eval point.
    """
    stats = train_log.get("training_stats", {})
    if not stats:
        print("  WARNING: training_stats empty -- skipping training curves.")
        return

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=False)

    # Panel 1: per-update losses
    ax = axes[0]
    for key, color, label in [
        ("critic_loss", "#8E44AD", "critic_loss"),
        ("actor_loss",  "#2980B9", "actor_loss"),
    ]:
        if key in stats and stats[key]:
            ax.plot(np.arange(len(stats[key])), stats[key],
                    lw=1.0, alpha=0.8, label=label, color=color)
    ax.set_ylabel("Loss", fontsize=11)
    ax.set_xlabel("Update Step", fontsize=11)
    ax.set_title("Critic and Actor Loss", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3)

    # Panel 2: validation metrics per eval epoch
    ax = axes[1]
    for key, color, label in [
        ("val_score",          "#C0392B", "val_score"),
        ("val_release_corr",   "#27AE60", "val_release_corr"),
        ("val_release_nrmse",  "#E67E22", "val_release_nrmse"),
    ]:
        if key in stats and stats[key]:
            val_x = np.arange(len(stats[key])) * eval_interval
            ax.plot(val_x, stats[key], lw=1.4, label=label, color=color)

    val_scores = stats.get("val_score", [])
    if val_scores:
        val_x    = np.arange(len(val_scores)) * eval_interval
        best_idx = int(np.argmax(val_scores))
        ax.axvline(val_x[best_idx], color="grey", lw=0.8, linestyle="--",
                   label=f"best @ epoch {val_x[best_idx]}")

    ax.set_xlabel("Joint Training Epoch", fontsize=11)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Validation Metrics", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {save_path.name}")


# =============================================================================
# Monthly Q-function (reward) contour plots
# =============================================================================

def _plot_reward_contours(
    critic:      IQCriticNetwork,
    test_df:     pd.DataFrame,
    normalizer,
    bounds:      dict,
    device:      torch.device,
    reservoir:   str,
    save_path:   Path,
    storage_col: str = "storage",
    inflow_col:  str = "net_inflow",
    action_col:  str = "release",
    grid_size:   int = 200,
) -> None:
    """
    Plot Q-function contours Q(s, a) for each of the 12 calendar months.

    State includes month encoding:
        s = [storage_norm, inflow_norm, sin_month, cos_month]

    For each month:
        1. Build grid over (storage_norm, release_norm).
        2. Extract consecutive observed inflow values for that month.
        3. For each inflow: build full state (storage_grid, inflow, sin_m, cos_m).
           Evaluate Q(state_grid, release_grid).
        4. Average Q over all inflow samples -> 2D Q map.
        5. Plot filled contour + black isolines + expert scatter.

    All 12 subplots share a global colour scale.
    """
    df = test_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # ---- Normalised grid bounds ----
    s_lo = float(normalizer.normalize(storage_col, np.array([bounds[storage_col]["min"]]))[0])
    s_hi = float(normalizer.normalize(storage_col, np.array([bounds[storage_col]["max"]]))[0])
    s_grid = np.linspace(s_lo, s_hi, grid_size, dtype=np.float32)
    r_grid = np.linspace(0.0, 1.0,  grid_size, dtype=np.float32)

    # meshgrid: axis-0 = release, axis-1 = storage
    storage_mesh, release_mesh = np.meshgrid(s_grid, r_grid)

    storage_flat = storage_mesh.flatten().astype(np.float32)
    release_flat = release_mesh.flatten().astype(np.float32)
    n_points     = len(storage_flat)

    # Denormalised grids for axis tick labels
    storage_denorm = normalizer.denormalize(storage_col, storage_mesh)
    release_denorm = normalizer.denormalize(action_col,  release_mesh)

    month_names = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ]

    fig, axes = plt.subplots(3, 4, figsize=(24, 16), sharex=True, sharey=True)
    axes_flat = axes.flatten()
    all_q: List[np.ndarray] = []

    for month in range(1, 13):
        month_mask = df["month"] == month
        month_idx  = df.index[month_mask].tolist()

        sin_m_val, cos_m_val = _month_to_sin_cos(np.array([month]))
        sin_flat = np.full(n_points, float(sin_m_val[0]), dtype=np.float32)
        cos_flat = np.full(n_points, float(cos_m_val[0]), dtype=np.float32)

        act_t = torch.tensor(
            release_flat.reshape(-1, 1), dtype=torch.float32, device=device
        )

        inflow_samples: List[float] = []
        for k in range(len(month_idx) - 1):
            ci = month_idx[k]
            ni = month_idx[k + 1]
            if (df.loc[ni, "date"] - df.loc[ci, "date"]).days != 1:
                continue
            v_c = df.loc[ci, inflow_col]
            if np.isnan(v_c):
                continue
            inflow_samples.append(float(v_c))

        if not inflow_samples:
            all_q.append(np.zeros((grid_size, grid_size), dtype=np.float32))
            continue

        q_across: List[np.ndarray] = []
        with torch.no_grad():
            for inflow_curr in inflow_samples:
                q_norm = float(
                    normalizer.normalize(inflow_col, np.array([inflow_curr]))[0]
                )
                inflow_flat_np = np.full(n_points, q_norm, dtype=np.float32)

                states_np = np.column_stack(
                    [storage_flat, inflow_flat_np, sin_flat, cos_flat]
                )
                states_t = torch.tensor(
                    states_np, dtype=torch.float32, device=device
                )

                q1, q2 = critic(states_t, act_t)
                q      = torch.min(q1, q2).squeeze(-1).cpu().numpy()
                q_across.append(q.reshape(grid_size, grid_size))

        q_2d = np.mean(q_across, axis=0)
        q_2d = gaussian_filter(q_2d, sigma=0.0)
        all_q.append(q_2d)

    # ---- Global colour scale ----
    global_min = float(min(q.min() for q in all_q))
    global_max = float(max(q.max() for q in all_q))
    if global_max <= global_min:
        global_max = global_min + 1.0
    levels = np.linspace(global_min, global_max, 50)

    # ---- Plot all 12 months ----
    for i, month in enumerate(range(1, 13)):
        ax   = axes_flat[i]
        q_2d = all_q[i]

        ax.contourf(
            release_denorm, storage_denorm, q_2d,
            levels=levels, cmap="RdYlGn",
            vmin=global_min, vmax=global_max, extend="both",
        )
        ax.contour(
            release_denorm, storage_denorm, q_2d,
            levels=15, colors="black", alpha=0.2, linewidths=0.5,
        )

        month_data = df[df["month"] == month]
        if not month_data.empty:
            ax.scatter(
                month_data[action_col].values,
                month_data[storage_col].values,
                c="blue", s=10, alpha=0.5,
                edgecolors="white", linewidth=0.3, zorder=5,
            )

        ax.set_title(month_names[i], fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3, linestyle="--")
        if i >= 8:
            ax.set_xlabel(f"{action_col.capitalize()} (m^3/s)", fontsize=10)
        if i % 4 == 0:
            ax.set_ylabel(f"{storage_col.capitalize()} (Mm^3)", fontsize=10)

    sm = plt.cm.ScalarMappable(
        cmap="RdYlGn", norm=plt.Normalize(global_min, global_max)
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, location="right", shrink=0.8, pad=0.02)
    cbar.set_label("Q(s, a)", fontsize=12, fontweight="bold")

    plt.suptitle(
        f"Learned Q-Function by Month -- "
        f"{reservoir.replace('_', ' ').title()}",
        fontsize=16, fontweight="bold", y=0.995,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {save_path.name}")


# =============================================================================
# SHAP wrappers  (nn.Module so GradientExplainer can trace gradients)
# =============================================================================

class _PolicySHAPWrapper(nn.Module):
    """Wraps actor policy: state → scalar deterministic action."""
    def __init__(self, policy: nn.Module) -> None:
        super().__init__()
        self.policy = policy

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        out = self.policy(state, deterministic=True)
        # Return (batch, 1) — GradientExplainer requires 2D output.
        return out.action   # (batch, 1)


class _QNetworkSHAPWrapper(nn.Module):
    """Wraps IQCriticNetwork: cat([state | action]) → scalar Q_min(s, a)."""
    def __init__(self, critic: IQCriticNetwork, state_dim: int) -> None:
        super().__init__()
        self.critic    = critic
        self.state_dim = state_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        state  = x[:, :self.state_dim]
        action = x[:, self.state_dim:]
        q1, q2 = self.critic(state, action)
        return torch.min(q1, q2)   # (batch, 1)


def _get_state_feature_names(res_cfg: dict) -> List[str]:
    """Return ordered state feature names, including month encoding if active."""
    state_cols = list(res_cfg["columns"]["state"])
    use_month  = bool(res_cfg["columns"].get("use_month_encoding", True))
    names = list(state_cols)
    if use_month:
        names += ["sin_month", "cos_month"]
    return names


def _run_shap(
    wrapper:    nn.Module,
    background: np.ndarray,
    test_data:  np.ndarray,
    n_bg:       int,
    n_test:     int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute GradientExplainer SHAP values on CPU.

    Always runs on CPU to avoid device-specific gradient edge cases.

    Returns
    -------
    shap_values : (n_test_subset, n_features)
    test_idx    : (n_test_subset,) indices into test_data used — pass back to
                  caller so month alignment uses the exact same subset.
    """
    try:
        import shap
    except ImportError:
        sys.exit(
            "\nERROR: shap is required for SHAP analysis.\n"
            "  Install with:  pip install shap\n"
        )

    cpu     = torch.device("cpu")
    wrapper = copy.deepcopy(wrapper).to(cpu).eval()

    rng      = np.random.default_rng(42)
    bg_idx   = rng.choice(len(background), size=min(n_bg,   len(background)), replace=False)
    test_idx = rng.choice(len(test_data),  size=min(n_test, len(test_data)),  replace=False)
    test_idx = np.sort(test_idx)   # keep temporal order

    bg_tensor   = torch.tensor(background[bg_idx],  dtype=torch.float32)
    test_tensor = torch.tensor(test_data[test_idx], dtype=torch.float32)

    explainer   = shap.GradientExplainer(wrapper, bg_tensor)
    shap_values = explainer.shap_values(test_tensor)

    if isinstance(shap_values, list):
        shap_values = shap_values[0]

    shap_values = np.array(shap_values, dtype=np.float32)
    # GradientExplainer may return (n_test, n_features, 1) when output is (batch, 1).
    if shap_values.ndim == 3 and shap_values.shape[-1] == 1:
        shap_values = shap_values.squeeze(-1)

    return shap_values, test_idx


def _plot_shap_total(
    shap_values:   np.ndarray,
    feature_names: List[str],
    title:         str,
    save_path:     Path,
) -> None:
    """Horizontal bar chart of mean |SHAP| per feature, sorted descending."""
    shap_clean = np.nan_to_num(shap_values, nan=0.0, posinf=0.0, neginf=0.0)
    mean_abs   = np.abs(shap_clean).mean(axis=0)
    total      = mean_abs.sum()
    pct        = (mean_abs / total * 100) if total > 0 else mean_abs
    order      = np.argsort(pct)   # ascending for horizontal bar

    names_sorted = [feature_names[i] for i in order]
    pct_sorted   = pct[order]

    fig, ax = plt.subplots(figsize=(7, max(3, len(feature_names) * 0.4)))
    bars = ax.barh(names_sorted, pct_sorted, color=_SHAP_COLOR,
                   edgecolor="white", height=0.6)

    for bar, val in zip(bars, pct_sorted):
        ax.text(
            bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}%", va="center", ha="left", fontsize=9,
        )

    ax.set_xlabel("Contribution to network output (%)", fontsize=12)
    max_pct = float(np.nanmax(pct_sorted)) if pct_sorted.size > 0 else 100.0
    ax.set_xlim(0, max_pct * 1.18)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.tick_params(labelsize=10)
    ax.grid(True, axis="x", alpha=0.3)
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {save_path.name}")


def _plot_shap_monthly(
    shap_values:   np.ndarray,
    feature_names: List[str],
    months:        np.ndarray,
    title:         str,
    save_path:     Path,
) -> None:
    """
    Heatmap of feature contribution (%) to network output, by month.

    Rows = features, columns = months 1-12.  Each column is normalised to sum
    to 100%.  sin_month / cos_month must be stripped by the caller before
    passing here — they are constant per month so showing monthly variation
    is circular.
    """
    n_features   = len(feature_names)
    month_labels = ["Jan","Feb","Mar","Apr","May","Jun",
                    "Jul","Aug","Sep","Oct","Nov","Dec"]

    shap_clean = np.nan_to_num(shap_values, nan=0.0, posinf=0.0, neginf=0.0)

    raw = np.zeros((n_features, 12), dtype=np.float32)
    for m_idx, m in enumerate(range(1, 13)):
        mask = (months == m)
        if mask.sum() > 0:
            raw[:, m_idx] = np.abs(shap_clean[mask]).mean(axis=0)

    col_totals = raw.sum(axis=0, keepdims=True)
    col_totals = np.where(col_totals == 0, 1, col_totals)
    matrix = raw / col_totals * 100   # (n_features, 12) in %

    fig, ax = plt.subplots(figsize=(11, max(3, n_features * 0.55)))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd",
                   interpolation="nearest", vmin=0, vmax=100)
    plt.colorbar(im, ax=ax, label="Contribution to output (%)")

    ax.set_xticks(range(12))
    ax.set_xticklabels(month_labels, fontsize=10)
    ax.set_yticks(range(n_features))
    ax.set_yticklabels(feature_names, fontsize=10)
    ax.set_xlabel("Month", fontsize=12)
    ax.set_ylabel("Feature", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")

    for i in range(n_features):
        for j in range(12):
            ax.text(j, i, f"{matrix[i, j]:.1f}%",
                    ha="center", va="center", fontsize=7,
                    color="black" if matrix[i, j] < 65 else "white")

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {save_path.name}")


# =============================================================================
# CLI
# =============================================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Evaluate the trained IQ-Learn agent on the test split and produce "
            "figures.  Requires iqlearn/train.py to have been run first."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--reservoir", required=True,
        help="Reservoir name -- must match configs/reservoirs/<name>.yaml.",
    )
    p.add_argument(
        "--policy_type", default=None,
        choices=["beta", "lognormal", "hardgating", "softgating"],
        help="Policy type -- inferred from the run folder name if omitted.",
    )
    p.add_argument(
        "--run_id", type=int, required=True,
        help=(
            "Integer run identifier matching the folder created by tune.py "
            "(e.g. 1 for folder '1_hardgating').  Required."
        ),
    )
    p.add_argument(
        "--device", default=None,
        help=(
            "Compute device.  Defaults to the device stored in model.pt.  "
            "Options: auto | cpu | cuda | cuda:N | mps."
        ),
    )
    p.add_argument(
        "--n_mc", type=int, default=100,
        help="Number of Monte Carlo rollouts for the trajectory ensemble.",
    )
    p.add_argument(
        "--shap_background", type=int, default=100,
        help="Number of training samples used as SHAP background.",
    )
    p.add_argument(
        "--shap_test_size", type=int, default=300,
        help="Number of test samples explained by SHAP.",
    )
    p.add_argument(
        "--skip_shap", action="store_true",
        help="Skip all SHAP computation (useful for quick diagnostic runs).",
    )
    return p.parse_args()


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args = _parse_args()

    if args.n_mc < 1:
        sys.exit("\nERROR: --n_mc must be a positive integer.\n")
    if args.shap_background < 1 or args.shap_test_size < 1:
        sys.exit("\nERROR: --shap_background and --shap_test_size must be positive integers.\n")

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------
    res_cfg_path = _ROOT / "configs" / "reservoirs" / f"{args.reservoir}.yaml"
    iq_base_dir  = _ROOT / "results" / args.reservoir / "iqlearn"
    results_dir  = _find_run_folder(iq_base_dir, args.run_id)

    if not res_cfg_path.exists():
        sys.exit(
            f"\nERROR: Reservoir config not found: {res_cfg_path}\n"
            f"  Available: configs/reservoirs/*.yaml\n"
        )

    with open(res_cfg_path, "r") as f:
        res_cfg = yaml.safe_load(f)

    # Column names from YAML
    date_col    = str(res_cfg["columns"]["date"])
    state_cols  = list(res_cfg["columns"]["state"])   # e.g. ["storage", "net_inflow"]
    action_col  = str(res_cfg["columns"]["action"])   # e.g. "release"
    storage_col = state_cols[0]                        # first state variable
    inflow_col  = state_cols[1]                        # second state variable

    # ------------------------------------------------------------------
    # Infer policy_type from folder name if not given on CLI
    # Folder convention: <run_id>_<policy_type>  e.g. 1_hardgating
    # ------------------------------------------------------------------
    if args.policy_type is None:
        parts = results_dir.name.split("_", 1)
        if len(parts) != 2 or not parts[1]:
            sys.exit(
                f"\nERROR: Cannot infer policy_type from folder '{results_dir.name}'.\n"
                f"  Pass --policy_type explicitly.\n"
            )
        args.policy_type = parts[1]
        print(f"Policy type inferred from folder: {args.policy_type}")

    # ------------------------------------------------------------------
    # Load checkpoint
    # ------------------------------------------------------------------
    ckpt = _load_checkpoint(results_dir, args.reservoir, args.policy_type)
    policy_type = str(ckpt["policy_type"])

    print(f"\nLoaded model.pt")
    print(f"  Reservoir      : {args.reservoir}")
    print(f"  Policy type    : {policy_type}")
    print(
        f"  Best val score : {ckpt['best_val_score']:.4f}  "
        f"(epoch {ckpt['best_epoch'] + 1})"
    )

    # ------------------------------------------------------------------
    # Load normalised data (for bounds + normalizer)
    # ------------------------------------------------------------------
    print("\nLoading data ...")
    data = load_reservoir_data(res_cfg, res_cfg_path)
    print(f"  state_dim  = {data.state_dim}")
    print(f"  test rows  = {len(data.test.states)}")

    # state_dim consistency check
    cfg_state_dim = int(ckpt["config"]["state_dim"])
    if cfg_state_dim != data.state_dim:
        sys.exit(
            f"\nERROR: state_dim mismatch: checkpoint={cfg_state_dim}, "
            f"data={data.state_dim}.  Re-run iqlearn/train.py.\n"
        )

    # ------------------------------------------------------------------
    # Rebuild raw test DataFrame (physical units, needed for MC rollouts
    # and reward-contour scatter)
    # ------------------------------------------------------------------
    raw_data_path = Path(res_cfg["data_path"].replace("\\", "/"))
    if not raw_data_path.is_absolute() and not raw_data_path.exists():
        raw_data_path = res_cfg_path.parent.parent.parent / raw_data_path
    if not raw_data_path.exists():
        sys.exit(
            f"\nERROR: Data file not found: {raw_data_path}\n"
            f"  Update data_path in {res_cfg_path}.\n"
        )

    df_raw = pd.read_csv(raw_data_path)
    df_raw[date_col] = pd.to_datetime(df_raw[date_col])
    df_raw = df_raw.sort_values(date_col).reset_index(drop=True)
    df_raw["_year"] = df_raw[date_col].dt.year
    df_raw["month"] = df_raw[date_col].dt.month

    if date_col != "date":
        df_raw = df_raw.rename(columns={date_col: "date"})

    years      = sorted(df_raw["_year"].unique())
    n_train    = int(res_cfg["split"]["train"])
    n_val      = int(res_cfg["split"]["val"])
    test_years = set(years[n_train + n_val:])
    test_df    = df_raw[df_raw["_year"].isin(test_years)].copy().reset_index(drop=True)

    print(f"  Test years : {sorted(test_years)}")
    print(f"  Test rows  : {len(test_df)}")

    # ------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------
    raw_device = (
        args.device if args.device is not None
        else str(ckpt["config"].get("device", "cpu"))
    )
    resolved = _resolve_device(raw_device)

    if resolved.startswith("cuda") and not torch.cuda.is_available():
        print(
            f"\nWARNING: Requested '{resolved}' but CUDA is not available.  "
            f"Falling back to CPU.\n",
            file=sys.stderr,
        )
        resolved = "cpu"
    elif resolved == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        print(
            "\nWARNING: Requested 'mps' but MPS is not available.  "
            "Falling back to CPU.\n",
            file=sys.stderr,
        )
        resolved = "cpu"

    device = torch.device(resolved)
    print(f"\nDevice : {device}")

    # ------------------------------------------------------------------
    # Build agent
    # ------------------------------------------------------------------
    actor, critic, cfg = _build_agent(ckpt, device)
    print(
        f"\nAgent reconstructed."
        f"\n  Actor  : {policy_type}  "
        f"hidden={cfg.actor_hidden_dim}  layers={cfg.actor_n_hidden_layers}"
        f"\n  Critic : hidden={cfg.critic_hidden_dim}  "
        f"layers={cfg.critic_n_hidden_layers}"
        f"\n  state_dim = {cfg.state_dim}  action_dim = {cfg.action_dim}"
    )

    # ------------------------------------------------------------------
    # Monte Carlo rollouts
    # ------------------------------------------------------------------
    print(f"\nRunning {args.n_mc} MC rollouts ...", end="", flush=True)
    mc_release, mc_storage = _mc_rollout(
        actor       = actor,
        test_df     = test_df,
        normalizer  = data.normalizer,
        bounds      = data.bounds,
        n_mc        = args.n_mc,
        device      = device,
        storage_col = storage_col,
        inflow_col  = inflow_col,
        action_col  = action_col,
    )
    print(" done.")

    # Align observed arrays to rollout length T = N - 1
    T           = mc_release.shape[1]
    obs_release = test_df[action_col].values[:T].astype(np.float32)
    obs_storage = test_df[storage_col].values[:T].astype(np.float32)
    test_dates  = pd.DatetimeIndex(test_df["date"].values[:T])

    # Summary statistics across rollouts
    median_rel  = np.median(mc_release, axis=0)
    median_stor = np.median(mc_storage, axis=0)

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    print("Computing metrics ...", end="", flush=True)
    metrics = _compute_metrics(mc_release, mc_storage, obs_release, obs_storage)
    print(" done.")

    print(
        f"\n  Release : r = {metrics['release_corr_mean']:.4f} "
        f"+/- {metrics['release_corr_std']:.4f}  |  "
        f"nRMSE = {metrics['release_nrmse_mean']:.4f} "
        f"+/- {metrics['release_nrmse_std']:.4f}"
        f"\n  Storage : r = {metrics['storage_corr_mean']:.4f} "
        f"+/- {metrics['storage_corr_std']:.4f}  |  "
        f"nRMSE = {metrics['storage_nrmse_mean']:.4f} "
        f"+/- {metrics['storage_nrmse_std']:.4f}"
    )

    # ------------------------------------------------------------------
    # Save test_metrics.json
    # ------------------------------------------------------------------
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load train_log.json for best_val_score (authoritative source is train_log;
    # also available in checkpoint, but train_log is more explicit)
    train_log_path = results_dir / "train_log.json"
    best_val_score = float(ckpt.get("best_val_score", 0.0))
    train_log_data: dict = {}
    if train_log_path.exists():
        try:
            with open(train_log_path, "r") as f:
                train_log_data = json.load(f)
            best_val_score = train_log_data.get("best_val_score", best_val_score)
        except json.JSONDecodeError:
            pass

    metrics_out = {
        "reservoir":      args.reservoir,
        "policy_type":    policy_type,
        "run_id":         args.run_id,
        "n_mc":           args.n_mc,
        "best_val_score": round(float(best_val_score), 6),
        "test_metrics":   {k: round(float(v), 6) for k, v in metrics.items()},
        "timestamp":      datetime.now().isoformat(timespec="seconds"),
    }
    with open(results_dir / "test_metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2)
    print(f"\nMetrics saved  -> test_metrics.json")

    # ------------------------------------------------------------------
    # Time-series plots (MC ensemble)
    # ------------------------------------------------------------------
    print("\nGenerating time-series plots ...")
    _plot_time_series(
        obs       = obs_release,
        mc_preds  = mc_release,
        ylabel    = f"{action_col.capitalize()} (m^3/s)",
        title     = (
            f"Release -- {args.reservoir.replace('_', ' ').title()}\n"
            f"r = {metrics['release_corr_mean']:.3f} "
            f"+/- {metrics['release_corr_std']:.3f},  "
            f"nRMSE = {metrics['release_nrmse_mean']:.3f} "
            f"+/- {metrics['release_nrmse_std']:.3f}"
        ),
        dates     = test_dates,
        save_path = results_dir / "release_test.png",
    )
    _plot_time_series(
        obs       = obs_storage,
        mc_preds  = mc_storage,
        ylabel    = f"{storage_col.capitalize()} (Mm^3)",
        title     = (
            f"Storage -- {args.reservoir.replace('_', ' ').title()}\n"
            f"r = {metrics['storage_corr_mean']:.3f} "
            f"+/- {metrics['storage_corr_std']:.3f},  "
            f"nRMSE = {metrics['storage_nrmse_mean']:.3f} "
            f"+/- {metrics['storage_nrmse_std']:.3f}"
        ),
        dates     = test_dates,
        save_path = results_dir / "storage_test.png",
    )

    # ------------------------------------------------------------------
    # Scatter plots (median MC as point estimate)
    # ------------------------------------------------------------------
    print("Generating scatter plots ...")
    rel_corr  = metrics["release_corr_mean"]
    stor_corr = metrics["storage_corr_mean"]
    _plot_scatter(
        expert    = obs_release,
        simulated = median_rel,
        xlabel    = f"Observed Release (m^3/s)",
        ylabel    = f"Simulated Release (m^3/s)",
        title     = f"Release  r={rel_corr:.3f}  (median MC)",
        save_path = results_dir / "scatter_release.png",
    )
    _plot_scatter(
        expert    = obs_storage,
        simulated = median_stor,
        xlabel    = "Observed Storage (Mm^3)",
        ylabel    = "Simulated Storage (Mm^3)",
        title     = f"Storage  r={stor_corr:.3f}  (median MC)",
        save_path = results_dir / "scatter_storage.png",
    )

    # ------------------------------------------------------------------
    # Training curves
    # ------------------------------------------------------------------
    print("Generating training curves ...")
    if train_log_data:
        _plot_training_curves(
            train_log     = train_log_data,
            eval_interval = cfg.eval_interval,
            save_path     = results_dir / "training_curves.png",
        )
    else:
        print("  WARNING: train_log.json not found -- skipping training curves.")

    # ------------------------------------------------------------------
    # Q-function reward contours
    # ------------------------------------------------------------------
    print("\nPlotting Q-function reward contours ...")
    _plot_reward_contours(
        critic      = critic,
        test_df     = test_df,
        normalizer  = data.normalizer,
        bounds      = data.bounds,
        device      = device,
        reservoir   = args.reservoir,
        save_path   = results_dir / "reward_contours.png",
        storage_col = storage_col,
        inflow_col  = inflow_col,
        action_col  = action_col,
    )

    # ------------------------------------------------------------------
    # SHAP  (policy network + Q-network, using GradientExplainer)
    # ------------------------------------------------------------------
    if args.skip_shap:
        print("\nSHAP skipped (--skip_shap).")
    else:
        print("\nComputing SHAP values ...")
        feature_names      = _get_state_feature_names(res_cfg)
        qnet_feature_names = feature_names + [action_col]
        _month_enc         = {"sin_month", "cos_month"}

        train_states  = data.train.states.astype(np.float32)
        test_states   = data.test.states.astype(np.float32)
        train_actions = data.train.actions.reshape(-1, 1).astype(np.float32)
        test_actions  = data.test.actions.reshape(-1, 1).astype(np.float32)
        train_combined = np.concatenate([train_states, train_actions], axis=1)
        test_combined  = np.concatenate([test_states,  test_actions],  axis=1)

        # Month labels aligned to test split rows (for monthly heatmaps)
        test_months = test_df["month"].values[:len(test_states)].astype(np.int32)

        # ---- Policy SHAP ----
        print("  Policy network ...", end="", flush=True)
        policy_wrapper = _PolicySHAPWrapper(actor)
        shap_policy, test_idx = _run_shap(
            wrapper    = policy_wrapper,
            background = train_states,
            test_data  = test_states,
            n_bg       = args.shap_background,
            n_test     = args.shap_test_size,
        )
        print(" done.")

        months_subset = test_months[test_idx]

        _plot_shap_total(
            shap_values   = shap_policy,
            feature_names = feature_names,
            title         = "Policy Network — Feature Importance (SHAP)",
            save_path     = results_dir / "shap_policy_total.png",
        )
        # Monthly: strip sin_month / cos_month — they ARE the month encoding,
        # so their SHAP varying by month is circular information.
        # They are still shown in the total bar chart above.
        pol_non_month_idx = [i for i, n in enumerate(feature_names)
                             if n not in _month_enc]
        _plot_shap_monthly(
            shap_values   = shap_policy[:, pol_non_month_idx],
            feature_names = [feature_names[i] for i in pol_non_month_idx],
            months        = months_subset,
            title         = "Policy Network — Monthly SHAP Contributions",
            save_path     = results_dir / "shap_policy_monthly.png",
        )

        # ---- Q-network SHAP ----
        print("  Q-network ...", end="", flush=True)
        qnet_wrapper = _QNetworkSHAPWrapper(critic, cfg.state_dim)
        shap_qnet, _ = _run_shap(
            wrapper    = qnet_wrapper,
            background = train_combined,
            test_data  = test_combined,
            n_bg       = args.shap_background,
            n_test     = args.shap_test_size,
        )
        print(" done.")

        _plot_shap_total(
            shap_values   = shap_qnet,
            feature_names = qnet_feature_names,
            title         = "Q-Network — Feature Importance (SHAP)",
            save_path     = results_dir / "shap_qnetwork_total.png",
        )
        qnet_non_month_idx = [i for i, n in enumerate(qnet_feature_names)
                              if n not in _month_enc]
        _plot_shap_monthly(
            shap_values   = shap_qnet[:, qnet_non_month_idx],
            feature_names = [qnet_feature_names[i] for i in qnet_non_month_idx],
            months        = months_subset,
            title         = "Q-Network — Monthly SHAP Contributions",
            save_path     = results_dir / "shap_qnetwork_monthly.png",
        )

    # ------------------------------------------------------------------
    # Update run_args.json
    # ------------------------------------------------------------------
    run_args_path = results_dir / "run_args.json"
    run_args: dict = {}
    if run_args_path.exists():
        try:
            with open(run_args_path, "r") as f:
                run_args = json.load(f)
        except json.JSONDecodeError:
            run_args = {}

    run_args["generate_results"] = {
        "reservoir":      args.reservoir,
        "policy_type":    policy_type,
        "run_id":         args.run_id,
        "device_cli":     args.device,
        "device_used":    resolved,
        "n_mc":           args.n_mc,
        "shap_background": args.shap_background,
        "shap_test_size":  args.shap_test_size,
        "skip_shap":      args.skip_shap,
        "timestamp":      datetime.now().isoformat(timespec="seconds"),
    }
    with open(run_args_path, "w") as f:
        json.dump(run_args, f, indent=2)
    print(f"Run args updated  -> run_args.json")

    print(f"\n{'=' * 60}")
    print(f"All outputs saved to: {results_dir}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
