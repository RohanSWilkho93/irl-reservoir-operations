"""
airl/generate_results.py
========================
Evaluate the trained AIRL model on the held-out TEST split and produce
publication-quality figures.

MUST be run AFTER airl/train.py.  Loads model.pt from the run folder.

Outputs
-------
results/<reservoir>/airl/<run_id>_<policy_type>/
    test_metrics.json         — release & storage nRMSE / Pearson r / RMSE.
    release_test.png          — simulated vs expert release time series.
    storage_test.png          — simulated vs expert storage time series.
    scatter_release.png       — scatter + 1:1 line (release).
    scatter_storage.png       — scatter + 1:1 line (storage).
    training_curves.png       — discriminator, PPO, and val-score history.
    reward_contour.png        — g(s,a) over storage × release grid.
    shap_policy_total.png     — mean |SHAP| per feature (policy network).
    shap_reward_total.png     — mean |SHAP| per feature (reward network).
    shap_policy_monthly.png   — SHAP heatmap by month (policy, if use_month=True).
    shap_reward_monthly.png   — SHAP heatmap by month (reward, if use_month=True).

Usage
-----
python airl/generate_results.py --reservoir garrison --policy_type beta --run_id 1

# Override device
python airl/generate_results.py --reservoir garrison --policy_type beta --run_id 1 \\
    --device cpu

# Control SHAP sample sizes
python airl/generate_results.py --reservoir garrison --policy_type beta --run_id 1 \\
    --shap_background 200 --shap_test_size 500
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml

# ---------------------------------------------------------------------------
# Project root on sys.path so sibling packages resolve correctly.
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from behavioral_cloning.tune import _resolve_device
from networks.policy         import build_policy_network
from networks.airl           import build_airl_networks
from utils.data              import load_reservoir_data
from utils.metrics           import nrmse, safe_pearsonr
from utils.runs              import _find_run_folder
from airl.core               import AIRLConfig, ReservoirEnvironment, _load_raw_splits

# ---------------------------------------------------------------------------
# Plot style constants
# ---------------------------------------------------------------------------
_OBSERVED_COLOR  = "#1565C0"   # navy blue  — observed
_SIMULATED_COLOR = "#C0392B"   # crimson    — simulated / median MC
_BAND_COLOR      = "#F4A582"   # warm salmon — IQR shading (matches BC generate_results)
_SCATTER_COLOR   = "#2C3E50"   # dark slate
_SHAP_COLOR      = "#2980B9"   # steel blue


# =============================================================================
# Checkpoint loader
# =============================================================================

def _load_checkpoint(results_dir: Path, reservoir: str, policy_type: str) -> dict:
    """Load and validate model.pt produced by airl/train.py."""
    path = results_dir / "model.pt"

    if not path.exists():
        sys.exit(
            f"\nERROR: model.pt not found.\n"
            f"  Expected: {path}\n\n"
            f"  airl/train.py must be run first.  Run:\n"
            f"    python airl/train.py --reservoir {reservoir} "
            f"--policy_type {policy_type} --run_id <id>\n"
        )

    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:
        sys.exit(
            f"\nERROR: Cannot load model.pt.\n"
            f"  Error: {e}\n"
            f"  File:  {path}\n\n"
            f"  The checkpoint may be corrupted.  Re-run airl/train.py.\n"
        )

    required = {"policy", "discriminator", "config", "policy_type"}
    missing  = required - set(ckpt.keys())
    if missing:
        sys.exit(
            f"\nERROR: model.pt is missing required keys: {sorted(missing)}\n"
            f"  The checkpoint may be from an older version of train.py.  "
            f"Re-run airl/train.py.\n"
        )

    saved_pt = str(ckpt["policy_type"]).lower().strip()
    if saved_pt != policy_type:
        sys.exit(
            f"\nERROR: model.pt has policy_type='{saved_pt}' but "
            f"--policy_type='{policy_type}' was requested.\n"
            f"  Pass --policy_type {saved_pt} or re-run train.py.\n"
        )

    return ckpt


# =============================================================================
# Model reconstruction
# =============================================================================

def _rebuild_models(ckpt: dict, device: torch.device):
    """
    Reconstruct policy + AIRLDiscriminator from a checkpoint and load weights.

    Returns
    -------
    policy        : nn.Module  — deterministic policy, eval mode, on device.
    discriminator : AIRLDiscriminator — eval mode, on device.
    config        : AIRLConfig
    policy_type   : str
    """
    cfg_dict    = ckpt["config"]
    policy_type = str(ckpt["policy_type"]).lower().strip()

    config = AIRLConfig(
        state_dim  = int(cfg_dict["state_dim"]),
        action_dim = int(cfg_dict.get("action_dim", 1)),
        hidden_dim      = int(cfg_dict["hidden_dim"]),
        n_hidden_layers = int(cfg_dict["n_hidden_layers"]),
        dropout         = float(cfg_dict["dropout"]),
        alpha_min       = float(cfg_dict.get("alpha_min",      1.0)),
        alpha_max       = float(cfg_dict.get("alpha_max",      50.0)),
        beta_min        = float(cfg_dict.get("beta_min",       1.0)),
        beta_max        = float(cfg_dict.get("beta_max",       50.0)),
        sigma_min       = float(cfg_dict.get("sigma_min",      0.1)),
        log_epsilon     = float(cfg_dict.get("log_epsilon",    1.0)),
        zero_threshold  = float(cfg_dict.get("zero_threshold", 0.01)),
        mse_weight      = float(cfg_dict.get("mse_weight",     10.0)),
        gate_weight     = float(cfg_dict.get("gate_weight",    5.0)),
        critic_hidden_dim      = int(cfg_dict["critic_hidden_dim"]),
        critic_n_hidden_layers = int(cfg_dict["critic_n_hidden_layers"]),
        disc_hidden_dim      = int(cfg_dict["disc_hidden_dim"]),
        disc_n_hidden_layers = int(cfg_dict["disc_n_hidden_layers"]),
        disc_dropout         = float(cfg_dict["disc_dropout"]),
        lr_policy        = float(cfg_dict["lr_policy"]),
        lr_critic        = float(cfg_dict["lr_critic"]),
        lr_discriminator = float(cfg_dict["lr_discriminator"]),
        disc_updates            = int(cfg_dict["disc_updates"]),
        warmup_disc_updates     = int(cfg_dict["warmup_disc_updates"]),
        gradient_penalty_coef   = float(cfg_dict["gradient_penalty_coef"]),
        label_smoothing_epsilon = float(cfg_dict["label_smoothing_epsilon"]),
        gamma        = float(cfg_dict["gamma"]),
        gae_lambda   = float(cfg_dict.get("gae_lambda",   0.95)),
        clip_epsilon = float(cfg_dict["clip_epsilon"]),
        entropy_coef = float(cfg_dict["entropy_coef"]),
        ppo_epochs   = int(cfg_dict["ppo_epochs"]),
        kl_regularization_coef = float(cfg_dict["kl_regularization_coef"]),
        warmup_iterations       = int(cfg_dict["warmup_iterations"]),
        num_iterations          = int(cfg_dict["num_iterations"]),
        steps_per_iteration     = int(cfg_dict["steps_per_iteration"]),
        batch_size              = int(cfg_dict["batch_size"]),
        early_stopping_patience = int(cfg_dict["early_stopping_patience"]),
        expert_buffer_size   = int(cfg_dict.get("expert_buffer_size",  60_000)),
        policy_buffer_size   = int(cfg_dict.get("policy_buffer_size", 120_000)),
        trajectory_years     = int(cfg_dict.get("trajectory_years",       1)),
        align_to_year_start  = bool(cfg_dict.get("align_to_year_start",  True)),
        end_at_year_boundary = bool(cfg_dict.get("end_at_year_boundary", True)),
        eval_interval        = int(cfg_dict.get("eval_interval",         10)),
        max_grad_norm        = float(cfg_dict.get("max_grad_norm",       0.5)),
        device  = str(device),
        seed    = int(cfg_dict.get("seed", 42)),
        verbose = False,
    )

    # Build and load policy
    policy = build_policy_network(policy_type, config)
    policy.load_state_dict(ckpt["policy"])
    policy.to(device).eval()

    # Build and load discriminator (reward_net, shaping_net, critic live inside)
    discriminator = build_airl_networks(config, policy)
    discriminator.load_state_dict(ckpt["discriminator"])
    discriminator.to(device).eval()

    return policy, discriminator, config, policy_type


# =============================================================================
# Feature name helper
# =============================================================================

def _get_feature_names(res_cfg: dict) -> list[str]:
    """Return ordered list of state feature names, including month encoding if active."""
    state_cols = list(res_cfg["columns"]["state"])
    use_month  = bool(res_cfg["columns"].get("use_month_encoding", True))
    names = list(state_cols)
    if use_month:
        names += ["sin_month", "cos_month"]
    return names


# =============================================================================
# Test rollout
# =============================================================================

def _rollout_test(
    policy,
    test_env:   ReservoirEnvironment,
    test_split,
    device:     torch.device,
):
    """
    Deterministic policy rollout over the full test split.

    Mirrors AIRLAgent.evaluate() but standalone (no agent required).
    Handles year-boundary resets so all test years are covered.

    Returns
    -------
    expert_release  : (N,) float32  — raw expert release (engineering units).
    expert_storage  : (N,) float32  — raw expert storage (Mm³).
    sim_release     : (N,) float32  — simulated release (engineering units).
    sim_storage     : (N,) float32  — simulated storage (Mm³).
    """
    policy.eval()
    N = len(test_split.states) - 1

    expert_release = test_split.raw_actions[:N]
    expert_storage = test_env.normalizer.denormalize(
        test_env.storage_col,
        test_split.states[:N, 0],
    )

    sim_release: list = []
    sim_storage: list = []

    state    = test_env.reset(0)
    sim_stor = test_env.storage

    for i in range(N):
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            out    = policy(state_t, deterministic=True)
            action = float(out.action.cpu().numpy()[0, 0])

        sim_storage.append(sim_stor)
        sim_release.append(
            float(test_env.normalizer.denormalize(
                test_env.action_col, np.array([action])
            )[0])
        )

        state, _, done, info = test_env.step(action)
        sim_stor = info["storage"]

        if done and i < N - 1:
            state    = test_env.reset(i + 1)
            sim_stor = test_env.storage

    return (
        np.array(expert_release, dtype=np.float32),
        np.array(expert_storage, dtype=np.float32),
        np.array(sim_release,    dtype=np.float32),
        np.array(sim_storage,    dtype=np.float32),
    )


# =============================================================================
# Monte Carlo rollouts  (closed-loop)
# =============================================================================

def _run_mc_rollout(
    policy,
    test_env:   ReservoirEnvironment,
    test_split,
    device:     torch.device,
    n_mc:       int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run n_mc independent stochastic closed-loop rollouts over the test split.

    At every timestep the policy samples from its output distribution
    (deterministic=False).  Because actions feed into the environment physics,
    each rollout produces a unique storage trajectory — this is fully closed-loop,
    unlike the open-loop BC MC rollout where states are always from the test set.

    Seeded with 42 for reproducibility.

    Parameters
    ----------
    policy     : Trained policy in eval mode.
    test_env   : ReservoirEnvironment initialised on the test split.
    test_split : DataSplit — provides N.
    device     : torch.device.
    n_mc       : Number of rollouts.

    Returns
    -------
    mc_release : (n_mc, N)  float32  Simulated release (engineering units, m³/s).
    mc_storage : (n_mc, N)  float32  Simulated storage (engineering units, Mm³).
    """
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    policy.eval()
    N = len(test_split.states) - 1

    mc_release = np.empty((n_mc, N), dtype=np.float32)
    mc_storage = np.empty((n_mc, N), dtype=np.float32)

    for k in range(n_mc):
        sim_rel:  list = []
        sim_stor: list = []

        state    = test_env.reset(0)
        stor_val = test_env.storage

        for i in range(N):
            state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                out    = policy(state_t, deterministic=False)
                action = float(out.action.cpu().numpy()[0, 0])

            sim_stor.append(stor_val)
            sim_rel.append(
                float(test_env.normalizer.denormalize(
                    test_env.action_col, np.array([action])
                )[0])
            )

            state, _, done, info = test_env.step(action)
            stor_val = info["storage"]

            if done and i < N - 1:
                state    = test_env.reset(i + 1)
                stor_val = test_env.storage

        mc_release[k] = np.array(sim_rel,  dtype=np.float32)
        mc_storage[k] = np.array(sim_stor, dtype=np.float32)

        if (k + 1) % max(1, n_mc // 10) == 0:
            print(f" {k + 1}/{n_mc}", end="", flush=True)

    return mc_release, mc_storage


# =============================================================================
# Metrics
# =============================================================================

def _compute_test_metrics(
    exp_rel: np.ndarray, sim_rel: np.ndarray,
    exp_stor: np.ndarray, sim_stor: np.ndarray,
) -> dict:
    """Pearson r, nRMSE, and raw RMSE for release and storage (denormalized)."""
    rel_corr,  _ = safe_pearsonr(exp_rel,  sim_rel)
    stor_corr, _ = safe_pearsonr(exp_stor, sim_stor)
    rel_nrmse    = nrmse(exp_rel,  sim_rel)
    stor_nrmse   = nrmse(exp_stor, sim_stor)
    rel_rmse     = float(np.sqrt(np.mean((exp_rel  - sim_rel)  ** 2)))
    stor_rmse    = float(np.sqrt(np.mean((exp_stor - sim_stor) ** 2)))
    return {
        "release_corr":   float(rel_corr),
        "release_nrmse":  float(rel_nrmse),
        "release_rmse":   float(rel_rmse),
        "storage_corr":   float(stor_corr),
        "storage_nrmse":  float(stor_nrmse),
        "storage_rmse":   float(stor_rmse),
    }


def _compute_mc_metrics(
    mc_release: np.ndarray,
    mc_storage: np.ndarray,
    exp_rel:    np.ndarray,
    exp_stor:   np.ndarray,
) -> dict:
    """
    Compute Pearson r and nRMSE for each MC rollout then aggregate mean / std.

    Both metrics are on denormalized (engineering-unit) values.
    nRMSE = sqrt(MSE) / (max_observed − min_observed).

    Parameters
    ----------
    mc_release : (n_mc, N)  Simulated release across rollouts.
    mc_storage : (n_mc, N)  Simulated storage across rollouts.
    exp_rel    : (N,)       Observed release.
    exp_stor   : (N,)       Observed storage.

    Returns
    -------
    dict with *_mean and *_std for release and storage corr + nRMSE.
    """
    n_mc = mc_release.shape[0]
    rel_corrs   = np.empty(n_mc)
    rel_nrmses  = np.empty(n_mc)
    stor_corrs  = np.empty(n_mc)
    stor_nrmses = np.empty(n_mc)

    for k in range(n_mc):
        rel_corrs[k],  _ = safe_pearsonr(exp_rel,  mc_release[k])
        rel_nrmses[k]    = nrmse(exp_rel,  mc_release[k])
        stor_corrs[k], _ = safe_pearsonr(exp_stor, mc_storage[k])
        stor_nrmses[k]   = nrmse(exp_stor, mc_storage[k])

    return {
        "release_corr_mean":  float(np.mean(rel_corrs)),
        "release_corr_std":   float(np.std(rel_corrs)),
        "release_nrmse_mean": float(np.mean(rel_nrmses)),
        "release_nrmse_std":  float(np.std(rel_nrmses)),
        "storage_corr_mean":  float(np.mean(stor_corrs)),
        "storage_corr_std":   float(np.std(stor_corrs)),
        "storage_nrmse_mean": float(np.mean(stor_nrmses)),
        "storage_nrmse_std":  float(np.std(stor_nrmses)),
    }


# =============================================================================
# Plotting helpers  (matplotlib imported lazily — errors gracefully on HPC)
# =============================================================================

def _require_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        sys.exit(
            "\nERROR: matplotlib is required for plotting.\n"
            "  Install with:  pip install matplotlib\n"
        )


def _plot_time_series(
    expert:    np.ndarray,
    simulated: np.ndarray,
    ylabel:    str,
    title:     str,
    save_path: Path,
) -> None:
    """Observed vs simulated time series (single deterministic line)."""
    plt = _require_matplotlib()
    import matplotlib.pyplot as mplt

    N = len(expert)
    x = np.arange(N)

    fig, ax = mplt.subplots(figsize=(14, 4))
    ax.plot(x, expert,    color=_OBSERVED_COLOR,  lw=1.5, label="Observed",   zorder=3)
    ax.plot(x, simulated, color=_SIMULATED_COLOR, lw=1.5, label="Simulated",
            linestyle="--", zorder=2)

    ax.set_title(title, fontsize=14, fontweight="bold", pad=8)
    ax.set_xlabel("Time Steps (Days)", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.tick_params(labelsize=11)
    ax.grid(True, color="grey", lw=0.4, alpha=0.35, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(fontsize=11, loc="upper right", framealpha=0.9)
    ax.set_xlim(0, N - 1)

    mplt.tight_layout()
    mplt.savefig(save_path, dpi=200, bbox_inches="tight")
    mplt.close(fig)
    print(f"  Saved → {save_path.name}")


def _plot_time_series_mc(
    expert:    np.ndarray,
    mc_preds:  np.ndarray,
    ylabel:    str,
    title:     str,
    save_path: Path,
) -> None:
    """
    Observed vs MC-rollout ensemble with IQR shading — matches BC generate_results style.

    Blue solid line    : Observed.
    Salmon shaded band : 25th–75th percentile across MC rollouts.
    Red dashed line    : Median MC rollout.
    """
    plt = _require_matplotlib()
    import matplotlib.pyplot as mplt

    N = len(expert)
    x = np.arange(N)

    median = np.median(mc_preds, axis=0)
    q25    = np.percentile(mc_preds, 25, axis=0)
    q75    = np.percentile(mc_preds, 75, axis=0)

    fig, ax = mplt.subplots(figsize=(14, 4))

    ax.fill_between(x, q25, q75, color=_BAND_COLOR, alpha=0.6,
                    label="25th–75th percentile (IQR)", zorder=1)
    ax.plot(x, median, color=_SIMULATED_COLOR, lw=1.5, linestyle="--",
            label="Median (MC rollouts)", zorder=2)
    ax.plot(x, expert, color=_OBSERVED_COLOR,  lw=1.5,
            label="Observed", zorder=3)

    ax.set_title(title, fontsize=14, fontweight="bold", pad=8)
    ax.set_xlabel("Time Steps (Days)", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.tick_params(labelsize=11)
    ax.grid(True, color="grey", lw=0.4, alpha=0.35, zorder=0)
    ax.set_axisbelow(True)

    # Legend order: Observed → IQR → Median
    handles, labels = ax.get_legend_handles_labels()
    order = [labels.index("Observed"),
             labels.index("25th–75th percentile (IQR)"),
             labels.index("Median (MC rollouts)")]
    ax.legend([handles[i] for i in order], [labels[i] for i in order],
              fontsize=11, loc="upper right", framealpha=0.9, edgecolor="grey")
    ax.set_xlim(0, N - 1)

    mplt.tight_layout()
    mplt.savefig(save_path, dpi=200, bbox_inches="tight")
    mplt.close(fig)
    print(f"  Saved → {save_path.name}")


def _plot_scatter(
    expert:    np.ndarray,
    simulated: np.ndarray,
    xlabel:    str,
    ylabel:    str,
    title:     str,
    save_path: Path,
) -> None:
    """Scatter plot of simulated vs expert with a 1:1 reference line."""
    plt = _require_matplotlib()
    import matplotlib.pyplot as mplt

    fig, ax = mplt.subplots(figsize=(5, 5))
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

    mplt.tight_layout()
    mplt.savefig(save_path, dpi=200, bbox_inches="tight")
    mplt.close(fig)
    print(f"  Saved → {save_path.name}")


def _plot_training_curves(train_log: dict, eval_interval: int, save_path: Path) -> None:
    """Three-panel training history: discriminator, PPO losses, validation score."""
    plt = _require_matplotlib()
    import matplotlib.pyplot as mplt

    stats = train_log.get("training_stats", {})
    if not stats:
        print("  WARNING: training_stats empty — skipping training curves.")
        return

    fig, axes = mplt.subplots(3, 1, figsize=(12, 9), sharex=False)

    # Panel 1: Discriminator
    ax = axes[0]
    for key, color, label in [
        ("disc_loss",  "#2C3E50", "disc_loss"),
        ("expert_acc", "#27AE60", "expert_acc"),
        ("policy_acc", "#E74C3C", "policy_acc"),
    ]:
        if key in stats:
            ax.plot(np.arange(len(stats[key])), stats[key],
                    lw=1.2, label=label, color=color)
    ax.axhline(0.5, color="grey", lw=0.8, linestyle=":")
    ax.set_ylabel("Value", fontsize=11)
    ax.set_title("Discriminator", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3)

    # Panel 2: PPO losses
    ax = axes[1]
    for key, color, label in [
        ("policy_loss", "#2980B9", "policy_loss"),
        ("critic_loss", "#8E44AD", "critic_loss"),
        ("kl_loss",     "#E67E22", "kl_loss"),
        ("entropy",     "#16A085", "entropy"),
    ]:
        if key in stats:
            ax.plot(np.arange(len(stats[key])), stats[key],
                    lw=1.2, label=label, color=color)
    ax.set_ylabel("Value", fontsize=11)
    ax.set_title("PPO Losses", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.3)

    # Panel 3: Validation score
    ax = axes[2]
    val_y = stats.get("val_score", [])
    if val_y:
        val_x = np.arange(len(val_y)) * eval_interval
        ax.plot(val_x, val_y, lw=1.4, color="#C0392B", label="val_score")
        best_idx = int(np.argmax(val_y))
        ax.axvline(val_x[best_idx], color="grey", lw=0.8, linestyle="--",
                   label=f"best @ iter {val_x[best_idx]}")
    ax.set_xlabel("Iteration", fontsize=11)
    ax.set_ylabel("Composite Score", fontsize=11)
    ax.set_title("Validation Score", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, alpha=0.3)

    mplt.tight_layout()
    mplt.savefig(save_path, dpi=200, bbox_inches="tight")
    mplt.close(fig)
    print(f"  Saved → {save_path.name}")


def _plot_reward_contour(
    discriminator,
    config:     AIRLConfig,
    data,
    res_cfg:    dict,
    device:     torch.device,
    save_path:  Path,
    grid_size:  int = 60,
) -> None:
    """
    2D contour of the learned reward g(s, a) over a storage × release grid.

    All state features except storage are fixed at their test-set mean.
    The action (release) ranges over [0, 1] (normalized).
    The storage ranges over [0, 1] (normalized).
    """
    plt = _require_matplotlib()
    import matplotlib.pyplot as mplt

    # Mean of all test state features — shape (state_dim,)
    mean_state = data.test.states.mean(axis=0)  # (state_dim,)

    storage_vals = np.linspace(0.0, 1.0, grid_size)
    release_vals = np.linspace(0.0, 1.0, grid_size)
    SS, RR = np.meshgrid(storage_vals, release_vals)  # each (grid_size, grid_size)

    # Build state tensor: repeat mean_state for all grid points, replace storage col
    n_points = grid_size * grid_size
    states_np = np.tile(mean_state, (n_points, 1)).astype(np.float32)
    states_np[:, 0] = SS.ravel()   # storage is always column 0
    actions_np = RR.ravel().reshape(-1, 1).astype(np.float32)

    states_t  = torch.tensor(states_np,  dtype=torch.float32).to(device)
    actions_t = torch.tensor(actions_np, dtype=torch.float32).to(device)

    with torch.no_grad():
        reward_vals = discriminator.extract_reward_function(states_t, actions_t)
        reward_np   = reward_vals.cpu().numpy().reshape(grid_size, grid_size)

    # Denormalize axes for labelling
    normalizer   = data.normalizer
    action_col   = str(res_cfg["columns"]["action"])
    storage_col  = list(res_cfg["columns"]["state"])[0]

    stor_lo  = normalizer.bounds[storage_col]["min"]
    stor_hi  = normalizer.bounds[storage_col]["max"]
    rel_lo   = normalizer.bounds[action_col]["min"]
    rel_hi   = normalizer.bounds[action_col]["max"]

    stor_axis = stor_lo + storage_vals * (stor_hi - stor_lo)
    rel_axis  = rel_lo  + release_vals * (rel_hi  - rel_lo)

    fig, ax = mplt.subplots(figsize=(7, 6))
    cf = ax.contourf(stor_axis, rel_axis, reward_np, levels=20, cmap="RdYlGn")
    mplt.colorbar(cf, ax=ax, label="g(s, a)  [learned reward]")
    cs = ax.contour(stor_axis, rel_axis, reward_np, levels=10,
                    colors="k", linewidths=0.4, alpha=0.5)
    ax.clabel(cs, inline=True, fontsize=7, fmt="%.2f")

    ax.set_xlabel(f"Storage (Mm³)", fontsize=12)
    ax.set_ylabel(f"Release (m³/s)", fontsize=12)
    ax.set_title("Learned Reward Function  g(s, a)", fontsize=13, fontweight="bold")
    ax.tick_params(labelsize=10)

    mplt.tight_layout()
    mplt.savefig(save_path, dpi=200, bbox_inches="tight")
    mplt.close(fig)
    print(f"  Saved → {save_path.name}")


# =============================================================================
# SHAP wrappers (nn.Module so GradientExplainer can trace gradients)
# =============================================================================

class _PolicySHAPWrapper(nn.Module):
    """Wraps policy: state → scalar deterministic action."""
    def __init__(self, policy: nn.Module) -> None:
        super().__init__()
        self.policy = policy

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        out = self.policy(state, deterministic=True)
        # Return (batch, 1) — GradientExplainer requires 2D output (batch, n_outputs).
        # squeeze(-1) would produce 1D (batch,) which makes shap crash on outputs[:, idx].
        return out.action   # (batch, 1)


class _RewardSHAPWrapper(nn.Module):
    """Wraps reward_net: [state | action] → scalar reward g(s,a)."""
    def __init__(self, reward_net: nn.Module, state_dim: int) -> None:
        super().__init__()
        self.reward_net = reward_net
        self.state_dim  = state_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        state  = x[:, :self.state_dim]
        action = x[:, self.state_dim:]
        return self.reward_net(state, action)  # (batch, 1) — keep 2D for GradientExplainer


def _run_shap(
    wrapper:    nn.Module,
    background: np.ndarray,
    test_data:  np.ndarray,
    n_bg:       int,
    n_test:     int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute GradientExplainer SHAP values on CPU.

    Parameters
    ----------
    wrapper    : nn.Module wrapping policy or reward net.
    background : (N_train, n_features)  background reference data (numpy).
    test_data  : (N_test,  n_features)  data to explain (numpy).
    n_bg       : number of background samples to draw.
    n_test     : number of test samples to explain.

    Returns
    -------
    shap_values : np.ndarray  (min(n_test, N_test), n_features)  — raw SHAP values.
    test_idx    : np.ndarray  (min(n_test, N_test),)             — row indices into
                  test_data used for explanation; pass to caller so month alignment
                  uses the exact same subset without re-drawing the RNG.
    """
    try:
        import shap
    except ImportError:
        sys.exit(
            "\nERROR: shap is required for SHAP analysis.\n"
            "  Install with:  pip install shap\n"
        )

    # Always run SHAP on CPU to avoid device-specific gradient edge cases
    cpu     = torch.device("cpu")
    wrapper = copy.deepcopy(wrapper).to(cpu).eval()

    rng      = np.random.default_rng(42)
    bg_idx   = rng.choice(len(background), size=min(n_bg,   len(background)), replace=False)
    test_idx = rng.choice(len(test_data),  size=min(n_test, len(test_data)),  replace=False)
    test_idx = np.sort(test_idx)   # keep temporal order

    bg_tensor   = torch.tensor(background[bg_idx],  dtype=torch.float32)
    test_tensor = torch.tensor(test_data[test_idx], dtype=torch.float32)

    explainer   = shap.GradientExplainer(wrapper, bg_tensor)
    shap_values = explainer.shap_values(test_tensor)   # (n_test, n_features)

    if isinstance(shap_values, list):
        shap_values = shap_values[0]   # scalar output — single array

    shap_values = np.array(shap_values, dtype=np.float32)
    # GradientExplainer may return (n_test, n_features, 1) when the model output
    # is (batch, 1) rather than (batch,).  Squeeze to (n_test, n_features).
    if shap_values.ndim == 3 and shap_values.shape[-1] == 1:
        shap_values = shap_values.squeeze(-1)

    return shap_values, test_idx


def _plot_shap_total(
    shap_values:   np.ndarray,
    feature_names: list[str],
    title:         str,
    save_path:     Path,
) -> None:
    """Horizontal bar chart of mean |SHAP| per feature, sorted descending."""
    plt = _require_matplotlib()
    import matplotlib.pyplot as mplt

    # Sanitize NaN/Inf before aggregating (can arise from boundary inputs in Beta dist)
    shap_clean = np.nan_to_num(shap_values, nan=0.0, posinf=0.0, neginf=0.0)

    mean_abs  = np.abs(shap_clean).mean(axis=0)           # (n_features,)
    total     = mean_abs.sum()
    pct       = (mean_abs / total * 100) if total > 0 else mean_abs   # (n_features,) in %
    order     = np.argsort(pct)                            # ascending for horizontal bar

    names_sorted = [feature_names[i] for i in order]
    pct_sorted   = pct[order]

    fig, ax = mplt.subplots(figsize=(7, max(3, len(feature_names) * 0.4)))
    bars = ax.barh(names_sorted, pct_sorted, color=_SHAP_COLOR, edgecolor="white", height=0.6)

    # Annotate each bar with its percentage
    for bar, val in zip(bars, pct_sorted):
        ax.text(
            bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}%", va="center", ha="left", fontsize=9,
        )

    ax.set_xlabel("Contribution to network output (%)", fontsize=12)
    max_pct = float(np.nanmax(pct_sorted)) if pct_sorted.size > 0 else 100.0
    ax.set_xlim(0, max_pct * 1.18)   # headroom for labels
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.tick_params(labelsize=10)
    ax.grid(True, axis="x", alpha=0.3)
    ax.set_axisbelow(True)

    mplt.tight_layout()
    mplt.savefig(save_path, dpi=200, bbox_inches="tight")
    mplt.close(fig)
    print(f"  Saved → {save_path.name}")


def _plot_shap_monthly(
    shap_values:   np.ndarray,
    feature_names: list[str],
    months:        np.ndarray,
    title:         str,
    save_path:     Path,
) -> None:
    """
    Heatmap of feature contribution (%) to network output, by month.

    Rows = features (sin/cos month excluded by caller), columns = months 1-12.
    Each column is normalised to sum to 100%: the cell value is the fraction of
    the model's total sensitivity in that month that came from each feature.
    """
    plt = _require_matplotlib()
    import matplotlib.pyplot as mplt

    n_features = len(feature_names)
    month_labels = ["Jan","Feb","Mar","Apr","May","Jun",
                    "Jul","Aug","Sep","Oct","Nov","Dec"]

    # Sanitize NaN/Inf before aggregating
    shap_clean = np.nan_to_num(shap_values, nan=0.0, posinf=0.0, neginf=0.0)

    # Build (n_features, 12) matrix of mean |SHAP| per month
    raw = np.zeros((n_features, 12), dtype=np.float32)
    for m_idx, m in enumerate(range(1, 13)):
        mask = (months == m)
        if mask.sum() > 0:
            raw[:, m_idx] = np.abs(shap_clean[mask]).mean(axis=0)

    # Normalise each month column to 100%
    col_totals = raw.sum(axis=0, keepdims=True)           # (1, 12)
    col_totals = np.where(col_totals == 0, 1, col_totals) # avoid div-by-zero
    matrix = raw / col_totals * 100                       # (n_features, 12) in %

    fig, ax = mplt.subplots(figsize=(11, max(3, n_features * 0.55)))
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd",
                   interpolation="nearest", vmin=0, vmax=100)
    mplt.colorbar(im, ax=ax, label="Contribution to output (%)")

    ax.set_xticks(range(12))
    ax.set_xticklabels(month_labels, fontsize=10)
    ax.set_yticks(range(n_features))
    ax.set_yticklabels(feature_names, fontsize=10)
    ax.set_xlabel("Month", fontsize=12)
    ax.set_ylabel("Feature", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")

    # Cell annotations showing the %
    for i in range(n_features):
        for j in range(12):
            ax.text(j, i, f"{matrix[i, j]:.1f}%",
                    ha="center", va="center", fontsize=7,
                    color="black" if matrix[i, j] < 65 else "white")

    mplt.tight_layout()
    mplt.savefig(save_path, dpi=200, bbox_inches="tight")
    mplt.close(fig)
    print(f"  Saved → {save_path.name}")


# =============================================================================
# CLI
# =============================================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Evaluate the trained AIRL model on the test split and produce "
            "figures.  Requires airl/train.py to have been run first."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--reservoir",   required=True,
                   help="Reservoir name.")
    p.add_argument("--policy_type", default=None,
                   choices=["beta", "lognormal", "hardgating", "softgating"],
                   help="Policy type — inferred from the run folder name if omitted.")
    p.add_argument("--run_id",      type=int, required=True,
                   help="Integer run_id matching the folder created by tune.py.")
    p.add_argument("--device",      default=None,
                   help="Compute device: auto | cpu | cuda | cuda:N | mps.")
    p.add_argument("--n_mc", type=int, default=500,
                   help="Number of stochastic Monte Carlo rollouts for the ensemble plots.")
    p.add_argument("--shap_background", type=int, default=100,
                   help="Number of training samples used as SHAP background.")
    p.add_argument("--shap_test_size",  type=int, default=300,
                   help="Number of test samples explained by SHAP.")
    p.add_argument("--skip_shap",   action="store_true",
                   help="Skip all SHAP computation (useful for quick runs).")
    return p.parse_args()


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args = _parse_args()

    res_cfg_path = _ROOT / "configs" / "reservoirs" / f"{args.reservoir}.yaml"
    airl_base    = _ROOT / "results" / args.reservoir / "airl"

    if not res_cfg_path.exists():
        sys.exit(f"\nERROR: Reservoir config not found: {res_cfg_path}\n")

    with open(res_cfg_path, "r") as f:
        res_cfg = yaml.safe_load(f)

    use_month  = bool(res_cfg["columns"].get("use_month_encoding", True))
    action_col = str(res_cfg["columns"]["action"])
    storage_col= list(res_cfg["columns"]["state"])[0]

    # ------------------------------------------------------------------
    # Locate run folder and load checkpoint
    # ------------------------------------------------------------------
    results_dir = _find_run_folder(airl_base, args.run_id)

    # Infer policy_type from folder name if not given on CLI (e.g. 1_hardgating → hardgating)
    if args.policy_type is None:
        parts = results_dir.name.split("_", 1)
        if len(parts) != 2 or not parts[1]:
            sys.exit(
                f"\nERROR: Cannot infer policy_type from folder '{results_dir.name}'.\n"
                f"  Pass --policy_type explicitly.\n"
            )
        args.policy_type = parts[1]
        print(f"Policy type inferred from folder: {args.policy_type}")

    ckpt = _load_checkpoint(results_dir, args.reservoir, args.policy_type)

    print(f"\nLoaded model.pt")
    print(f"  Reservoir   : {args.reservoir}")
    print(f"  Policy type : {ckpt['policy_type']}")

    # ------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------
    raw_device = args.device if args.device is not None else ckpt["config"]["device"]
    resolved   = _resolve_device(raw_device)
    if resolved.startswith("cuda") and not torch.cuda.is_available():
        print(f"\nWARNING: CUDA not available, falling back to CPU.\n",
              file=sys.stderr)
        resolved = "cpu"
    elif resolved == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        print(f"\nWARNING: MPS not available, falling back to CPU.\n",
              file=sys.stderr)
        resolved = "cpu"
    device = torch.device(resolved)
    print(f"  Device      : {device}")

    # ------------------------------------------------------------------
    # Rebuild models
    # ------------------------------------------------------------------
    policy, discriminator, config, policy_type = _rebuild_models(ckpt, device)

    # ------------------------------------------------------------------
    # Load data + raw splits
    # ------------------------------------------------------------------
    print(f"\nLoading data …")
    data       = load_reservoir_data(res_cfg, res_cfg_path)
    raw_splits = _load_raw_splits(res_cfg, res_cfg_path)
    _, _, test_df = raw_splits
    print(f"  state_dim = {data.state_dim},  test rows = {len(data.test.states)}")

    # state_dim consistency check
    if config.state_dim != data.state_dim:
        sys.exit(
            f"\nERROR: state_dim mismatch: checkpoint={config.state_dim}, "
            f"data={data.state_dim}.  Re-run airl/train.py.\n"
        )

    # ------------------------------------------------------------------
    # Expert values (directly from data — no deterministic rollout needed)
    # ------------------------------------------------------------------
    N        = len(data.test.states) - 1
    exp_rel  = data.test.raw_actions[:N]
    exp_stor = data.normalizer.denormalize(storage_col, data.test.states[:N, 0])

    # ------------------------------------------------------------------
    # Monte Carlo rollouts
    # ------------------------------------------------------------------
    if args.n_mc < 1:
        sys.exit("\nERROR: --n_mc must be a positive integer.\n")

    print(f"\nRunning {args.n_mc} MC rollouts …", end="", flush=True)
    test_env = ReservoirEnvironment(test_df, config, data.normalizer, res_cfg)
    mc_release, mc_storage = _run_mc_rollout(
        policy, test_env, data.test, device, args.n_mc
    )
    print(" done.")

    # Summary statistics across rollouts
    median_rel  = np.median(mc_release, axis=0)
    median_stor = np.median(mc_storage, axis=0)

    # ------------------------------------------------------------------
    # Metrics (mean / std across MC rollouts)
    # ------------------------------------------------------------------
    print("Computing metrics …", end="", flush=True)
    metrics = _compute_mc_metrics(mc_release, mc_storage, exp_rel, exp_stor)
    print(" done.")

    print(
        f"\n  Release — r={metrics['release_corr_mean']:.4f} "
        f"(±{metrics['release_corr_std']:.4f})  "
        f"nRMSE={metrics['release_nrmse_mean']:.4f} "
        f"(±{metrics['release_nrmse_std']:.4f})"
    )
    print(
        f"  Storage — r={metrics['storage_corr_mean']:.4f} "
        f"(±{metrics['storage_corr_std']:.4f})  "
        f"nRMSE={metrics['storage_nrmse_mean']:.4f} "
        f"(±{metrics['storage_nrmse_std']:.4f})"
    )

    # ------------------------------------------------------------------
    # Save test_metrics.json
    # ------------------------------------------------------------------
    train_log_path = results_dir / "train_log.json"
    best_val_score = None
    if train_log_path.exists():
        with open(train_log_path, "r") as f:
            train_log_data = json.load(f)
        best_val_score = train_log_data.get("best_val_score")

    metrics_out = {
        "reservoir":      args.reservoir,
        "policy_type":    policy_type,
        "run_id":         args.run_id,
        "n_mc":           args.n_mc,
        "best_val_score": best_val_score,
        "test_metrics":   {k: round(float(v), 6) for k, v in metrics.items()},
        "timestamp":      datetime.now().isoformat(timespec="seconds"),
    }
    metrics_path = results_dir / "test_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_out, f, indent=2)
    print(f"\nMetrics saved → {metrics_path.name}")

    # ------------------------------------------------------------------
    # Time-series plots (MC envelope)
    # ------------------------------------------------------------------
    print("\nGenerating time-series plots …")
    rel_corr   = metrics["release_corr_mean"]
    rel_nrmse  = metrics["release_nrmse_mean"]
    stor_corr  = metrics["storage_corr_mean"]
    stor_nrmse = metrics["storage_nrmse_mean"]

    _plot_time_series_mc(
        expert    = exp_rel,
        mc_preds  = mc_release,
        ylabel    = f"{action_col.capitalize()} (m³/s)",
        title     = (f"Release — Test Set  ({args.n_mc} MC rollouts)\n"
                     f"r = {rel_corr:.3f} ± {metrics['release_corr_std']:.3f},  "
                     f"nRMSE = {rel_nrmse:.3f} ± {metrics['release_nrmse_std']:.3f}"),
        save_path = results_dir / "release_test.png",
    )
    _plot_time_series_mc(
        expert    = exp_stor,
        mc_preds  = mc_storage,
        ylabel    = "Storage (Mm³)",
        title     = (f"Storage — Test Set  ({args.n_mc} MC rollouts)\n"
                     f"r = {stor_corr:.3f} ± {metrics['storage_corr_std']:.3f},  "
                     f"nRMSE = {stor_nrmse:.3f} ± {metrics['storage_nrmse_std']:.3f}"),
        save_path = results_dir / "storage_test.png",
    )

    # ------------------------------------------------------------------
    # Scatter plots (median MC as point estimate)
    # ------------------------------------------------------------------
    print("Generating scatter plots …")
    _plot_scatter(
        expert    = exp_rel,
        simulated = median_rel,
        xlabel    = f"Observed Release (m³/s)",
        ylabel    = f"Simulated Release (m³/s)",
        title     = f"Release  r={rel_corr:.3f}  (median MC)",
        save_path = results_dir / "scatter_release.png",
    )
    _plot_scatter(
        expert    = exp_stor,
        simulated = median_stor,
        xlabel    = "Observed Storage (Mm³)",
        ylabel    = "Simulated Storage (Mm³)",
        title     = f"Storage  r={stor_corr:.3f}  (median MC)",
        save_path = results_dir / "scatter_storage.png",
    )

    # ------------------------------------------------------------------
    # Training curves
    # ------------------------------------------------------------------
    print("Generating training curves …")
    if train_log_path.exists():
        _plot_training_curves(
            train_log     = train_log_data,
            eval_interval = config.eval_interval,
            save_path     = results_dir / "training_curves.png",
        )
    else:
        print("  WARNING: train_log.json not found — skipping training curves.")

    # ------------------------------------------------------------------
    # Reward contour
    # ------------------------------------------------------------------
    print("Generating reward contour plot …")
    _plot_reward_contour(
        discriminator = discriminator,
        config        = config,
        data          = data,
        res_cfg       = res_cfg,
        device        = device,
        save_path     = results_dir / "reward_contour.png",
    )

    # ------------------------------------------------------------------
    # SHAP
    # ------------------------------------------------------------------
    if args.skip_shap:
        print("\nSHAP skipped (--skip_shap).")
    else:
        print("\nComputing SHAP values …")
        feature_names = _get_feature_names(res_cfg)
        reward_feature_names = feature_names + [action_col]

        # Prepare background (training split) and test data
        train_states = data.train.states.astype(np.float32)  # (N_train, state_dim)
        test_states  = data.test.states.astype(np.float32)   # (N_test,  state_dim)

        # For reward net: concatenate [state | action]
        train_actions = data.train.actions.reshape(-1, 1).astype(np.float32)
        test_actions  = data.test.actions.reshape(-1, 1).astype(np.float32)
        train_combined = np.concatenate([train_states, train_actions], axis=1)
        test_combined  = np.concatenate([test_states,  test_actions],  axis=1)

        # Month indices for test set (used if use_month=True)
        test_months = data.test.dates.month.values  # (N_test,) integers 1-12

        # ---- Policy SHAP ----
        print("  Policy network …", end="", flush=True)
        policy_wrapper = _PolicySHAPWrapper(policy)
        shap_policy, test_idx = _run_shap(
            wrapper    = policy_wrapper,
            background = train_states,
            test_data  = test_states,
            n_bg       = args.shap_background,
            n_test     = args.shap_test_size,
        )
        print(" done.")

        # test_idx returned by _run_shap is the exact subset used for explanation;
        # use it directly so month alignment is guaranteed correct.
        months_subset = test_months[test_idx]

        _plot_shap_total(
            shap_values   = shap_policy,
            feature_names = feature_names,
            title         = "Policy Network — Feature Importance (SHAP)",
            save_path     = results_dir / "shap_policy_total.png",
        )
        if use_month:
            # Strip sin_month / cos_month from the monthly heatmap — they ARE the
            # month encoding, so showing how their SHAP varies by month is circular.
            # They still appear in the total importance bar chart above.
            _month_enc = {"sin_month", "cos_month"}
            pol_non_month_idx = [i for i, n in enumerate(feature_names)
                                 if n not in _month_enc]
            _plot_shap_monthly(
                shap_values   = shap_policy[:, pol_non_month_idx],
                feature_names = [feature_names[i] for i in pol_non_month_idx],
                months        = months_subset,
                title         = "Policy Network — Monthly SHAP Contributions",
                save_path     = results_dir / "shap_policy_monthly.png",
            )

        # ---- Reward network SHAP ----
        print("  Reward network …", end="", flush=True)
        reward_wrapper = _RewardSHAPWrapper(
            reward_net = discriminator.reward_net,
            state_dim  = config.state_dim,
        )
        shap_reward, _ = _run_shap(
            wrapper    = reward_wrapper,
            background = train_combined,
            test_data  = test_combined,
            n_bg       = args.shap_background,
            n_test     = args.shap_test_size,
        )
        print(" done.")

        _plot_shap_total(
            shap_values   = shap_reward,
            feature_names = reward_feature_names,
            title         = "Reward Network — Feature Importance (SHAP)",
            save_path     = results_dir / "shap_reward_total.png",
        )
        if use_month:
            rew_non_month_idx = [i for i, n in enumerate(reward_feature_names)
                                 if n not in _month_enc]
            _plot_shap_monthly(
                shap_values   = shap_reward[:, rew_non_month_idx],
                feature_names = [reward_feature_names[i] for i in rew_non_month_idx],
                months        = months_subset,
                title         = "Reward Network — Monthly SHAP Contributions",
                save_path     = results_dir / "shap_reward_monthly.png",
            )

    # ------------------------------------------------------------------
    # Update run_args.json
    # ------------------------------------------------------------------
    run_args_path = results_dir / "run_args.json"
    run_args: dict = {}
    if run_args_path.exists():
        with open(run_args_path, "r") as f:
            run_args = json.load(f)

    run_args["generate_results"] = {
        "reservoir":        args.reservoir,
        "policy_type":      policy_type,
        "run_id":           args.run_id,
        "device_cli":       args.device,
        "device_used":      resolved,
        "n_mc":             args.n_mc,
        "shap_background":  args.shap_background,
        "shap_test_size":   args.shap_test_size,
        "skip_shap":        args.skip_shap,
        "use_month":        use_month,
        "timestamp":        datetime.now().isoformat(timespec="seconds"),
    }
    with open(run_args_path, "w") as f:
        json.dump(run_args, f, indent=2)
    print(f"\nRun args updated → {run_args_path.name}")
    print("\nDone.\n")


if __name__ == "__main__":
    main()
