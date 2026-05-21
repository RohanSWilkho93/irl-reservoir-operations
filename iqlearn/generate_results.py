"""
iqlearn/generate_results.py
============================
Evaluate the trained IQ-Learn agent on the held-out TEST split and produce
publication-quality figures.

MUST be run AFTER iqlearn/train.py.  Loads agent.pt from the results directory.
If agent.pt is not found the script exits with a clear error.

Expected checkpoint keys (written by iqlearn/train.py)
-------------------------------------------------------
    actor           : ActorNetwork state_dict
    critic          : CriticNetwork state_dict
    critic_target   : CriticNetwork state_dict  (not used here)
    config          : dict  (from dataclasses.asdict(IQLearnConfig))
    policy_type     : str   (e.g. 'hardgating')
    best_epoch      : int
    best_val_score  : float
    reservoir       : str

Architecture convention
-----------------------
Actor  input  : cat([storage_norm, inflow_norm, sin_month, cos_month]) — 4D
Actor  output : zero-inflated Beta action in [0, 1]  (gate × Beta sample)
Critic input  : cat([storage_norm, inflow_norm, release_norm,
                     sin_month, cos_month]) — 5D
Critic output : scalar Q-value  (twin Q: take min)

Monte Carlo rollouts
--------------------
Each rollout simulates a full trajectory through the reservoir water-balance
physics, starting from the first observed test state.  Observed inflow is
used at each step (weather is exogenous and cannot be predicted).  Release
is sampled stochastically from the actor at each step.

Outputs
-------
results/<reservoir>/iqlearn/<run_id>_*/
    test_metrics.json                   — release / storage Pearson r + nRMSE
    release_test.png                    — observed vs. MC median + IQR band
    storage_test.png                    — observed vs. MC median + IQR band
    reward_contours.png                 — 3x4 monthly Q-function contour grid
    shap_qnetwork_monthly/
        importance_heatmap.png          — 12-month x 5-feature SHAP heatmap
    run_args.json                       — updated with generate_results arguments

Usage
-----
python iqlearn/generate_results.py --reservoir conchas --run_id 1
python iqlearn/generate_results.py --reservoir conchas --run_id 1 \\
    --n_mc 100 --device cpu --n_shap_bg 100 --n_shap_explain 50
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------------
import argparse
import json
import sys
import warnings
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Third-party
# ---------------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — safe for headless runs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import yaml
from scipy.ndimage import gaussian_filter

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Project root on sys.path so sibling packages resolve correctly
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils.data    import load_reservoir_data
from utils.metrics import nrmse, safe_pearsonr
from utils.runs    import _find_run_folder

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Plot colours — match BC / DeepMaxEnt style
# ---------------------------------------------------------------------------
_BAND_COLOR     = "#F4A582"   # warm salmon  — IQR shading
_OBSERVED_COLOR = "#1565C0"   # navy blue    — observed line
_MEDIAN_COLOR   = "#C0392B"   # crimson      — median MC line

# ---------------------------------------------------------------------------
# Physical constant
# ---------------------------------------------------------------------------
_SPD = 86_400.0   # seconds per day (release m³/s → volume Mm³ via * _SPD / 1e6)


# =============================================================================
# IQ-Learn configuration
# =============================================================================

@dataclass
class IQLearnConfig:
    """
    IQ-Learn hyperparameters.  Reconstructed from checkpoint['config'].

    Actor  input_dim = state_dim + context_dim  = 2 + 2 = 4
    Critic input_dim = state_dim + action_dim + context_dim = 2 + 1 + 2 = 5
    """
    seed:                   int   = 2048
    state_dim:              int   = 2       # storage + inflow (no month in state)
    action_dim:             int   = 1
    context_dim:            int   = 2       # sin_month + cos_month
    # Actor (matches BC architecture — weights transferred)
    actor_hidden_dim:       int   = 512
    actor_n_hidden_layers:  int   = 5
    # Critic (tuned independently)
    critic_hidden_dim:      int   = 128
    critic_n_hidden_layers: int   = 4
    # Training schedule
    batch_size:             int   = 128
    critic_warm_up_epochs:  int   = 200
    n_epochs:               int   = 500
    # Learning rates
    learning_rate_actor:    float = 4.324e-5
    learning_rate_critic:   float = 2.168e-4
    # IQ-Learn loss coefficients
    gamma:                  float = 0.90
    tau:                    float = 0.00195
    alpha_entropy:          float = 0.01619
    alpha_regularization:   float = 0.09192
    lambda_bc:              float = 0.01906
    # Beta bounds (transferred from BC)
    alpha_min:              float = 1.0
    alpha_max:              float = 10.0
    beta_min:               float = 1.0
    beta_max:               float = 100.0
    # Logging
    log_interval:           int   = 50
    eval_interval:          int   = 50
    device:                 str   = "cpu"

    # Legacy compatibility — code that does config.hidden_dim still works
    @property
    def hidden_dim(self) -> int:
        return self.actor_hidden_dim

    @property
    def n_hidden_layers(self) -> int:
        return self.actor_n_hidden_layers

    @classmethod
    def from_dict(cls, d: dict) -> "IQLearnConfig":
        """Reconstruct from checkpoint['config'] dict; silently drops unknown keys."""
        valid = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in valid})


# =============================================================================
# Actor network — zero-inflated Beta (two-stage gate × Beta)
# =============================================================================

class ActorNetwork(nn.Module):
    """
    Two-stage actor for zero-inflated release distributions.

    Stage 1 — Bernoulli gate  : P(release > 0)
    Stage 2 — Beta head       : continuous amount given gate = 1

    input_dim = state_dim + context_dim = 2 + 2 = 4
    Input  : cat([storage_norm, inflow_norm, sin_month, cos_month])
    Output : (action, log_prob, gate_prob, alpha, beta)
    """

    def __init__(self, cfg: IQLearnConfig) -> None:
        super().__init__()
        input_dim = cfg.state_dim + cfg.context_dim   # 4

        # Shared MLP encoder
        layers: List[nn.Module] = [
            nn.Linear(input_dim, cfg.actor_hidden_dim), nn.ReLU()
        ]
        for _ in range(cfg.actor_n_hidden_layers - 1):
            layers += [
                nn.Linear(cfg.actor_hidden_dim, cfg.actor_hidden_dim), nn.ReLU()
            ]
        self.encoder = nn.Sequential(*layers)

        # Output heads
        self.gate_head  = nn.Linear(cfg.actor_hidden_dim, 1)
        self.alpha_head = nn.Linear(cfg.actor_hidden_dim, cfg.action_dim)
        self.beta_head  = nn.Linear(cfg.actor_hidden_dim, cfg.action_dim)

    def forward(
        self,
        state:        torch.Tensor,   # (B, state_dim)   [storage_norm, inflow_norm]
        context:      torch.Tensor,   # (B, context_dim) [sin_month, cos_month]
        deterministic: bool = False,
    ) -> Tuple[
        torch.Tensor,           # action       (B, action_dim)
        Optional[torch.Tensor], # log_prob     (B, 1) or None in deterministic mode
        torch.Tensor,           # gate_prob    (B, 1)
        torch.Tensor,           # alpha        (B, action_dim)
        torch.Tensor,           # beta         (B, action_dim)
    ]:
        x        = torch.cat([state, context], dim=-1)  # (B, 4)
        features = self.encoder(x)

        # ---- Stage 1: gate ----
        gate_prob = torch.clamp(
            torch.sigmoid(self.gate_head(features)), 0.01, 0.99
        )   # (B, 1)
        gate_dist = torch.distributions.Bernoulli(probs=gate_prob)

        if deterministic:
            gate = gate_prob                        # soft gate — continuous
        else:
            gate = gate_dist.sample()              # hard Bernoulli sample

        # ---- Stage 2: Beta parameters ----
        alpha = torch.clamp(
            F.softplus(self.alpha_head(features)) + 1.0, 1.0, 20.0
        )   # (B, action_dim)
        beta = torch.clamp(
            F.softplus(self.beta_head(features))  + 1.0, 1.0, 20.0
        )   # (B, action_dim)
        beta_dist = torch.distributions.Beta(alpha, beta)

        if deterministic:
            cont = torch.clamp(
                (alpha - 1.0) / (alpha + beta - 2.0), 0.05, 0.99
            )
        else:
            cont = torch.clamp(beta_dist.rsample(), 0.01, 0.99)

        action = gate * cont   # (B, action_dim)

        # ---- Log probability (stochastic path only) ----
        if deterministic:
            log_prob = None
        else:
            gate_lp  = gate_dist.log_prob(gate)          # (B, 1)
            beta_lp  = beta_dist.log_prob(cont)          # (B, action_dim)
            log_prob = torch.clamp(
                (gate_lp + gate * beta_lp).sum(dim=-1, keepdim=True),
                -20.0, 2.0,
            )   # (B, 1)

        return action, log_prob, gate_prob, alpha, beta


# =============================================================================
# Critic network — twin Q-network
# =============================================================================

class CriticNetwork(nn.Module):
    """
    Twin Q-network for IQ-Learn.

    input_dim = state_dim + action_dim + context_dim = 2 + 1 + 2 = 5
    Input  : cat([storage_norm, inflow_norm, release_norm, sin_month, cos_month])
    Output : (Q1, Q2) — take min for pessimistic Q-estimate
    """

    def __init__(self, cfg: IQLearnConfig) -> None:
        super().__init__()
        input_dim = cfg.state_dim + cfg.action_dim + cfg.context_dim  # 5
        self.q1 = self._build_q(input_dim, cfg.critic_hidden_dim, cfg.critic_n_hidden_layers)
        self.q2 = self._build_q(input_dim, cfg.critic_hidden_dim, cfg.critic_n_hidden_layers)

    @staticmethod
    def _build_q(in_dim: int, hidden: int, n_layers: int) -> nn.Sequential:
        layers = [nn.Linear(in_dim, hidden), nn.ReLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.ReLU()]
        layers.append(nn.Linear(hidden, 1))
        return nn.Sequential(*layers)

    def forward(
        self,
        state:   torch.Tensor,   # (B, 2) — [storage_norm, inflow_norm]
        action:  torch.Tensor,   # (B, 1) — release_norm
        context: torch.Tensor,   # (B, 2) — [sin_month, cos_month]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([state, action, context], dim=-1)   # (B, 5)
        return self.q1(x), self.q2(x)


# =============================================================================
# Utilities
# =============================================================================

def _month_to_context(months: np.ndarray) -> np.ndarray:
    """
    Convert integer months (1–12) to circular [sin, cos] encoding.

    Returns
    -------
    np.ndarray  shape (N, 2), dtype float32
    """
    m = np.asarray(months, dtype=np.float32)
    return np.stack([
        np.sin(2.0 * np.pi * m / 12.0),
        np.cos(2.0 * np.pi * m / 12.0),
    ], axis=-1).astype(np.float32)


def _resolve_device(raw: Optional[str]) -> str:
    """
    Resolve CLI device string to a canonical torch device string.
    Mirrors behavioral_cloning.tune._resolve_device.
    """
    if raw is None or raw.lower() == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return raw.lower().strip()


# =============================================================================
# Checkpoint loader
# =============================================================================

def _load_checkpoint(results_dir: Path, reservoir: str) -> dict:
    """
    Load and validate agent.pt produced by iqlearn/train.py.

    Checks
    ------
    1. File exists.
    2. File is loadable by torch.load.
    3. Required keys are present.
    4. Reservoir name in checkpoint matches the CLI argument.
    """
    path = results_dir / "agent.pt"

    # 1. Existence
    if not path.exists():
        sys.exit(
            f"\nERROR: agent.pt not found.\n"
            f"  Expected : {path}\n\n"
            f"  iqlearn/train.py must be run before generate_results.py.  Run:\n"
            f"    python iqlearn/train.py --reservoir {reservoir} --run_id <id>\n"
        )

    # 2. Loadable
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as e:
        sys.exit(
            f"\nERROR: Cannot load agent.pt.\n"
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
            f"\nERROR: agent.pt is missing required keys: {sorted(missing)}\n"
            f"  File : {path}\n\n"
            f"  The checkpoint may be from an older version of train.py.  "
            f"Re-run train.py.\n"
        )

    # 4. Reservoir match
    saved_res = str(ckpt["reservoir"]).lower().strip()
    if saved_res != reservoir.lower().strip():
        sys.exit(
            f"\nERROR: Reservoir mismatch.\n"
            f"  agent.pt was trained on '{ckpt['reservoir']}', "
            f"but --reservoir '{reservoir}' was requested.\n"
        )

    return ckpt


# =============================================================================
# Agent builder
# =============================================================================

def _build_agent(
    ckpt:   dict,
    device: torch.device,
) -> Tuple[ActorNetwork, CriticNetwork, IQLearnConfig]:
    """
    Reconstruct actor + critic from checkpoint and move to device.
    Both networks are put into eval mode.
    """
    cfg    = IQLearnConfig.from_dict(ckpt["config"])
    actor  = ActorNetwork(cfg)
    critic = CriticNetwork(cfg)

    actor.load_state_dict(ckpt["actor"])
    critic.load_state_dict(ckpt["critic"])

    actor.to(device).eval()
    critic.to(device).eval()

    return actor, critic, cfg


# =============================================================================
# Monte Carlo rollouts
# =============================================================================

def _mc_rollout(
    actor:       ActorNetwork,
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

    At each step:
        1. Normalize current (storage, inflow) → actor input.
        2. Sample release action from the actor (stochastic).
        3. Denormalize release → physical units.
        4. Apply water-balance: storage_next = storage + inflow_vol - release_vol.
        5. Clip storage to [min, max] training bounds.
        6. Advance to next observed inflow from test data.

    Returns
    -------
    mc_release : (n_mc, T)  release trajectories  [m³/s]
    mc_storage : (n_mc, T)  storage trajectories  [Mm³]

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

                # Normalize current state
                s_norm = float(
                    normalizer.normalize(storage_col, np.array([storage_cur]))[0]
                )
                q_norm = float(
                    normalizer.normalize(inflow_col, np.array([inflow_cur]))[0]
                )

                state_t = torch.tensor(
                    [[s_norm, q_norm]], dtype=torch.float32, device=device
                )   # (1, 2)
                ctx_t   = torch.tensor(
                    _month_to_context(np.array([month_t])),
                    dtype=torch.float32, device=device,
                )   # (1, 2)

                # Sample stochastic action
                action_norm, _, _, _, _ = actor(state_t, ctx_t, deterministic=False)
                action_norm_val = float(action_norm.squeeze().cpu().item())

                # Denormalize release
                release = float(
                    normalizer.denormalize(action_col, np.array([action_norm_val]))[0]
                )
                release = float(np.clip(release, r_min, r_max))

                # Record CURRENT step values (storage before action, release at action)
                mc_release[i, t] = release
                mc_storage[i, t] = storage_cur

                # Water-balance physics: storage [Mm³], flows [m³/s] → [Mm³]
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
    obs_release: np.ndarray,   # (T,) observed release  [m³/s]
    obs_storage: np.ndarray,   # (T,) observed storage  [Mm³]
) -> dict:
    """
    Compute Pearson r and nRMSE for release and storage across MC rollouts.

    All metrics are computed on DENORMALIZED (original engineering-unit) values.
    nRMSE = sqrt(MSE) / (max_observed − min_observed).

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
# Time-series plot (shared by release and storage)
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

    Same visual style as behavioral_cloning/generate_results.py.
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

    # IQR shading (lowest z-order)
    ax.fill_between(
        x, q25, q75,
        color=_BAND_COLOR, alpha=0.6,
        label="25th–75th percentile (IQR)",
        zorder=1,
    )
    # Median MC line
    ax.plot(
        x, median_pred,
        color=_MEDIAN_COLOR, linestyle="--", linewidth=1.5,
        label="Median (MC rollouts)",
        zorder=2,
    )
    # Observed (top layer)
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
    ax.set_xlim(0, T - 1)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure saved  → {save_path}")


# =============================================================================
# Monthly Q-function (reward) contour plots
# =============================================================================

def _plot_reward_contours(
    critic:      CriticNetwork,
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
    Plot Q-function contours Q(s, a, context) for each of the 12 calendar months.

    For each month:
        1. Build grid over (storage_norm, release_norm).
        2. Extract consecutive observed inflow pairs (preserves autocorrelation).
        3. For each inflow pair: Q(storage_grid, release_grid, inflow_curr, month).
        4. Average Q over all inflow pairs → 2D Q map.
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
    r_grid = np.linspace(0.0, 1.0,  grid_size, dtype=np.float32)  # release in [0, 1]

    # meshgrid: axis-0 = release, axis-1 = storage
    # storage_mesh[i, j] = s_grid[j],  release_mesh[i, j] = r_grid[i]
    storage_mesh, release_mesh = np.meshgrid(s_grid, r_grid)

    storage_flat = storage_mesh.flatten().astype(np.float32)  # (grid_size²,)
    release_flat = release_mesh.flatten().astype(np.float32)  # (grid_size²,)
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

        # Collect consecutive daily inflow pairs for this month
        inflow_pairs: List[float] = []
        for k in range(len(month_idx) - 1):
            ci = month_idx[k]
            ni = month_idx[k + 1]
            if (df.loc[ni, "date"] - df.loc[ci, "date"]).days != 1:
                continue
            v_c = df.loc[ci, inflow_col]
            v_n = df.loc[ni, inflow_col]
            if np.isnan(v_c) or np.isnan(v_n):
                continue
            inflow_pairs.append(float(v_c))

        if not inflow_pairs:
            all_q.append(np.zeros((grid_size, grid_size), dtype=np.float32))
            continue

        # Context tensor fixed for this month (same for all grid points)
        ctx_np = _month_to_context(np.full(n_points, month))   # (n_points, 2)
        ctx_t  = torch.tensor(ctx_np, dtype=torch.float32, device=device)
        act_t  = torch.tensor(
            release_flat.reshape(-1, 1), dtype=torch.float32, device=device
        )   # (n_points, 1)

        q_across: List[np.ndarray] = []
        with torch.no_grad():
            for inflow_curr in inflow_pairs:
                q_norm = float(
                    normalizer.normalize(inflow_col, np.array([inflow_curr]))[0]
                )
                inflow_flat_np = np.full(n_points, q_norm, dtype=np.float32)
                states_np      = np.column_stack((storage_flat, inflow_flat_np))
                states_t       = torch.tensor(
                    states_np, dtype=torch.float32, device=device
                )   # (n_points, 2)

                q1, q2 = critic(states_t, act_t, ctx_t)
                q      = torch.min(q1, q2).squeeze(-1).cpu().numpy()   # (n_points,)
                q_across.append(q.reshape(grid_size, grid_size))

        q_2d = np.mean(q_across, axis=0)
        q_2d = gaussian_filter(q_2d, sigma=0.0)   # no smoothing; set sigma>0 if desired
        all_q.append(q_2d)

    # ---- Global colour scale ----
    global_min = float(min(q.min() for q in all_q))
    global_max = float(max(q.max() for q in all_q))
    # Guard against degenerate case (all-zero Q maps)
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

        # Expert demonstrations overlaid as scatter
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
            ax.set_xlabel(f"{action_col.capitalize()} (m³/s)", fontsize=10)
        if i % 4 == 0:
            ax.set_ylabel(f"{storage_col.capitalize()} (Mm³)", fontsize=10)

    # ---- Shared colorbar ----
    sm = plt.cm.ScalarMappable(
        cmap="RdYlGn", norm=plt.Normalize(global_min, global_max)
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, location="right", shrink=0.8, pad=0.02)
    cbar.set_label("Q(s, a)", fontsize=12, fontweight="bold")

    plt.suptitle(
        f"Learned Q-Function by Month — "
        f"{reservoir.replace('_', ' ').title()}",
        fontsize=16, fontweight="bold", y=0.995,
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure saved  → {save_path}")


# =============================================================================
# Monthly SHAP importance heatmap (Q-network)
# =============================================================================

def _run_shap_monthly_heatmap(
    critic:       CriticNetwork,
    test_df:      pd.DataFrame,
    normalizer,
    device:       torch.device,
    save_dir:     Path,
    storage_col:  str = "storage",
    inflow_col:   str = "net_inflow",
    action_col:   str = "release",
    n_background: int = 100,
    n_explain:    int = 50,
) -> None:
    """
    Compute monthly SHAP feature importance for the Q-network critic.

    For each month (1–12):
        • Background  : random sample of n_background points from all test data.
        • Explain set : random sample of n_explain points from that month only.
        • KernelSHAP  : compute SHAP values for Q_min(s, a, context).
        • Importance  : mean |SHAP| per feature, normalised to sum = 1 (fraction).

    Feature order (matches CriticNetwork.forward input):
        [storage_norm, inflow_norm, release_norm, sin_month, cos_month]

    Saves
    -----
    save_dir/importance_heatmap.png
        Rows = months (Jan–Dec), Columns = features.
        Colour = normalised mean |SHAP|.  Cells annotated with values.
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    df         = test_df.reset_index(drop=True)
    months_arr = df["month"].values.astype(np.int32)
    ctx_arr    = _month_to_context(months_arr)   # (N, 2)

    # Build full normalised feature matrix — shape (N, 5)
    # Column order MUST match CriticNetwork.forward: state[:2], action, context
    s_norm = normalizer.normalize(storage_col, df[storage_col].values)
    q_norm = normalizer.normalize(inflow_col,  df[inflow_col].values)
    r_norm = normalizer.normalize(action_col,  df[action_col].values)

    X_all = np.column_stack([s_norm, q_norm, r_norm, ctx_arr]).astype(np.float64)
    # (N, 5): [storage_norm, inflow_norm, release_norm, sin_month, cos_month]

    # Q-function callable for KernelSHAP — takes (M, 5) float64, returns (M,) float64
    def _q_fn(X: np.ndarray) -> np.ndarray:
        X_f = torch.tensor(X.astype(np.float32), device=device)
        with torch.no_grad():
            state_t  = X_f[:, :2]     # storage, inflow
            action_t = X_f[:, 2:3]    # release
            ctx_t    = X_f[:, 3:]     # sin_month, cos_month
            q1, q2   = critic(state_t, action_t, ctx_t)
            q_min    = torch.min(q1, q2).squeeze(-1)
        return q_min.cpu().numpy().astype(np.float64)

    month_names     = [
        "Jan", "Feb", "Mar", "Apr", "May",
        "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ]
    feature_labels  = ["Storage", "Net Inflow", "Release", "sin(Month)", "cos(Month)"]
    n_features      = 5
    importance_mat  = np.full((12, n_features), np.nan, dtype=np.float64)

    print("  Computing SHAP for each month:")
    rng = np.random.default_rng(seed=42)

    for m_idx, month in enumerate(range(1, 13)):
        month_mask = months_arr == month
        n_month    = int(month_mask.sum())

        if n_month < 2:
            print(f"    Month {month:2d} ({month_names[m_idx]}) — skipped (< 2 samples)")
            continue

        X_month = X_all[month_mask]   # (n_month, 5)

        # Background: random sample from ALL test data (not just this month)
        bg_size = min(n_background, len(X_all))
        bg_idx  = rng.choice(len(X_all), size=bg_size, replace=False)
        X_bg    = X_all[bg_idx]       # (bg_size, 5)

        # Explain: random sample from this month's data
        exp_size = min(n_explain, n_month)
        exp_idx  = rng.choice(n_month, size=exp_size, replace=False)
        X_exp    = X_month[exp_idx]   # (exp_size, 5)

        explainer = shap.KernelExplainer(_q_fn, X_bg)
        shap_vals = explainer.shap_values(X_exp, nsamples=50, silent=True)
        # shape: (exp_size, n_features)

        # Mean absolute SHAP per feature, normalised to sum = 1
        mean_abs = np.abs(shap_vals).mean(axis=0)   # (n_features,)
        total    = mean_abs.sum()
        importance_mat[m_idx] = mean_abs / total if total > 0 else mean_abs

        print(f"    Month {month:2d} ({month_names[m_idx]}) — done "
              f"({exp_size} explain, {bg_size} background)")

    # ---- Heatmap ----
    valid_vals = importance_mat[~np.isnan(importance_mat)]
    vmax       = float(valid_vals.max()) if valid_vals.size > 0 else 1.0

    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(
        importance_mat,
        aspect="auto",
        cmap="YlOrRd",
        vmin=0.0,
        vmax=vmax,
    )

    ax.set_xticks(np.arange(n_features))
    ax.set_xticklabels(feature_labels, fontsize=11, rotation=30, ha="right")
    ax.set_yticks(np.arange(12))
    ax.set_yticklabels(month_names, fontsize=11)

    # Cell annotations
    threshold = 0.4 * vmax  # switch to white text above this value
    for i in range(12):
        for j in range(n_features):
            val = importance_mat[i, j]
            if np.isnan(val):
                continue
            text_color = "white" if val > threshold else "black"
            ax.text(
                j, i, f"{val:.2f}",
                ha="center", va="center",
                fontsize=9, color=text_color,
            )

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Mean |SHAP| (normalised)", fontsize=11)
    ax.set_title(
        "Q-Network Monthly Feature Importance (SHAP)",
        fontsize=14, fontweight="bold", pad=12,
    )
    plt.tight_layout()

    out_path = save_dir / "importance_heatmap.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Heatmap saved → {out_path}")


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
        help="Reservoir name — must match configs/reservoirs/<name>.yaml.",
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
            "Compute device.  Defaults to the device stored in agent.pt.  "
            "Options: auto | cpu | cuda | cuda:N | mps."
        ),
    )
    p.add_argument(
        "--n_mc", type=int, default=100,
        help="Number of Monte Carlo rollouts for the trajectory ensemble.",
    )
    p.add_argument(
        "--n_shap_bg", type=int, default=100,
        help="Number of SHAP background samples (from all test data) per month.",
    )
    p.add_argument(
        "--n_shap_explain", type=int, default=50,
        help="Number of SHAP explain samples (from that month only) per month.",
    )
    return p.parse_args()


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args = _parse_args()

    if args.n_mc < 1:
        sys.exit("\nERROR: --n_mc must be a positive integer.\n")
    if args.n_shap_bg < 1 or args.n_shap_explain < 1:
        sys.exit("\nERROR: --n_shap_bg and --n_shap_explain must be positive integers.\n")

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
    # Load checkpoint
    # ------------------------------------------------------------------
    ckpt = _load_checkpoint(results_dir, args.reservoir)

    print(f"\nLoaded agent.pt")
    print(f"  Reservoir      : {args.reservoir}")
    print(f"  Policy type    : {ckpt['policy_type']}")
    print(
        f"  Best val score : {ckpt['best_val_score']:.4f}  "
        f"(epoch {ckpt['best_epoch'] + 1})"
    )

    # ------------------------------------------------------------------
    # Load normalised data (for bounds + normalizer)
    # ------------------------------------------------------------------
    print("\nLoading data …")
    data = load_reservoir_data(res_cfg, res_cfg_path)
    print(f"  state_dim  = {data.state_dim}")
    print(f"  test rows  = {len(data.test.states)}")

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

    # Rename date column to "date" for consistent internal reference
    if date_col != "date":
        df_raw = df_raw.rename(columns={date_col: "date"})

    # Identify test years (same logic as load_reservoir_data)
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
        f"\n  Actor  : hidden={cfg.actor_hidden_dim}  "
        f"layers={cfg.actor_n_hidden_layers}"
        f"\n  Critic : hidden={cfg.critic_hidden_dim}  "
        f"layers={cfg.critic_n_hidden_layers}"
    )

    # ------------------------------------------------------------------
    # Monte Carlo rollouts
    # ------------------------------------------------------------------
    print(f"\nRunning {args.n_mc} MC rollouts …", end="", flush=True)
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

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    print("Computing metrics …", end="", flush=True)
    metrics = _compute_metrics(mc_release, mc_storage, obs_release, obs_storage)
    print(" done.")

    print(
        f"\n  Release : r = {metrics['release_corr_mean']:.4f} "
        f"± {metrics['release_corr_std']:.4f}  |  "
        f"nRMSE = {metrics['release_nrmse_mean']:.4f} "
        f"± {metrics['release_nrmse_std']:.4f}"
        f"\n  Storage : r = {metrics['storage_corr_mean']:.4f} "
        f"± {metrics['storage_corr_std']:.4f}  |  "
        f"nRMSE = {metrics['storage_nrmse_mean']:.4f} "
        f"± {metrics['storage_nrmse_std']:.4f}"
    )

    # ------------------------------------------------------------------
    # Save test_metrics.json
    # ------------------------------------------------------------------
    results_dir.mkdir(parents=True, exist_ok=True)

    metrics_out = {
        "reservoir":   args.reservoir,
        "policy_type": str(ckpt["policy_type"]),
        "run_id":      args.run_id,
        "n_mc":        args.n_mc,
        "metrics":     {k: round(v, 6) for k, v in metrics.items()},
    }
    with open(results_dir / "test_metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2)
    print(f"\nMetrics saved  → {results_dir / 'test_metrics.json'}")

    # ------------------------------------------------------------------
    # release_test.png
    # ------------------------------------------------------------------
    print("\nPlotting release time series …")
    _plot_time_series(
        obs       = obs_release,
        mc_preds  = mc_release,
        ylabel    = f"{action_col.capitalize()} (m³/s)",
        title     = (
            f"Release — {args.reservoir.replace('_', ' ').title()}\n"
            f"$r$ = {metrics['release_corr_mean']:.3f},  "
            f"nRMSE = {metrics['release_nrmse_mean']:.3f}"
        ),
        dates     = test_dates,
        save_path = results_dir / "release_test.png",
    )

    # ------------------------------------------------------------------
    # storage_test.png
    # ------------------------------------------------------------------
    print("Plotting storage time series …")
    _plot_time_series(
        obs       = obs_storage,
        mc_preds  = mc_storage,
        ylabel    = f"{storage_col.capitalize()} (Mm³)",
        title     = (
            f"Storage — {args.reservoir.replace('_', ' ').title()}\n"
            f"$r$ = {metrics['storage_corr_mean']:.3f},  "
            f"nRMSE = {metrics['storage_nrmse_mean']:.3f}"
        ),
        dates     = test_dates,
        save_path = results_dir / "storage_test.png",
    )

    # ------------------------------------------------------------------
    # reward_contours.png
    # ------------------------------------------------------------------
    print("\nPlotting Q-function reward contours …")
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
    # SHAP monthly importance heatmap
    # ------------------------------------------------------------------
    print("\nRunning SHAP monthly analysis (Q-network) …")
    _run_shap_monthly_heatmap(
        critic        = critic,
        test_df       = test_df,
        normalizer    = data.normalizer,
        device        = device,
        save_dir      = results_dir / "shap_qnetwork_monthly",
        storage_col   = storage_col,
        inflow_col    = inflow_col,
        action_col    = action_col,
        n_background  = args.n_shap_bg,
        n_explain     = args.n_shap_explain,
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
        "run_id":         args.run_id,
        "device_cli":     args.device,
        "device_used":    resolved,
        "n_mc":           args.n_mc,
        "n_shap_bg":      args.n_shap_bg,
        "n_shap_explain": args.n_shap_explain,
        "timestamp":      datetime.now().isoformat(timespec="seconds"),
    }
    with open(run_args_path, "w") as f:
        json.dump(run_args, f, indent=2)
    print(f"Run args updated  → {run_args_path}")

    print(f"\n{'=' * 60}")
    print(f"All outputs saved to: {results_dir}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
