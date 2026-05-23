"""
airl/core.py
============
Core logic for Adversarial Inverse Reinforcement Learning (AIRL) applied to
reservoir operations.

Components
----------
AIRLConfig          — flat dataclass holding all hyperparameters for one trial.
_load_raw_splits    — load and split the raw CSV for the reservoir environment.
ReservoirEnvironment — on-policy rollout environment backed by raw timestep data.
RolloutBuffer       — on-policy PPO buffer (states, actions, rewards, log_probs, values).
ReplayBuffer        — fixed-capacity replay buffer for discriminator updates.
load_expert_from_split — extract (s, a, s', done) from a normalized DataSplits Split.
AIRLAgent           — orchestrates rollout collection, discriminator updates, and PPO.

Training flow
-------------
1. Load data via ``load_reservoir_data`` (normalized DataSplits) and
   ``_load_raw_splits`` (raw DataFrames for the environment simulator).
2. Instantiate ``AIRLAgent(config, policy, policy_type)`` where ``policy`` is a BC-pretrained
   network loaded from ``results/<reservoir>/behavioral_cloning/<run_id>/model.pt``.
3. Call ``agent.add_expert_data(train_split)`` to populate the expert buffer.
4. Call ``agent.warmup_discriminator(train_env, config.warmup_iterations)`` to
   pre-train the discriminator with the policy frozen.
5. Call ``agent.train(train_env, val_env, val_split, trial=trial)`` for the main
   adversarial loop.  Pass an Optuna trial for pruning support (airl/tune.py);
   pass ``trial=None`` for the final training run (airl/train.py).

Optimizer construction
----------------------
``build_airl_networks`` attaches ``policy``, ``reward_net``, ``shaping_net``, and
``critic`` as sub-modules of the returned ``AIRLDiscriminator``.  Calling
``discriminator.parameters()`` therefore yields ALL of their parameters.

``AIRLAgent`` builds THREE separate optimizers to prevent cross-contamination:

    disc_optimizer   — ``reward_net`` + ``shaping_net`` only
    critic_optimizer — ``critic`` only
    policy_optimizer — ``policy`` only

Gradient clipping is applied per-optimizer before each ``step()``.

Train / eval mode
-----------------
``discriminator.policy.eval()`` is set before every discriminator update so that
BC dropout is deterministic during log-probability computation inside
``discriminator.forward()``.  It is restored to ``train()`` before every PPO
update so that policy dropout remains active during the actor update.
"""

from __future__ import annotations

import copy
import gc
import random
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# ---------------------------------------------------------------------------
# Project-root on sys.path (needed when airl/core.py is imported directly).
# ---------------------------------------------------------------------------
import sys
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from networks.airl   import build_airl_networks
from networks.policy import build_policy_network
from utils.data      import Split, Normalizer
from utils.metrics   import rmse, nrmse, safe_pearsonr

# ---------------------------------------------------------------------------
# Physical constant
# ---------------------------------------------------------------------------
_FLOW_TO_VOLUME = 86_400.0 / 1_000_000.0   # m³/s × 1 day  →  Mm³


# =============================================================================
# AIRLConfig
# =============================================================================

@dataclass
class AIRLConfig:
    """
    All hyperparameters needed to build and run one AIRL trial.

    BC architecture fields (hidden_dim, n_hidden_layers, dropout, and the
    distribution-specific fields) are carried forward from best_config.json so
    that ``build_policy_network`` reproduces the exact architecture that was
    pretrained by Behavioral Cloning.  These should NOT be tuned by Optuna in
    airl/tune.py — they are loaded and forwarded verbatim.

    AIRL-specific fields (critic_*, disc_*, lr_*, PPO, KL, schedule) are tuned
    by Optuna in airl/tune.py using the search space defined in
    configs/algorithms/airl.yaml.
    """

    # ------------------------------------------------------------------
    # Data / architecture — always required, no sensible universal default
    # ------------------------------------------------------------------
    state_dim:  int
    action_dim: int = 1

    # ------------------------------------------------------------------
    # BC policy architecture (loaded from best_config.json, not tuned here)
    # ------------------------------------------------------------------
    hidden_dim:      int   = 128
    n_hidden_layers: int   = 3
    dropout:         float = 0.1

    # Beta / Hardgating / Softgating
    alpha_min: float = 1.0
    alpha_max: float = 50.0
    beta_min:  float = 1.0
    beta_max:  float = 50.0

    # Lognormal
    sigma_min:   float = 0.1
    log_epsilon: float = 1.0

    # Hardgating / Softgating
    zero_threshold: float = 0.01

    # Softgating only
    mse_weight:  float = 10.0
    gate_weight: float = 5.0

    # ------------------------------------------------------------------
    # Critic (AIRL-specific, Optuna-tuned)
    # ------------------------------------------------------------------
    critic_hidden_dim:      int = 256
    critic_n_hidden_layers: int = 3

    # ------------------------------------------------------------------
    # Discriminator networks (AIRL-specific, Optuna-tuned)
    # ------------------------------------------------------------------
    disc_hidden_dim:      int   = 256
    disc_n_hidden_layers: int   = 3
    disc_dropout:         float = 0.1

    # ------------------------------------------------------------------
    # Learning rates (Optuna-tuned)
    # ------------------------------------------------------------------
    lr_policy:        float = 3e-5
    lr_critic:        float = 3e-4
    lr_discriminator: float = 1e-4

    # ------------------------------------------------------------------
    # Discriminator training (Optuna-tuned)
    # ------------------------------------------------------------------
    disc_updates:            int   = 5
    warmup_disc_updates:     int   = 10
    gradient_penalty_coef:   float = 10.0
    label_smoothing_epsilon: float = 0.05

    # ------------------------------------------------------------------
    # PPO (Optuna-tuned)
    # ------------------------------------------------------------------
    gamma:        float = 0.99
    gae_lambda:   float = 0.95
    clip_epsilon: float = 0.20
    entropy_coef: float = 0.01
    ppo_epochs:   int   = 5

    # ------------------------------------------------------------------
    # KL regularisation toward BC prior (Optuna-tuned)
    # ------------------------------------------------------------------
    kl_regularization_coef: float = 0.5

    # ------------------------------------------------------------------
    # Training schedule (Optuna-tuned)
    # ------------------------------------------------------------------
    warmup_iterations:       int = 50
    num_iterations:          int = 300
    steps_per_iteration:     int = 2048
    batch_size:              int = 512
    early_stopping_patience: int = 50

    # ------------------------------------------------------------------
    # Replay buffers
    # ------------------------------------------------------------------
    expert_buffer_size: int = 60_000
    policy_buffer_size: int = 120_000

    # ------------------------------------------------------------------
    # Trajectory / rollout parameters
    # ------------------------------------------------------------------
    trajectory_years:     int  = 1
    num_expert_traj:      int  = 100
    align_to_year_start:  bool = True
    end_at_year_boundary: bool = True

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    eval_interval: int   = 10
    max_grad_norm: float = 0.5

    # ------------------------------------------------------------------
    # Runtime
    # ------------------------------------------------------------------
    device:  str  = "cpu"
    seed:    int  = 42
    verbose: bool = False

    @property
    def trajectory_length(self) -> int:
        """Number of timesteps per trajectory (365 × trajectory_years)."""
        return 365 * self.trajectory_years


# =============================================================================
# Data helpers
# =============================================================================

def _load_raw_splits(
    res_cfg:  dict,
    cfg_path: str | Path,
) -> Tuple["pd.DataFrame", "pd.DataFrame", "pd.DataFrame"]:
    """
    Load and split the raw reservoir CSV into train / val / test DataFrames.

    Mirrors the year-based chronological split in ``utils/data.py`` but returns
    the raw (un-normalised) DataFrames required by ``ReservoirEnvironment``.
    Adds helper columns ``_year``, ``_month``, ``_day`` for the environment.

    Parameters
    ----------
    res_cfg  : reservoir config dict from configs/reservoirs/<name>.yaml.
    cfg_path : path to that YAML (used to resolve relative data_path).

    Returns
    -------
    (train_df, val_df, test_df) — raw DataFrames, indexed 0…N-1.
    """
    cfg_path  = Path(cfg_path).resolve()
    data_path = Path(res_cfg["data_path"])

    if not data_path.is_absolute() and not data_path.exists():
        # configs/reservoirs/<name>.yaml is two levels below the repo root
        data_path = cfg_path.parent.parent.parent / data_path

    if not data_path.exists():
        raise FileNotFoundError(
            f"Data file not found: {data_path}\n"
            f"Update data_path in {cfg_path} or pass --data_path."
        )

    date_col = res_cfg["columns"]["date"]
    df       = pd.read_csv(data_path)
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)

    # Helper columns used by ReservoirEnvironment
    df["_year"]  = df[date_col].dt.year
    df["_month"] = df[date_col].dt.month
    df["_day"]   = df[date_col].dt.day

    years   = sorted(df["_year"].unique())
    n_train = int(res_cfg["split"]["train"])
    n_val   = int(res_cfg["split"]["val"])
    n_test  = int(res_cfg["split"]["test"])

    train_years = set(years[:n_train])
    val_years   = set(years[n_train : n_train + n_val])
    test_years  = set(years[n_train + n_val : n_train + n_val + n_test])

    return (
        df[df["_year"].isin(train_years)].reset_index(drop=True),
        df[df["_year"].isin(val_years)].reset_index(drop=True),
        df[df["_year"].isin(test_years)].reset_index(drop=True),
    )


def load_expert_from_split(split: Split) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract expert demonstrations from a normalised ``Split``.

    Returns
    -------
    states      : (N, state_dim)  float32 — normalised states.
    actions     : (N, 1)          float32 — normalised actions ∈ [0, 1].
    next_states : (N, state_dim)  float32 — normalised next states.
    dones       : (N,)            float32 — 1.0 at terminal steps.
    """
    states      = split.states
    actions     = split.actions.reshape(-1, 1).astype(np.float32)
    next_states = split.next_states
    dones       = split.dones.astype(np.float32)
    return states, actions, next_states, dones


# =============================================================================
# ReservoirEnvironment
# =============================================================================

class ReservoirEnvironment:
    """
    On-policy rollout environment for reservoir simulation.

    Stores the policy's simulated storage and reads exogenous inflow timestep-
    by-timestep from the raw DataFrame.  On each ``step(action)``:

        release  = denormalize(action)
        storage += (inflow − release) × FLOW_TO_VOLUME
        storage  = clip(storage, storage_min, storage_max)

    The state returned is identical in format to the vectors produced by
    ``utils/data.py``:  [norm(storage), norm(inflow), sin(2π·m/12), cos(2π·m/12), …].

    Parameters
    ----------
    df         : Raw split DataFrame with columns for all state variables,
                 the action column, and helper columns ``_year``, ``_month``,
                 ``_day``.  Produced by ``_load_raw_splits``.
    config     : ``AIRLConfig`` (for trajectory parameters).
    normalizer : ``Normalizer`` from ``DataSplits`` (column-keyed bounds).
    res_cfg    : Reservoir config dict (column names, month-encoding flag).
    """

    def __init__(
        self,
        df:         pd.DataFrame,
        config:     AIRLConfig,
        normalizer: Normalizer,
        res_cfg:    dict,
    ) -> None:
        self.df         = df.copy().reset_index(drop=True)
        self.config     = config
        self.normalizer = normalizer

        self.state_cols  = list(res_cfg["columns"]["state"])
        self.action_col  = str(res_cfg["columns"]["action"])
        self.storage_col = self.state_cols[0]   # first state col is always storage
        self.use_month   = bool(res_cfg["columns"].get("use_month_encoding", True))

        # Action bounds (for clipping denormalized release)
        b = normalizer.bounds[self.action_col]
        self.action_lo = float(b["min"])
        self.action_hi = float(b["max"])

        # Storage bounds (for clipping simulated storage)
        sb = normalizer.bounds[self.storage_col]
        self.storage_lo = float(sb["min"])
        self.storage_hi = float(sb["max"])

        self._build_year_index()
        self.current_idx:             int   = 0
        self.episode_start_idx:       int   = 0
        self.steps_in_episode:        int   = 0
        self.storage:                 float = 0.0
        self._year_boundaries_crossed: int  = 0

    # ------------------------------------------------------------------
    # Year index
    # ------------------------------------------------------------------

    def _build_year_index(self) -> None:
        """Map year → first row index of that year."""
        self.year_starts: Dict[int, int] = {}
        for idx, row in self.df.iterrows():
            yr = int(row["_year"])
            if int(row["_month"]) == 1 and int(row["_day"]) == 1:
                if yr not in self.year_starts:
                    self.year_starts[yr] = int(idx)

    def _is_year_boundary(self, idx: int) -> bool:
        """True if ``idx`` is the last timestep before a new calendar year."""
        if idx >= len(self.df) - 1:
            return True
        c = self.df.iloc[idx]
        n = self.df.iloc[idx + 1]
        return (
            int(c["_month"]) == 12 and int(c["_day"]) == 31
            and int(n["_month"]) == 1 and int(n["_day"]) == 1
        )

    def get_valid_start_indices(self) -> List[int]:
        """Row indices at which a full trajectory of length ``trajectory_length`` fits."""
        return [
            idx for idx in sorted(self.year_starts.values())
            if idx + self.config.trajectory_length <= len(self.df)
        ]

    # ------------------------------------------------------------------
    # State construction
    # ------------------------------------------------------------------

    def _get_state(self, idx: int) -> np.ndarray:
        """Build a normalised state vector matching the utils/data.py format."""
        row   = self.df.iloc[idx]
        parts: List[float] = []

        for col in self.state_cols:
            if col == self.storage_col:
                raw = self.storage
            else:
                raw = float(row[col])
            parts.append(
                float(self.normalizer.normalize(col, np.array([raw]))[0])
            )

        if self.use_month:
            m     = float(row["_month"])
            angle = 2.0 * np.pi * m / 12.0
            parts.append(float(np.sin(angle)))
            parts.append(float(np.cos(angle)))

        return np.array(parts, dtype=np.float32)

    # ------------------------------------------------------------------
    # Gym-style interface
    # ------------------------------------------------------------------

    def reset(self, start_idx: Optional[int] = None) -> np.ndarray:
        """
        Reset the environment to a (possibly random) year-start index.

        If ``align_to_year_start`` is True and ``start_idx`` is not a valid
        year start, snap to the nearest valid start.
        """
        valid = self.get_valid_start_indices()

        if not valid:
            start_idx = 0 if start_idx is None else min(start_idx, len(self.df) - 2)
        elif start_idx is None:
            start_idx = random.choice(valid)
        elif start_idx not in valid and self.config.align_to_year_start:
            start_idx = min(valid, key=lambda x: abs(x - start_idx))

        self.current_idx              = start_idx
        self.episode_start_idx        = start_idx
        self.steps_in_episode         = 0
        self._year_boundaries_crossed = 0          # O(1) boundary counter
        self.storage = float(self.df.iloc[start_idx][self.storage_col])
        return self._get_state(start_idx)

    def step(self, action: np.ndarray | float) -> Tuple[np.ndarray, float, bool, dict]:
        """
        Advance one timestep using the policy's normalised action.

        Parameters
        ----------
        action : Normalised release in [0, 1].  Accepts a scalar or a numpy
                 array (shape ``(1,)`` or ``(1, 1)``).

        Returns
        -------
        next_state : np.ndarray (state_dim,).  All-zeros tensor when done.
        reward     : 0.0  (filled in by the discriminator in the agent loop).
        done       : bool
        info       : dict with ``storage`` (raw Mm³) and ``release`` (raw m³/s).
        """
        # Denormalize action → actual release (m³/s)
        if isinstance(action, np.ndarray):
            action_val = float(action.flatten()[0])
        else:
            action_val = float(action)
        action_val = float(np.clip(action_val, 0.0, 1.0))

        release = float(
            self.normalizer.denormalize(self.action_col, np.array([action_val]))[0]
        )
        release = float(np.clip(release, self.action_lo, self.action_hi))

        # Water balance (convert flow × time → volume)
        inflow    = float(self.df.iloc[self.current_idx][self.state_cols[1]])
        self.storage = float(np.clip(
            self.storage + (inflow - release) * _FLOW_TO_VOLUME,
            self.storage_lo,
            self.storage_hi,
        ))

        # Determine terminal condition
        at_boundary = self._is_year_boundary(self.current_idx)
        self.current_idx      += 1
        self.steps_in_episode += 1

        # O(1) boundary tracking — increment counter instead of rescanning history
        if at_boundary:
            self._year_boundaries_crossed += 1

        done = (
            self.current_idx >= len(self.df) - 1
            or self.steps_in_episode >= self.config.trajectory_length
            or (
                self.config.end_at_year_boundary
                and at_boundary
                and self._year_boundaries_crossed >= self.config.trajectory_years
            )
        )

        next_state = (
            np.zeros(len(self.state_cols) + (2 if self.use_month else 0), dtype=np.float32)
            if done
            else self._get_state(self.current_idx)
        )

        return next_state, 0.0, done, {"storage": self.storage, "release": release}


# =============================================================================
# Buffers
# =============================================================================

class RolloutBuffer:
    """On-policy PPO buffer — stores a single rollout, cleared after each update."""

    def __init__(self) -> None:
        self.clear()

    def push(
        self,
        state:     np.ndarray,
        action:    np.ndarray,
        reward:    float,
        next_state: np.ndarray,
        done:      float,
        log_prob:  float,
        value:     float,
    ) -> None:
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)

    def get(self, device: torch.device) -> Dict[str, torch.Tensor]:
        """Return all fields as float32 tensors on ``device``."""
        return {
            k: torch.tensor(np.array(v), dtype=torch.float32).to(device)
            for k, v in [
                ("states",      self.states),
                ("actions",     self.actions),
                ("rewards",     self.rewards),
                ("next_states", self.next_states),
                ("dones",       self.dones),
                ("log_probs",   self.log_probs),
                ("values",      self.values),
            ]
        }

    def clear(self) -> None:
        self.states:      list = []
        self.actions:     list = []
        self.rewards:     list = []
        self.next_states: list = []
        self.dones:       list = []
        self.log_probs:   list = []
        self.values:      list = []

    def __len__(self) -> int:
        return len(self.states)


class ReplayBuffer:
    """Fixed-capacity replay buffer for discriminator updates."""

    def __init__(self, capacity: int = 100_000) -> None:
        self.buffer: deque = deque(maxlen=capacity)

    def push(
        self,
        state:      np.ndarray,
        action:     np.ndarray | float,
        next_state: np.ndarray,
        done:       bool | float = False,
    ) -> None:
        if isinstance(action, (int, float)):
            action = np.array([action], dtype=np.float32)
        elif isinstance(action, np.ndarray) and action.ndim == 0:
            action = action.reshape(1).astype(np.float32)
        self.buffer.append((
            state.astype(np.float32),
            action.astype(np.float32),
            next_state.astype(np.float32),
            float(done),
        ))

    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        batch  = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        s, a, ns, d = zip(*batch)
        return np.array(s), np.array(a), np.array(ns), np.array(d)

    def clear(self) -> None:
        self.buffer.clear()

    def __len__(self) -> int:
        return len(self.buffer)


# =============================================================================
# Training utilities
# =============================================================================


def _compute_composite_score(
    release_corr:   float,
    storage_corr:   float,
    release_nrmse:  float,
    storage_nrmse:  float,
    expert_acc:     float,
    policy_acc:     float,
) -> float:
    """
    Composite validation score used for Optuna optimisation and early stopping.

    Weights:
        50 % — discriminator balance score (penalises deviation from D=0.5)
        25 % — release performance (corr + 1-nrmse)
        25 % — storage performance (corr + 1-nrmse)

    Returns a value in [0, 1].  0.0 on NaN inputs.
    """
    if any(
        x != x  # NaN check without importing math
        for x in [release_corr, storage_corr, release_nrmse, storage_nrmse,
                  expert_acc, policy_acc]
    ):
        return 0.0

    disc_score     = max(0.0, 1.0 - abs(expert_acc - 0.5) - abs(policy_acc - 0.5))
    rel_corr_norm  = (release_corr + 1.0) / 2.0
    stor_corr_norm = (storage_corr + 1.0) / 2.0
    rel_nrmse_sc   = max(0.0, min(1.0, 1.0 - release_nrmse))
    stor_nrmse_sc  = max(0.0, min(1.0, 1.0 - storage_nrmse))

    return (
        0.50 * disc_score
        + 0.125 * rel_corr_norm  + 0.125 * rel_nrmse_sc
        + 0.125 * stor_corr_norm + 0.125 * stor_nrmse_sc
    )


# =============================================================================
# AIRLAgent
# =============================================================================

class AIRLAgent:
    """
    AIRL training agent.

    Combines:
      • A policy network (initialised from BC weights).
      • A frozen BC prior (deep copy of the policy at construction time).
      • A CriticNetwork for PPO advantage estimation.
      • An AIRLDiscriminator (reward net + shaping net + policy reference).

    Parameters
    ----------
    config    : ``AIRLConfig`` for this trial.
    policy    : Pre-loaded policy network (any of the four types from
                networks/policy.py).  The agent initialises its own policy
                with the same weights and stores a frozen deep copy as
                ``self.bc_policy``.

    Attributes
    ----------
    discriminator : AIRLDiscriminator returned by ``build_airl_networks``.
    critic        : CriticNetwork attached as ``discriminator.critic``.
    bc_policy     : Frozen deep copy of the input policy.  Parameters have
                    ``requires_grad=False``.
    expert_buffer : ReplayBuffer for expert transitions.
    policy_buffer : ReplayBuffer for policy rollout transitions.
    training_stats: defaultdict(list) — accumulated per-update stats.
    """

    def __init__(
        self,
        config:      AIRLConfig,
        policy:      nn.Module,
        policy_type: str,
    ) -> None:
        self.config      = config
        self.device      = torch.device(config.device)
        self.policy_type = policy_type

        # ------------------------------------------------------------------
        # Sanity-check: verify the supplied network actually matches the
        # declared policy_type before loading any weights.
        # ------------------------------------------------------------------
        inferred = self._infer_policy_type(policy)
        if inferred != policy_type:
            raise ValueError(
                f"policy_type mismatch: declared '{policy_type}' but the "
                f"supplied network class '{type(policy).__name__}' maps to "
                f"'{inferred}'.\n"
                f"Ensure the policy was built with "
                f"build_policy_network('{policy_type}', config)."
            )

        # ------------------------------------------------------------------
        # Policy: clone BC weights into a trainable copy
        # ------------------------------------------------------------------
        self.policy = build_policy_network(
            policy_type = policy_type,   # explicit — no class-name inference
            config      = config,
        ).to(self.device)
        self.policy.load_state_dict(policy.state_dict())

        # Frozen BC prior — used only for KL regularisation
        self.bc_policy = copy.deepcopy(self.policy)
        self.bc_policy.eval()
        for p in self.bc_policy.parameters():
            p.requires_grad = False

        # ------------------------------------------------------------------
        # Discriminator (wraps reward_net, shaping_net, policy)
        # Critic is attached as discriminator.critic by build_airl_networks
        # ------------------------------------------------------------------
        self.discriminator = build_airl_networks(config, self.policy).to(self.device)
        self.critic = self.discriminator.critic  # type: ignore[attr-defined]

        # ------------------------------------------------------------------
        # Separate optimizers — see module docstring for rationale
        # ------------------------------------------------------------------
        disc_params = (
            list(self.discriminator.reward_net.parameters())
            + list(self.discriminator.shaping_net.parameters())
        )
        self.disc_optimizer   = optim.Adam(disc_params,                       lr=config.lr_discriminator)
        self.critic_optimizer = optim.Adam(self.critic.parameters(),          lr=config.lr_critic)
        self.policy_optimizer = optim.Adam(self.policy.parameters(),          lr=config.lr_policy)

        # ------------------------------------------------------------------
        # Replay buffers
        # ------------------------------------------------------------------
        self.expert_buffer = ReplayBuffer(config.expert_buffer_size)
        self.policy_buffer = ReplayBuffer(config.policy_buffer_size)

        # ------------------------------------------------------------------
        # Logging
        # ------------------------------------------------------------------
        self.training_stats: Dict[str, list] = defaultdict(list)

    # ------------------------------------------------------------------
    # Policy-type inference
    # ------------------------------------------------------------------

    @staticmethod
    def _infer_policy_type(policy: nn.Module) -> str:
        """
        Infer the policy type string from the network class name.

        Raises ValueError if the class name is not one of the four known types.
        """
        cls = type(policy).__name__.lower()
        if   "beta"     in cls and "hard" not in cls and "soft" not in cls:
            return "beta"
        elif "lognormal" in cls:
            return "lognormal"
        elif "hard"  in cls:
            return "hardgating"
        elif "soft"  in cls:
            return "softgating"
        raise ValueError(
            f"Cannot infer policy type from class name '{type(policy).__name__}'.\n"
            f"Pass a network created by build_policy_network()."
        )

    # ------------------------------------------------------------------
    # Expert data loading
    # ------------------------------------------------------------------

    def add_expert_data(
        self,
        states:      np.ndarray,
        actions:     np.ndarray,
        next_states: np.ndarray,
        dones:       Optional[np.ndarray] = None,
    ) -> None:
        """Populate the expert replay buffer from pre-extracted arrays."""
        if dones is None:
            dones = np.zeros(len(states), dtype=np.float32)
        for s, a, ns, d in zip(states, actions, next_states, dones):
            self.expert_buffer.push(s, a, ns, d)

    def add_expert_from_split(self, split: Split) -> None:
        """Convenience wrapper: extract arrays from a ``Split`` and add."""
        s, a, ns, d = load_expert_from_split(split)
        self.add_expert_data(s, a, ns, d)

    # ------------------------------------------------------------------
    # Stratified sampling (balances zero / nonzero actions)
    # ------------------------------------------------------------------

    def _sample_stratified(
        self,
        buffer:     ReplayBuffer,
        batch_size: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample ``batch_size`` transitions, balancing zero / nonzero actions.

        Falls back to uniform random sampling when the buffer is smaller than
        ``batch_size``.  The stratification threshold is fixed at 0.01
        (normalised), consistent with the reference implementations.
        """
        samples = list(buffer.buffer)
        if len(samples) <= batch_size:
            return buffer.sample(len(samples))

        actions = np.array([s[1] for s in samples]).flatten()
        zero_idx    = np.where(actions < 0.01)[0]
        nonzero_idx = np.where(actions >= 0.01)[0]
        n_zero      = batch_size // 2

        fallback = np.arange(len(samples))
        idx_zero    = np.random.choice(zero_idx    if len(zero_idx)    > 0 else fallback, n_zero,                replace=True)
        idx_nonzero = np.random.choice(nonzero_idx if len(nonzero_idx) > 0 else fallback, batch_size - n_zero, replace=True)

        batch = [samples[i] for i in np.concatenate([idx_zero, idx_nonzero])[:batch_size]]
        s, a, ns, d = zip(*batch)
        return np.array(s), np.array(a), np.array(ns), np.array(d)

    # ------------------------------------------------------------------
    # Gradient penalty (WGAN-GP style)
    # ------------------------------------------------------------------

    def compute_gradient_penalty(
        self,
        e_batch: Tuple[np.ndarray, ...],
        p_batch: Tuple[np.ndarray, ...],
    ) -> torch.Tensor:
        """
        WGAN-GP gradient penalty on the AIRL advantage function f(s, a, s').

        Interpolates uniformly between expert and policy transitions, computes
        f on the interpolated inputs, then penalises ‖∇f‖ deviating from 1.

        Returns
        -------
        Scalar torch.Tensor.
        """
        e_s, e_a, e_ns, _ = [torch.tensor(x, dtype=torch.float32).to(self.device) for x in e_batch]
        p_s, p_a, p_ns, _ = [torch.tensor(x, dtype=torch.float32).to(self.device) for x in p_batch]

        alpha = torch.rand(e_s.shape[0], 1, device=self.device)
        # .detach() makes each tensor a proper leaf node so autograd.grad
        # computes df/d(i_x) cleanly without retaining the interpolation graph.
        i_s  = (alpha * e_s  + (1.0 - alpha) * p_s).detach().requires_grad_(True)
        i_a  = (alpha * e_a  + (1.0 - alpha) * p_a).detach().requires_grad_(True)
        i_ns = (alpha * e_ns + (1.0 - alpha) * p_ns).detach().requires_grad_(True)

        f = self.discriminator.compute_f(i_s, i_a, i_ns)   # (B, 1)

        grads = torch.autograd.grad(
            outputs     = f,
            inputs      = [i_s, i_a, i_ns],
            grad_outputs= torch.ones_like(f),
            create_graph= True,
            retain_graph= True,
        )
        # Concatenate gradients along feature dim, compute per-sample L2 norm
        grad_cat  = torch.cat([g.view(e_s.shape[0], -1) for g in grads], dim=1)
        penalty   = ((grad_cat.norm(2, dim=1) - 1.0) ** 2).mean()
        return penalty

    # ------------------------------------------------------------------
    # Discriminator update
    # ------------------------------------------------------------------

    def update_discriminator(
        self,
        batch_size:  int,
        num_updates: int = 1,
    ) -> Dict[str, float]:
        """
        Run ``num_updates`` gradient steps on the discriminator.

        Uses binary cross-entropy with label smoothing plus a WGAN-GP penalty.
        Returns a dict of averaged statistics over all updates.
        """
        half = batch_size // 2
        if len(self.expert_buffer) < half or len(self.policy_buffer) < half:
            return {"disc_loss": 0.0, "expert_acc": 0.5, "policy_acc": 0.5}

        eps         = self.config.label_smoothing_epsilon
        total_loss  = 0.0
        total_e_acc = 0.0
        total_p_acc = 0.0

        # Set policy to eval so BC dropout is deterministic during log_pi computation
        self.discriminator.policy.eval()

        for _ in range(num_updates):
            e_batch = self._sample_stratified(self.expert_buffer, half)
            p_batch = self._sample_stratified(self.policy_buffer, half)

            e_s, e_a, e_ns, _ = [torch.tensor(x, dtype=torch.float32).to(self.device) for x in e_batch]
            p_s, p_a, p_ns, _ = [torch.tensor(x, dtype=torch.float32).to(self.device) for x in p_batch]

            e_out = self.discriminator(e_s, e_a, e_ns)
            p_out = self.discriminator(p_s, p_a, p_ns)

            bce = (
                F.binary_cross_entropy(e_out, torch.ones_like(e_out)  * (1.0 - eps))
                + F.binary_cross_entropy(p_out, torch.zeros_like(p_out) + eps)
            )
            gp   = self.compute_gradient_penalty(e_batch, p_batch)
            loss = bce + self.config.gradient_penalty_coef * gp

            self.disc_optimizer.zero_grad()
            self.policy_optimizer.zero_grad()   # prevent discriminator backward from leaving stale grads on policy
            loss.backward()
            disc_params = (
                list(self.discriminator.reward_net.parameters())
                + list(self.discriminator.shaping_net.parameters())
            )
            torch.nn.utils.clip_grad_norm_(disc_params, self.config.max_grad_norm)
            self.disc_optimizer.step()
            self.policy_optimizer.zero_grad()   # clear any policy grads accumulated via log_π in BCE

            with torch.no_grad():
                total_loss  += bce.item()
                total_e_acc += (self.discriminator(e_s, e_a, e_ns) > 0.5).float().mean().item()
                total_p_acc += (self.discriminator(p_s, p_a, p_ns) < 0.5).float().mean().item()

        self.discriminator.policy.train()

        stats = {
            "disc_loss":  total_loss  / num_updates,
            "expert_acc": total_e_acc / num_updates,
            "policy_acc": total_p_acc / num_updates,
        }
        for k, v in stats.items():
            self.training_stats[k].append(v)
        return stats

    # ------------------------------------------------------------------
    # GAE advantage estimation
    # ------------------------------------------------------------------

    def compute_gae(
        self,
        rewards:     torch.Tensor,
        values:      torch.Tensor,
        next_values: torch.Tensor,
        dones:       torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Generalised Advantage Estimates (GAE-λ).

        Parameters
        ----------
        rewards, values, next_values, dones : all shape (T,).

        Returns
        -------
        advantages : (T,)
        returns    : (T,)  = advantages + values
        """
        gamma, lam = self.config.gamma, self.config.gae_lambda
        T          = len(rewards)
        advantages = torch.zeros_like(rewards)
        last_gae   = 0.0

        for t in reversed(range(T)):
            nv      = 0.0 if dones[t] > 0.5 else float(next_values[t])
            delta   = float(rewards[t]) + gamma * nv - float(values[t])
            last_gae = delta if dones[t] > 0.5 else delta + gamma * lam * last_gae
            advantages[t] = last_gae

        return advantages, advantages + values

    # ------------------------------------------------------------------
    # PPO policy update
    # ------------------------------------------------------------------

    def update_policy_ppo(self, rollout: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Run ``ppo_epochs`` gradient steps on the policy and critic.

        Uses clipped surrogate objective + KL regularisation toward BC prior.
        Entropy bonus is estimated as −E[log π] from the rollout actions.

        Returns per-epoch averaged statistics.
        """
        states     = rollout["states"]
        actions    = rollout["actions"]
        old_lp     = rollout["log_probs"]     # (T,) — 1D
        rewards    = rollout["rewards"]        # (T,)
        next_states= rollout["next_states"]
        dones      = rollout["dones"]
        old_values = rollout["values"]         # (T,)

        with torch.no_grad():
            next_values = self.critic(next_states).squeeze(-1)  # (T,)

        advantages, returns = self.compute_gae(rewards, old_values, next_values, dones)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        self.discriminator.policy.train()

        total_p, total_c, total_kl, total_ent = 0.0, 0.0, 0.0, 0.0

        for _ in range(self.config.ppo_epochs):
            new_lp = self.policy.get_log_prob(states, actions).squeeze(-1)  # (T,)
            new_v  = self.critic(states).squeeze(-1)                         # (T,)

            # Entropy estimate: H ≈ −E[log π]
            entropy = -new_lp.mean()

            # PPO clipped surrogate
            ratio = torch.exp(new_lp - old_lp)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - self.config.clip_epsilon,
                                        1.0 + self.config.clip_epsilon) * advantages
            p_loss = -torch.min(surr1, surr2).mean()

            # KL regularisation — reuse new_lp from the forward pass already done
            # above.  Calling _compute_kl_from_bc here would trigger a THIRD
            # separate encoder pass with yet another dropout mask, making the
            # gradient of total_loss internally inconsistent.
            if self.config.kl_regularization_coef > 0.0:
                with torch.no_grad():
                    bc_lp = self.bc_policy.get_log_prob(states, actions).squeeze(-1)  # (T,)
                kl_loss = self.config.kl_regularization_coef * (
                    (new_lp - bc_lp).clamp(min=0.0).mean()
                )
            else:
                kl_loss = torch.tensor(0.0, device=self.device)

            total_loss = p_loss + kl_loss - self.config.entropy_coef * entropy

            # Critic loss
            c_loss = F.mse_loss(new_v, returns)

            # Policy step
            self.policy_optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_grad_norm)
            self.policy_optimizer.step()

            # Critic step
            self.critic_optimizer.zero_grad()
            c_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.max_grad_norm)
            self.critic_optimizer.step()

            total_p   += p_loss.item()
            total_c   += c_loss.item()
            total_kl  += kl_loss.item() if isinstance(kl_loss, torch.Tensor) else kl_loss
            total_ent += entropy.item()

        n     = self.config.ppo_epochs
        stats = {
            "policy_loss": total_p   / n,
            "critic_loss": total_c   / n,
            "kl_loss":     total_kl  / n,
            "entropy":     total_ent / n,
            "mean_reward": float(rollout["rewards"].mean()),
        }
        for k, v in stats.items():
            self.training_stats[k].append(v)
        return stats

    # ------------------------------------------------------------------
    # Rollout collection
    # ------------------------------------------------------------------

    def collect_rollout(
        self,
        env:           ReservoirEnvironment,
        min_steps:     int  = 2048,
        deterministic: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Collect at least ``min_steps`` on-policy transitions.

        Stores transitions in both the PPO ``RolloutBuffer`` (for advantage
        estimation) and the ``policy_buffer`` (for discriminator updates).

        Returns
        -------
        Dict of tensors on ``self.device`` — keys match ``RolloutBuffer.get()``.
        """
        rollout     = RolloutBuffer()
        total_steps = 0

        # Switch policy to eval so both the action-sampling forward pass and the
        # log_prob forward pass use the same deterministic dropout state.  With
        # policy in train mode, two separate calls to the encoder produce different
        # dropout masks → different alpha/beta → the stored log_prob does not match
        # the distribution that generated the action, corrupting PPO ratios.
        was_training = self.policy.training
        self.policy.eval()

        while total_steps < min_steps:
            state = env.reset()
            done  = False

            while not done:
                state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)

                with torch.no_grad():
                    out      = self.policy(state_t, deterministic=deterministic)
                    action_t = out.action                              # (1, action_dim)
                    log_prob = self.policy.get_log_prob(state_t, action_t)   # (1, 1)
                    value    = self.critic(state_t)                    # (1, 1)

                action_np  = action_t.cpu().numpy()[0]                 # (action_dim,)
                next_state, _, done, _ = env.step(action_np)

                with torch.no_grad():
                    ns_t   = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(self.device)
                    reward = self.discriminator.get_reward(state_t, action_t, ns_t).item()

                rollout.push(
                    state      = state,
                    action     = action_np,
                    reward     = reward,
                    next_state = next_state,
                    done       = float(done),
                    log_prob   = float(log_prob.item()),
                    value      = float(value.item()),
                )
                self.policy_buffer.push(state, action_np, next_state, done)

                state        = next_state
                total_steps += 1

        # Restore original train/eval mode so callers (e.g. warmup_discriminator,
        # which sets eval before calling collect_rollout) aren't affected.
        if was_training:
            self.policy.train()

        return rollout.get(self.device)

    # ------------------------------------------------------------------
    # Discriminator warmup
    # ------------------------------------------------------------------

    def warmup_discriminator(
        self,
        env:        ReservoirEnvironment,
        iterations: int,
    ) -> None:
        """
        Pre-train the discriminator for ``iterations`` iterations with the
        policy frozen.  Fills the policy buffer before discriminator updates.
        """
        if self.config.verbose:
            print(f"  Warming up discriminator for {iterations} iterations …")

        self.policy.eval()
        for i in range(iterations):
            rollout = self.collect_rollout(
                env, self.config.steps_per_iteration, deterministic=True
            )
            self.update_discriminator(self.config.batch_size, self.config.warmup_disc_updates)
            del rollout
            if (i + 1) % 10 == 0:
                gc.collect()
                if self.config.verbose:
                    print(f"    Warmup {i + 1}/{iterations}")

        self.policy.train()
        gc.collect()

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        env:   ReservoirEnvironment,
        split: Split,
    ) -> Dict[str, Any]:
        """
        Roll out the deterministic policy on ``env`` and compare to expert data.

        Parameters
        ----------
        env   : Environment initialised with the same split (val or test).
        split : Corresponding ``Split`` from ``DataSplits``.

        Returns
        -------
        dict with keys:
            metrics        — release and storage nrmse / rmse / corr.
            expert_release — raw expert release (engineering units).
            learned_release— raw learned release (engineering units).
            expert_storage — raw expert storage (Mm³).
            learned_storage— raw learned storage (Mm³).
        """
        self.policy.eval()
        expert_release  = split.raw_actions                               # (N,) engineering units
        expert_storage  = env.normalizer.denormalize(
            env.storage_col,
            split.states[:, 0],                                           # storage is col 0
        )                                                                  # (N,)

        learned_release: list = []
        learned_storage: list = []

        state    = env.reset(0)
        sim_stor = env.storage          # read directly from env after reset

        for i in range(len(split.states) - 1):
            state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            with torch.no_grad():
                out    = self.policy(state_t, deterministic=True)
                action = out.action.cpu().numpy()[0, 0]

            learned_storage.append(sim_stor)
            learned_release.append(
                float(env.normalizer.denormalize(env.action_col, np.array([action]))[0])
            )

            state, _, done, info = env.step(action)
            sim_stor = info["storage"]

            if done and i < len(split.states) - 2:
                state    = env.reset(i + 1)
                sim_stor = env.storage  # read directly from env after reset

        self.policy.train()

        learned_release = np.array(learned_release, dtype=np.float32)
        learned_storage = np.array(learned_storage, dtype=np.float32)
        exp_rel         = expert_release[:len(learned_release)]
        exp_stor        = expert_storage[:len(learned_storage)]

        rel_corr, _ = safe_pearsonr(exp_rel, learned_release)
        rel_nrmse   = nrmse(exp_rel, learned_release)    # denormalized data → use nrmse (range-normalised)
        stor_corr,_ = safe_pearsonr(exp_stor, learned_storage)
        stor_nrmse  = nrmse(exp_stor, learned_storage)   # denormalized data → use nrmse (range-normalised)

        return {
            "metrics": {
                "release_corr":  float(rel_corr),
                "release_nrmse": float(rel_nrmse),
                "storage_corr":  float(stor_corr),
                "storage_nrmse": float(stor_nrmse),
            },
            "expert_release":  exp_rel,
            "learned_release": learned_release,
            "expert_storage":  exp_stor,
            "learned_storage": learned_storage,
        }

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def train(
        self,
        train_env:  ReservoirEnvironment,
        val_env:    ReservoirEnvironment,
        val_split:  Split,
        trial       = None,
    ) -> Dict[str, Any]:
        """
        Full adversarial training loop with validation and early stopping.

        Parameters
        ----------
        train_env  : Training environment (backed by training split raw data).
        val_env    : Validation environment.
        val_split  : Normalised ``Split`` for evaluation metrics.
        trial      : Optuna ``Trial`` for pruning (pass ``None`` in train.py).

        Returns
        -------
        dict with keys:
            best_val_score        — float
            training_stats        — dict of lists (accumulated per-update)
            iterations_completed  — int
        """
        # Lazy import — optuna is only needed when a trial is passed (tune.py).
        # Importing here rather than inside the hot loop avoids repeated dict
        # lookups and keeps the dependency optional for train.py (trial=None).
        if trial is not None:
            import optuna as _optuna

        best_score   = -float("inf")
        best_weights = None
        patience_ctr = 0
        last_disc    = {"disc_loss": 0.0, "expert_acc": 0.5, "policy_acc": 0.5}

        if self.config.verbose:
            print(f"\n  Starting AIRL training: {self.config.num_iterations} iterations …")

        for it in range(self.config.num_iterations):
            # Periodically clear the policy buffer to avoid staleness
            if it > 0 and it % 10 == 0:
                self.policy_buffer.clear()
                gc.collect()

            # Collect on-policy rollout
            rollout   = self.collect_rollout(train_env, self.config.steps_per_iteration)

            # Discriminator update (policy set to eval inside)
            last_disc = self.update_discriminator(self.config.batch_size, self.config.disc_updates)

            # PPO update (policy set to train inside)
            self.update_policy_ppo(rollout)

            del rollout
            gc.collect()

            # ------------------------------------------------------------------
            # Periodic validation
            # ------------------------------------------------------------------
            if it % self.config.eval_interval == 0:
                val_res = self.evaluate(val_env, val_split)
                m       = val_res["metrics"]
                score   = _compute_composite_score(
                    m["release_corr"], m["storage_corr"],
                    m["release_nrmse"], m["storage_nrmse"],
                    last_disc["expert_acc"], last_disc["policy_acc"],
                )

                self.training_stats["val_score"].append(score)
                self.training_stats["val_release_corr"].append(m["release_corr"])
                self.training_stats["val_storage_corr"].append(m["storage_corr"])
                self.training_stats["val_release_nrmse"].append(m["release_nrmse"])
                self.training_stats["val_storage_nrmse"].append(m["storage_nrmse"])

                if score > best_score:
                    best_score   = score
                    best_weights = {k: v.clone() for k, v in self.policy.state_dict().items()}
                    patience_ctr = 0
                else:
                    patience_ctr += 1
                    if patience_ctr >= self.config.early_stopping_patience:
                        if self.config.verbose:
                            print(f"  Early stopping at iteration {it}.")
                        break

                # Optuna pruning
                if trial is not None:
                    trial.report(score, it)
                    if trial.should_prune():
                        raise _optuna.exceptions.TrialPruned()

                if self.config.verbose:
                    print(
                        f"  Iter {it:4d}  score={score:.4f}  "
                        f"rel_corr={m['release_corr']:.4f}  "
                        f"stor_corr={m['storage_corr']:.4f}  "
                        f"disc_loss={last_disc['disc_loss']:.4f}"
                    )

                del val_res
                gc.collect()

        # Restore best weights
        if best_weights is not None:
            self.policy.load_state_dict(best_weights)

        return {
            "best_val_score":       best_score,
            "training_stats":       dict(self.training_stats),
            "iterations_completed": len(self.training_stats.get("disc_loss", [])),
        }

    # ------------------------------------------------------------------
    # Checkpoint I/O
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save policy, critic, discriminator, and BC policy weights."""
        torch.save(
            {
                "policy":         self.policy.state_dict(),
                "critic":         self.critic.state_dict(),
                "discriminator":  self.discriminator.state_dict(),
                "bc_policy":      self.bc_policy.state_dict(),
                "training_stats": dict(self.training_stats),
                "config":         asdict(self.config),
                "policy_type":    self.policy_type,      # needed by train.py to reload agent
            },
            path,
        )

    def load(self, path: str | Path) -> dict:
        """Load weights from a checkpoint created by save()."""
        ckpt = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(ckpt["policy"])
        self.critic.load_state_dict(ckpt["critic"])
        self.discriminator.load_state_dict(ckpt["discriminator"])
        if "bc_policy" in ckpt and ckpt["bc_policy"] is not None:
            self.bc_policy.load_state_dict(ckpt["bc_policy"])
        return ckpt
