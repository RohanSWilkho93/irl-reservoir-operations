"""
iqlearn/environment.py
======================
Closed-loop mass-balance rollout for prediction_fidelity scoring.

Only STORAGE is propagated by the model; INFLOW and MONTH are exogenous and read
from the data at every step.  Starting from the data's initial storage:

    release_norm = policy(state)                        # deterministic expected value
    release_eng  = denorm(release_norm, release bounds) # NORMALIZATION bounds (data)
    release_eng  = clip(release_eng, min_release, max_release)   # PHYSICAL bounds
    storage_eng  = storage_eng + (inflow_eng - release_eng) * seconds_per_day / volume_factor
    storage_eng  = clip(storage_eng, min_storage, max_storage)   # PHYSICAL (clamp/spill)
    next state   = [ norm(storage_eng) | data inflow_norm | data month sin/cos ]

NORMALIZATION is plain min-max from the TRAIN split (confirmed in data.py):
    norm(x)   = (x - lo) / (hi - lo)
    denorm(z) =  z * (hi - lo) + lo
where (lo, hi) are the train min/max for that column.  We compute it inline
from explicit bounds passed by the caller, so this module depends on NO external
Normalizer API.  (val/test values outside [0, 1] are fine and intended — the
round-trip is exact regardless of range.)

The DUAL-BOUNDS RULE (do not merge):
  * NORMALIZATION bounds (train min/max) map storage/release <-> [0, 1] so the
    policy sees the same scaling it trained on.  ALWAYS the data values.
  * PHYSICAL bounds (resolved CLI > config > data) clamp/spill storage and clamp
    release.  When physical == data (the default) the clamps are no-ops.

INVARIANT: overwriting only the storage slot of the state is complete ONLY
because no other state feature is a function of storage (state is
[storage, inflow, sin_month, cos_month]).  If a storage-derived feature is ever
added, its slot must be recomputed here too.

Shapes: T = #split timesteps, D = state_dim.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch


# =============================================================================
# Resolved physical / mass-balance configuration
# =============================================================================

@dataclass
class MassBalanceConfig:
    """
    Fully-resolved PHYSICAL config for one reservoir (no nulls — the caller has
    already applied CLI > config > data resolution for the four bounds).

    storage_col / inflow_col / action_col : column NAMES.  storage_col and
    inflow_col must be in the state column list; all three index `norm_bounds`.
    """
    storage_col:     str
    inflow_col:      str
    action_col:      str
    seconds_per_day: float
    volume_factor:   float
    max_storage:     float
    min_storage:     float
    max_release:     float
    min_release:     float

    def validate(self) -> None:
        if not (self.min_storage < self.max_storage):
            raise ValueError(
                f"min_storage ({self.min_storage}) must be < max_storage ({self.max_storage})."
            )
        if not (self.min_release < self.max_release):
            raise ValueError(
                f"min_release ({self.min_release}) must be < max_release ({self.max_release})."
            )


# =============================================================================
# Rollout
# =============================================================================

class ReservoirRollout:
    """
    Mass-balance simulator over one chronological split (val for tuning, test
    for final reporting).

    Parameters
    ----------
    split       : utils.data.Split (val or test) — uses .states (normalized,
                  (T, D)) and .raw_actions (observed release in engineering units).
    state_cols  : ordered state-column names INCLUDING sin_month/cos_month, i.e.
                  DataSplits.state_cols.
    mb          : resolved MassBalanceConfig (PHYSICAL bounds + constants).
    norm_bounds : {col_name: (lo, hi)} train min/max for AT LEAST mb.storage_col,
                  mb.inflow_col, mb.action_col.  Used for min-max norm/denorm.
    device      : device for the policy forward pass.
    """

    def __init__(
        self,
        split,
        state_cols:  List[str],
        mb:          MassBalanceConfig,
        norm_bounds: Dict[str, Tuple[float, float]],
        device:      str | torch.device,
    ):
        mb.validate()
        self.mb     = mb
        self.device = torch.device(device)

        # ---- locate roles in the state vector (invariant: must be present) ----
        if mb.storage_col not in state_cols:
            raise ValueError(f"storage column '{mb.storage_col}' not in state_cols {state_cols}.")
        if mb.inflow_col not in state_cols:
            raise ValueError(f"inflow column '{mb.inflow_col}' not in state_cols {state_cols}.")
        self.storage_idx = state_cols.index(mb.storage_col)
        self.inflow_idx  = state_cols.index(mb.inflow_col)

        # ---- normalization (min-max) bounds for the three roles ----
        self._s_lo, self._s_hi = self._check_span(norm_bounds, mb.storage_col)
        self._i_lo, self._i_hi = self._check_span(norm_bounds, mb.inflow_col)
        self._r_lo, self._r_hi = self._check_span(norm_bounds, mb.action_col)

        states = np.asarray(split.states, dtype=np.float32)   # (T, D) normalized
        if states.ndim != 2:
            raise ValueError(f"split.states must be 2-D (T, D); got shape {states.shape}.")
        self.T = states.shape[0]

        # Template state reused every step; only the storage slot is overwritten
        # with the simulated value (inflow + month stay from data).
        self._state_template = states.copy()                  # (T, D)

        # ---- exogenous series in ENGINEERING units (min-max denorm) ----
        self.obs_storage = self._denorm(states[:, self.storage_idx], self._s_lo, self._s_hi)
        self.inflow_eng  = self._denorm(states[:, self.inflow_idx],  self._i_lo, self._i_hi)
        self.obs_release = np.asarray(split.raw_actions, dtype=np.float32)   # (T,) engineering

        if self.obs_release.shape[0] != self.T:
            raise ValueError(
                f"raw_actions length ({self.obs_release.shape[0]}) != states length ({self.T})."
            )

    # ---- min-max helpers --------------------------------------------------

    @staticmethod
    def _check_span(norm_bounds: Dict[str, Tuple[float, float]], col: str) -> Tuple[float, float]:
        if col not in norm_bounds:
            raise ValueError(f"norm_bounds is missing column '{col}'.")
        lo, hi = float(norm_bounds[col][0]), float(norm_bounds[col][1])
        if not (hi > lo):
            raise ValueError(f"degenerate norm bounds for '{col}': hi ({hi}) must be > lo ({lo}).")
        return lo, hi

    @staticmethod
    def _denorm(z, lo: float, hi: float):
        return z * (hi - lo) + lo

    @staticmethod
    def _norm(x, lo: float, hi: float):
        return (x - lo) / (hi - lo)

    # -----------------------------------------------------------------------

    @torch.no_grad()
    def rollout(self, agent, *, deterministic: bool = True,
            generator: torch.Generator | None = None) -> Dict[str, np.ndarray]:
        """
        Simulate the policy closed-loop and return engineering-unit trajectories.

        Returns dict with sim_storage, sim_release, obs_storage, obs_release
        (each (T-1,)).  We score the T-1 transitions (the last step has no
        successor storage).
        """
        mb = self.mb
        n = self.T - 1
        if n <= 0:
            empty = np.empty(0, dtype=np.float32)
            return {"sim_storage": empty, "sim_release": empty,
                    "obs_storage": empty, "obs_release": empty}

        sim_storage = np.empty(n, dtype=np.float32)
        sim_release = np.empty(n, dtype=np.float32)

        storage_eng = float(self.obs_storage[0])              # initial state from data
        conv = mb.seconds_per_day / mb.volume_factor

        for t in range(n):
            # Build the policy's state: data row t, but storage slot = simulated.
            state = self._state_template[t].copy()
            state[self.storage_idx] = self._norm(storage_eng, self._s_lo, self._s_hi)
            state_t = torch.from_numpy(state).to(self.device).unsqueeze(0)   # (1, D)

            # Policy -> normalized release -> engineering -> physical clamp.
            release_norm = float(agent.select_action(state_t, deterministic=deterministic, generator=generator).item())
            release_eng = float(self._denorm(release_norm, self._r_lo, self._r_hi))
            release_eng = min(max(release_eng, mb.min_release), mb.max_release)

            # Mass balance (inflow from data), clamp/spill on PHYSICAL bounds.
            storage_after = storage_eng + (float(self.inflow_eng[t]) - release_eng) * conv
            storage_eng = min(max(storage_after, mb.min_storage), mb.max_storage)

            sim_release[t] = release_eng
            sim_storage[t] = storage_eng

        return {
            "sim_storage": sim_storage,
            "sim_release": sim_release,
            "obs_storage": self.obs_storage[1:n + 1],   # storage AFTER each step
            "obs_release": self.obs_release[:n],         # release taken at each step
        }