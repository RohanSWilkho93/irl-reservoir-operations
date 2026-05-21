"""
deepmaxent/core.py
==================
Core logic for Deep Maximum Entropy IRL applied to reservoir operations.

Responsibilities
----------------
  * Data loading and chronological train/val/test splitting (by year count).
  * Discretisation helpers and MDP trajectory construction.
  * Empirical inflow transition matrix estimation.
  * Feature engineering: z-score normalisation of MDP state variables, optional
    reward-conditioning features, and the release action.
  * ''DeepMaxEntConfig'' — typed dataclass holding every hyperparameter.
  * ''MaxEntTrainer'' — the main IRL training object exposing:
        train_fast()  — SAVF-diff early stopping; no Monte-Carlo per epoch.
                        Called by deepmaxent/tune.py during 2000-trial Optuna
                        hyperparameter search.
        train_full()  — per-epoch S_DeepMaxEnt tracking with light MC (n=5);
                        early stopping maximises validation S_DeepMaxEnt.
                        Called by deepmaxent/train.py for the final run.
  * ''compute_s_deepmaxent()'' — weighted composite score used for model
    selection in both tune.py and train.py.

MDP state
---------
  The MDP state is always (storage, net_inflow).  These two variables define
  the state space, the transition tensor P, and the SAVF.  They cannot be
  swapped out or extended — the water-balance physics only makes sense with
  this pair.

Month handling (use_month_encoding)
------------------------------------
  True  → month enters BOTH the MDP structure (12-slot P tensor and SAVF)
           AND the reward-network features (sin/cos encoding).
  False → month is absent from both.  The MDP collapses to a single
          time-invariant slot (n_months = 1); no sin/cos features.

  Month data (col 1) is always stored in the trajectory rows regardless of
  this flag — it comes from the CSV and is harmless to carry.  The MDP and
  feature builder only consume it when use_month_encoding=True.

Reward features (reward_features)
----------------------------------
  Additional columns from the reservoir CSV that condition the reward network
  but do NOT enter the MDP state or SAVF.  This is "conditional IRL": the
  reward function can depend on e.g. temperature, but the policy (solved via
  value iteration over the discrete MDP) remains over the (storage, inflow)
  state space only.

  During reward-table computation the reward features are held at their
  training mean for every (state, action, month) triplet.  This is a
  deliberate approximation — expanding the MDP state to include continuous
  extra variables would cause exponential state-space growth.

  If reward_features is empty (the default), the model reduces exactly to the
  original Deep MaxEnt IRL formulation from the paper.

Trajectory row layout (fixed regardless of config flags)
---------------------------------------------------------
  discretized rows  [used by SAVF and MDP]:
      col 0  storage (discretized)
      col 1  month   (1–12, always present; used only if use_month_encoding)
      col 2  release (discretized)
      col 3  net_inflow (discretized)

  Raw rows  [used by feature builder and MC simulation]:
      col 0  storage
      col 1  month   (1–12, always present; used only if use_month_encoding)
      col 2  release
      col 3  net_inflow
      col 4+ reward_features in the order given in cfg.reward_features

Reward feature vector order (inputs to the reward network)
----------------------------------------------------------
  [z(storage),  z(net_inflow),  z(rf1),  z(rf2),  ...,  z(release),
   sin(2π·m/12),  cos(2π·m/12)]          ← last two only if use_month_encoding

  n_inputs = 2 + len(reward_features) + 1 + (2 if use_month_encoding else 0)

Physical constant
-----------------
  FLOW_TO_VOLUME = 86 400 / 1e6   converts  m³/s × 1 day → Mm³.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.optim as optim

from networks.reward_deepmax import RewardNet, to_torch
from utils.metrics import nrmse, rmse, safe_pearsonr

# ---------------------------------------------------------------------------
# Physical constant
# ---------------------------------------------------------------------------
FLOW_TO_VOLUME: float = 86_400 / 1e6  # m³/s × day → Mm³


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------

def discretize(val: float, step: float) -> float:
    """Round *val* to the nearest grid point spaced by *step*."""
    if pd.isna(val):
        return np.nan
    return round(val / step) * step


def get_state_idx(s_idx: int, i_idx: int, n_s: int) -> int:
    """Flat state index from (storage_bin, inflow_bin).

    State layout: state = i_idx * n_s + s_idx
    This means storage varies fastest (inner dimension).
    """
    return i_idx * n_s + s_idx


def get_si_fi(flat_idx: int, n_s: int) -> Tuple[int, int]:
    """Recover (storage_bin, inflow_bin) from a flat state index."""
    return flat_idx % n_s, flat_idx // n_s


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_and_split_data(
    data_path: str,
    date_col: str,
    n_train: int,
    n_val: int,
    n_test: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame,
           List[int], List[int], List[int]]:
    """
    Load a reservoir CSV and split chronologically by calendar year.

    The CSV must contain at minimum the columns:
        <date_col>, storage, release, net_inflow

    ''year'' and ''month'' are derived from ''date_col'' via
    ''pd.to_datetime''; they must not be pre-computed in the CSV.

    All four parameters after ''data_path'' are read directly from the
    reservoir YAML by the caller and passed in:
        date_col → cfg["columns"]["date"]
        n_train  → cfg["split"]["train"]
        n_val    → cfg["split"]["val"]
        n_test   → cfg["split"]["test"]

    Any additional columns listed in ''cfg.reward_features'' are preserved
    and will be included in raw trajectories at col 4+.

    Parameters
    ----------
    data_path : path to the reservoir CSV.
    date_col  : name of the date column.
    n_train   : number of calendar years for training.
    n_val     : number of calendar years for validation.
    n_test    : number of calendar years for test.

    Returns
    -------
    data       : full DataFrame (all years), with ''year'' and ''month''
                 columns appended.
    train_data : training-split DataFrame.
    val_data   : validation-split DataFrame.
    test_data  : test-split DataFrame.
    train_years, val_years, test_years : sorted lists of calendar years.
    """
    data = pd.read_csv(data_path)

    if date_col not in data.columns:
        raise ValueError(
            f"Date column '{date_col}' not found in {data_path}.\n"
            f"Available columns: {list(data.columns)}"
        )

    data[date_col] = pd.to_datetime(data[date_col])
    data = data.sort_values(date_col).reset_index(drop=True)
    data["year"]  = data[date_col].dt.year
    data["month"] = data[date_col].dt.month

    unique_years = sorted(data["year"].unique())
    n_total = n_train + n_val + n_test

    if len(unique_years) < n_total:
        raise ValueError(
            f"Split requires {n_total} years "
            f"(train={n_train}, val={n_val}, test={n_test}) "
            f"but the dataset only contains {len(unique_years)} unique years."
        )

    train_years = unique_years[:n_train]
    val_years   = unique_years[n_train : n_train + n_val]
    test_years  = unique_years[n_train + n_val : n_total]

    train_data = data[data["year"].isin(train_years)].copy()
    val_data   = data[data["year"].isin(val_years)].copy()
    test_data  = data[data["year"].isin(test_years)].copy()

    return data, train_data, val_data, test_data, train_years, val_years, test_years


# ---------------------------------------------------------------------------
# Discretisation and trajectory construction
# ---------------------------------------------------------------------------

def create_spaces(
    df: pd.DataFrame,
    cfg: "DeepMaxEntConfig",
    storage_col: str,
    action_col: str,
    inflow_col: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build storage, release, and inflow grids from observed data ranges.

    Grid lower edges are snapped to the nearest lower multiple of the step
    so that every observed value maps cleanly to a bin.

    Parameters
    ----------
    storage_col : CSV column name for reservoir storage
                  (''cfg["columns"]["state"][0]'' in the reservoir YAML).
    action_col  : CSV column name for release / action
                  (''cfg["columns"]["action"]'' in the reservoir YAML).
    inflow_col  : CSV column name for net inflow
                  (''cfg["columns"]["state"][1]'' in the reservoir YAML).

    Returns
    -------
    s_space, r_space, i_space : 1-D grid vectors (np.ndarray).
    """
    s_min, s_max = df[storage_col].min(), df[storage_col].max()
    r_min, r_max = df[action_col].min(),  df[action_col].max()
    i_min, i_max = df[inflow_col].min(),  df[inflow_col].max()

    s_space = np.arange(
        np.floor(s_min / cfg.storage_step) * cfg.storage_step,
        s_max + cfg.storage_step,
        cfg.storage_step,
    )
    r_space = np.arange(
        np.floor(r_min / cfg.release_step) * cfg.release_step,
        r_max + cfg.release_step,
        cfg.release_step,
    )
    i_space = np.arange(
        np.floor(i_min / cfg.inflow_step) * cfg.inflow_step,
        i_max + cfg.inflow_step,
        cfg.inflow_step,
    )
    return s_space, r_space, i_space


def create_trajectories(
    df: pd.DataFrame,
    s_space: np.ndarray,
    r_space: np.ndarray,
    i_space: np.ndarray,
    cfg: "DeepMaxEntConfig",
    storage_col: str,
    action_col: str,
    inflow_col: str,
) -> Tuple[List, List, Dict, Dict, Dict]:
    """
    Build per-year discretized and raw trajectories from a data split.

    discretized trajectory row  (cols 0–3, fixed):
        [s_d, month, r_d, i_d]
        Used exclusively for SAVF computation and MDP state initialisation.
        Month is always stored; it is only consumed by the MDP when
        cfg.use_month_encoding is True.

    Raw trajectory row  (cols 0–3 fixed; cols 4+ variable):
        [storage, month, release, net_inflow, rf1, rf2, ...]
        The values come from the CSV columns named by storage_col, action_col,
        inflow_col; the internal position order is always fixed as above.
        Reward features (cfg.reward_features) are appended at col 4+ in
        the order they appear in the config list.

    Parameters
    ----------
    df          : DataFrame for one split (train / val / test).
                  Must contain all columns referenced by cfg.reward_features.
    storage_col : CSV column name for reservoir storage.
    action_col  : CSV column name for release / action.
    inflow_col  : CSV column name for net inflow.

    Returns
    -------
    trajs     : list[list[list]]  one inner list per year (discretized rows).
    trajs_raw : list[list[list]]  one inner list per year (raw rows).
    s_map, r_map, i_map : {value: bin_index} lookup dicts.
    """
    # Validate reward-feature columns
    missing = [rf for rf in cfg.reward_features if rf not in df.columns]
    if missing:
        raise ValueError(
            f"Reward features {missing} not found in DataFrame. "
            f"Available columns: {list(df.columns)}"
        )

    df_d = df.copy()
    df_d["s_d"] = df_d[storage_col].apply(lambda x: discretize(x, cfg.storage_step))
    df_d["r_d"] = df_d[action_col].apply(lambda x: discretize(x, cfg.release_step))
    df_d["i_d"] = df_d[inflow_col].apply(lambda x: discretize(x, cfg.inflow_step))
    df_d.dropna(subset=["s_d", "r_d", "i_d"], inplace=True)

    s_map = {v: i for i, v in enumerate(s_space)}
    r_map = {v: i for i, v in enumerate(r_space)}
    i_map = {v: i for i, v in enumerate(i_space)}

    # Raw column list: fixed internal order [storage, month, release, net_inflow, ...]
    # Values read from the user-specified column names; internal positions are fixed.
    raw_cols = [storage_col, "month", action_col, inflow_col] + list(cfg.reward_features)

    trajs:     List[List] = []
    trajs_raw: List[List] = []

    for year in sorted(df_d["year"].unique()):
        yd = df_d[df_d["year"] == year]
        t     = yd[["s_d", "month", "r_d", "i_d"]].values.tolist()
        t_raw = yd[raw_cols].values.tolist()
        if t:
            trajs.append(t)
            trajs_raw.append(t_raw)

    return trajs, trajs_raw, s_map, r_map, i_map


# ---------------------------------------------------------------------------
# Transition matrix
# ---------------------------------------------------------------------------

def build_inflow_transitions(
    df: pd.DataFrame,
    i_space: np.ndarray,
    i_map: Dict,
    cfg: "DeepMaxEntConfig",
    inflow_col: str,
) -> np.ndarray:
    """
    Estimate month-to-month inflow transition probabilities from training data.

    Counts are pooled across all months (the matrix is stationary in time).
    Laplace smoothing (α = 0.1) ensures every transition has positive mass.

    Parameters
    ----------
    inflow_col : CSV column name for net inflow
                 (''cfg["columns"]["state"][1]'' in the reservoir YAML).

    Returns
    -------
    trans : (n_inflow_bins, n_inflow_bins) row-stochastic transition matrix.
    """
    n_i  = len(i_space)
    trans = np.zeros((n_i, n_i))

    for year in sorted(df["year"].unique()):
        year_data = df[df["year"] == year].sort_values("month")
        for k in range(len(year_data) - 1):
            curr_i = discretize(year_data.iloc[k][inflow_col],     cfg.inflow_step)
            next_i = discretize(year_data.iloc[k + 1][inflow_col], cfg.inflow_step)
            if curr_i in i_map and next_i in i_map:
                trans[i_map[curr_i], i_map[next_i]] += 1

    trans += 0.1  # Laplace smoothing
    return trans / trans.sum(axis=1, keepdims=True)


def build_transition_matrix(
    s_space: np.ndarray,
    r_space: np.ndarray,
    i_space: np.ndarray,
    inflow_trans: np.ndarray,
    n_months: int = 12,
) -> Tuple[np.ndarray, int]:
    """
    Construct the full MDP transition tensor P[state, month, action, next_state].

    Storage dynamics follow the water-balance equation:
        s_{t+1} = clip(s_t + FLOW_TO_VOLUME × (inflow_t − release_t), s_min, s_max)

    The inflow transitions are stochastic, governed by the empirical
    month-to-month matrix from training data.  Importantly, the water-balance
    physics is month-agnostic; the month dimension of P carries identical
    physics across all m when n_months > 1 (the policy, however, will produce
    different actions per month because the reward function is month-dependent).

    Parameters
    ----------
    s_space, r_space, i_space : 1-D grids from ''create_spaces''.
    inflow_trans : (n_i, n_i) row-stochastic inflow transition matrix.
    n_months     : 12 when use_month_encoding=True, 1 otherwise.
        Pass ''12 if cfg.use_month_encoding else 1'' at the call site.

    Returns
    -------
    P   : np.ndarray, shape (n_states, n_months, n_actions, n_states), float32.
    n_s : number of storage bins (needed to decode flat state indices later).
    """
    n_s, n_i, n_r = len(s_space), len(i_space), len(r_space)
    n_states = n_s * n_i
    P = np.zeros((n_states, n_months, n_r, n_states), dtype=np.float32)
    s_min, s_max = s_space[0], s_space[-1]

    for si in range(n_s):
        s_val = s_space[si]
        for fi in range(n_i):
            i_val     = i_space[fi]
            curr_state = get_state_idx(si, fi, n_s)
            for m in range(n_months):
                for ri, r_val in enumerate(r_space):
                    s_next  = np.clip(
                        s_val + FLOW_TO_VOLUME * (i_val - r_val), s_min, s_max
                    )
                    si_next = int(np.argmin(np.abs(s_space - s_next)))
                    for fi_next in range(n_i):
                        prob = inflow_trans[fi, fi_next]
                        if prob > 1e-6:
                            P[curr_state, m, ri,
                              get_state_idx(si_next, fi_next, n_s)] += prob

    return P, n_s


# ---------------------------------------------------------------------------
# Score function helpers
# ---------------------------------------------------------------------------

def _nrmse_std(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    RMSE normalised by the standard deviation of the observed series.

    Used *inside* ''compute_s_deepmaxent'' — matches the original DeepMaxEnt
    source.  The range-normalised nRMSE from ''utils/metrics.py'' is used only
    for final reporting tables (different normalisation, different purpose).

    Returns np.inf when the observed series is effectively constant.
    """
    y_true   = np.asarray(y_true, dtype=float).flatten()
    y_pred   = np.asarray(y_pred, dtype=float).flatten()
    std_true = float(np.std(y_true))
    if std_true < 1e-10:
        return np.inf
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)) / std_true)


def compute_s_deepmaxent(
    savf_diff:     float,
    release_corr:  float,
    storage_corr:  float,
    release_nrmse: float,
    storage_nrmse: float,
    savf_diff_bounds:   Tuple[float, float] = (0.0, 1000.0),
    w_savf:             float = 0.50,
    w_release_corr:     float = 0.125,
    w_storage_corr:     float = 0.125,
    w_release_nrmse:    float = 0.125,
    w_storage_nrmse:    float = 0.125,
) -> Tuple[float, Dict[str, float]]:
    """
    Composite model-selection score for Deep MaxEnt IRL (S_DeepMaxEnt).

    Theory-first weighting (weights sum to 1.0):
      SAVF diff (0.50)     — primary IRL objective; lower is better.
      Release corr (0.125)  — timing of the control action.
      Storage corr (0.125)  — timing of reservoir state evolution.
      Release nRMSE (0.125) — magnitude of the control action.
      Storage nRMSE (0.125) — magnitude of state evolution.

    All five components are mapped to [0, 1] (higher is always better) before
    the weighted sum.  nRMSE components use the std-based variant to match the
    original source.

    Parameters
    ----------
    savf_diff      : SAVF L1 difference (lower is better).
    release_corr   : Pearson r for release, in [-1, 1] (higher is better).
    storage_corr   : Pearson r for storage, in [-1, 1] (higher is better).
    release_nrmse  : std-normalised RMSE for release (lower is better).
    storage_nrmse  : std-normalised RMSE for storage (lower is better).
    savf_diff_bounds : (min, max) used for SAVF normalisation clipping.

    Returns
    -------
    score      : float in [0, 1], higher is better.
    components : dict of the five normalised component values.
    """
    savf_min, savf_max = savf_diff_bounds
    savf_norm = 1.0 - (
        np.clip(savf_diff, savf_min, savf_max) - savf_min
    ) / (savf_max - savf_min + 1e-8)

    release_corr_norm  = (release_corr  + 1.0) / 2.0
    storage_corr_norm  = (storage_corr  + 1.0) / 2.0
    release_nrmse_norm = 1.0 - np.clip(release_nrmse, 0.0, 2.0) / 2.0
    storage_nrmse_norm = 1.0 - np.clip(storage_nrmse, 0.0, 2.0) / 2.0

    score = (
        w_savf          * savf_norm
        + w_release_corr  * release_corr_norm
        + w_storage_corr  * storage_corr_norm
        + w_release_nrmse * release_nrmse_norm
        + w_storage_nrmse * storage_nrmse_norm
    )

    components = {
        "savf_normalized":         float(savf_norm),
        "release_corr_normalized": float(release_corr_norm),
        "storage_corr_normalized": float(storage_corr_norm),
        "release_nrmse_normalized": float(release_nrmse_norm),
        "storage_nrmse_normalized": float(storage_nrmse_norm),
    }
    return float(score), components


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class DeepMaxEntConfig:
    """
    All hyperparameters and feature-space settings for Deep MaxEnt IRL.

    MDP state is always (storage, net_inflow) — this is not configurable.

    Parameters
    ----------
    use_month_encoding : bool
        True  → month enters both the MDP (12-slot P and SAVF) and the reward
                 network features (sin/cos encoding).
        False → month is absent from both.  The MDP uses a single time-invariant
                 slot (n_months = 1); no sin/cos features are added.
    reward_features : list of str
        Extra column names from the reservoir CSV to include as reward-network
        inputs only.  These condition the reward function ("conditional IRL")
        but do NOT expand the MDP state space.  During reward-table computation
        they are held at their training mean.  Empty by default (reproduces the
        original paper exactly).
    """

    # Reproducibility
    seed: int = 42

    # Discretisation (also tuned by Optuna — determines state-space size)
    storage_step: float = 10.0
    release_step: float  = 2.0
    inflow_step:  float  = 2.0

    # MDP
    gamma: float = 0.95   # discount factor
    tau:   float = 0.10   # soft-policy temperature (lower → sharper policy)

    # Reward-network architecture
    hidden_dim1: int   = 256
    hidden_dim2: int   = 256
    dropout:     float = 0.10

    # Training
    lr:           float = 1e-4
    n_iterations: int   = 200
    batch_size:   int   = 5_000

    # Early stopping
    early_stop_patience:    int   = 50
    convergence_threshold:  float = 0.01   # stop if train SAVF diff < this
    tolerance:              float = 1e-6   # value-iteration convergence

    # Monte-Carlo (final evaluation only; train_full uses n_mc_per_epoch)
    n_mc_simulations: int = 50

    # Feature space
    use_month_encoding: bool      = True
    reward_features:    List[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Derived property
    # ------------------------------------------------------------------
    @property
    def n_inputs(self) -> int:
        """Number of inputs to the reward network.

        Formula:
            2                                    # storage + net_inflow
            + len(reward_features)               # extra reward-only features
            + 1                                  # release (action)
            + (2 if use_month_encoding else 0)   # sin/cos month

        Base case (no extras, with month): 2 + 0 + 1 + 2 = 5  (original paper).
        """
        return 2 + len(self.reward_features) + 1 + (2 if self.use_month_encoding else 0)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable dict (excludes the derived property)."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "DeepMaxEntConfig":
        """Reconstruct a config from a saved dict, ignoring unknown keys."""
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in known})

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> "DeepMaxEntConfig":
        with open(path) as f:
            return cls.from_dict(json.load(f))


# ---------------------------------------------------------------------------
# MaxEntTrainer
# ---------------------------------------------------------------------------

class MaxEntTrainer:
    """
    Deep Maximum Entropy IRL trainer.

    Constructs the reward network, computes the expert SAVF once at
    initialisation, then exposes two training entry points:

        train_fast(val_trajs)
            Optuna tuning: early-stop on validation SAVF diff.
            No Monte-Carlo per epoch — fast enough for 2 000 trials.

        train_full(val_trajs, val_trajs_raw)
            Final run: per-epoch MC (n=5) → S_DeepMaxEnt; early-stop on
            validation S_DeepMaxEnt (maximise).

    The MDP month structure (n_months) is inferred from P.shape[1]:
        P.shape[1] = 12  when cfg.use_month_encoding = True
        P.shape[1] = 1   when cfg.use_month_encoding = False

    Build P with ''build_transition_matrix(..., n_months=12 if cfg.use_month_encoding else 1)''
    before constructing this trainer.

    Parameters
    ----------
    trajs     : discretized training trajectories from ''create_trajectories''.
    trajs_raw : raw training trajectories from ''create_trajectories''.
                Used to compute z-score statistics for feature engineering.
    """

    def __init__(
        self,
        cfg:          DeepMaxEntConfig,
        P:            np.ndarray,
        trajs:        List,
        trajs_raw:    List,
        s_space:      np.ndarray,
        r_space:      np.ndarray,
        i_space:      np.ndarray,
        s_map:        Dict,
        r_map:        Dict,
        i_map:        Dict,
        n_s_bins:     int,
        inflow_trans: np.ndarray,
        device:       torch.device,
        verbose:      bool = True,
    ) -> None:
        self.cfg          = cfg
        self.P            = P
        self.trajs        = trajs
        self.trajs_raw    = trajs_raw
        self.s_space      = s_space
        self.r_space      = r_space
        self.i_space      = i_space
        self.s_map        = s_map
        self.r_map        = r_map
        self.i_map        = i_map
        self.n_s_bins     = n_s_bins
        self.inflow_trans = inflow_trans
        self.device       = device
        self.verbose      = verbose

        # Infer month structure from the transition tensor
        self.n_states  = P.shape[0]
        self.n_months  = P.shape[1]   # 12 or 1
        self.n_actions = P.shape[2]

        # Validate consistency with config
        expected_n_months = 12 if cfg.use_month_encoding else 1
        if self.n_months != expected_n_months:
            raise ValueError(
                f"P.shape[1] = {self.n_months} does not match "
                f"expected n_months = {expected_n_months} for "
                f"use_month_encoding = {cfg.use_month_encoding}. "
                f"Rebuild P with build_transition_matrix("
                f"n_months={expected_n_months})."
            )

        # Warn user about reward features (approximation notice)
        if cfg.reward_features:
            warnings.warn(
                f"Reward features {cfg.reward_features} condition the reward "
                f"network but are held at their training mean during reward-table "
                f"computation.  The policy cannot condition on these variables at "
                f"decision time (they do not enter the MDP state).",
                UserWarning,
                stacklevel=2,
            )

        # Feature normalisation statistics (from training raw trajectories)
        self._stats: Dict[str, Dict[str, float]] = {}
        self._compute_stats(trajs_raw)

        # Reward network
        self.r_net = RewardNet(
            n_inputs = cfg.n_inputs,
            h1       = cfg.hidden_dim1,
            h2       = cfg.hidden_dim2,
            dropout  = cfg.dropout,
        ).to(device)

        self.opt   = optim.Adam(self.r_net.parameters(), lr=cfg.lr)
        self.sched = optim.lr_scheduler.ReduceLROnPlateau(
            self.opt, mode="min", factor=0.7, patience=15
        )

        # Expert SAVF — computed once from training discretized trajectories
        self.savf_expert = self._calc_expert_savf(trajs)

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------

    def _compute_stats(self, trajs_raw: List) -> None:
        """
        Compute z-score statistics (mean, std) from raw training trajectories.

        Statistics are computed for: storage (col 0), net_inflow (col 3),
        release (col 2), and each reward feature (col 4, 5, ...).
        Month is encoded as sin/cos and is never z-scored.
        """
        accum: Dict[str, List[float]] = {
            "storage":    [],
            "net_inflow": [],
            "release":    [],
        }
        for k, rf in enumerate(self.cfg.reward_features):
            accum[rf] = []

        for traj in trajs_raw:
            for row in traj:
                accum["storage"].append(row[0])
                # row[1] = month — not accumulated (encoded as sin/cos)
                accum["release"].append(row[2])
                accum["net_inflow"].append(row[3])
                for k, rf in enumerate(self.cfg.reward_features):
                    accum[rf].append(row[4 + k])

        self._stats = {}
        for key, vals in accum.items():
            arr = np.asarray(vals, dtype=float)
            self._stats[key] = {
                "mean": float(np.mean(arr)),
                "std":  float(np.std(arr) + 1e-6),
            }

    def _build_features(
        self,
        storage:           np.ndarray,
        net_inflow:        np.ndarray,
        reward_feat_arrays: Dict[str, np.ndarray],
        r:                 np.ndarray,
        m:                 np.ndarray,
    ) -> np.ndarray:
        """
        Assemble the feature matrix fed to the reward network.

        Column order:
            [z(storage),  z(net_inflow),  z(rf1),  z(rf2),  ...,
             z(release),  sin(2π·m/12),   cos(2π·m/12)]
            (sin/cos only if use_month_encoding=True)

        Parameters
        ----------
        storage, net_inflow : arrays of shape (N,) — MDP state variables.
        reward_feat_arrays  : {name: array of shape (N,)} for each reward
                              feature in cfg.reward_features (in that order).
        r                   : release values, shape (N,).
        m                   : month indices, shape (N,).
                              0-indexed (0 = January) when n_months = 12.
                              Always 0 when n_months = 1; values are ignored
                              because sin/cos block is skipped.

        Returns
        -------
        np.ndarray of shape (N, n_inputs), float32.
        """
        cols = []

        # MDP state variables — always first
        cols.append(
            (storage    - self._stats["storage"]["mean"])    / self._stats["storage"]["std"]
        )
        cols.append(
            (net_inflow - self._stats["net_inflow"]["mean"]) / self._stats["net_inflow"]["std"]
        )

        # Reward-only features
        for rf in self.cfg.reward_features:
            cols.append(
                (reward_feat_arrays[rf] - self._stats[rf]["mean"]) / self._stats[rf]["std"]
            )

        # Release action
        cols.append(
            (r - self._stats["release"]["mean"]) / self._stats["release"]["std"]
        )

        # Optional sin/cos month encoding
        if self.cfg.use_month_encoding:
            theta = m / 12.0 * 2.0 * np.pi
            cols.append(np.sin(theta))
            cols.append(np.cos(theta))

        return np.column_stack(cols).astype(np.float32)

    # ------------------------------------------------------------------
    # SAVF computation
    # ------------------------------------------------------------------

    def _calc_expert_savf(self, trajectories: List) -> np.ndarray:
        """
        Compute the expert SAVF from discretized trajectories.

        Shape: (n_states, n_months, n_actions).

        Month index mapping:
            n_months = 12 → m_idx = int(m_val) - 1  (1-based col → 0-based index)
            n_months =  1 → m_idx = 0 always (all observations pool into slot 0)
        """
        mu = np.zeros((self.n_states, self.n_months, self.n_actions))

        for traj in trajectories:
            for row in traj:
                s_val, m_val, r_val, i_val = row
                if (
                    s_val not in self.s_map
                    or r_val not in self.r_map
                    or i_val not in self.i_map
                ):
                    continue

                si    = self.s_map[s_val]
                ri    = self.r_map[r_val]
                fi    = self.i_map[i_val]
                s_idx = get_state_idx(si, fi, self.n_s_bins)

                if self.n_months == 12:
                    m_idx = int(m_val) - 1
                    if not (0 <= m_idx < 12):
                        continue
                else:
                    m_idx = 0

                mu[s_idx, m_idx, ri] += 1.0

        return mu / max(1, len(trajectories))

    def _calc_learned_savf(
        self, Pi: np.ndarray, trajectories: List
    ) -> np.ndarray:
        """
        Compute the learned policy's SAVF by rolling the MDP forward.

        The initial state distribution D is constructed from the first step of
        each trajectory.  The horizon T equals the length of the longest
        trajectory (typically 365 days).

        When n_months = 12, a cumulative-days mapping converts the timestep
        counter into a calendar month (0-indexed, non-leap year).
        When n_months =  1, every timestep uses m = 0.

        Returns
        -------
        mu : np.ndarray, shape (n_states, n_months, n_actions).
        """
        T  = max(len(t) for t in trajectories) if trajectories else 365
        mu = np.zeros((self.n_states, self.n_months, self.n_actions))

        # Initial state distribution from trajectory starting points
        D = np.zeros(self.n_states)
        for traj in trajectories:
            if not traj:
                continue
            s0, _, _, f0 = traj[0]
            if s0 in self.s_map and f0 in self.i_map:
                D[get_state_idx(self.s_map[s0], self.i_map[f0], self.n_s_bins)] += 1
        if D.sum() == 0:
            return mu
        D /= D.sum()
        curr_D = D.copy()

        # Build a function that maps timestep → month index
        if self.n_months == 12:
            _days_per_m = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
            _c_days     = np.cumsum([0] + _days_per_m)   # [0, 31, 59, ..., 365]

            def _month_fn(t: int) -> int:
                d = t % 365
                for i in range(12):
                    if _c_days[i] <= d < _c_days[i + 1]:
                        return i
                return 11
        else:
            def _month_fn(t: int) -> int:  # type: ignore[misc]
                return 0

        for t in range(T):
            m = _month_fn(t)
            mu[:, m, :] += curr_D.reshape(-1, 1) * Pi[:, m, :]
            T_mat   = np.einsum("sa,san->sn", Pi[:, m, :], self.P[:, m, :, :])
            curr_D  = curr_D @ T_mat

        return mu

    def _compute_savf_metrics(
        self, mu_expert: np.ndarray, mu_learned: np.ndarray
    ) -> Tuple[float, float, float]:
        """Return (L1 diff, overlap sum, overlap %)."""
        diff        = float(np.abs(mu_expert - mu_learned).sum())
        overlap     = float(np.minimum(mu_expert, mu_learned).sum())
        overlap_pct = 100.0 * overlap / (mu_expert.sum() + 1e-8)
        return diff, overlap, overlap_pct

    def evaluate_savf(
        self,
        trajectories: List,
        Pi: Optional[np.ndarray] = None,
    ) -> Tuple[float, float]:
        """
        Evaluate SAVF matching on a trajectory set.

        Parameters
        ----------
        trajectories : discretized trajectories (val or test split).
        Pi           : pre-computed policy.  If None, computes from scratch.

        Returns
        -------
        savf_diff   : L1 difference (lower is better).
        overlap_pct : percentage overlap (higher is better).
        """
        if Pi is None:
            R  = self._calc_rewards()
            Pi = self._solve_mdp(R)

        mu_expert  = self._calc_expert_savf(trajectories)
        mu_learned = self._calc_learned_savf(Pi, trajectories)
        diff, _, overlap_pct = self._compute_savf_metrics(mu_expert, mu_learned)
        return diff, overlap_pct

    # ------------------------------------------------------------------
    # Reward table and MDP solver
    # ------------------------------------------------------------------

    def _calc_rewards(self) -> np.ndarray:
        """
        Evaluate the reward network over the full discrete
        (state, action, month) grid and return R[state, action, month].

        Storage and net_inflow come from MDP state decomposition via
        get_si_fi.  Reward features are held at their training mean — a
        deliberate approximation to keep the reward table tractable
        (see module docstring for rationale).
        """
        was_training = self.r_net.training
        self.r_net.eval()

        R = np.zeros((self.n_states, self.n_actions, self.n_months))

        # ri_grid: (n_actions, n_months); mi_grid: (n_actions, n_months)
        ri_grid, mi_grid = np.meshgrid(
            np.arange(self.n_actions),
            np.arange(self.n_months),
            indexing="ij",
        )
        n_points = ri_grid.size   # n_actions × n_months

        # Reward features: training mean replicated for every grid point
        rf_arrays: Dict[str, np.ndarray] = {
            rf: np.full(n_points, self._stats[rf]["mean"])
            for rf in self.cfg.reward_features
        }

        for s_idx in range(self.n_states):
            si, fi = get_si_fi(s_idx, self.n_s_bins)
            s_val  = self.s_space[si]
            i_val  = self.i_space[fi]

            feats = self._build_features(
                np.full(n_points, s_val),
                np.full(n_points, i_val),
                rf_arrays,
                self.r_space[ri_grid.flatten()],
                mi_grid.flatten().astype(float),
            )
            with torch.no_grad():
                R[s_idx] = (
                    self.r_net(to_torch(feats, self.device))
                    .cpu()
                    .numpy()
                    .reshape(self.n_actions, self.n_months)
                )

        if was_training:
            self.r_net.train()
        return R

    def _solve_mdp(self, R: np.ndarray) -> np.ndarray:
        """
        Soft value iteration.

        V[s, m] = τ · log Σ_a exp(Q[s, m, a] / τ)

        When n_months = 12, month wraps: m_next = (m + 1) % 12.
        When n_months =  1, m_next = 0 — the single-period stationary MDP.

        Returns
        -------
        Pi : (n_states, n_months, n_actions) softmax policy.
        """
        V = np.zeros((self.n_states, self.n_months))
        Q = np.zeros((self.n_states, self.n_months, self.n_actions))

        for _ in range(100):
            V_prev = V.copy()
            for m in range(self.n_months):
                m_next    = (m + 1) % self.n_months
                Q[:, m, :] = R[:, :, m] + self.cfg.gamma * np.einsum(
                    "san,n->sa",
                    self.P[:, m, :, :],
                    V_prev[:, m_next],
                )
            Q_sc  = Q / self.cfg.tau
            Q_max = Q_sc.max(axis=2, keepdims=True)          # (n_states, n_months, 1)
            V     = self.cfg.tau * (
                Q_max.squeeze(axis=2)
                + np.log(np.exp(Q_sc - Q_max).sum(axis=2))
            )
            if np.abs(V - V_prev).max() < self.cfg.tolerance:
                break

        Pi = np.zeros_like(Q)
        for m in range(self.n_months):
            Q_m   = Q[:, m, :] / self.cfg.tau
            Q_max = Q_m.max(axis=1, keepdims=True)
            exp_Q = np.exp(Q_m - Q_max)
            Pi[:, m, :] = exp_Q / exp_Q.sum(axis=1, keepdims=True)

        return Pi

    # ------------------------------------------------------------------
    # Monte-Carlo simulation and full evaluation
    # ------------------------------------------------------------------

    def monte_carlo_simulate(
        self,
        trajs_d:   List,
        trajs_raw: List,
        n_sims:    int = 50,
        Pi:        Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Roll out the learned stochastic policy via Monte Carlo.

        For each year trajectory, ''n_sims'' independent simulations are run
        from the observed initial storage with the expert inflow sequence
        (teacher-forcing inflow only).  The mean across simulations is compared
        to the expert.

        Month lookup for Pi:
            n_months = 12 → m = expert_months[t] - 1  (1-based → 0-based)
            n_months =  1 → m = 0 always

        Returns
        -------
        dict with keys: expert_storage, expert_release, sim_storage, sim_release.
        """
        if Pi is None:
            Pi = self._solve_mdp(self._calc_rewards())

        all_expert_storage: List[float] = []
        all_expert_release: List[float] = []
        all_sim_storage:    List[float] = []
        all_sim_release:    List[float] = []
        s_min, s_max = self.s_space[0], self.s_space[-1]

        for traj_d, traj_raw in zip(trajs_d, trajs_raw):
            T              = len(traj_d)
            expert_storage = [row[0] for row in traj_raw]
            expert_release = [row[2] for row in traj_raw]
            expert_months  = [int(row[1]) for row in traj_raw]
            expert_inflows = [row[3] for row in traj_raw]

            mc_storage = np.zeros((n_sims, T))
            mc_release = np.zeros((n_sims, T))

            for sim in range(n_sims):
                # Initialise simulation from the first discretized step
                i_val = traj_d[0][3]
                fi    = self.i_map.get(
                    i_val, int(np.argmin(np.abs(self.i_space - i_val)))
                )
                s_val_sim = float(traj_raw[0][0])

                for t in range(T):
                    m = expert_months[t] - 1 if self.n_months == 12 else 0

                    si_curr = int(np.argmin(np.abs(self.s_space - s_val_sim)))
                    s_idx   = get_state_idx(si_curr, fi, self.n_s_bins)
                    ri      = int(np.random.choice(self.n_actions, p=Pi[s_idx, m, :]))
                    r_val   = self.r_space[ri]

                    mc_storage[sim, t] = s_val_sim
                    mc_release[sim, t] = r_val

                    if t < T - 1:
                        s_val_sim = float(np.clip(
                            s_val_sim + FLOW_TO_VOLUME * (expert_inflows[t] - r_val),
                            s_min, s_max,
                        ))
                        fi = int(np.argmin(np.abs(self.i_space - expert_inflows[t + 1])))

            all_expert_storage.extend(expert_storage)
            all_expert_release.extend(expert_release)
            all_sim_storage.extend(mc_storage.mean(axis=0).tolist())
            all_sim_release.extend(mc_release.mean(axis=0).tolist())

        return {
            "expert_storage": np.array(all_expert_storage),
            "expert_release": np.array(all_expert_release),
            "sim_storage":    np.array(all_sim_storage),
            "sim_release":    np.array(all_sim_release),
        }

    def evaluate_full(
        self,
        trajs_d:   List,
        trajs_raw: List,
        savf_diff: Optional[float] = None,
        Pi:        Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Full evaluation: Monte-Carlo simulation followed by all metrics.

        Returns
        -------
        dict with keys:
            release_corr, storage_corr          (Pearson r)
            release_rmse, storage_rmse          (RMSE in engineering units)
            release_nrmse, storage_nrmse        (range-normalised; for reporting)
            release_nrmse_std, storage_nrmse_std (std-normalised; for S score)
            release_mae, storage_mae
            s_deepmaxent                        (None if savf_diff not supplied)
            s_deepmaxent_components
            results                             (raw MC arrays)
        """
        results = self.monte_carlo_simulate(
            trajs_d, trajs_raw, self.cfg.n_mc_simulations, Pi
        )

        release_corr, _ = safe_pearsonr(
            results["expert_release"], results["sim_release"]
        )
        storage_corr, _ = safe_pearsonr(
            results["expert_storage"], results["sim_storage"]
        )

        release_rmse = float(rmse(results["expert_release"], results["sim_release"]))
        storage_rmse = float(rmse(results["expert_storage"], results["sim_storage"]))

        # Range-normalised nRMSE — for reporting tables (utils/metrics.py)
        release_nrmse_range = float(nrmse(results["expert_release"], results["sim_release"]))
        storage_nrmse_range = float(nrmse(results["expert_storage"], results["sim_storage"]))

        # Std-normalised nRMSE — for S_DeepMaxEnt computation (original source)
        release_nrmse_std = _nrmse_std(results["expert_release"], results["sim_release"])
        storage_nrmse_std = _nrmse_std(results["expert_storage"], results["sim_storage"])

        s_score, s_components = None, None
        if savf_diff is not None:
            s_score, s_components = compute_s_deepmaxent(
                savf_diff,
                float(release_corr), float(storage_corr),
                release_nrmse_std,   storage_nrmse_std,
            )

        return {
            "release_corr":          float(release_corr),
            "storage_corr":          float(storage_corr),
            "release_rmse":          release_rmse,
            "storage_rmse":          storage_rmse,
            "release_nrmse":         release_nrmse_range,   # range-based → reporting
            "storage_nrmse":         storage_nrmse_range,
            "release_nrmse_std":     release_nrmse_std,     # std-based   → S score
            "storage_nrmse_std":     storage_nrmse_std,
            "release_mae":           float(np.mean(np.abs(
                results["expert_release"] - results["sim_release"]
            ))),
            "storage_mae":           float(np.mean(np.abs(
                results["expert_storage"] - results["sim_storage"]
            ))),
            "s_deepmaxent":          float(s_score) if s_score is not None else None,
            "s_deepmaxent_components": s_components,
            "results":               results,
        }

    # ------------------------------------------------------------------
    # Gradient update (shared by both train methods)
    # ------------------------------------------------------------------

    def _gradient_step(self, grad: np.ndarray) -> None:
        """
        Apply one MaxEnt gradient step: loss = −⟨reward(s, a, m), SAVF_grad⟩.

        Only grid cells with |grad| > 1e-6 contribute, keeping the effective
        batch sparse and fast.

        grad shape: (n_states, n_months, n_actions).
        When n_months = 1, m_idxs is all zeros; sin/cos block is skipped.
        When n_months = 12, m_idxs are 0–11; sin/cos is computed normally.
        Reward features are held at training mean (same approximation as in
        _calc_rewards).
        """
        idxs = np.where(np.abs(grad) > 1e-6)
        if len(idxs[0]) == 0:
            return

        s_idxs, m_idxs, a_idxs = idxs
        grad_vals = grad[s_idxs, m_idxs, a_idxs]

        storage_vals:    List[float] = []
        net_inflow_vals: List[float] = []
        r_vals:          List[float] = []
        m_vals:          List[int]   = []

        for k, s_idx in enumerate(s_idxs):
            si, fi = get_si_fi(int(s_idx), self.n_s_bins)
            storage_vals.append(self.s_space[si])
            net_inflow_vals.append(self.i_space[fi])
            r_vals.append(self.r_space[a_idxs[k]])
            m_vals.append(int(m_idxs[k]))   # 0-indexed; always 0 when n_months=1

        # Reward features at training mean
        rf_arrays: Dict[str, np.ndarray] = {
            rf: np.full(len(storage_vals), self._stats[rf]["mean"])
            for rf in self.cfg.reward_features
        }

        feats = self._build_features(
            np.array(storage_vals),
            np.array(net_inflow_vals),
            rf_arrays,
            np.array(r_vals),
            np.array(m_vals, dtype=float),
        )

        self.r_net.train()
        self.opt.zero_grad()
        bs = self.cfg.batch_size
        for i in range(0, len(feats), bs):
            x    = to_torch(feats[i : i + bs], self.device)
            g    = to_torch(grad_vals[i : i + bs].astype(np.float32), self.device)
            loss = -(self.r_net(x).squeeze() * g).sum()
            loss.backward()

        torch.nn.utils.clip_grad_norm_(self.r_net.parameters(), max_norm=1.0)
        self.opt.step()

    # ------------------------------------------------------------------
    # train_fast  —  Optuna tuning (2 000 trials)
    # ------------------------------------------------------------------

    def train_fast(
        self,
        val_trajs: List,
    ) -> Tuple[np.ndarray, np.ndarray, int, List[Dict], Dict]:
        """
        Train with early stopping on validation SAVF diff.

        No Monte-Carlo simulation is run during training — this keeps each
        trial fast enough for 2 000-trial Optuna search.

        Returns
        -------
        best_R           : reward table R[state, action, month] at best epoch.
        best_Pi          : policy Pi[state, month, action] at best epoch.
        best_epoch       : epoch index with lowest val_savf_diff.
        history          : list of per-epoch dicts with keys:
                               epoch, train_savf_diff, train_savf_overlap,
                               val_savf_diff, val_savf_overlap, learning_rate.
        best_model_state : r_net state_dict at the best epoch.
        """
        best_R:            Optional[np.ndarray] = None
        best_Pi:           Optional[np.ndarray] = None
        best_val_savf_diff = np.inf
        best_epoch         = 0
        epochs_no_improve  = 0
        best_model_state:  Optional[Dict]       = None
        history:           List[Dict]           = []

        for epoch in range(self.cfg.n_iterations):
            R          = self._calc_rewards()
            Pi         = self._solve_mdp(R)
            savf_learned = self._calc_learned_savf(Pi, self.trajs)
            grad         = self.savf_expert - savf_learned
            train_diff, _, train_overlap = self._compute_savf_metrics(
                self.savf_expert, savf_learned
            )

            val_savf_diff, val_overlap = self.evaluate_savf(val_trajs, Pi)
            self.sched.step(val_savf_diff)
            current_lr = float(self.opt.param_groups[0]["lr"])

            history.append({
                "epoch":              epoch,
                "train_savf_diff":    float(train_diff),
                "train_savf_overlap": float(train_overlap),
                "val_savf_diff":      float(val_savf_diff),
                "val_savf_overlap":   float(val_overlap),
                "learning_rate":      current_lr,
            })

            if self.verbose and epoch % 10 == 0:
                print(
                    f"  [{epoch:4d}]  train_savf={train_diff:.2f}  "
                    f"val_savf={val_savf_diff:.2f}  lr={current_lr:.2e}"
                )

            # Track best by validation SAVF diff (minimise)
            if val_savf_diff < best_val_savf_diff:
                best_val_savf_diff = val_savf_diff
                best_R             = R.copy()
                best_Pi            = Pi.copy()
                best_epoch         = epoch
                epochs_no_improve  = 0
                best_model_state   = {
                    k: v.cpu().clone() for k, v in self.r_net.state_dict().items()
                }
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= self.cfg.early_stop_patience:
                if self.verbose:
                    print(f"  Early stopping at epoch {epoch}  (best: {best_epoch})")
                break

            if train_diff < self.cfg.convergence_threshold:
                if self.verbose:
                    print(f"  Converged at epoch {epoch}")
                break

            self._gradient_step(grad)

        # Restore best model weights
        if best_model_state is not None:
            self.r_net.load_state_dict(best_model_state)
            self.r_net.to(self.device)

        return best_R, best_Pi, best_epoch, history, best_model_state

    # ------------------------------------------------------------------
    # train_full  —  final training run
    # ------------------------------------------------------------------

    def train_full(
        self,
        val_trajs:     List,
        val_trajs_raw: List,
        n_mc_per_epoch: int = 5,
    ) -> Tuple[np.ndarray, np.ndarray, int, List[Dict], Dict]:
        """
        Train with early stopping on validation S_DeepMaxEnt.

        A lightweight Monte-Carlo simulation (''n_mc_per_epoch'' rollouts) is
        run every epoch so that S_DeepMaxEnt can be tracked and used for model
        selection.  The best model is the one with the highest validation
        S_DeepMaxEnt — not just the lowest SAVF diff.

        Parameters
        ----------
        val_trajs, val_trajs_raw : validation discretized and raw trajectories.
        n_mc_per_epoch : MC rollouts per epoch (default 5).  Keep small to
            control wall-clock time; the full evaluation uses n_mc_simulations.

        Returns
        -------
        Same structure as ''train_fast''.  History dicts additionally contain:
            val_s_deepmaxent, val_release_corr, val_storage_corr,
            val_release_nrmse_std, val_storage_nrmse_std.
        """
        best_R:           Optional[np.ndarray] = None
        best_Pi:          Optional[np.ndarray] = None
        best_val_s        = -np.inf
        best_epoch        = 0
        epochs_no_improve = 0
        best_model_state: Optional[Dict]       = None
        history:          List[Dict]           = []

        for epoch in range(self.cfg.n_iterations):
            R            = self._calc_rewards()
            Pi           = self._solve_mdp(R)
            savf_learned = self._calc_learned_savf(Pi, self.trajs)
            grad         = self.savf_expert - savf_learned
            train_diff, _, train_overlap = self._compute_savf_metrics(
                self.savf_expert, savf_learned
            )

            # Validation SAVF
            val_savf_diff, val_overlap = self.evaluate_savf(val_trajs, Pi)

            # Validation MC (lightweight, n=n_mc_per_epoch)
            val_mc = self.monte_carlo_simulate(
                val_trajs, val_trajs_raw, n_mc_per_epoch, Pi
            )
            release_corr, _ = safe_pearsonr(
                val_mc["expert_release"], val_mc["sim_release"]
            )
            storage_corr, _ = safe_pearsonr(
                val_mc["expert_storage"], val_mc["sim_storage"]
            )
            release_nrmse_std = _nrmse_std(
                val_mc["expert_release"], val_mc["sim_release"]
            )
            storage_nrmse_std = _nrmse_std(
                val_mc["expert_storage"], val_mc["sim_storage"]
            )
            val_s, _ = compute_s_deepmaxent(
                val_savf_diff,
                float(release_corr), float(storage_corr),
                release_nrmse_std,   storage_nrmse_std,
            )

            self.sched.step(val_savf_diff)
            current_lr = float(self.opt.param_groups[0]["lr"])

            history.append({
                "epoch":                  epoch,
                "train_savf_diff":        float(train_diff),
                "train_savf_overlap":     float(train_overlap),
                "val_savf_diff":          float(val_savf_diff),
                "val_savf_overlap":       float(val_overlap),
                "val_s_deepmaxent":       float(val_s),
                "val_release_corr":       float(release_corr),
                "val_storage_corr":       float(storage_corr),
                "val_release_nrmse_std":  float(release_nrmse_std),
                "val_storage_nrmse_std":  float(storage_nrmse_std),
                "learning_rate":          current_lr,
            })

            if self.verbose and epoch % 10 == 0:
                print(
                    f"  [{epoch:4d}]  train_savf={train_diff:.2f}  "
                    f"val_savf={val_savf_diff:.2f}  "
                    f"val_S={val_s:.4f}  lr={current_lr:.2e}"
                )

            # Track best by validation S_DeepMaxEnt (maximise)
            if val_s > best_val_s:
                best_val_s        = val_s
                best_R            = R.copy()
                best_Pi           = Pi.copy()
                best_epoch        = epoch
                epochs_no_improve = 0
                best_model_state  = {
                    k: v.cpu().clone() for k, v in self.r_net.state_dict().items()
                }
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= self.cfg.early_stop_patience:
                if self.verbose:
                    print(f"  Early stopping at epoch {epoch}  (best: {best_epoch})")
                break

            if train_diff < self.cfg.convergence_threshold:
                if self.verbose:
                    print(f"  Converged at epoch {epoch}")
                break

            self._gradient_step(grad)

        # Restore best model weights
        if best_model_state is not None:
            self.r_net.load_state_dict(best_model_state)
            self.r_net.to(self.device)

        return best_R, best_Pi, best_epoch, history, best_model_state
