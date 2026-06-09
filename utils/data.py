"""
utils/data.py
=============
Data loading, splitting, normalization, and month encoding for reservoir
time series.

Pipeline (runs once before any algorithm):
    1. Load CSV and validate all required columns.
    2. Split chronologically into train / val / test by calendar year.
    3. Compute min-max normalization bounds from the TRAINING split only.
    4. Normalize all three splits using the training bounds.
    5. Optionally append sin/cos month encoding to state vectors.
    6. Build next_state and done arrays (required by AIRL).

Normalization note
------------------
Bounds are computed from training data only to avoid leaking val/test
information. Val and test samples are normalized with the same training
bounds as-is — values can fall outside [0, 1] if a val/test extreme was
not seen in training. These values are NOT clipped; the model must handle
them at inference time.

Month encoding
--------------
If columns.use_month_encoding is true, two extra features are appended:
    sin(2π · month / 12)  and  cos(2π · month / 12)
These are already in [-1, 1] and are not normalized further.
    state_dim = len(columns.state) + (2 if use_month_encoding else 0)

Dependencies
------------
    pip install numpy pandas pyyaml ruamel.yaml
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import yaml


# ---------------------------------------------------------------------------
# Normalizer
# ---------------------------------------------------------------------------

class Normalizer:
    """
    Min-max normalizer built from training-split bounds.

    Used by algorithm modules to denormalize predicted actions back to
    engineering units for evaluation and plotting.

    Parameters
    ----------
    bounds : dict  {column_name: {min: float, max: float}}
    """

    def __init__(self, bounds: Dict[str, Dict[str, float]]):
        self.bounds = bounds

    def normalize(self, col: str, values: np.ndarray) -> np.ndarray:
        """Scale values to [0, 1] using training bounds for `col`."""
        lo = self.bounds[col]["min"]
        hi = self.bounds[col]["max"]
        if hi == lo:
            return np.zeros_like(values, dtype=np.float32)
        return ((values - lo) / (hi - lo)).astype(np.float32)

    def denormalize(self, col: str, values: np.ndarray) -> np.ndarray:
        """Invert normalization: [0, 1] → original engineering units."""
        lo = self.bounds[col]["min"]
        hi = self.bounds[col]["max"]
        return (values * (hi - lo) + lo).astype(np.float32)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class Split:
    """One chronological data split (train, val, or test)."""
    states:      np.ndarray       # (N, state_dim)  — normalized
    actions:     np.ndarray       # (N,)             — normalized to [0, 1]
    next_states: np.ndarray       # (N, state_dim)  — normalized; last row repeats
    dones:       np.ndarray       # (N,) bool        — True at last step of each year
    raw_actions: np.ndarray       # (N,)             — original engineering units
    dates:       pd.DatetimeIndex # for reference / plotting


@dataclass
class DataSplits:
    """All three splits plus shared metadata."""
    train:      Split
    val:        Split
    test:       Split
    bounds:     Dict[str, Dict[str, float]]  # {col: {min: float, max: float}}
    state_dim:  int
    state_cols: List[str]   # column names in order, including sin/cos_month if used
    normalizer: Normalizer


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def load_reservoir_data(cfg: dict, cfg_path: str | Path) -> DataSplits:
    """
    Load, split, normalize, and encode reservoir time series.

    Parameters
    ----------
    cfg      : reservoir config dict loaded from configs/reservoirs/<name>.yaml
    cfg_path : path to that YAML file (needed to write bounds back)

    Returns
    -------
    DataSplits
        train / val / test splits ready for model training, plus the
        Normalizer instance for denormalization at evaluation time.
    """
    cfg_path = Path(cfg_path).resolve()  # make absolute so .parent traversal is reliable

    # ------------------------------------------------------------------
    # 1. Load CSV and validate columns
    # ------------------------------------------------------------------
    data_path  = Path(cfg["data_path"])
    date_col   = cfg["columns"]["date"]
    state_cols = list(cfg["columns"]["state"])
    action_col = str(cfg["columns"]["action"])
    use_month  = bool(cfg["columns"].get("use_month_encoding", True))

    # Resolve relative paths: try as-is first (relative to cwd),
    # then relative to the config file's directory (repo root pattern).
    if not data_path.is_absolute() and not data_path.exists():
        data_path = cfg_path.parent.parent.parent / data_path  # configs/reservoirs/ → repo root
    if not data_path.exists():
        raise FileNotFoundError(
            f"Data file not found: {data_path}\n"
            f"Update data_path in {cfg_path} or pass --data_path."
        )

    df = pd.read_csv(data_path)

    required_cols = [date_col] + state_cols + [action_col]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Columns missing from {data_path.name}: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)
    df["_year"] = df[date_col].dt.year

    # ------------------------------------------------------------------
    # 2. Chronological split by calendar year
    # ------------------------------------------------------------------
    years   = sorted(df["_year"].unique())
    n_train = int(cfg["split"]["train"])
    n_val   = int(cfg["split"]["val"])
    n_test  = int(cfg["split"]["test"])
    n_total = n_train + n_val + n_test

    if len(years) < n_total:
        raise ValueError(
            f"Split requires {n_total} years "
            f"(train={n_train}, val={n_val}, test={n_test}) "
            f"but the dataset only contains {len(years)} unique years."
        )

    train_years = set(years[:n_train])
    val_years   = set(years[n_train : n_train + n_val])
    test_years  = set(years[n_train + n_val : n_total])

    df_train = df[df["_year"].isin(train_years)].reset_index(drop=True)
    df_val   = df[df["_year"].isin(val_years)].reset_index(drop=True)
    df_test  = df[df["_year"].isin(test_years)].reset_index(drop=True)

    # ------------------------------------------------------------------
    # 3. Compute normalization bounds from TRAINING split only
    # ------------------------------------------------------------------
    norm_cols = list(dict.fromkeys(state_cols + [action_col]))  # preserve order, drop duplicates
    bounds: Dict[str, Dict[str, float]] = {
        col: {
            "min": float(df_train[col].min()),
            "max": float(df_train[col].max()),
        }
        for col in norm_cols
    }

    # ------------------------------------------------------------------
    # 4. Write bounds back to reservoir config (preserves YAML comments)
    # ------------------------------------------------------------------
    _write_bounds_to_config(cfg_path, bounds)

    # ------------------------------------------------------------------
    # 5 & 6. Normalize splits and build state/action arrays
    # ------------------------------------------------------------------
    normalizer = Normalizer(bounds)

    def _build_split(df_split: pd.DataFrame) -> Split:
        # Normalize each state column
        state_arrays: List[np.ndarray] = [
            normalizer.normalize(col, df_split[col].values)
            for col in state_cols
        ]

        # Append sin/cos month encoding if requested
        if use_month:
            month   = df_split[date_col].dt.month.values.astype(np.float32)
            sin_m   = np.sin(2.0 * np.pi * month / 12.0).astype(np.float32)
            cos_m   = np.cos(2.0 * np.pi * month / 12.0).astype(np.float32)
            state_arrays += [sin_m, cos_m]

        states = np.column_stack(state_arrays).astype(np.float32)  # (N, state_dim)

        # Normalize action
        raw_actions = df_split[action_col].values.astype(np.float32)
        actions     = normalizer.normalize(action_col, raw_actions)   # (N,)

        # ------------------------------------------------------------------
        # 7. next_states and done flags
        # ------------------------------------------------------------------
        # next_state[i] = state[i+1]; at the final row, repeat current state
        next_states        = np.empty_like(states)
        next_states[:-1]   = states[1:]
        next_states[-1]    = states[-1]

        # done = True at the last timestep of each year and at the very last row
        year_arr       = df_split[date_col].dt.year.values
        dones          = np.zeros(len(df_split), dtype=bool)
        dones[:-1]     = year_arr[:-1] != year_arr[1:]
        dones[-1]      = True

        return Split(
            states      = states,
            actions     = actions,
            next_states = next_states,
            dones       = dones,
            raw_actions = raw_actions,
            dates       = pd.DatetimeIndex(df_split[date_col]),
        )

    train_split = _build_split(df_train)
    val_split   = _build_split(df_val)
    test_split  = _build_split(df_test)

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------
    state_dim       = len(state_cols) + (2 if use_month else 0)
    state_cols_out  = state_cols + (["sin_month", "cos_month"] if use_month else [])

    return DataSplits(
        train      = train_split,
        val        = val_split,
        test       = test_split,
        bounds     = bounds,
        state_dim  = state_dim,
        state_cols = state_cols_out,
        normalizer = normalizer,
    )


# ---------------------------------------------------------------------------
# Config write-back (comment-preserving)
# ---------------------------------------------------------------------------

def _write_bounds_to_config(
    cfg_path: Path,
    bounds: Dict[str, Dict[str, float]],
) -> None:
    """
    Write training-computed bounds into the reservoir YAML under
    reservoir.bounds, preserving all existing comments.

    Skips writing if bounds are already populated — they are computed
    from the training split and never change between runs, so a one-time
    write is sufficient.  This prevents repeated ruamel.yaml round-trips
    from silently altering YAML formatting on every tune/train call.

    Uses ruamel.yaml if available (comment-preserving); falls back to
    plain PyYAML with a warning if ruamel is not installed.
    """
    # ------------------------------------------------------------------
    # Early-exit: skip if bounds already written
    # ------------------------------------------------------------------
    with open(cfg_path, "r") as f:
        _existing = yaml.safe_load(f)
    if _existing.get("reservoir", {}).get("bounds"):
        return

    try:
        from ruamel.yaml import YAML
        _use_ruamel = True
    except ImportError:
        _use_ruamel = False

    if _use_ruamel:
        ryaml = YAML()
        ryaml.preserve_quotes = True
        ryaml.best_width = 4096  # prevent unwanted line wrapping

        with open(cfg_path, "r") as f:
            cfg = ryaml.load(f)

        # Build the bounds mapping in ruamel's CommentedMap style
        from ruamel.yaml.comments import CommentedMap
        bounds_map = CommentedMap()
        for col, vals in bounds.items():
            entry = CommentedMap()
            entry["min"] = round(vals["min"], 6)
            entry["max"] = round(vals["max"], 6)
            bounds_map[col] = entry

        cfg["reservoir"]["bounds"] = bounds_map

        with open(cfg_path, "w") as f:
            ryaml.dump(cfg, f)

    else:
        # Fallback: plain PyYAML (destroys comments — warn the user)
        import warnings
        warnings.warn(
            "ruamel.yaml not installed — bounds written with plain PyYAML. "
            "YAML comments in the reservoir config will be lost. "
            "Install ruamel.yaml to preserve them: pip install ruamel.yaml",
            UserWarning,
            stacklevel=2,
        )
        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f)

        cfg["reservoir"]["bounds"] = {
            col: {"min": round(v["min"], 6), "max": round(v["max"], 6)}
            for col, v in bounds.items()
        }

        with open(cfg_path, "w") as f:
            yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
