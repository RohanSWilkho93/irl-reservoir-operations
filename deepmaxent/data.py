"""
deepmaxent/data.py
==================
Raw (engineering-unit) data loading for Deep MaxEnt IRL.

Unlike IQ-Learn (which normalises to [0,1]), Deep MaxEnt discretizes the RAW
storage / release / inflow, so this loader keeps engineering units and just adds
`year` / `month` and an `inflow` alias for the configured inflow column. The
chronological train/val/test split uses the SAME year counts as the reservoir
config (configs/reservoirs/<name>.yaml → split), so it lines up with IQ-Learn.

`flow_to_volume_factor = seconds_per_day / volume_factor` is read from the
reservoir config's mass_balance block (default 86400 / 1e6).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import pandas as pd


@dataclass
class DMData:
    full:   pd.DataFrame
    train:  pd.DataFrame
    val:    pd.DataFrame
    test:   pd.DataFrame
    train_years: List[int]
    val_years:   List[int]
    test_years:  List[int]
    flow_to_volume_factor: float
    use_month_encoding: bool


def load_raw_reservoir_data(cfg: dict, cfg_path: str | Path) -> DMData:
    """Read the reservoir CSV in engineering units and split chronologically by year."""
    cfg_path = Path(cfg_path).resolve()
    cols = cfg["columns"]
    date_col   = cols["date"]
    storage_col = cols.get("storage", "storage")
    action_col  = str(cols["action"])
    inflow_col  = cols.get("inflow", "net_inflow")
    use_month   = bool(cols.get("use_month_encoding", True))

    data_path = Path(cfg["data_path"])
    if not data_path.is_absolute() and not data_path.exists():
        data_path = cfg_path.parent.parent.parent / data_path     # configs/reservoirs/ -> repo root
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    df = pd.read_csv(data_path)
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(date_col).reset_index(drop=True)
    if "month" not in df.columns:
        df["month"] = df[date_col].dt.month
    if "year" not in df.columns:
        df["year"] = df[date_col].dt.year

    # Canonical engineering-unit columns the MDP code expects.
    df["storage"] = df[storage_col].astype(float)
    df["release"] = df[action_col].astype(float)
    df["inflow"]  = df[inflow_col].astype(float)

    years = sorted(df["year"].unique())
    n_tr = int(cfg["split"]["train"]); n_va = int(cfg["split"]["val"]); n_te = int(cfg["split"]["test"])
    if len(years) < n_tr + n_va + n_te:
        raise ValueError(f"Split needs {n_tr+n_va+n_te} years; dataset has {len(years)}.")
    tr_y = years[:n_tr]; va_y = years[n_tr:n_tr + n_va]; te_y = years[n_tr + n_va:n_tr + n_va + n_te]

    mb = (cfg.get("reservoir", {}) or {}).get("mass_balance", {}) or {}
    spd = float(mb.get("seconds_per_day") or 86400.0)
    vf  = float(mb.get("volume_factor") or 1.0e6)

    pick = lambda ys: df[df["year"].isin(ys)].copy()
    return DMData(
        full=df, train=pick(tr_y), val=pick(va_y), test=pick(te_y),
        train_years=tr_y, val_years=va_y, test_years=te_y,
        flow_to_volume_factor=spd / vf, use_month_encoding=use_month,
    )
