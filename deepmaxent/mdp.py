"""
deepmaxent/mdp.py
=================
Discretization, expert trajectories, and the transition model for the gridded
reservoir MDP used by Deep MaxEnt IRL.

State  = (storage-bin, inflow-bin) flattened to a single index;
         month (1-12) is an exogenous, cyclic context.
Action = release-bin.

Dynamics:
  * inflow follows a data-estimated Markov chain (Laplace-smoothed),
  * storage is deterministic mass balance, S' = clip(S + fvf*(inflow - release)),
where fvf = seconds_per_day / volume_factor.

Flat-index helpers: state = inflow_idx * n_storage + storage_idx.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def discretize(val: float, step: float) -> float:
    if pd.isna(val):
        return np.nan
    return round(val / step) * step


def get_state_idx(s_idx: int, i_idx: int, n_s: int) -> int:
    return i_idx * n_s + s_idx


def get_si_fi(flat_idx: int, n_s: int) -> Tuple[int, int]:
    return flat_idx % n_s, flat_idx // n_s


def create_spaces(df: pd.DataFrame, storage_step: float, release_step: float,
                  inflow_step: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    s_min, s_max = df["storage"].min(), df["storage"].max()
    r_min, r_max = df["release"].min(), df["release"].max()
    i_min, i_max = df["inflow"].min(), df["inflow"].max()
    s_space = np.arange(np.floor(s_min / storage_step) * storage_step, s_max + storage_step, storage_step)
    r_space = np.arange(np.floor(r_min / release_step) * release_step, r_max + release_step, release_step)
    i_space = np.arange(np.floor(i_min / inflow_step) * inflow_step, i_max + inflow_step, inflow_step)
    return s_space, r_space, i_space


def create_trajectories(df: pd.DataFrame, s_space, r_space, i_space,
                        storage_step, release_step, inflow_step
                        ) -> Tuple[List, List, Dict, Dict, Dict]:
    d = df.copy()
    d["s_d"] = d["storage"].apply(lambda x: discretize(x, storage_step))
    d["r_d"] = d["release"].apply(lambda x: discretize(x, release_step))
    d["i_d"] = d["inflow"].apply(lambda x: discretize(x, inflow_step))
    d.dropna(subset=["s_d", "r_d", "i_d"], inplace=True)

    s_map = {v: i for i, v in enumerate(s_space)}
    r_map = {v: i for i, v in enumerate(r_space)}
    i_map = {v: i for i, v in enumerate(i_space)}

    trajs, trajs_raw = [], []
    for year in sorted(d["year"].unique()):
        yd = d[d["year"] == year]
        t = yd[["s_d", "month", "r_d", "i_d"]].values.tolist()
        t_raw = yd[["storage", "month", "release", "inflow"]].values.tolist()
        if t:
            trajs.append(t); trajs_raw.append(t_raw)
    return trajs, trajs_raw, s_map, r_map, i_map


def build_inflow_transitions(df: pd.DataFrame, i_map: Dict, inflow_step: float) -> np.ndarray:
    n_i = len(i_map)
    trans = np.zeros((n_i, n_i))
    for year in sorted(df["year"].unique()):
        yd = df[df["year"] == year].sort_values("month")
        for k in range(len(yd) - 1):
            ci = discretize(yd.iloc[k]["inflow"], inflow_step)
            ni = discretize(yd.iloc[k + 1]["inflow"], inflow_step)
            if ci in i_map and ni in i_map:
                trans[i_map[ci], i_map[ni]] += 1
    trans += 0.1  # Laplace smoothing
    return trans / trans.sum(axis=1, keepdims=True)


def build_transition_matrix(s_space, r_space, i_space, inflow_trans: np.ndarray,
                            flow_to_volume_factor: float) -> Tuple[np.ndarray, int]:
    n_s, n_i, n_r = len(s_space), len(i_space), len(r_space)
    n_states = n_s * n_i
    P = np.zeros((n_states, 12, n_r, n_states), dtype=np.float32)
    s_min, s_max = s_space[0], s_space[-1]

    for si in range(n_s):
        s_val = s_space[si]
        for fi in range(n_i):
            i_val = i_space[fi]
            curr_state = get_state_idx(si, fi, n_s)
            for m in range(12):
                for ri, r_val in enumerate(r_space):
                    s_next = np.clip(s_val + flow_to_volume_factor * (i_val - r_val), s_min, s_max)
                    si_next = int(np.argmin(np.abs(s_space - s_next)))
                    for fi_next in range(n_i):
                        prob = inflow_trans[fi, fi_next]
                        if prob > 1e-6:
                            P[curr_state, m, ri, get_state_idx(si_next, fi_next, n_s)] += prob
    return P, n_s


def grid_sizes(df: pd.DataFrame, storage_step, release_step, inflow_step) -> Tuple[int, int, int, int]:
    """(n_storage, n_release, n_inflow, n_states) for a candidate discretization."""
    s, r, i = create_spaces(df, storage_step, release_step, inflow_step)
    return len(s), len(r), len(i), len(s) * len(i)
