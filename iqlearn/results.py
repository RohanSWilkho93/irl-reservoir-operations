"""
iqlearn/generate_results.py
===========================
Final-figure generation for a trained IQ-Learn agent, on the TEST split.

Loads the agent with IQLearnAgent.from_checkpoint and the SAME resolved physics
training used (read from iq_best_config.json -> mass_balance), then produces
four deliverables into  results/<reservoir>/iqlearn/<run_id>/figures/ :

  1. Reward-contours  (reward_contours.png)
     Learned Q(s, a) = min(Q1, Q2) over a (storage x release) grid.  The critic
     takes a continuous normalised release, so the release axis is a true
     continuous grid (no bin hack).  Q is averaged over the ENTIRE trajectory's
     observed inflows per month; sin/cos are read from a real row of that month
     (the encoding is never re-derived).  ALL historical observations
     (train+val+test) are overlaid.  use_month_encoding -> 12 panels; else 1.

  2. MONTE-CARLO ROLLOUT FANS  (mc_fan_test.png and mc_fan_full.png)
     n_mc stochastic closed-loop rollouts (select_action deterministic=False),
     produced for the TEST split ("Test data") AND the FULL trajectory
     ("All data", train+val+test).  One stacked Storage/Release figure per scope:
     blue solid Observed, red dashed Median, salmon IQR(25-75) band; each panel
     title reports Pearson r and nRMSE of the MEDIAN trajectory vs observed.

  3. SHAP OVERALL BARS  (shap_policy_overall.png, shap_critic_overall.png)
     mean|SHAP| per feature (raw units, matching the attached bar figure), on
     the chosen split (shap_split: train/val/test/all, default all).
       policy : features = state  [storage, inflow (, sin_month, cos_month)]
                explains the expected release  E[a|s] = sum_k p_k bin_means[k]
       critic : features = state + release  [.., release]
                explains  min(Q1, Q2)(s, a)            (Q is a function of s AND a)

  4. SHAP MONTHLY HEATMAPS  (shap_policy_monthly.png, shap_critic_monthly.png)
     Only when use_month_encoding.  Per-month mean|SHAP|, with sin_month/cos_month
     ROWS DROPPED and the remaining features renormalised to 100% per month
     (matching the attached heatmap, minus the trivially-zero seasonal rows).
       policy rows : [storage, inflow]
       critic rows : [storage, inflow, release]

Runnable standalone (CLI; --run_id optional, defaults to the latest run) and
importable as run_generate_results(...) for run.py to call after the IQ stage.

Shapes: N = #test rows, D = state_dim, K = n_bins, G = grid_size.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Dict, List, Sequence

import numpy as np
import pandas as pd
import torch

import matplotlib
matplotlib.use("Agg")            # headless: save only, never display
import matplotlib.pyplot as plt

# ---- repo root on path so `iqlearn.*` and `utils.*` import cleanly ----
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from iqlearn.agent import IQLearnAgent
from iqlearn.environment import ReservoirRollout, MassBalanceConfig
from iqlearn.utils.runs import _resolve_device, _find_run_folder, _RUN_FOLDER_PATTERN
from utils.data import load_reservoir_data


_MONTH_ABBR = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
_MONTH_FULL = ["January", "February", "March", "April", "May", "June",
               "July", "August", "September", "October", "November", "December"]
_DROP_FEATURES = ("sin_month", "cos_month")   # removed from the MONTHLY heatmaps


# =============================================================================
# Small helpers
# =============================================================================

def _latest_run_id(base_dir: Path) -> int:
    """Largest integer run-folder under base_dir (default for --run_id)."""
    if not base_dir.exists():
        sys.exit(f"\nERROR: results directory does not exist: {base_dir}\n"
                 f"  Run the BC->IQ pipeline (run.py) for this reservoir first.\n")
    ids = [int(m.group(1)) for d in base_dir.iterdir()
           if d.is_dir() and (m := _RUN_FOLDER_PATTERN.match(d.name))]
    if not ids:
        sys.exit(f"\nERROR: no run folders found under {base_dir}\n")
    return max(ids)


def _bounds_pair(bounds: dict, col: str) -> tuple[float, float]:
    """(lo, hi) train min/max for `col` from data.bounds (handles {min,max} or pair)."""
    if col not in bounds:
        raise ValueError(f"data.bounds is missing column '{col}'.")
    b = bounds[col]
    lo, hi = (b["min"], b["max"]) if isinstance(b, dict) else (b[0], b[1])
    return float(lo), float(hi)


def _denorm(z, lo: float, hi: float):
    return z * (hi - lo) + lo


def _months_of(dates) -> np.ndarray:
    """Calendar month (1-12) for each row, robust to datetime64 / str / Timestamp."""
    return pd.DatetimeIndex(pd.to_datetime(pd.Series(np.asarray(dates)).values)).month.to_numpy()


def _display_names(state_cols: Sequence[str], storage_col: str, inflow_col: str) -> List[str]:
    """Map the storage/inflow role columns to 'storage'/'inflow'; keep the rest."""
    rename = {storage_col: "storage", inflow_col: "inflow"}
    return [rename.get(c, c) for c in state_cols]


def _split_arrays(data, which: str):
    """
    (states, actions, raw_actions, dates) for which in {'train','val','test','all'}.

    'all' concatenates train+val+test in chronological order (the pipeline's
    chronological split puts train earliest, test latest), giving the entire
    available trajectory.
    """
    if which == "all":
        parts = [data.train, data.val, data.test]
        st = np.concatenate([np.asarray(p.states, np.float32) for p in parts], axis=0)
        ac = np.concatenate([np.asarray(p.actions, np.float32) for p in parts], axis=0)
        rw = np.concatenate([np.asarray(p.raw_actions, np.float32) for p in parts], axis=0)
        dt = np.concatenate([np.asarray(p.dates) for p in parts], axis=0)
        return st, ac, rw, dt
    sp = getattr(data, which)
    return (np.asarray(sp.states, np.float32), np.asarray(sp.actions, np.float32),
            np.asarray(sp.raw_actions, np.float32), np.asarray(sp.dates))


def _r_and_nrmse(sim: np.ndarray, obs: np.ndarray) -> tuple[float, float]:
    """Pearson r and normalised RMSE (rmse / observed range) of sim vs obs."""
    if sim.size < 2 or np.std(sim) < 1e-12 or np.std(obs) < 1e-12:
        r = 0.0
    else:
        r = float(np.corrcoef(sim, obs)[0, 1])
        if not np.isfinite(r):
            r = 0.0
    rmse = float(np.sqrt(np.mean((sim - obs) ** 2)))
    rng = float(obs.max() - obs.min())
    nrmse = rmse / rng if rng > 1e-12 else float("inf")
    return r, nrmse


# =============================================================================
# SHAP
# =============================================================================

def _shap_matrix(f: Callable[[np.ndarray], np.ndarray],
                 x_explain: np.ndarray, x_background: np.ndarray,
                 nsamples: int) -> np.ndarray:
    """KernelSHAP values, normalised to a (n_explain, n_features) float array."""
    import shap   # imported lazily so the module loads even if shap is absent
    explainer = shap.KernelExplainer(f, x_background)
    sv = explainer.shap_values(x_explain, nsamples=nsamples, silent=True)
    if isinstance(sv, list):                 # multi-output guard (we are scalar)
        sv = sv[0]
    sv = np.asarray(sv, dtype=np.float64)
    if sv.ndim > 2:                          # squeeze any trailing singleton output dim
        sv = sv.reshape(x_explain.shape[0], x_explain.shape[1])
    return sv


def _subsample(n: int, k: int, rng: np.random.Generator) -> np.ndarray:
    """Up to k distinct indices in [0, n)."""
    k = min(k, n)
    return rng.choice(n, size=k, replace=False)


def _plot_overall_bar(feature_names: List[str], mean_abs: np.ndarray,
                      network_label: str, path: Path) -> None:
    """Horizontal mean|SHAP| bar (raw units), largest on top."""
    order = np.argsort(mean_abs)             # ascending -> largest ends up on top in barh
    names = [feature_names[i] for i in order]
    vals = mean_abs[order]

    plt.figure(figsize=(10, 6))
    plt.barh(range(len(names)), vals, color="#1f9bff")
    plt.yticks(range(len(names)), names, fontsize=12)
    plt.xlabel("mean(|SHAP value|)  (average impact on model output magnitude)", fontsize=12)
    plt.title(f"Overall Feature Importance: {network_label} (mean |SHAP|)",
              fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def _plot_monthly_heatmap(per_month: Dict[int, Dict[str, float]],
                          kept_names: List[str], network_label: str, path: Path) -> None:
    """Per-month importance heatmap over kept features, renormalised to 100%/month."""
    present = sorted(per_month)
    data = np.zeros((len(kept_names), len(present)), dtype=np.float64)
    for j, m in enumerate(present):
        vals = np.array([per_month[m][n] for n in kept_names], dtype=np.float64)
        tot = vals.sum()
        data[:, j] = (vals / tot * 100.0) if tot > 1e-12 else 0.0

    fig_w = max(13, 1.7 * len(present))
    fig_h = max(4.5, 2.4 * len(kept_names))
    plt.figure(figsize=(fig_w, fig_h))
    im = plt.imshow(data, cmap="YlOrRd", aspect="auto", vmin=0.0)
    plt.xticks(range(len(present)), [_MONTH_ABBR[m - 1] for m in present],
               fontsize=24, fontweight="bold")
    plt.yticks(range(len(kept_names)), [n.capitalize() for n in kept_names],
               fontsize=24, fontweight="bold")

    hi = data.max() if data.size else 1.0
    for i in range(len(kept_names)):
        for j in range(len(present)):
            color = "white" if data[i, j] > hi * 0.6 else "black"
            plt.text(j, i, f"{data[i, j]:.1f}", ha="center", va="center",
                     color=color, fontsize=24, fontweight="bold")

    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def _shap_block(*, network_label: str, f: Callable[[np.ndarray], np.ndarray],
                X: np.ndarray, feature_names: List[str], months: np.ndarray,
                month_encoded: bool, figdir: Path, stem: str,
                n_background: int, nsamples: int, max_explain: int,
                rng: np.random.Generator) -> List[Path]:
    """Overall bar (always) + monthly heatmap (only if month_encoded)."""
    saved: List[Path] = []
    n = X.shape[0]

    # ---- overall ----
    ex_idx = np.arange(n) if (max_explain <= 0 or max_explain >= n) else _subsample(n, max_explain, rng)
    x_explain = X[ex_idx]
    x_background = X[_subsample(n, n_background, rng)]
    print(f"  [{network_label}] overall SHAP: explain {len(x_explain)} rows, "
          f"background {len(x_background)} (nsamples={nsamples})")
    sv = _shap_matrix(f, x_explain, x_background, nsamples)
    mean_abs = np.abs(sv).mean(axis=0)
    p = figdir / f"{stem}_overall.png"
    _plot_overall_bar(feature_names, mean_abs, network_label, p)
    saved.append(p)

    if not month_encoded:
        return saved

    # ---- monthly ----
    per_month: Dict[int, Dict[str, float]] = {}
    for m in range(1, 13):
        rows = np.where(months == m)[0]
        if len(rows) < 10:                                   # too few rows to explain
            continue
        ex = rows if (max_explain <= 0 or max_explain >= len(rows)) \
            else rows[_subsample(len(rows), max_explain, rng)]
        bg = rows[_subsample(len(rows), n_background, rng)]
        sv_m = _shap_matrix(f, X[ex], X[bg], nsamples)
        ma = np.abs(sv_m).mean(axis=0)
        per_month[m] = {name: float(ma[i]) for i, name in enumerate(feature_names)}
        print(f"  [{network_label}] monthly SHAP {_MONTH_ABBR[m - 1]}: {len(ex)} rows")

    if per_month:
        kept = [nm for nm in feature_names if nm not in _DROP_FEATURES]
        p = figdir / f"{stem}_monthly.png"
        _plot_monthly_heatmap(per_month, kept, network_label, p)
        saved.append(p)
    return saved


# =============================================================================
# Monte-Carlo rollout fans
# =============================================================================

def _mc_fans(agent: IQLearnAgent, env: ReservoirRollout, figdir: Path,
             n_mc: int, seed: int, *, suffix: str, scope_title: str) -> List[Path]:
    """n_mc stochastic rollouts -> one stacked Storage/Release figure (median + IQR band)."""
    sims_storage, sims_release = [], []
    obs_storage = obs_release = None
    for i in range(n_mc):
        gen = torch.Generator(device=agent.device).manual_seed(seed + i)
        out = env.rollout(agent, deterministic=False, generator=gen)
        sims_storage.append(out["sim_storage"])
        sims_release.append(out["sim_release"])
        obs_storage, obs_release = out["obs_storage"], out["obs_release"]

    sims_storage = np.stack(sims_storage, axis=0)   # (n_mc, T-1)
    sims_release = np.stack(sims_release, axis=0)

    fig, (ax_s, ax_r) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    for ax, label, ylab, sims, obs in (
            (ax_s, "Storage", "Storage (Mm$^3$)", sims_storage, obs_storage),
            (ax_r, "Release", "Release (m$^3$/s)", sims_release, obs_release)):
        median = np.median(sims, axis=0)
        q25 = np.percentile(sims, 25, axis=0)
        q75 = np.percentile(sims, 75, axis=0)
        r, nrmse = _r_and_nrmse(median, obs)
        t = np.arange(len(obs))

        ax.fill_between(t, q25, q75, color="lightcoral", alpha=0.6,
                        label="25th-75th percentile (IQR)")
        ax.plot(t, obs, color="#1f5fbf", linewidth=1.3, label="Observed")
        ax.plot(t, median, color="#d62728", linewidth=2.0, linestyle="--",
                label="Median (MC rollouts)")
        ax.set_ylabel(ylab, fontsize=18)
        ax.set_title(f"{label}    r = {r:.3f},  nRMSE = {nrmse:.3f}",
                     fontsize=20, fontweight="bold")
        ax.tick_params(labelsize=14)
        ax.legend(loc="upper right", fontsize=14)
        ax.grid(True, alpha=0.3)
        print(f"  MC fan [{label}, {scope_title}]: r={r:.3f}, nRMSE={nrmse:.3f}")

    ax_r.set_xlabel("Time Steps (Days)", fontsize=18)
    fig.suptitle(f"Monte-Carlo Rollout - {scope_title}", fontsize=24, fontweight="bold")
    plt.tight_layout()
    p = figdir / f"mc_fan_{suffix}.png"
    plt.savefig(p, dpi=300, bbox_inches="tight")
    plt.close()
    return [p]


# =============================================================================
# Reward-contours
# =============================================================================

def _reward_contours(agent: IQLearnAgent, *, obs_states: np.ndarray, raw_actions: np.ndarray,
                state_cols: List[str], storage_col: str, inflow_col: str,
                norm_bounds: Dict[str, tuple], month_encoded: bool, months: np.ndarray,
                figdir: Path, grid_size: int, max_inflows: int,
                rng: np.random.Generator) -> List[Path]:
    """Learned Q(s,a)=min(Q1,Q2) contours over (storage x release); monthly or single.

    `obs_states` / `raw_actions` / `months` cover the ENTIRE trajectory: their
    inflows are averaged into the Q surface and their (release, storage) points
    are overlaid as expert observations.
    """
    device = agent.device
    s_idx = state_cols.index(storage_col)
    i_idx = state_cols.index(inflow_col)
    s_lo, s_hi = norm_bounds[storage_col]

    # release bounds come from the action-column entry in norm_bounds
    # (norm_bounds is keyed by storage_col, inflow_col, action_col): the action
    # key is the one that is neither storage nor inflow.
    action_keys = [k for k in norm_bounds if k not in (storage_col, inflow_col)]
    a_lo, a_hi = norm_bounds[action_keys[0]]

    storage_grid = np.linspace(0.0, 1.0, grid_size)        # normalised storage axis
    release_grid = np.linspace(0.0, 1.0, grid_size)        # normalised release axis
    SN, RN = np.meshgrid(storage_grid, release_grid)       # (G, G); storage->cols, release->rows
    storage_flat = SN.ravel().astype(np.float32)
    release_flat = RN.ravel().astype(np.float32)
    storage_eng = _denorm(SN, s_lo, s_hi)                  # axis labels (engineering units)
    release_eng = _denorm(RN, a_lo, a_hi)
    actions_t = torch.from_numpy(release_flat).to(device)

    panel_months = list(range(1, 13)) if month_encoded else [None]

    @torch.no_grad()
    def q_map_for(template_row: np.ndarray, inflow_norms: np.ndarray) -> np.ndarray:
        """Average min(Q1,Q2) over the given inflow values -> (G, G)."""
        acc = np.zeros((grid_size, grid_size), dtype=np.float64)
        for inflow_norm in inflow_norms:
            states = np.tile(template_row, (storage_flat.shape[0], 1)).astype(np.float32)
            states[:, s_idx] = storage_flat
            states[:, i_idx] = float(inflow_norm)
            s_t = torch.from_numpy(states).to(device)
            q1, q2 = agent.critic(s_t, actions_t)
            q = torch.min(q1, q2).cpu().numpy().reshape(grid_size, grid_size)
            acc += q
        return acc / max(1, len(inflow_norms))

    # ---- build each panel's Q map + expert scatter ----
    maps: List[np.ndarray | None] = []
    scatters: List[tuple] = []
    for m in panel_months:
        rows = np.arange(len(obs_states)) if m is None else np.where(months == m)[0]
        if len(rows) == 0:
            maps.append(None)
            scatters.append((np.array([]), np.array([])))
            continue
        template = obs_states[rows[0]].astype(np.float32)          # real row -> correct sin/cos
        inflow_norms = obs_states[rows, i_idx]
        if len(inflow_norms) > max_inflows:
            inflow_norms = inflow_norms[_subsample(len(inflow_norms), max_inflows, rng)]
        maps.append(q_map_for(template, inflow_norms))
        scatters.append((raw_actions[rows],                                  # release (eng)
                         _denorm(obs_states[rows, s_idx], s_lo, s_hi)))      # storage (eng)
        if m is not None:
            print(f"  Reward-contour {_MONTH_ABBR[m - 1]}: {len(inflow_norms)} inflow samples")

    valid = [q for q in maps if q is not None]
    if not valid:
        print("  Reward-contour: no test rows; skipping.")
        return []
    vmin = float(min(q.min() for q in valid))
    vmax = float(max(q.max() for q in valid))
    levels = np.linspace(vmin, vmax, 50)

    # ---- plot ----
    if month_encoded:
        fig, axes = plt.subplots(3, 4, figsize=(26, 16), sharex=True, sharey=True)
        axes_flat = axes.flatten()
    else:
        fig, ax0 = plt.subplots(1, 1, figsize=(10, 8))
        axes_flat = [ax0]

    for idx, (m, q_2d, (sx, sy)) in enumerate(zip(panel_months, maps, scatters)):
        ax = axes_flat[idx]
        title = "Reward function" if m is None else _MONTH_FULL[m - 1]
        if q_2d is None:
            ax.set_title(f"{title} (no data)", fontsize=20, fontweight="bold")
            continue
        ax.contourf(release_eng, storage_eng, q_2d, levels=levels,
                    cmap="RdYlGn", vmin=vmin, vmax=vmax, extend="both")
        ax.contour(release_eng, storage_eng, q_2d, levels=15,
                   colors="black", alpha=0.2, linewidths=0.5)
        if sx.size:
            ax.scatter(sx, sy, c="magenta", s=24, alpha=0.6,
                       edgecolors="none", zorder=5)
        ax.set_title(title, fontsize=20, fontweight="bold")
        ax.tick_params(labelsize=14)
        ax.grid(True, alpha=0.3, linestyle="--")
        if not month_encoded or idx >= 8:
            ax.set_xlabel("Release (m$^3$/s)", fontsize=17)
        if not month_encoded or idx % 4 == 0:
            ax.set_ylabel("Storage (Mm$^3$)", fontsize=17)

    sm = plt.cm.ScalarMappable(cmap="RdYlGn", norm=plt.Normalize(vmin, vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes_flat if month_encoded else axes_flat[0],
                        location="right", shrink=0.8, pad=0.02)
    cbar.set_label("Reward", fontsize=20, fontweight="bold")
    cbar.ax.tick_params(labelsize=14)
    fig.suptitle("Reward Function Contours", fontsize=26, fontweight="bold", y=0.997)

    p = figdir / "reward_contours.png"
    plt.savefig(p, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"  Reward-contours saved ({'12 panels' if month_encoded else '1 panel'}).")
    return [p]


# =============================================================================
# Orchestrator
# =============================================================================

def run_generate_results(*, reservoir: str, res_cfg: dict, res_cfg_path: str,
                         algo_cfg: dict, data, device_str: str, run_id: int,
                         n_mc: int = 200, shap_n_background: int = 100,
                         shap_nsamples: int = 100, shap_max_explain: int = 300,
                         shap_split: str = "all", grid_size: int = 120,
                         contour_max_inflows: int = 80, seed: int | None = None) -> dict:
    """
    Generate result figures for run_id into <run_folder>/figures/.

    Scope: Reward-contours overlay the ENTIRE trajectory; Monte-Carlo fans are drawn
    for the TEST split and the FULL trajectory; SHAP runs on `shap_split`
    (train/val/test/all, default all).  Reads iq_agent.pt + iq_best_config.json
    from the run folder and uses the SAME resolved mass-balance the agent trained
    with.  Returns a dict of saved paths.
    """
    base_dir = Path("results") / reservoir / "iqlearn"
    run_folder = _find_run_folder(base_dir, run_id)
    agent_path = run_folder / "iq_agent.pt"
    cfg_path = run_folder / "iq_best_config.json"
    if not agent_path.exists():
        sys.exit(f"\nERROR: {agent_path} not found. Run the IQ stage for run {run_id} first.\n")
    if not cfg_path.exists():
        sys.exit(f"\nERROR: {cfg_path} not found (needed for resolved physics).\n")

    # ---- agent + the exact physics training used ----
    agent = IQLearnAgent.from_checkpoint(agent_path, device_str)
    best_cfg = json.loads(cfg_path.read_text())
    mb = MassBalanceConfig(**best_cfg["mass_balance"])
    if seed is None:
        seed = int(best_cfg.get("seed", 42))
    rng = np.random.default_rng(seed)

    # ---- data views ----
    state_cols: List[str] = list(data.state_cols)
    month_encoded = ("sin_month" in state_cols) and ("cos_month" in state_cols)
    storage_col, inflow_col, action_col = mb.storage_col, mb.inflow_col, mb.action_col
    norm_bounds = {
        storage_col: _bounds_pair(data.bounds, storage_col),
        inflow_col:  _bounds_pair(data.bounds, inflow_col),
        action_col:  _bounds_pair(data.bounds, action_col),
    }

    # ---- split arrays ----
    all_states, all_actions, all_raw, all_dates = _split_arrays(data, "all")
    all_months = _months_of(all_dates)
    D = all_states.shape[1]
    disp = _display_names(state_cols, storage_col, inflow_col)        # display feature names

    figdir = run_folder / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating results for {reservoir} run {run_id}  "
          f"(month_encoding={month_encoded}, N_all={len(all_states)}, "
          f"N_test={len(data.test.states)}, shap_split={shap_split})")
    print(f"  figures -> {figdir}")

    saved: Dict[str, List[Path]] = {}

    # ---- 1. Reward-contours: overlay + inflow-averaging over the ENTIRE trajectory ----
    saved["reward_contours"] = _reward_contours(
        agent, obs_states=all_states, raw_actions=all_raw, state_cols=state_cols,
        storage_col=storage_col, inflow_col=inflow_col, norm_bounds=norm_bounds,
        month_encoded=month_encoded, months=all_months, figdir=figdir,
        grid_size=grid_size, max_inflows=contour_max_inflows, rng=rng,
    )

    # ---- 2. Monte-Carlo fans: TEST split AND the full trajectory ----
    env_test = ReservoirRollout(data.test, state_cols, mb, norm_bounds, device_str)
    saved["mc_fans_test"] = _mc_fans(agent, env_test, figdir, n_mc=n_mc, seed=seed,
                                     suffix="test", scope_title="Test data")

    full_split = SimpleNamespace(states=all_states, raw_actions=all_raw)
    env_full = ReservoirRollout(full_split, state_cols, mb, norm_bounds, device_str)
    saved["mc_fans_full"] = _mc_fans(agent, env_full, figdir, n_mc=n_mc, seed=seed,
                                     suffix="full", scope_title="All data")

    # ---- 3+4. SHAP on the chosen split (default: entire trajectory) ----
    shap_states, shap_actions, _, shap_dates = _split_arrays(data, shap_split)
    shap_months = _months_of(shap_dates)

    @torch.no_grad()
    def policy_f(Xnp: np.ndarray) -> np.ndarray:
        s = torch.from_numpy(np.ascontiguousarray(np.asarray(Xnp, np.float32))).to(agent.device)
        return agent.select_action(s, deterministic=True).cpu().numpy()

    @torch.no_grad()
    def critic_f(Xnp: np.ndarray) -> np.ndarray:
        Xnp = np.asarray(Xnp, dtype=np.float32)
        s = torch.from_numpy(np.ascontiguousarray(Xnp[:, :D])).to(agent.device)
        a = torch.from_numpy(np.ascontiguousarray(Xnp[:, D])).to(agent.device)
        q1, q2 = agent.critic(s, a)
        return torch.min(q1, q2).cpu().numpy()

    try:
        saved["shap_policy"] = _shap_block(
            network_label="Policy Network", f=policy_f, X=shap_states,
            feature_names=disp, months=shap_months, month_encoded=month_encoded,
            figdir=figdir, stem="shap_policy", n_background=shap_n_background,
            nsamples=shap_nsamples, max_explain=shap_max_explain, rng=rng,
        )
        X_critic = np.hstack([shap_states, shap_actions.reshape(-1, 1)]).astype(np.float32)
        saved["shap_critic"] = _shap_block(
            network_label="Q-Network", f=critic_f, X=X_critic,
            feature_names=disp + ["release"], months=shap_months, month_encoded=month_encoded,
            figdir=figdir, stem="shap_critic", n_background=shap_n_background,
            nsamples=shap_nsamples, max_explain=shap_max_explain, rng=rng,
        )
    except ImportError:
        print("  WARNING: `shap` is not installed - skipping SHAP figures. "
              "Install with `pip install shap` to generate them.")
        saved["shap_policy"] = []
        saved["shap_critic"] = []

    all_paths = [str(p) for group in saved.values() for p in group]
    print(f"\nDone. {len(all_paths)} figures in {figdir}")
    return {"run_folder": run_folder, "figures_dir": figdir, "saved": saved, "paths": all_paths}


# =============================================================================
# Standalone CLI
# =============================================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate IQ-Learn test-split result figures.")
    p.add_argument("--reservoir", required=True)
    p.add_argument("--run_id", type=int, default=None,
                   help="Run id to visualise (default: latest run found).")
    p.add_argument("--device", default="cpu", help="cpu | cuda | auto (default cpu).")
    p.add_argument("--n_mc", type=int, default=200, help="Monte-Carlo rollouts for the fans.")
    p.add_argument("--shap_n_background", type=int, default=100)
    p.add_argument("--shap_nsamples", type=int, default=100)
    p.add_argument("--shap_max_explain", type=int, default=300,
                   help="Cap on rows explained per SHAP run (<=0 means all).")
    p.add_argument("--shap_split", choices=["train", "val", "test", "all"], default="all",
                   help="Which split SHAP explains (default: all = entire trajectory).")
    p.add_argument("--grid_size", type=int, default=120, help="Reward-contour grid resolution.")
    p.add_argument("--contour_max_inflows", type=int, default=80,
                   help="Max observed inflows averaged per contour panel.")
    p.add_argument("--seed", type=int, default=None,
                   help="RNG seed (default: the seed stored in iq_best_config.json).")
    return p.parse_args()


def main() -> None:
    import yaml
    a = _parse_args()

    res_cfg_path = f"configs/reservoirs/{a.reservoir}.yaml"
    algo_cfg_path = "configs/algorithms/iqlearn.yaml"
    if not Path(res_cfg_path).exists():
        sys.exit(f"\nERROR: reservoir config not found: {res_cfg_path}\n")
    res_cfg = yaml.safe_load(open(res_cfg_path))
    algo_cfg = yaml.safe_load(open(algo_cfg_path)) if Path(algo_cfg_path).exists() else {}

    device_str = _resolve_device(a.device)

    run_id = a.run_id if a.run_id is not None \
        else _latest_run_id(Path("results") / a.reservoir / "iqlearn")

    data = load_reservoir_data(res_cfg, res_cfg_path)

    run_generate_results(
        reservoir=a.reservoir, res_cfg=res_cfg, res_cfg_path=res_cfg_path,
        algo_cfg=algo_cfg, data=data, device_str=device_str, run_id=run_id,
        n_mc=a.n_mc, shap_n_background=a.shap_n_background, shap_nsamples=a.shap_nsamples,
        shap_max_explain=a.shap_max_explain, shap_split=a.shap_split, grid_size=a.grid_size,
        contour_max_inflows=a.contour_max_inflows, seed=a.seed,
    )


if __name__ == "__main__":
    main()