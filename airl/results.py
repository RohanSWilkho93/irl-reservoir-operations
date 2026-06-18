"""
airl/results.py
===============
Figures for a trained AIRL run (auto-run by run.py; also standalone), into
results/<reservoir>/airl/<run_id>/figures/ :

  1. mc_fan_test.png / mc_fan_full.png  — stochastic Monte-Carlo rollout fans
     (median + 25-75% IQR) of storage and release: test split, and full record.
  2. reward_contours.png                — the recovered reward g(s,a) over a
     storage x release grid, per month (when month-encoded), expert obs overlaid.
  3. shap_policy_overall.png (+ _monthly) — SHAP of the policy's expected release.
  4. shap_reward_overall.png  (+ _monthly) — SHAP of the recovered reward g(s,a).

Reuses IQ-Learn's MC-fan + SHAP plotting helpers; only the reward source differs
(AIRL reward_net g(s,a) instead of the IQ critic).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils.data import load_reservoir_data
from iqlearn.environment import ReservoirRollout, MassBalanceConfig
from iqlearn.utils.runs import _resolve_device, _find_run_folder, _RUN_FOLDER_PATTERN
from iqlearn.results import (_mc_fans, _shap_block, _split_arrays, _months_of,
                             _display_names, _bounds_pair, _subsample, _MONTH_FULL)
from airl.agent import AIRLAgent


# =============================================================================
# Reward contours from the recovered reward g(s, a)
# =============================================================================

def _reward_contours(agent, *, obs_states, raw_actions, state_cols, storage_col, inflow_col,
                     norm_bounds, month_encoded, months, figdir, grid_size, max_inflows, rng):
    device = agent.device
    s_idx = state_cols.index(storage_col); i_idx = state_cols.index(inflow_col)
    s_lo, s_hi = norm_bounds[storage_col]
    a_key = [k for k in norm_bounds if k not in (storage_col, inflow_col)][0]
    a_lo, a_hi = norm_bounds[a_key]

    grid = np.linspace(0.0, 1.0, grid_size)
    SN, RN = np.meshgrid(grid, grid)
    storage_flat = SN.ravel().astype(np.float32); release_flat = RN.ravel().astype(np.float32)
    storage_eng = SN * (s_hi - s_lo) + s_lo; release_eng = RN * (a_hi - a_lo) + a_lo
    actions_t = torch.from_numpy(release_flat).to(device)
    panel_months = list(range(1, 13)) if month_encoded else [None]

    @torch.no_grad()
    def reward_map(template_row, inflow_norms):
        acc = np.zeros((grid_size, grid_size), dtype=np.float64)
        for inflow_norm in inflow_norms:
            states = np.tile(template_row, (storage_flat.shape[0], 1)).astype(np.float32)
            states[:, s_idx] = storage_flat; states[:, i_idx] = float(inflow_norm)
            g = agent.discriminator.extract_reward(torch.from_numpy(states).to(device), actions_t)
            acc += g.cpu().numpy().reshape(grid_size, grid_size)
        return acc / max(1, len(inflow_norms))

    maps, scatters = [], []
    for m in panel_months:
        rows = np.arange(len(obs_states)) if m is None else np.where(months == m)[0]
        if len(rows) == 0:
            maps.append(None); scatters.append((np.array([]), np.array([]))); continue
        template = obs_states[rows[0]].astype(np.float32)
        inflow_norms = obs_states[rows, i_idx]
        if len(inflow_norms) > max_inflows:
            inflow_norms = inflow_norms[_subsample(len(inflow_norms), max_inflows, rng)]
        maps.append(reward_map(template, inflow_norms))
        scatters.append((raw_actions[rows], obs_states[rows, s_idx] * (s_hi - s_lo) + s_lo))

    valid = [g for g in maps if g is not None]
    if not valid:
        return
    vmin = float(min(g.min() for g in valid)); vmax = float(max(g.max() for g in valid))
    levels = np.linspace(vmin, vmax, 50)
    if month_encoded:
        fig, axes = plt.subplots(3, 4, figsize=(26, 16), sharex=True, sharey=True); axf = axes.flatten()
    else:
        fig, ax0 = plt.subplots(1, 1, figsize=(10, 8)); axf = [ax0]
    for idx, (m, g2d, (sx, sy)) in enumerate(zip(panel_months, maps, scatters)):
        ax = axf[idx]; title = "Reward g(s,a)" if m is None else _MONTH_FULL[m - 1]
        if g2d is None:
            ax.set_title(f"{title} (no data)", fontsize=20, fontweight="bold"); continue
        ax.contourf(release_eng, storage_eng, g2d, levels=levels, cmap="RdYlGn", vmin=vmin, vmax=vmax, extend="both")
        ax.contour(release_eng, storage_eng, g2d, levels=15, colors="black", alpha=0.2, linewidths=0.5)
        if sx.size:
            ax.scatter(sx, sy, c="magenta", s=24, alpha=0.6, edgecolors="none", zorder=5)
        ax.set_title(title, fontsize=20, fontweight="bold"); ax.grid(True, alpha=0.3, linestyle="--")
        if not month_encoded or idx >= 8: ax.set_xlabel("Release (m³/s)", fontsize=17)
        if not month_encoded or idx % 4 == 0: ax.set_ylabel("Storage (Mm³)", fontsize=17)
    sm = plt.cm.ScalarMappable(cmap="RdYlGn", norm=plt.Normalize(vmin, vmax)); sm.set_array([])
    cbar = fig.colorbar(sm, ax=axf if month_encoded else axf[0], location="right", shrink=0.8, pad=0.02)
    cbar.set_label("Recovered Reward  g(s,a)", fontsize=20, fontweight="bold")
    fig.suptitle("AIRL Recovered Reward Contours", fontsize=26, fontweight="bold", y=0.997)
    plt.savefig(figdir / "reward_contours.png", dpi=300, bbox_inches="tight"); plt.close()


# =============================================================================
# Orchestrator
# =============================================================================

def run_generate_results(*, reservoir, res_cfg, res_cfg_path, algo_cfg, data, device_str, run_id,
                         n_mc=200, shap_n_background=100, shap_nsamples=100, shap_max_explain=300,
                         shap_split="all", grid_size=120, contour_max_inflows=80, seed=None) -> dict:
    base = _ROOT / "results" / reservoir / "airl"
    folder = _find_run_folder(base, run_id)
    agent_path = folder / "airl_agent.pt"
    cfg_path = folder / "airl_best_config.json"
    if not agent_path.exists():
        sys.exit(f"\nERROR: {agent_path} not found. Run the AIRL stage for run {run_id} first.\n")

    agent = AIRLAgent.from_checkpoint(agent_path, device_str)
    best_cfg = json.loads(cfg_path.read_text())
    mb = MassBalanceConfig(**best_cfg["mass_balance"])
    if seed is None:
        seed = int(best_cfg.get("seed", 42))
    rng = np.random.default_rng(seed)

    state_cols: List[str] = list(data.state_cols)
    month_encoded = ("sin_month" in state_cols) and ("cos_month" in state_cols)
    storage_col, inflow_col, action_col = mb.storage_col, mb.inflow_col, mb.action_col
    norm_bounds = {storage_col: _bounds_pair(data.bounds, storage_col),
                   inflow_col: _bounds_pair(data.bounds, inflow_col),
                   action_col: _bounds_pair(data.bounds, action_col)}

    all_states, all_actions, all_raw, all_dates = _split_arrays(data, "all")
    all_months = _months_of(all_dates)
    D = all_states.shape[1]
    disp = _display_names(state_cols, storage_col, inflow_col)

    figdir = folder / "figures"; figdir.mkdir(parents=True, exist_ok=True)
    print(f"\nGenerating AIRL results for {reservoir} run {run_id} → {figdir}")
    saved: Dict[str, list] = {}

    # ---- 1. Monte-Carlo fans: test + full ----
    env_test = ReservoirRollout(data.test, state_cols, mb, norm_bounds, device_str)
    saved["mc_test"] = _mc_fans(agent, env_test, figdir, n_mc=n_mc, seed=seed, suffix="test", scope_title="Test data")
    full = SimpleNamespace(states=all_states, raw_actions=all_raw)
    env_full = ReservoirRollout(full, state_cols, mb, norm_bounds, device_str)
    saved["mc_full"] = _mc_fans(agent, env_full, figdir, n_mc=n_mc, seed=seed, suffix="full", scope_title="All data")

    # ---- 2. Recovered-reward contours ----
    _reward_contours(agent, obs_states=all_states, raw_actions=all_raw, state_cols=state_cols,
                     storage_col=storage_col, inflow_col=inflow_col, norm_bounds=norm_bounds,
                     month_encoded=month_encoded, months=all_months, figdir=figdir,
                     grid_size=grid_size, max_inflows=contour_max_inflows, rng=rng)
    print("  saved reward_contours.png")

    # ---- 3+4. SHAP — policy (state features) and reward g(s,a) (state + release) ----
    shap_states, shap_actions, _, shap_dates = _split_arrays(data, shap_split)
    shap_months = _months_of(shap_dates)

    @torch.no_grad()
    def policy_f(Xnp):
        s = torch.from_numpy(np.ascontiguousarray(np.asarray(Xnp, np.float32))).to(agent.device)
        return agent.select_action(s, deterministic=True).cpu().numpy()

    @torch.no_grad()
    def reward_f(Xnp):
        Xnp = np.asarray(Xnp, np.float32)
        s = torch.from_numpy(np.ascontiguousarray(Xnp[:, :D])).to(agent.device)
        a = torch.from_numpy(np.ascontiguousarray(Xnp[:, D])).to(agent.device)
        return agent.discriminator.extract_reward(s, a).cpu().numpy().ravel()

    try:
        _shap_block(network_label="Policy Network", f=policy_f, X=shap_states, feature_names=disp,
                    months=shap_months, month_encoded=month_encoded, figdir=figdir, stem="shap_policy",
                    n_background=shap_n_background, nsamples=shap_nsamples, max_explain=shap_max_explain, rng=rng)
        X_reward = np.hstack([shap_states, shap_actions.reshape(-1, 1)]).astype(np.float32)
        _shap_block(network_label="Reward g(s,a)", f=reward_f, X=X_reward, feature_names=disp + ["release"],
                    months=shap_months, month_encoded=month_encoded, figdir=figdir, stem="shap_reward",
                    n_background=shap_n_background, nsamples=shap_nsamples, max_explain=shap_max_explain, rng=rng)
        print("  saved shap_policy_* and shap_reward_*")
    except ImportError:
        print("  WARNING: `shap` not installed — skipping SHAP (pip install shap).")

    print(f"\nDone. Figures in {figdir}")
    return {"run_folder": folder, "figures_dir": figdir}


def _parse_args():
    p = argparse.ArgumentParser(description="Generate AIRL result figures.")
    p.add_argument("--reservoir", required=True)
    p.add_argument("--run_id", type=int, default=None)
    p.add_argument("--device", default="cpu")
    p.add_argument("--n_mc", type=int, default=200)
    p.add_argument("--shap_n_background", type=int, default=100)
    p.add_argument("--shap_nsamples", type=int, default=100)
    p.add_argument("--shap_max_explain", type=int, default=300)
    p.add_argument("--shap_split", choices=["train", "val", "test", "all"], default="all")
    p.add_argument("--grid_size", type=int, default=120)
    p.add_argument("--contour_max_inflows", type=int, default=80)
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()


def main():
    import yaml
    a = _parse_args()
    res_cfg_path = _ROOT / "configs" / "reservoirs" / f"{a.reservoir}.yaml"
    algo_cfg_path = _ROOT / "configs" / "algorithms" / "airl.yaml"
    res_cfg = yaml.safe_load(open(res_cfg_path))
    algo_cfg = yaml.safe_load(open(algo_cfg_path)) if algo_cfg_path.exists() else {}
    device_str = _resolve_device(a.device)
    base = _ROOT / "results" / a.reservoir / "airl"
    run_id = a.run_id if a.run_id is not None else max(
        (int(d.name) for d in base.iterdir() if d.is_dir() and _RUN_FOLDER_PATTERN.match(d.name)), default=None)
    if run_id is None:
        sys.exit(f"No run found under {base}")
    data = load_reservoir_data(res_cfg, res_cfg_path)
    run_generate_results(reservoir=a.reservoir, res_cfg=res_cfg, res_cfg_path=res_cfg_path, algo_cfg=algo_cfg,
                         data=data, device_str=device_str, run_id=run_id, n_mc=a.n_mc,
                         shap_n_background=a.shap_n_background, shap_nsamples=a.shap_nsamples,
                         shap_max_explain=a.shap_max_explain, shap_split=a.shap_split, grid_size=a.grid_size,
                         contour_max_inflows=a.contour_max_inflows, seed=a.seed)


if __name__ == "__main__":
    main()
