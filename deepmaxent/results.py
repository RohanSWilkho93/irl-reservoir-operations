"""
deepmaxent/results.py
=====================
Figures for a trained Deep MaxEnt IRL run (auto-run by run.py; also standalone).

Reloads reward_net.pt + spaces + policy from results/<reservoir>/deepmaxent/<run_id>/
and produces, into figures/:

  1. mc_fan_test.png / mc_fan_full.png   — Monte-Carlo rollout fans (median + IQR)
     of storage and release (test split, and the full record).
  2. reward_maps.png                      — the learned REWARD STRUCTURE: 12 monthly
     storage×release reward contours with expert observations overlaid.
  3. shap_reward_overall.png              — reward-only SHAP, combined.
     shap_reward_monthly.png              — reward-only SHAP per month (drops the
     seasonal sin/cos rows), ONLY when use_month_encoding is true.

Everything is reward-network based — there is no separate policy/critic SHAP.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from deepmaxent.config import DMConfig
from deepmaxent.data import load_raw_reservoir_data
from deepmaxent.networks import RewardNet, FEATURE_NAMES
from deepmaxent.mdp import create_trajectories, get_state_idx
from deepmaxent.scoring import safe_pearsonr, compute_nrmse
from iqlearn.utils.runs import _resolve_device, _find_run_folder

_MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


# =============================================================================
# Monte-Carlo rollout fan (returns ALL sims so we can show median + IQR)
# =============================================================================

def _mc_fan(Pi, s_space, r_space, i_space, i_map, n_s_bins, fvf,
            trajs_d: List, trajs_raw: List, n_sims: int, seed: int) -> Dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    s_min, s_max = s_space[0], s_space[-1]
    sims_s, sims_r, obs_s, obs_r = [], [], [], []
    for traj_d, traj_raw in zip(trajs_d, trajs_raw):
        T = len(traj_d)
        months = [int(row[1]) for row in traj_raw]
        inflows = [row[3] for row in traj_raw]
        mc_s = np.zeros((n_sims, T)); mc_r = np.zeros((n_sims, T))
        for sim in range(n_sims):
            fi = i_map.get(traj_d[0][3], int(np.argmin(np.abs(i_space - traj_d[0][3]))))
            s_val = traj_raw[0][0]
            for t in range(T):
                m = months[t] - 1
                si = int(np.argmin(np.abs(s_space - s_val)))
                s_idx = get_state_idx(si, fi, n_s_bins)
                p = Pi[s_idx, m, :]; p = p / p.sum()           # guard float drift in choice()
                ri = rng.choice(len(r_space), p=p)
                rel = r_space[ri]
                mc_s[sim, t] = s_val
                mc_r[sim, t] = rel
                if t < T - 1:
                    s_val = np.clip(s_val + (inflows[t] - rel) * fvf, s_min, s_max)
                    fi = int(np.argmin(np.abs(i_space - inflows[t + 1])))
        sims_s.append(mc_s); sims_r.append(mc_r)
        obs_s.extend([row[0] for row in traj_raw]); obs_r.extend([row[2] for row in traj_raw])
    return {"sim_storage": np.concatenate(sims_s, axis=1), "sim_release": np.concatenate(sims_r, axis=1),
            "obs_storage": np.array(obs_s), "obs_release": np.array(obs_r)}


def _plot_fan(fan: Dict[str, np.ndarray], title: str, path: Path):
    fig, (a_s, a_r) = plt.subplots(2, 1, figsize=(16, 9), sharex=True)
    for ax, key, yl in [(a_s, "storage", "Storage (Mm³)"), (a_r, "release", "Release (m³/s)")]:
        sims = fan[f"sim_{key}"]; obs = fan[f"obs_{key}"]
        med = np.median(sims, axis=0); q25 = np.percentile(sims, 25, axis=0); q75 = np.percentile(sims, 75, axis=0)
        r, _ = safe_pearsonr(obs, med); nrmse = compute_nrmse(obs, med)
        t = np.arange(len(obs))
        ax.fill_between(t, q25, q75, color="lightcoral", alpha=0.6, label="25–75% IQR")
        ax.plot(t, obs, color="#1f5fbf", lw=1.0, label="Observed")
        ax.plot(t, med, color="#d62728", lw=1.6, ls="--", label="Median (MC)")
        ax.set_ylabel(yl); ax.set_title(f"{key.capitalize()} — r={r:.3f}  nRMSE={nrmse:.3f}", fontweight="bold")
        ax.legend(loc="upper right"); ax.grid(True, alpha=0.3)
    a_r.set_xlabel("Day"); fig.suptitle(title, fontsize=15, fontweight="bold")
    plt.tight_layout(); plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()


# =============================================================================
# Reward structure maps (12 monthly contours)
# =============================================================================

def _reward_maps(best_R, s_space, r_space, i_space, n_s_bins, train_df, trajs_raw_train, path: Path):
    monthly_inflow = train_df.groupby("month")["inflow"].mean()
    grids = []
    for m in range(12):
        fi = int(np.argmin(np.abs(i_space - monthly_inflow.get(m + 1, np.mean(i_space)))))
        g = np.array([best_R[get_state_idx(si, fi, n_s_bins), :, m] for si in range(len(s_space))])
        grids.append(g)
    vmin = min(g.min() for g in grids); vmax = max(g.max() for g in grids)

    by_month = {m: {"s": [], "r": []} for m in range(1, 13)}
    for traj in trajs_raw_train:
        for s_val, m_val, r_val, _ in traj:
            mi = int(m_val)
            if 1 <= mi <= 12:
                by_month[mi]["s"].append(s_val); by_month[mi]["r"].append(r_val)

    fig = plt.figure(figsize=(20, 15)); gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.3)
    R_mesh, S_mesh = np.meshgrid(r_space, s_space)
    for m in range(12):
        ax = fig.add_subplot(gs[m // 4, m % 4])
        ax.contourf(R_mesh, S_mesh, gaussian_filter(grids[m], sigma=1.5), levels=20,
                    cmap="RdYlGn", vmin=vmin, vmax=vmax, extend="both")
        if by_month[m + 1]["r"]:
            ax.scatter(by_month[m + 1]["r"], by_month[m + 1]["s"], c="steelblue", s=14, alpha=0.65,
                       edgecolors="white", linewidths=0.3, zorder=5)
        ax.set_xlabel("Release (m³/s)", fontsize=8); ax.set_ylabel("Storage (Mm³)", fontsize=8)
        ax.set_title(_MONTHS[m], fontsize=10, fontweight="bold"); ax.tick_params(labelsize=7)
    cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    sm = plt.cm.ScalarMappable(cmap="RdYlGn", norm=plt.Normalize(vmin=vmin, vmax=vmax)); sm.set_array([])
    plt.colorbar(sm, cax=cax).set_label("Learned Reward", fontsize=11, rotation=270, labelpad=20)
    fig.suptitle("Learned Reward Structure by Month (expert observations overlaid)",
                 fontsize=14, fontweight="bold", y=0.98)
    plt.savefig(path, dpi=150, bbox_inches="tight"); plt.close()


# =============================================================================
# Reward SHAP (combined + monthly)
# =============================================================================

def _reward_shap(r_net, feats_all: np.ndarray, months_all: np.ndarray,
                 use_month_encoding: bool, figdir: Path, seed: int,
                 n_bg: int = 100, nsamples: int = 100, max_explain: int = 300):
    import shap
    rng = np.random.default_rng(seed)
    dev = next(r_net.parameters()).device

    def f(X):
        with torch.no_grad():
            return r_net(torch.tensor(np.asarray(X, np.float32), device=dev)).cpu().numpy().ravel()

    def sub(n, k):
        return rng.choice(n, size=min(k, n), replace=False)

    # ---- combined ----
    n = feats_all.shape[0]
    ex = feats_all[sub(n, max_explain)]; bg = feats_all[sub(n, n_bg)]
    sv = np.abs(shap.KernelExplainer(f, bg).shap_values(ex, nsamples=nsamples, silent=True)).mean(0)
    order = np.argsort(sv)
    plt.figure(figsize=(10, 6)); plt.barh(range(len(sv)), sv[order], color="#2e8b57")
    plt.yticks(range(len(sv)), [FEATURE_NAMES[i] for i in order], fontsize=12)
    plt.xlabel("mean(|SHAP value|) on learned reward", fontsize=12)
    plt.title("Reward Network — Overall Feature Importance", fontsize=14, fontweight="bold")
    plt.tight_layout(); plt.savefig(figdir / "shap_reward_overall.png", dpi=300, bbox_inches="tight"); plt.close()
    print(f"  reward SHAP (combined): inflow={sv[FEATURE_NAMES.index('inflow')]:.3f} "
          f"storage={sv[FEATURE_NAMES.index('storage')]:.3f} release={sv[FEATURE_NAMES.index('release')]:.3f}")

    if not use_month_encoding:
        return
    # ---- monthly (drop sin/cos rows; renormalise per month) ----
    kept = ["storage", "release", "inflow"]; kept_idx = [FEATURE_NAMES.index(k) for k in kept]
    per_month = {}
    for m in range(1, 13):
        rows = np.where(months_all == m)[0]
        if len(rows) < 10:
            continue
        exm = feats_all[rows[sub(len(rows), max_explain)]]; bgm = feats_all[rows[sub(len(rows), n_bg)]]
        svm = np.abs(shap.KernelExplainer(f, bgm).shap_values(exm, nsamples=nsamples, silent=True)).mean(0)
        per_month[m] = svm[kept_idx]
    if not per_month:
        return
    present = sorted(per_month); data = np.zeros((len(kept), len(present)))
    for j, m in enumerate(present):
        v = per_month[m]; tot = v.sum(); data[:, j] = (v / tot * 100.0) if tot > 1e-12 else 0.0
    plt.figure(figsize=(max(13, 1.6 * len(present)), 4.5))
    im = plt.imshow(data, cmap="YlOrRd", aspect="auto", vmin=0.0)
    plt.xticks(range(len(present)), [_MONTHS[m - 1] for m in present], fontsize=16, fontweight="bold")
    plt.yticks(range(len(kept)), [k.capitalize() for k in kept], fontsize=16, fontweight="bold")
    hi = data.max() if data.size else 1.0
    for i in range(len(kept)):
        for j in range(len(present)):
            plt.text(j, i, f"{data[i, j]:.0f}", ha="center", va="center",
                     color="white" if data[i, j] > hi * 0.6 else "black", fontsize=14, fontweight="bold")
    plt.title("Reward Network — Monthly Feature Importance (%)", fontsize=14, fontweight="bold")
    plt.tight_layout(); plt.savefig(figdir / "shap_reward_monthly.png", dpi=300, bbox_inches="tight"); plt.close()
    print(f"  reward SHAP (monthly): {len(present)} months")


# =============================================================================
# Orchestrator
# =============================================================================

def run_generate_results(*, reservoir, res_cfg, res_cfg_path, algo_cfg, data, device_str,
                         run_id, n_mc=50, shap_n_background=100, shap_nsamples=100,
                         shap_max_explain=300, seed=None) -> dict:
    base = _ROOT / "results" / reservoir / "deepmaxent"
    folder = _find_run_folder(base, run_id)
    if not (folder / "reward_net.pt").exists():
        sys.exit(f"\nERROR: reward_net.pt not found in {folder}. Run tuning for run {run_id} first.\n")

    best = json.loads((folder / "best_config.json").read_text())
    cfg = DMConfig(**best["config"])
    if seed is None:
        seed = cfg.seed
    s_space = np.load(folder / "s_space.npy"); r_space = np.load(folder / "r_space.npy")
    i_space = np.load(folder / "i_space.npy")
    Pi = np.load(folder / "policy_Pi.npy"); best_R = np.load(folder / "reward_table_R.npy")
    n_s_bins = len(s_space)

    ck = torch.load(folder / "reward_net.pt", map_location=device_str, weights_only=False)
    r_net = RewardNet(cfg.hidden_dim1, cfg.hidden_dim2, cfg.dropout).to(device_str)
    r_net.load_state_dict(ck["state_dict"]); r_net.load_stats(ck["stats"]); r_net.eval()

    figdir = folder / "figures"; figdir.mkdir(parents=True, exist_ok=True)
    print(f"\nGenerating Deep MaxEnt results for {reservoir} run {run_id} → {figdir}")

    # trajectories per split (cheap; no transition matrix needed here)
    def trajs(df):
        return create_trajectories(df, s_space, r_space, i_space,
                                   cfg.storage_step, cfg.release_step, cfg.inflow_step)
    tr_d, tr_raw, s_map, r_map, i_map = trajs(data.train)
    te_d, te_raw, *_ = trajs(data.test)
    va_d, va_raw, *_ = trajs(data.val)
    full_d = tr_d + va_d + te_d; full_raw = tr_raw + va_raw + te_raw

    # ---- 1. Monte-Carlo fans (test + full) ----
    _plot_fan(_mc_fan(Pi, s_space, r_space, i_space, i_map, n_s_bins, cfg.flow_to_volume_factor,
                      te_d, te_raw, n_mc, seed), "Monte-Carlo Rollout — Test", figdir / "mc_fan_test.png")
    _plot_fan(_mc_fan(Pi, s_space, r_space, i_space, i_map, n_s_bins, cfg.flow_to_volume_factor,
                      full_d, full_raw, n_mc, seed), "Monte-Carlo Rollout — All data", figdir / "mc_fan_full.png")
    print("  saved mc_fan_test.png, mc_fan_full.png")

    # ---- 2. Reward structure maps ----
    _reward_maps(best_R, s_space, r_space, i_space, n_s_bins, data.train, tr_raw, figdir / "reward_maps.png")
    print("  saved reward_maps.png")

    # ---- 3. Reward SHAP (combined + monthly) ----
    s_vals, r_vals, m_vals, i_vals = [], [], [], []
    for traj in tr_raw:
        for s_val, m_val, r_val, i_val in traj:
            s_vals.append(s_val); r_vals.append(r_val); m_vals.append(int(m_val)); i_vals.append(i_val)
    feats_all = r_net.get_features(np.array(s_vals), np.array(r_vals), np.array(m_vals), np.array(i_vals))
    months_all = np.array(m_vals)
    try:
        _reward_shap(r_net, feats_all, months_all, data.use_month_encoding, figdir, seed,
                     shap_n_background, shap_nsamples, shap_max_explain)
        print("  saved shap_reward_overall.png" + (" + shap_reward_monthly.png" if data.use_month_encoding else ""))
    except ImportError:
        print("  WARNING: `shap` not installed — skipping reward SHAP (pip install shap).")

    return {"run_folder": folder, "figures_dir": figdir}


def _parse_args():
    import yaml  # noqa
    p = argparse.ArgumentParser(description="Generate Deep MaxEnt IRL result figures.")
    p.add_argument("--reservoir", required=True)
    p.add_argument("--run_id", type=int, default=None, help="default: latest")
    p.add_argument("--device", default="cpu")
    p.add_argument("--n_mc", type=int, default=50)
    p.add_argument("--shap_n_background", type=int, default=100)
    p.add_argument("--shap_nsamples", type=int, default=100)
    p.add_argument("--shap_max_explain", type=int, default=300)
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()


def main():
    import yaml
    a = _parse_args()
    res_cfg_path = _ROOT / "configs" / "reservoirs" / f"{a.reservoir}.yaml"
    algo_cfg_path = _ROOT / "configs" / "algorithms" / "deepmaxent.yaml"
    res_cfg = yaml.safe_load(open(res_cfg_path))
    algo_cfg = yaml.safe_load(open(algo_cfg_path)) if algo_cfg_path.exists() else {}
    device_str = _resolve_device(a.device)
    base = _ROOT / "results" / a.reservoir / "deepmaxent"
    run_id = a.run_id if a.run_id is not None else max(
        (int(d.name) for d in base.iterdir() if d.is_dir() and d.name.isdigit()), default=None)
    if run_id is None:
        sys.exit(f"No run found under {base}")
    data = load_raw_reservoir_data(res_cfg, res_cfg_path)
    run_generate_results(reservoir=a.reservoir, res_cfg=res_cfg, res_cfg_path=res_cfg_path,
                         algo_cfg=algo_cfg, data=data, device_str=device_str, run_id=run_id,
                         n_mc=a.n_mc, shap_n_background=a.shap_n_background, shap_nsamples=a.shap_nsamples,
                         shap_max_explain=a.shap_max_explain, seed=a.seed)


if __name__ == "__main__":
    main()
