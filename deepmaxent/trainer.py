"""
deepmaxent/trainer.py
=====================
The Deep MaxEnt IRL trainer.

Loop per iteration:
  1. R = reward_net over the whole (state, action, month) grid,
  2. Pi = soft value iteration on R (entropy temperature tau, discount gamma,
     cyclic 12-month horizon) -> softmax policy,
  3. mu_L = expected state-visitation frequency of Pi (forward propagation),
  4. grad = mu_E - mu_L  (the MaxEnt gradient),
  5. step the reward net by loss = -(R * grad).

Early-stop on validation SVF L1 difference. The best reward net + policy are kept.

Closed-loop fidelity (Monte-Carlo) and the unified score reuse the same Pi.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.optim as optim

from deepmaxent.config import DMConfig
from deepmaxent.networks import RewardNet
from deepmaxent.mdp import get_state_idx, get_si_fi
from deepmaxent.scoring import (safe_pearsonr, compute_rmse, compute_nrmse,
                                compute_mae, svf_metrics, compute_unified_score)


def _to_torch(arr, device):
    return torch.tensor(np.ascontiguousarray(arr, dtype=np.float32), dtype=torch.float32, device=device)


class MaxEntTrainer:
    def __init__(self, cfg: DMConfig, P: np.ndarray, trajs: List,
                 s_space, r_space, i_space, s_map, r_map, i_map,
                 n_s_bins: int, device, verbose: bool = True):
        self.cfg = cfg
        self.P = P
        self.trajs = trajs
        self.s_space, self.r_space, self.i_space = s_space, r_space, i_space
        self.s_map, self.r_map, self.i_map = s_map, r_map, i_map
        self.n_s_bins = n_s_bins
        self.device = device
        self.verbose = verbose
        self.fvf = cfg.flow_to_volume_factor

        self.n_states = P.shape[0]
        self.n_actions = P.shape[2]

        self.r_net = RewardNet(cfg.hidden_dim1, cfg.hidden_dim2, cfg.dropout).to(device)
        self.r_net.set_stats(trajs)
        self.opt = optim.Adam(self.r_net.parameters(), lr=cfg.lr)
        self.sched = optim.lr_scheduler.ReduceLROnPlateau(self.opt, mode="min", factor=0.7, patience=15)

        self.mu_E = self._calc_expert_svf(trajs)

    # ---- state-visitation frequencies -------------------------------------

    def _calc_expert_svf(self, trajectories: List) -> np.ndarray:
        mu = np.zeros((self.n_states, 12, self.n_actions))
        for t in trajectories:
            for row in t:
                s_val, m_val, r_val, i_val = row
                if s_val not in self.s_map or r_val not in self.r_map or i_val not in self.i_map:
                    continue
                s_idx = get_state_idx(self.s_map[s_val], self.i_map[i_val], self.n_s_bins)
                m_idx = int(m_val) - 1
                if 0 <= m_idx < 12:
                    mu[s_idx, m_idx, self.r_map[r_val]] += 1.0
        return mu / max(1, len(trajectories))

    def _calc_learned_svf(self, Pi: np.ndarray, trajectories: List) -> np.ndarray:
        T = max((len(t) for t in trajectories), default=365)
        days_per_m = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        c_days = np.cumsum([0] + days_per_m)

        def get_month(t):
            d = t % 365
            for i in range(12):
                if c_days[i] <= d < c_days[i + 1]:
                    return i
            return 11

        mu = np.zeros((self.n_states, 12, self.n_actions))
        D = np.zeros(self.n_states)
        for t in trajectories:
            if not t:
                continue
            s0, _, _, f0 = t[0]
            if s0 in self.s_map and f0 in self.i_map:
                D[get_state_idx(self.s_map[s0], self.i_map[f0], self.n_s_bins)] += 1
        if D.sum() == 0:
            return mu
        D /= D.sum()
        curr_D = D.copy()
        for t in range(T):
            m = get_month(t)
            mu[:, m, :] += curr_D.reshape(-1, 1) * Pi[:, m, :]
            T_mat = np.einsum("sa,san->sn", Pi[:, m, :], self.P[:, m, :, :])
            curr_D = curr_D @ T_mat
        return mu

    # ---- reward grid + soft value iteration -------------------------------

    def _calc_rewards(self) -> np.ndarray:
        was_training = self.r_net.training
        self.r_net.eval()
        R = np.zeros((self.n_states, self.n_actions, 12))
        ri_grid, mi_grid = np.meshgrid(np.arange(self.n_actions), np.arange(12), indexing="ij")
        for s_idx in range(self.n_states):
            si, fi = get_si_fi(s_idx, self.n_s_bins)
            n_pts = ri_grid.size
            feats = self.r_net.get_features(
                np.full(n_pts, self.s_space[si]), self.r_space[ri_grid.flatten()],
                mi_grid.flatten(), np.full(n_pts, self.i_space[fi]))
            with torch.no_grad():
                R[s_idx] = self.r_net(_to_torch(feats, self.device)).cpu().numpy().reshape(self.n_actions, 12)
        if was_training:
            self.r_net.train()
        return R

    def _solve_mdp(self, R: np.ndarray) -> np.ndarray:
        V = np.zeros((self.n_states, 12))
        Q = np.zeros((self.n_states, 12, self.n_actions))
        for _ in range(100):
            V_prev = V.copy()
            for m in range(12):
                m_next = (m + 1) % 12
                Q[:, m, :] = R[:, :, m] + self.cfg.gamma * np.einsum("san,n->sa", self.P[:, m, :, :], V_prev[:, m_next])
            Q_scaled = Q / self.cfg.tau
            Q_max = np.max(Q_scaled, axis=2, keepdims=True)
            V = self.cfg.tau * (Q_max.squeeze(-1) + np.log(np.sum(np.exp(Q_scaled - Q_max), axis=2)))
            if np.max(np.abs(V - V_prev)) < self.cfg.tolerance:
                break
        Pi = np.zeros_like(Q)
        for m in range(12):
            Q_m = Q[:, m, :] / self.cfg.tau
            Q_max = np.max(Q_m, axis=1, keepdims=True)
            exp_Q = np.exp(Q_m - Q_max)
            Pi[:, m, :] = exp_Q / np.sum(exp_Q, axis=1, keepdims=True)
        return Pi

    # ---- evaluation -------------------------------------------------------

    def evaluate_svf(self, trajectories: List, Pi: np.ndarray = None) -> Tuple[float, float]:
        if Pi is None:
            Pi = self._solve_mdp(self._calc_rewards())
        mu_expert = self._calc_expert_svf(trajectories)
        mu_learned = self._calc_learned_svf(Pi, trajectories)
        diff, _, overlap_pct = svf_metrics(mu_expert, mu_learned)
        return diff, overlap_pct

    def monte_carlo_simulate(self, trajs_d: List, trajs_raw: List,
                             n_sims: int = 50, Pi: np.ndarray = None) -> Dict[str, np.ndarray]:
        if Pi is None:
            Pi = self._solve_mdp(self._calc_rewards())
        all_es, all_er, all_ss, all_sr = [], [], [], []
        s_min, s_max = self.s_space[0], self.s_space[-1]
        for traj_d, traj_raw in zip(trajs_d, trajs_raw):
            T = len(traj_d)
            expert_storage = [row[0] for row in traj_raw]
            expert_release = [row[2] for row in traj_raw]
            expert_months  = [int(row[1]) for row in traj_raw]
            expert_inflows = [row[3] for row in traj_raw]
            mc_storage = np.zeros((n_sims, T)); mc_release = np.zeros((n_sims, T))
            for sim in range(n_sims):
                i_val = traj_d[0][3]
                fi = self.i_map.get(i_val, int(np.argmin(np.abs(self.i_space - i_val))))
                s_val_sim = traj_raw[0][0]
                for t in range(T):
                    m = expert_months[t] - 1
                    si_curr = int(np.argmin(np.abs(self.s_space - s_val_sim)))
                    s_idx = get_state_idx(si_curr, fi, self.n_s_bins)
                    p = Pi[s_idx, m, :]; p = p / p.sum()       # guard float drift in choice()
                    ri = np.random.choice(len(self.r_space), p=p)
                    r_val = self.r_space[ri]
                    mc_storage[sim, t] = s_val_sim
                    mc_release[sim, t] = r_val
                    if t < T - 1:
                        s_val_sim = np.clip(s_val_sim + (expert_inflows[t] - r_val) * self.fvf, s_min, s_max)
                        fi = int(np.argmin(np.abs(self.i_space - expert_inflows[t + 1])))
            all_es.extend(expert_storage); all_er.extend(expert_release)
            all_ss.extend(mc_storage.mean(axis=0).tolist()); all_sr.extend(mc_release.mean(axis=0).tolist())
        return {"expert_storage": np.array(all_es), "expert_release": np.array(all_er),
                "sim_storage": np.array(all_ss), "sim_release": np.array(all_sr)}

    def evaluate_full(self, trajs_d, trajs_raw, svf_diff=None, Pi=None) -> Dict[str, Any]:
        res = self.monte_carlo_simulate(trajs_d, trajs_raw, self.cfg.n_mc_simulations, Pi)
        rc, _ = safe_pearsonr(res["expert_release"], res["sim_release"])
        sc, _ = safe_pearsonr(res["expert_storage"], res["sim_storage"])
        r_nrmse = compute_nrmse(res["expert_release"], res["sim_release"])
        s_nrmse = compute_nrmse(res["expert_storage"], res["sim_storage"])
        unified, comps = (None, None)
        if svf_diff is not None:
            unified, comps = compute_unified_score(svf_diff, rc, sc, r_nrmse, s_nrmse)
        return {"release_corr": rc, "storage_corr": sc,
                "release_rmse": compute_rmse(res["expert_release"], res["sim_release"]),
                "storage_rmse": compute_rmse(res["expert_storage"], res["sim_storage"]),
                "release_nrmse": r_nrmse, "storage_nrmse": s_nrmse,
                "release_mae": compute_mae(res["expert_release"], res["sim_release"]),
                "storage_mae": compute_mae(res["expert_storage"], res["sim_storage"]),
                "unified_score": unified, "unified_components": comps, "results": res}

    # ---- training ---------------------------------------------------------

    def train(self, val_trajs: List):
        best_R, best_Pi, best_state = None, None, None
        best_val = np.inf; best_epoch = 0; stall = 0
        history = []
        for epoch in range(self.cfg.n_iterations):
            R = self._calc_rewards()
            Pi = self._solve_mdp(R)
            mu_L = self._calc_learned_svf(Pi, self.trajs)
            grad = self.mu_E - mu_L
            train_diff, _, train_overlap = svf_metrics(self.mu_E, mu_L)
            val_diff, val_overlap = self.evaluate_svf(val_trajs, Pi)
            self.sched.step(val_diff)
            lr = self.opt.param_groups[0]["lr"]
            history.append({"epoch": epoch, "train_svf_diff": float(train_diff),
                            "train_overlap": float(train_overlap), "val_svf_diff": float(val_diff),
                            "val_overlap": float(val_overlap), "learning_rate": float(lr)})
            if self.verbose and epoch % 10 == 0:
                print(f"  Epoch {epoch:4d}: train_diff={train_diff:.2f}  val_diff={val_diff:.2f}  lr={lr:.2e}")

            if val_diff < best_val:
                best_val, best_R, best_Pi, best_epoch, stall = val_diff, R.copy(), Pi.copy(), epoch, 0
                best_state = {k: v.cpu().clone() for k, v in self.r_net.state_dict().items()}
            else:
                stall += 1
                if stall >= self.cfg.val_early_stop_patience:
                    if self.verbose:
                        print(f"  Early stopping at epoch {epoch} (best {best_epoch})")
                    break
            if train_diff < self.cfg.convergence_threshold:
                if self.verbose:
                    print(f"  Converged at epoch {epoch}")
                break

            # MaxEnt gradient step on the reward net.
            idxs = np.where(np.abs(grad) > 1e-6)
            if len(idxs[0]) == 0:
                continue
            s_idxs, m_idxs, a_idxs = idxs
            grad_vals = grad[s_idxs, m_idxs, a_idxs]
            s_vals, i_vals, r_vals, m_vals = [], [], [], []
            for k, s_idx in enumerate(s_idxs):
                si, fi = get_si_fi(s_idx, self.n_s_bins)
                s_vals.append(self.s_space[si]); i_vals.append(self.i_space[fi])
                r_vals.append(self.r_space[a_idxs[k]]); m_vals.append(m_idxs[k])
            feats = self.r_net.get_features(np.array(s_vals), np.array(r_vals), np.array(m_vals), np.array(i_vals))

            self.r_net.train()
            self.opt.zero_grad()
            bs = self.cfg.batch_size
            for k in range(0, len(feats), bs):
                x = _to_torch(feats[k:k + bs], self.device)
                g = _to_torch(grad_vals[k:k + bs], self.device)
                loss = -(self.r_net(x).squeeze(-1) * g).sum()
                loss.backward()
            torch.nn.utils.clip_grad_norm_(self.r_net.parameters(), max_norm=1.0)
            self.opt.step()

        if best_state is not None:
            self.r_net.load_state_dict(best_state); self.r_net.to(self.device)
        return best_R, best_Pi, best_epoch, history, best_state
