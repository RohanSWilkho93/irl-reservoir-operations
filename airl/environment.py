"""
airl/environment.py
===================
A lightweight gym-style stepper for AIRL's PPO rollouts, built on the repo's
data convention (utils.data states: [storage, inflow, sin_month, cos_month],
train-normalized) so the BC-warm-started actor sees exactly what it trained on.

Episodes are calendar years: reset() starts at a year boundary; step() applies
the action, propagates storage by mass balance, and reads exogenous inflow/month
from the data row (only the storage slot of the state is simulated — same
invariant as iqlearn.environment.ReservoirRollout).

Also provides expert transitions (straight from the Split) and the two buffers
PPO/AIRL need.
"""

from __future__ import annotations

import random
from collections import deque
from typing import Dict, List

import numpy as np
import pandas as pd
import torch


# =============================================================================
# Buffers
# =============================================================================

class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, next_state, done=False):
        action = np.atleast_1d(np.asarray(action, dtype=np.float32))
        self.buffer.append((state.astype(np.float32), action,
                            next_state.astype(np.float32), float(done)))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        s, a, ns, d = zip(*batch)
        return np.array(s), np.array(a), np.array(ns), np.array(d)

    def __len__(self):
        return len(self.buffer)

    def clear(self):
        self.buffer.clear()


class RolloutBuffer:
    def __init__(self):
        self.clear()

    def push(self, state, action, reward, next_state, done, log_prob, value):
        self.states.append(state); self.actions.append(action); self.rewards.append(reward)
        self.next_states.append(next_state); self.dones.append(done)
        self.log_probs.append(log_prob); self.values.append(value)

    def get(self, device):
        f = lambda v: torch.as_tensor(np.array(v), dtype=torch.float32, device=device)
        return {"states": f(self.states), "actions": f(self.actions), "rewards": f(self.rewards),
                "next_states": f(self.next_states), "dones": f(self.dones),
                "log_probs": f(self.log_probs), "values": f(self.values)}

    def clear(self):
        self.states, self.actions, self.rewards = [], [], []
        self.next_states, self.dones, self.log_probs, self.values = [], [], [], []


# =============================================================================
# Expert transitions (straight from the Split)
# =============================================================================

def expert_transitions(split):
    """(states, actions(N,1), next_states, dones) for the expert buffer."""
    return (np.asarray(split.states, np.float32),
            np.asarray(split.actions, np.float32).reshape(-1, 1),
            np.asarray(split.next_states, np.float32),
            np.asarray(split.dones, np.float32))


# =============================================================================
# PPO environment
# =============================================================================

class AIRLEnv:
    def __init__(self, split, state_cols: List[str], mb, norm_bounds: Dict, device,
                 trajectory_length: int = 365):
        self.states = np.asarray(split.states, dtype=np.float32)       # (T, D), normalized
        self.T, self.D = self.states.shape
        self.dates = pd.DatetimeIndex(pd.to_datetime(split.dates))
        self.years = self.dates.year.to_numpy()
        self.traj_len = trajectory_length
        self.device = device
        self.mb = mb

        self.s_idx = state_cols.index(mb.storage_col)
        self.i_idx = state_cols.index(mb.inflow_col)
        self.s_lo, self.s_hi = norm_bounds[mb.storage_col]
        self.i_lo, self.i_hi = norm_bounds[mb.inflow_col]
        self.r_lo, self.r_hi = norm_bounds[mb.action_col]
        self.conv = mb.seconds_per_day / mb.volume_factor

        self.inflow_eng = self._denorm(self.states[:, self.i_idx], self.i_lo, self.i_hi)
        self.obs_storage_eng = self._denorm(self.states[:, self.s_idx], self.s_lo, self.s_hi)

        # year-start indices with at least 2 steps of room
        starts = [0] + [k for k in range(1, self.T) if self.years[k] != self.years[k - 1]]
        self.valid_starts = [k for k in starts if k < self.T - 1] or [0]

    @staticmethod
    def _denorm(z, lo, hi): return z * (hi - lo) + lo
    @staticmethod
    def _norm(x, lo, hi): return (x - lo) / (hi - lo) if hi > lo else 0.0

    def _state_at(self, idx, storage_eng):
        row = self.states[idx].copy()
        row[self.s_idx] = np.float32(self._norm(storage_eng, self.s_lo, self.s_hi))
        return row

    def reset(self, start_idx=None):
        self.cur = random.choice(self.valid_starts) if start_idx is None else min(start_idx, self.T - 2)
        self.steps = 0
        self.sim_storage = float(self.obs_storage_eng[self.cur])
        self.state = self._state_at(self.cur, self.sim_storage)
        return self.state

    def step(self, action_norm):
        a = float(np.clip(action_norm.flatten()[0] if isinstance(action_norm, np.ndarray) else action_norm, 0, 1))
        rel = float(np.clip(self._denorm(a, self.r_lo, self.r_hi), self.mb.min_release, self.mb.max_release))
        self.sim_storage = float(np.clip(self.sim_storage + (self.inflow_eng[self.cur] - rel) * self.conv,
                                         self.mb.min_storage, self.mb.max_storage))
        prev = self.cur
        self.cur += 1
        self.steps += 1
        done = (self.cur >= self.T - 1 or self.steps >= self.traj_len
                or self.years[self.cur] != self.years[prev])
        nxt = np.zeros(self.D, np.float32) if done else self._state_at(self.cur, self.sim_storage)
        self.state = nxt
        return nxt, 0.0, done, {"storage": self.sim_storage, "release": rel}
