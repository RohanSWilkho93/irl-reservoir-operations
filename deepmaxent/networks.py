"""
deepmaxent/networks.py
======================
The reward network for Deep MaxEnt IRL.

R(s, a) is a small MLP over 5 features built from the (engineering-unit)
state-action-month tuple:

    [ (storage-s_m)/s_s, (release-r_m)/r_s, sin(2pi*month/12), cos(2pi*month/12),
      (inflow-i_m)/i_s ]  ->  scalar reward

The per-feature mean/std (`stats`) are computed once from the training
trajectories and frozen; they are serialised with the weights so the reward can
be re-queried (and SHAP-explained) after training.
"""

from __future__ import annotations

from typing import List

import numpy as np
import torch
import torch.nn as nn


class RewardNet(nn.Module):
    def __init__(self, h1: int, h2: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, h1), nn.ReLU(), nn.LayerNorm(h1), nn.Dropout(dropout),
            nn.Linear(h1, h2), nn.ReLU(), nn.LayerNorm(h2), nn.Dropout(dropout),
            nn.Linear(h2, 1),
        )
        self.stats: dict = {}

    # ---- feature normalization stats (frozen after fit) -------------------

    def set_stats(self, trajs: List) -> None:
        s, r, i = [], [], []
        for t in trajs:
            for row in t:
                s.append(row[0]); r.append(row[2]); i.append(row[3])
        self.stats = {
            "s_m": float(np.mean(s)), "s_s": float(np.std(s) + 1e-6),
            "r_m": float(np.mean(r)), "r_s": float(np.std(r) + 1e-6),
            "i_m": float(np.mean(i)), "i_s": float(np.std(i) + 1e-6),
        }

    def load_stats(self, stats: dict) -> None:
        self.stats = dict(stats)

    # ---- forward + feature map --------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def get_features(self, s, r, m, i) -> np.ndarray:
        sn = (s - self.stats["s_m"]) / self.stats["s_s"]
        rn = (r - self.stats["r_m"]) / self.stats["r_s"]
        in_ = (i - self.stats["i_m"]) / self.stats["i_s"]
        th = np.asarray(m) / 12.0 * 2 * np.pi
        return np.column_stack([sn, rn, np.sin(th), np.cos(th), in_]).astype(np.float32)


# Human-readable feature names (column order of get_features).
FEATURE_NAMES = ["storage", "release", "sin_month", "cos_month", "inflow"]
