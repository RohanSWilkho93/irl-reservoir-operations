"""
airl/networks.py
================
The AIRL-specific networks (the actor is the shared iqlearn ParametricPolicy):

  * CriticNetwork    — V(s) value function for PPO/GAE.
  * RewardNetwork    — g(s, a), spectral-normalised  (the RECOVERED reward).
  * ShapingNetwork   — h(s), potential-based shaping.
  * AIRLDiscriminator — D = sigmoid(f - log pi),  f = g(s,a) + gamma*h(s') - h(s).

The discriminator holds the (shared) policy + its PolicyDistribution so it can
read log pi(a|s) for any family. `extract_reward(s, a)` returns g(s,a) — what the
reward contour and reward-SHAP explain.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


class CriticNetwork(nn.Module):
    """Value function V(s) for PPO."""
    def __init__(self, state_dim: int, hidden_dim: int = 256, n_hidden_layers: int = 3):
        super().__init__()
        layers = [nn.Linear(state_dim, hidden_dim), nn.ReLU()]
        for _ in range(n_hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, state):
        return self.net(state)


class RewardNetwork(nn.Module):
    """Recovered reward g(s, a) with spectral normalization (the AIRL reward)."""
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256, n_hidden_layers: int = 3):
        super().__init__()
        layers = [spectral_norm(nn.Linear(state_dim + action_dim, hidden_dim)), nn.ReLU(), nn.Dropout(0.1)]
        for _ in range(max(0, n_hidden_layers - 2)):
            layers += [spectral_norm(nn.Linear(hidden_dim, hidden_dim)), nn.ReLU(), nn.Dropout(0.1)]
        layers += [spectral_norm(nn.Linear(hidden_dim, hidden_dim)), nn.ReLU(),
                   spectral_norm(nn.Linear(hidden_dim, 1))]
        self.net = nn.Sequential(*layers)

    def forward(self, state, action):
        if action.dim() == 1:
            action = action.unsqueeze(-1)
        return self.net(torch.cat([state, action], dim=-1))


class ShapingNetwork(nn.Module):
    """Potential-based shaping h(s)."""
    def __init__(self, state_dim: int, hidden_dim: int = 256, n_hidden_layers: int = 3):
        super().__init__()
        layers = [nn.Linear(state_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.2)]
        for _ in range(max(0, n_hidden_layers - 2)):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.2)]
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, state):
        return self.net(state)


class AIRLDiscriminator(nn.Module):
    """D(s,a,s') = sigmoid(f - log pi),  f = g(s,a) + gamma*h(s') - h(s)."""
    def __init__(self, state_dim, action_dim, policy, distribution,
                 hidden_dim=256, n_hidden_layers=3, gamma=0.99):
        super().__init__()
        self.gamma = gamma
        self.policy = policy                 # shared ParametricPolicy (not owned/saved here)
        self.distribution = distribution
        self.reward_net = RewardNetwork(state_dim, action_dim, hidden_dim, n_hidden_layers)
        self.shaping_net = ShapingNetwork(state_dim, hidden_dim, n_hidden_layers)

    def compute_f(self, s, a, ns):
        return torch.clamp(self.reward_net(s, a) + self.gamma * self.shaping_net(ns) - self.shaping_net(s), -20, 20)

    def _log_pi(self, s, a):
        params = self.policy(s)
        return torch.clamp(self.distribution.log_prob(params, a), -20, 2).unsqueeze(-1)

    def forward(self, s, a, ns):
        with torch.no_grad():
            log_pi = self._log_pi(s, a)
        return torch.sigmoid(self.compute_f(s, a, ns) - log_pi)

    def get_reward(self, s, a, ns):
        """AIRL training reward = logit(D), clamped."""
        with torch.no_grad():
            D = torch.clamp(self.forward(s, a, ns), 0.01, 0.99)
            return torch.clamp(torch.logit(D), -10, 10)

    def extract_reward(self, s, a):
        """The recovered reward g(s,a) — used for contours and SHAP."""
        with torch.no_grad():
            return self.reward_net(s, a)
