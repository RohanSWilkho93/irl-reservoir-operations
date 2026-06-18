"""
iqlearn/distributions/beta.py
=============================
Beta policy over the normalised release [0, 1].

For reservoirs whose release is (almost) never exactly zero.  alpha/beta are
produced by softplus + lower-bound, clamped to [min, max] (the max bounds are
BC hyperparameters: alpha_max / beta_max).  The deterministic action is the Beta
mean alpha / (alpha + beta); the stochastic action is a reparameterised Beta
sample.  The BC-anchor KL is the closed-form Beta-Beta divergence (digamma/lgamma).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import PolicyDistribution

_LOG_PROB_CLAMP = (-20.0, 2.0)


def beta_kl(a1: torch.Tensor, b1: torch.Tensor,
            a2: torch.Tensor, b2: torch.Tensor) -> torch.Tensor:
    """Closed-form KL( Beta(a1,b1) || Beta(a2,b2) )  (elementwise)."""
    from torch.special import digamma
    log_b2 = torch.lgamma(a2) + torch.lgamma(b2) - torch.lgamma(a2 + b2)
    log_b1 = torch.lgamma(a1) + torch.lgamma(b1) - torch.lgamma(a1 + b1)
    return (log_b2 - log_b1
            + (a1 - a2) * digamma(a1)
            + (b1 - b2) * digamma(b1)
            + (a2 - a1 + b2 - b1) * digamma(a1 + b1))


class BetaDistribution(PolicyDistribution):
    name = "beta"

    def __init__(self, alpha_min: float = 1.0, alpha_max: float = 20.0,
                 beta_min: float = 1.0, beta_max: float = 20.0):
        super().__init__(alpha_min=alpha_min, alpha_max=alpha_max,
                         beta_min=beta_min, beta_max=beta_max)

    # ---- heads -------------------------------------------------------------

    def make_heads(self, hidden_dim: int, action_dim: int = 1) -> nn.ModuleDict:
        return nn.ModuleDict({
            "alpha": nn.Linear(hidden_dim, action_dim),
            "beta":  nn.Linear(hidden_dim, action_dim),
        })

    def init_heads(self, heads: nn.ModuleDict) -> None:
        # Bias toward Beta(2, 8) -> mean 0.2 (a calm, low-release prior).
        nn.init.zeros_(heads["alpha"].weight); nn.init.constant_(heads["alpha"].bias, 2.0)
        nn.init.zeros_(heads["beta"].weight);  nn.init.constant_(heads["beta"].bias, 8.0)

    def params_from_features(self, features, heads):
        amin, amax = self.hp["alpha_min"], self.hp["alpha_max"]
        bmin, bmax = self.hp["beta_min"], self.hp["beta_max"]
        alpha = torch.clamp(F.softplus(heads["alpha"](features)) + amin, amin, amax)
        beta  = torch.clamp(F.softplus(heads["beta"](features))  + bmin, bmin, bmax)
        return {"alpha": alpha, "beta": beta}

    # ---- distribution helpers ---------------------------------------------

    @staticmethod
    def _dist(params):
        return torch.distributions.Beta(params["alpha"], params["beta"])

    def mean_action(self, params):
        a, b = params["alpha"], params["beta"]
        return (a / (a + b)).squeeze(-1)

    def rsample(self, params, generator=None):
        dist = self._dist(params)
        action = torch.clamp(dist.rsample(), 1e-6, 1.0 - 1e-6)
        log_prob = torch.clamp(dist.log_prob(action), *_LOG_PROB_CLAMP).sum(dim=-1)
        return action.squeeze(-1), log_prob

    # ---- losses ------------------------------------------------------------

    def nll(self, params, expert_action):
        ea = torch.clamp(expert_action.view(-1, 1), 1e-6, 1.0 - 1e-6)
        return -self._dist(params).log_prob(ea).mean()

    def kl(self, params, ref_params):
        kl = beta_kl(params["alpha"], params["beta"],
                     ref_params["alpha"], ref_params["beta"])
        return kl.clamp(min=0.0).sum(dim=-1)

    def log_prob(self, params, action):
        a = torch.clamp(action.view(*params["alpha"].shape), 1e-6, 1.0 - 1e-6)
        return torch.clamp(self._dist(params).log_prob(a), -20, 2).sum(dim=-1)

    def entropy(self, params):
        return self._dist(params).entropy().sum(dim=-1)
