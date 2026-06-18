"""
iqlearn/distributions/_gating.py
================================
Shared machinery for the two zero-inflated families (HardGating / SoftGating).

A two-stage head models a release that is *exactly zero* on many days:
  Stage 1 — a Bernoulli "gate": P(release > 0) = sigmoid(gate_head).
  Stage 2 — a Beta over (0, 1) for the amount, conditional on the gate.

The BC loss is the proper zero-inflated negative log-likelihood
  zero day      ->  log(1 - gate)
  positive day  ->  log(gate) + log Beta_pdf(release)
and the BC-anchor KL is the Bernoulli gate KL plus the gate-weighted Beta KL.

HardGating vs SoftGating differ only in how the gate combines with the amount
(implemented in the subclasses):
  * HardGating — the gate is a hard 0/1 (threshold when deterministic, Bernoulli
    sample when stochastic), so the policy emits *exact* zeros.
  * SoftGating — the gate probability multiplies the amount continuously
    (action = gate_prob * amount), so releases shrink toward — but never hit —
    zero; gradients flow smoothly through the gate.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import PolicyDistribution
from .beta import beta_kl

_LOG_PROB_CLAMP = (-20.0, 2.0)
_KL_CLAMP = (0.0, 10.0)


class GatingDistribution(PolicyDistribution):
    """Base for the zero-inflated (gate x Beta) families."""

    def __init__(self, zero_threshold: float = 0.01,
                 alpha_min: float = 1.0, alpha_max: float = 20.0,
                 beta_min: float = 1.0, beta_max: float = 20.0,
                 gate_min: float = 0.01, gate_max: float = 0.99):
        super().__init__(zero_threshold=zero_threshold,
                         alpha_min=alpha_min, alpha_max=alpha_max,
                         beta_min=beta_min, beta_max=beta_max,
                         gate_min=gate_min, gate_max=gate_max)

    # ---- heads -------------------------------------------------------------

    def make_heads(self, hidden_dim: int, action_dim: int = 1) -> nn.ModuleDict:
        return nn.ModuleDict({
            "gate":  nn.Linear(hidden_dim, 1),
            "alpha": nn.Linear(hidden_dim, action_dim),
            "beta":  nn.Linear(hidden_dim, action_dim),
        })

    def init_heads(self, heads: nn.ModuleDict) -> None:
        nn.init.zeros_(heads["gate"].weight);  nn.init.constant_(heads["gate"].bias, -1.4)   # p~0.2
        nn.init.zeros_(heads["alpha"].weight); nn.init.constant_(heads["alpha"].bias, 2.0)
        nn.init.zeros_(heads["beta"].weight);  nn.init.constant_(heads["beta"].bias, 8.0)

    def params_from_features(self, features, heads):
        gmin, gmax = self.hp["gate_min"], self.hp["gate_max"]
        amin, amax = self.hp["alpha_min"], self.hp["alpha_max"]
        bmin, bmax = self.hp["beta_min"], self.hp["beta_max"]
        gate = torch.clamp(torch.sigmoid(heads["gate"](features)), gmin, gmax)
        alpha = torch.clamp(F.softplus(heads["alpha"](features)) + amin, amin, amax)
        beta  = torch.clamp(F.softplus(heads["beta"](features))  + bmin, bmin, bmax)
        return {"gate": gate, "alpha": alpha, "beta": beta}

    # ---- shared helpers ----------------------------------------------------

    @staticmethod
    def _beta_dist(params):
        return torch.distributions.Beta(params["alpha"], params["beta"])

    def _beta_mode(self, params):
        a, b = params["alpha"], params["beta"]
        return torch.clamp((a - 1.0) / (a + b - 2.0), 0.05, 0.99)

    # ---- losses (shared by both gating families) ---------------------------

    def nll(self, params, expert_action):
        zt = self.hp["zero_threshold"]
        ea = expert_action.view(-1, 1)
        gate = torch.clamp(params["gate"], 1e-6, 1.0 - 1e-6)
        is_zero = (ea < zt).float()
        ea_c = torch.clamp(ea, 1e-6, 1.0 - 1e-6)

        log_zero = torch.log(1.0 - gate)
        log_nonzero = torch.log(gate) + self._beta_dist(params).log_prob(ea_c)
        log_lik = is_zero * log_zero + (1.0 - is_zero) * log_nonzero
        return -log_lik.mean()

    def kl(self, params, ref_params):
        g, gr = params["gate"], ref_params["gate"]
        gate_kl = (g * torch.log((g + 1e-8) / (gr + 1e-8))
                   + (1.0 - g) * torch.log((1.0 - g + 1e-8) / (1.0 - gr + 1e-8)))
        gate_kl = gate_kl.clamp(*_KL_CLAMP)
        b_kl = beta_kl(params["alpha"], params["beta"],
                       ref_params["alpha"], ref_params["beta"]).clamp(*_KL_CLAMP)
        return (gate_kl + g * b_kl).sum(dim=-1)

    def log_prob(self, params, action):
        # zero-inflated density: log(1-gate) at ~zero release, else log(gate)+log Beta(a)
        gate = torch.clamp(params["gate"], 1e-6, 1.0 - 1e-6)
        a = action.view(*gate.shape)
        is_zero = (a < self.hp["zero_threshold"]).float()
        a_c = torch.clamp(a, 1e-6, 1.0 - 1e-6)
        lp = is_zero * torch.log(1.0 - gate) + (1.0 - is_zero) * (torch.log(gate) + self._beta_dist(params).log_prob(a_c))
        return torch.clamp(lp, -20, 2).sum(dim=-1)

    def entropy(self, params):
        # Bernoulli gate entropy + gate-weighted Beta entropy
        g = torch.clamp(params["gate"], 1e-6, 1.0 - 1e-6)
        gate_ent = -(g * torch.log(g) + (1.0 - g) * torch.log(1.0 - g))
        return (gate_ent + g * self._beta_dist(params).entropy()).sum(dim=-1)
