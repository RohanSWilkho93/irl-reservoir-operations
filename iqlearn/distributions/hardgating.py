"""
iqlearn/distributions/hardgating.py
===================================
Zero-inflated policy with a HARD gate: the gate is a 0/1 decision, so the policy
emits exact zeros.  Deterministic decode thresholds the gate probability at 0.5;
stochastic sampling draws the gate from a Bernoulli.  See _gating.py for the
shared head, NLL, and KL.
"""

from __future__ import annotations

import torch

from ._gating import GatingDistribution, _LOG_PROB_CLAMP


class HardGatingDistribution(GatingDistribution):
    name = "hardgating"

    def mean_action(self, params):
        gate_hard = (params["gate"] > 0.5).float()
        return (gate_hard * self._beta_mode(params)).squeeze(-1)

    def rsample(self, params, generator=None):
        gate_dist = torch.distributions.Bernoulli(probs=params["gate"])
        gate = gate_dist.sample()                                       # hard 0/1, non-differentiable
        beta_dist = self._beta_dist(params)
        cont = torch.clamp(beta_dist.rsample(), 0.01, 0.99)
        action = gate * cont
        log_prob = gate_dist.log_prob(gate) + gate * beta_dist.log_prob(cont)
        log_prob = torch.clamp(log_prob, *_LOG_PROB_CLAMP).sum(dim=-1)
        return action.squeeze(-1), log_prob
