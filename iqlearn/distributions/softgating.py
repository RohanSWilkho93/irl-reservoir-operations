"""
iqlearn/distributions/softgating.py
===================================
Zero-inflated policy with a SOFT gate: the gate *probability* multiplies the Beta
amount continuously (action = gate_prob * amount), so releases shrink smoothly
toward zero without a hard cut, and gradients flow through the gate.  Shares the
two-stage head, zero-inflated NLL, and gate+Beta KL from _gating.py.
"""

from __future__ import annotations

import torch

from ._gating import GatingDistribution, _LOG_PROB_CLAMP


class SoftGatingDistribution(GatingDistribution):
    name = "softgating"

    def mean_action(self, params):
        return (params["gate"] * self._beta_mode(params)).squeeze(-1)

    def rsample(self, params, generator=None):
        beta_dist = self._beta_dist(params)
        cont = torch.clamp(beta_dist.rsample(), 0.01, 0.99)
        action = params["gate"] * cont                      # soft, continuous weighting
        # The gate is a deterministic scale here; the stochastic density is the Beta part.
        log_prob = torch.clamp(beta_dist.log_prob(cont), *_LOG_PROB_CLAMP).sum(dim=-1)
        return action.squeeze(-1), log_prob
