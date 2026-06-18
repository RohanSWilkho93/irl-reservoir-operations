"""
iqlearn/distributions/lognormal.py
==================================
Log-space Gaussian policy over the normalised release.

mu is unbounded; sigma = softplus(head) + sigma_min.  The action is recovered
from a log-space sample as  exp(log_action) - log_epsilon  (clamped >= 0).  The
BC loss is the Gaussian NLL in log-space (expert release log-transformed with
the same epsilon), and the BC-anchor KL is the closed-form Gaussian KL (the
log-space Gaussians fully determine the LogNormals).

For reservoirs whose release is continuous and (almost) never exactly zero;
the heavy right-skew of release is well captured in log-space.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import PolicyDistribution

_LOG_PROB_CLAMP = (-20.0, 2.0)


class LogNormalDistribution(PolicyDistribution):
    name = "lognormal"

    def __init__(self, sigma_min: float = 0.1, log_epsilon: float = 1.0):
        super().__init__(sigma_min=sigma_min, log_epsilon=log_epsilon)

    # ---- heads -------------------------------------------------------------

    def make_heads(self, hidden_dim: int, action_dim: int = 1) -> nn.ModuleDict:
        return nn.ModuleDict({
            "mu":    nn.Linear(hidden_dim, action_dim),
            "sigma": nn.Linear(hidden_dim, action_dim),
        })

    def init_heads(self, heads: nn.ModuleDict) -> None:
        # mu biased to log(0.2 + epsilon): a calm, low-release prior in log-space.
        import math
        nn.init.zeros_(heads["mu"].weight)
        nn.init.constant_(heads["mu"].bias, float(math.log(0.2 + self.hp["log_epsilon"])))
        nn.init.zeros_(heads["sigma"].weight); nn.init.zeros_(heads["sigma"].bias)

    def params_from_features(self, features, heads):
        mu = heads["mu"](features)
        sigma = F.softplus(heads["sigma"](features)) + self.hp["sigma_min"]
        return {"mu": mu, "sigma": sigma}

    # ---- distribution helpers ---------------------------------------------

    @staticmethod
    def _dist(params):
        return torch.distributions.Normal(params["mu"], params["sigma"])

    def mean_action(self, params):
        action = torch.exp(params["mu"]) - self.hp["log_epsilon"]
        return torch.clamp(action, min=0.0).squeeze(-1)

    def rsample(self, params, generator=None):
        dist = self._dist(params)
        log_action = dist.rsample()                                   # (B,1) in log-space
        log_prob_normal = dist.log_prob(log_action).sum(dim=-1, keepdim=True)
        action = torch.clamp(torch.exp(log_action) - self.hp["log_epsilon"], min=1e-6)
        # change-of-variables: log p(a) = log p(log_a) - log_a  (d log_a / d a = 1/(a+eps))
        jacobian = log_action.sum(dim=-1, keepdim=True)
        log_prob = torch.clamp(log_prob_normal - jacobian, *_LOG_PROB_CLAMP)
        return action.squeeze(-1), log_prob.squeeze(-1)

    # ---- losses ------------------------------------------------------------

    def nll(self, params, expert_action):
        log_ea = torch.log(expert_action.view(-1, 1) + self.hp["log_epsilon"])
        return -self._dist(params).log_prob(log_ea).mean()

    def kl(self, params, ref_params):
        mu1, s1 = params["mu"], params["sigma"]
        mu2, s2 = ref_params["mu"], ref_params["sigma"]
        kl = torch.log(s2 / s1) + (s1 ** 2 + (mu1 - mu2) ** 2) / (2.0 * s2 ** 2) - 0.5
        return kl.clamp(min=0.0).sum(dim=-1)

    def log_prob(self, params, action):
        # change of variables for a = exp(l) - eps:  log p_A(a) = log p_L(l) - log(a + eps)
        log_action = torch.log(action.view(*params["mu"].shape) + self.hp["log_epsilon"])
        lp = self._dist(params).log_prob(log_action) - log_action
        return torch.clamp(lp, -20, 2).sum(dim=-1)

    def entropy(self, params):
        # entropy of the underlying log-space Gaussian (PPO bonus; constant Jacobian term dropped)
        return self._dist(params).entropy().sum(dim=-1)
