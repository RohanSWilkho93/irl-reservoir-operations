"""
iqlearn/networks/policy.py
==========================
The parametric Behavioral-Cloning / IQ-Learn actor.

A shared MLP backbone maps the state to features; a family-specific set of heads
(supplied by a PolicyDistribution) maps the features to that family's
distribution parameters.  The policy is fully determined by `config` —
(state_dim, hidden_dim, n_hidden_layers, dropout, policy_family, dist_params) —
so a saved state_dict reloads by rebuilding with the same config (the IQ-Learn
warm-start relies on this).

The state already contains the optional sin/cos month encoding (utils/data.py
folds it in), so the actor takes ONLY the state — there is no separate context.

forward(states) returns the distribution's parameter dict; the strategy turns
those parameters into a sampled / deterministic action, the BC NLL, and the KL.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from iqlearn.distributions import make_distribution


class ParametricPolicy(nn.Module):
    """MLP backbone + a PolicyDistribution's parameter heads."""

    def __init__(self, state_dim: int, hidden_dim: int, n_hidden_layers: int,
                 dropout: float, distribution, action_dim: int = 1):
        super().__init__()
        if state_dim < 1:
            raise ValueError(f"state_dim must be >= 1, got {state_dim}.")
        if n_hidden_layers < 1:
            raise ValueError(f"n_hidden_layers must be >= 1, got {n_hidden_layers}.")

        layers: list[nn.Module] = []
        in_dim = state_dim
        for _ in range(n_hidden_layers):
            layers += [nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = hidden_dim
        self.encoder = nn.Sequential(*layers)

        self.distribution = distribution                       # math only (no params)
        self.heads = distribution.make_heads(hidden_dim, action_dim)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for m in self.encoder:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        self.distribution.init_heads(self.heads)

    def forward(self, states: torch.Tensor) -> dict:
        """states (B, state_dim) -> distribution parameter dict (each (B, action_dim))."""
        features = self.encoder(states)
        return self.distribution.params_from_features(features, self.heads)


def build_policy_network(config) -> ParametricPolicy:
    """
    Construct the actor from a BC/IQ config.

    Reads (duck-typed): state_dim, hidden_dim, n_hidden_layers, dropout,
    policy_family, dist_params, action_dim (optional, default 1).
    """
    distribution = make_distribution(config.policy_family,
                                     getattr(config, "dist_params", {}) or {})
    return ParametricPolicy(
        state_dim       = config.state_dim,
        hidden_dim      = config.hidden_dim,
        n_hidden_layers = config.n_hidden_layers,
        dropout         = getattr(config, "dropout", 0.0),
        distribution    = distribution,
        action_dim      = getattr(config, "action_dim", 1),
    )
