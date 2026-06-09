"""
policy.py for AIRL and IQ-Learn
=========
The Behavioral Cloning policy network: a quantile-binned/logartihmic-binned categorical head.

A plain MLP backbone maps the state to K raw logits, one per release bin.  The
output is UN-normalised on purpose — the softmax is applied downstream:
    - training : _compute_loss() uses log_softmax(logits)
    - decode   : the expected value uses softmax(logits) @ bin_means
Applying a softmax inside forward() would double-count it, so forward returns
logits directly.

Contract (used by tune.py / iqlearn)
-----------------------------------
    model = build_policy_network(config).to(device)
    logits = model(states)        # states: (B, state_dim)  ->  logits: (B, n_bins)

The architecture is fully determined by `config`, so a saved state_dict can be
reloaded by rebuilding with the same config and calling load_state_dict — which
is what the IQLearn warm-start relies on.  Because the output layer width is
n_bins, a checkpoint is K-specific: it only loads into a network built with the
same n_bins (guaranteed by the frozen n_bins in bc_best_config.json).
"""

from __future__ import annotations

import torch
import torch.nn as nn


# =============================================================================
# Categorical policy network
# =============================================================================

class CategoricalPolicy(nn.Module):
    """
    MLP backbone + linear head producing K bin logits.

    Structure
    ---------
    state_dim
        -> [Linear -> ReLU -> Dropout] x n_hidden_layers   (width = hidden_dim)
        -> Linear(hidden_dim -> n_bins)                     (raw logits)

    Parameters
    ----------
    state_dim       : input feature dimension (storage, inflow, sin/cos month, ...).
    n_bins          : K — number of release bins == output-layer width.
    hidden_dim      : hidden-layer width.
    n_hidden_layers : number of hidden layers (>= 1).
    dropout         : dropout probability applied after each hidden activation
                      (0.0 -> identity; automatically disabled under .eval()).
    """

    def __init__(
        self,
        state_dim:       int,
        n_bins:          int,
        hidden_dim:      int   = 128,
        n_hidden_layers: int   = 3,
        dropout:         float = 0.1,
    ):
        super().__init__()
        if state_dim < 1:
            raise ValueError(f"state_dim must be >= 1, got {state_dim}.")
        if n_bins < 1:
            raise ValueError(f"n_bins must be >= 1, got {n_bins}.")
        if n_hidden_layers < 1:
            raise ValueError(f"n_hidden_layers must be >= 1, got {n_hidden_layers}.")

        layers: list[nn.Module] = []
        in_dim = state_dim
        for _ in range(n_hidden_layers):
            layers += [nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, n_bins))   # output head: raw logits

        self.net    = nn.Sequential(*layers)
        self.n_bins = n_bins

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        states : (B, state_dim) float tensor.

        Returns
        -------
        (B, n_bins) raw logits (NOT softmaxed).
        """
        return self.net(states)


# =============================================================================
# Factory
# =============================================================================

def build_policy_network(config) -> CategoricalPolicy:
    """
    Construct the categorical policy from a BCConfig.

    The returned module is on CPU; the caller moves it to the target device
    (tune.py does build_policy_network(config).to(device)).

    Parameters
    ----------
    config : BCConfig
        Provides state_dim, n_bins, hidden_dim, n_hidden_layers, dropout.

    Returns
    -------
    CategoricalPolicy
    """
    return CategoricalPolicy(
        state_dim       = config.state_dim,
        n_bins          = config.n_bins,
        hidden_dim      = config.hidden_dim,
        n_hidden_layers = config.n_hidden_layers,
        dropout         = config.dropout,
    )