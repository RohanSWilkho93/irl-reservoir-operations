"""
networks/reward_deepmax.py
==========================
RewardNet for Deep MaxEnt IRL — a pure MLP reward function approximator.

The network is intentionally data-agnostic: it takes a pre-built, pre-normalised
feature tensor and returns a scalar reward.  All feature engineering (z-scoring
state variables, sin/cos encoding of month, assembling the input matrix) lives
in ``deepmaxent/core.py :: MaxEntTrainer``, not here.

Architecture
------------
Two hidden layers — both width and dropout are hyperparameters tuned per
reservoir by ``deepmaxent/tune.py``:

    Linear(n_inputs → h1) → ReLU → LayerNorm(h1) → Dropout(p)
    → Linear(h1 → h2) → ReLU → LayerNorm(h2) → Dropout(p)
    → Linear(h2 → 1)

Input dimension
---------------
``n_inputs`` is computed by the caller from the reservoir config:

    n_inputs = len(state_variables) + 1            # state vars + action (release)
             + (2 if use_month_encoding else 0)    # optional sin/cos month

Examples:
  state=[storage, net_inflow], use_month_encoding=True  → n_inputs = 5
  state=[storage, net_inflow], use_month_encoding=False → n_inputs = 3
  state=[storage, net_inflow, temperature], use_month_encoding=True  → n_inputs = 6

Tunable hyperparameters (all three are searched by deepmaxent/tune.py)
-----------------------------------------------------------------------
  h1       : neurons in the first hidden layer
  h2       : neurons in the second hidden layer
  dropout  : dropout probability applied after each hidden layer
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Physical constant: converts (m³/s × day) → Mm³
# ---------------------------------------------------------------------------
FLOW_TO_VOLUME: float = 86_400 / 1e6


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def to_torch(arr: np.ndarray, device: torch.device) -> torch.Tensor:
    """Convert a NumPy array to a contiguous float32 tensor on *device*."""
    return torch.tensor(
        np.ascontiguousarray(arr, dtype=np.float32),
        dtype=torch.float32,
        device=device,
    )


# ---------------------------------------------------------------------------
# RewardNet
# ---------------------------------------------------------------------------

class RewardNet(nn.Module):
    """
    Pure MLP scalar reward function for Deep MaxEnt IRL.

    Parameters
    ----------
    n_inputs : int
        Number of input features.  Computed from the reservoir config by the
        caller — see module docstring for the formula.
    h1, h2 : int
        Neurons in the first and second hidden layers (hyperparameters).
    dropout : float
        Dropout probability after each hidden layer (hyperparameter).
    """

    def __init__(
        self,
        n_inputs: int,
        h1: int,
        h2: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.n_inputs = n_inputs

        self.net = nn.Sequential(
            nn.Linear(n_inputs, h1), nn.ReLU(), nn.LayerNorm(h1), nn.Dropout(dropout),
            nn.Linear(h1, h2),       nn.ReLU(), nn.LayerNorm(h2), nn.Dropout(dropout),
            nn.Linear(h2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor of shape (N, n_inputs)
            Pre-normalised feature matrix assembled by MaxEntTrainer.

        Returns
        -------
        torch.Tensor of shape (N, 1)  — raw (un-bounded) scalar rewards.
        """
        return self.net(x)
