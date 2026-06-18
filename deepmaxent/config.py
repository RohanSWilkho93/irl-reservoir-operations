"""
deepmaxent/config.py
====================
Hyperparameter container for one Deep MaxEnt IRL run.

Mirrors the Paper-1 `Config` dataclass; values are either fixed (evaluation
consistency) or sampled by Optuna during tuning (see configs/algorithms/
deepmaxent.yaml and tuning.py).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict


@dataclass
class DMConfig:
    """All hyperparameters needed to build + train one Deep MaxEnt IRL model."""
    seed: int = 123

    # Discretization (engineering units)
    storage_step: float = 5.0
    release_step: float = 50.0
    inflow_step:  float = 50.0

    # MDP
    gamma: float = 0.95
    tau:   float = 0.05          # soft-VI temperature (log-scale in search)

    # Reward network
    hidden_dim1: int = 128
    hidden_dim2: int = 512
    dropout:     float = 0.2

    # Training
    lr:           float = 1e-3
    n_iterations: int   = 300
    batch_size:   int   = 1000
    val_early_stop_patience: int = 150

    # Fixed (evaluation consistency)
    convergence_threshold: float = 0.01
    tolerance:             float = 1e-6
    n_mc_simulations:      int   = 50

    # Physics (mass balance): S' = clip(S + flow_to_volume_factor*(inflow - release))
    flow_to_volume_factor: float = 86400.0 / 1.0e6

    def to_dict(self) -> Dict:
        return asdict(self)
