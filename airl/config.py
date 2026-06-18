"""
airl/config.py
==============
Hyperparameters for one AIRL run.  The actor architecture and distribution
family are NOT here — they are inherited from the BC checkpoint (bc_policy.pt).
Only the AIRL-specific machinery (critic, discriminator, PPO, adversarial
schedule) is configured / tuned here.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict


@dataclass
class AIRLConfig:
    state_dim:  int = 4
    action_dim: int = 1

    # value critic (PPO) + discriminator architectures
    critic_hidden_dim:      int = 256
    critic_n_hidden_layers: int = 3
    disc_hidden_dim:        int = 256
    disc_n_hidden_layers:   int = 3

    # learning rates
    lr_policy:        float = 3e-5
    lr_critic:        float = 3e-4
    lr_discriminator: float = 1e-4

    # discriminator training
    disc_updates:            int   = 5
    warmup_disc_updates:     int   = 10
    gradient_penalty_coef:   float = 10.0
    label_smoothing_epsilon: float = 0.05

    # PPO
    gamma:        float = 0.99
    gae_lambda:   float = 0.95
    clip_epsilon: float = 0.2
    entropy_coef: float = 0.01
    ppo_epochs:   int   = 5

    # KL-to-BC anchor (uses the family's closed-form KL)
    kl_regularization_coef: float = 0.5

    # schedule
    batch_size:          int = 512
    expert_buffer_size:  int = 60000
    policy_buffer_size:  int = 120000
    warmup_iterations:   int = 50
    num_iterations:      int = 300
    steps_per_iteration: int = 2048
    early_stopping_patience: int = 50
    eval_interval:       int = 10
    max_grad_norm:       float = 0.5

    # provenance / runtime
    policy_family: str = "beta"
    seed:          int = 42
    device:        str = "cpu"

    def to_dict(self) -> Dict:
        return asdict(self)
