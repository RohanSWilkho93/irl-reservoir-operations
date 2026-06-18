"""
iqlearn/agent.py
================
IQ-Learn agent for a parametric policy: builds and owns the networks,
optimisers, and loss, and exposes the train step.

Construction (from a tuned BC checkpoint)
-----------------------------------------
  * actor      : rebuilt from the BC config inside bc_policy.pt (which carries
                 policy_family + dist_params) and warm-started with the BC
                 weights.  Put in eval() so its Dropout is OFF, but it still
                 trains (params keep requires_grad=True).
  * bc_policy  : a frozen clone from the same checkpoint — the KL anchor target.
  * critic     : TwinCritic from the IQ config (trained from scratch, tunable).
  * critic_tgt : frozen Polyak copy for stable bootstrap targets.

The actor's distribution family is inherited from BC, so IQ-Learn automatically
runs Beta / LogNormal / HardGating / SoftGating without any extra flag.

Training
--------
  update(batch, update_actor):
      critic step  ->  (optional) actor step  ->  Polyak target sync.
  update_actor=False is the critic warm-up phase (actor held at BC weights).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from types import SimpleNamespace
from typing import Any

import torch
from torch.nn.utils import clip_grad_norm_

from iqlearn.networks.policy import build_policy_network
from iqlearn.networks.critic import build_critic_network
from iqlearn.loss import IQLearnLoss
from iqlearn.expert_buffer import Batch
from iqlearn.distributions import make_distribution


POLICY_TYPE = "parametric"


# =============================================================================
# IQ config
# =============================================================================

@dataclass
class IQConfig:
    """
    All hyperparameters for one IQ-Learn run.

    The actor architecture + distribution family are NOT here — they are
    inherited from the BC checkpoint.  state_dim builds the critic and matches
    the warm-started actor.  policy_family is mirrored here only for logging /
    reproducibility (the authoritative copy lives in the BC config).
    """
    state_dim:  int
    action_dim: int = 1

    # critic architecture (trained from scratch — tunable)
    critic_hidden_dim:      int = 256
    critic_n_hidden_layers: int = 3

    # IQ-Learn hyperparameters
    gamma:            float = 0.99
    tau:              float = 0.005
    alpha_entropy:    float = 0.25
    alpha_reg:        float = 0.5
    lambda_bc:        float = 0.5
    lr_actor:         float = 3e-4
    lr_critic:        float = 3e-4
    n_action_samples: int   = 10     # Monte-Carlo samples for the soft value

    # training loop (used by iq_tuning; carried here for reproducibility)
    batch_size:            int = 256
    critic_warm_up_epochs: int = 100
    n_epochs:              int = 250
    seed:                  int = 42

    # LR schedule + early stopping
    scheduler_type:  str = "none"      # "cosine" | "plateau" | "none"
    warmup_patience: int = 20
    joint_patience:  int = 30

    # provenance (informational)
    policy_family: str = "beta"

    # runtime / numerical
    device:     str   = "cpu"
    q_clip:     float = 100.0
    q_reg_coef: float = 1e-3
    grad_clip:  float = 1.0


# =============================================================================
# Agent
# =============================================================================

class IQLearnAgent:
    """
    Parameters
    ----------
    iq_config : IQConfig
    bc_ckpt   : dict loaded from bc_policy.pt
                {"state_dict": ..., "config": <BCConfig asdict with policy_family
                 + dist_params>, "policy_type": ...}
    device    : target device.
    """

    def __init__(self, iq_config: IQConfig, bc_ckpt: dict, device: str | torch.device):
        self.config = iq_config
        self.device = torch.device(device)

        # BC config (actor architecture + family) — duck-typed for build_policy_network.
        self.bc_config_dict = bc_ckpt["config"]
        bc_cfg = SimpleNamespace(**self.bc_config_dict)
        self.policy_family = bc_cfg.policy_family
        self.dist_params   = getattr(bc_cfg, "dist_params", {}) or {}

        # Shared distribution (math only) — sampling + KL for the loss / inference.
        self.distribution = make_distribution(self.policy_family, self.dist_params)

        # ---- actor: warm-start + eval (dropout off) but trainable ----
        self.actor = build_policy_network(bc_cfg)
        self.actor.load_state_dict(bc_ckpt["state_dict"])
        self.actor.to(self.device)
        self.actor.eval()

        # ---- frozen BC prior (KL anchor) ----
        self.bc_policy = build_policy_network(bc_cfg)
        self.bc_policy.load_state_dict(bc_ckpt["state_dict"])
        self.bc_policy.to(self.device)
        self.bc_policy.eval()
        for p in self.bc_policy.parameters():
            p.requires_grad_(False)

        # ---- critic + frozen target ----
        self.critic = build_critic_network(iq_config).to(self.device)
        self.critic_target = build_critic_network(iq_config).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        for p in self.critic_target.parameters():
            p.requires_grad_(False)

        # ---- optimisers ----
        self.actor_opt  = torch.optim.Adam(self.actor.parameters(),  lr=iq_config.lr_actor)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=iq_config.lr_critic)

        # ---- loss ----
        self.loss_fn = IQLearnLoss(
            actor            = self.actor,
            critic           = self.critic,
            critic_target    = self.critic_target,
            bc_policy        = self.bc_policy,
            distribution     = self.distribution,
            gamma            = iq_config.gamma,
            alpha_entropy    = iq_config.alpha_entropy,
            alpha_reg        = iq_config.alpha_reg,
            lambda_bc        = iq_config.lambda_bc,
            n_action_samples = iq_config.n_action_samples,
            q_clip           = iq_config.q_clip,
            q_reg_coef       = iq_config.q_reg_coef,
        )

    # -----------------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------------

    def update(self, batch: Batch, update_actor: bool = True) -> dict[str, Any]:
        """One IQ-Learn step: critic update, optional actor update, target sync."""
        critic_loss, metrics = self.loss_fn.critic_loss(batch)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        clip_grad_norm_(self.critic.parameters(), self.config.grad_clip)
        self.critic_opt.step()

        if update_actor:
            actor_loss, a_metrics = self.loss_fn.actor_loss(batch)
            self.actor_opt.zero_grad(set_to_none=True)
            actor_loss.backward()
            clip_grad_norm_(self.actor.parameters(), self.config.grad_clip)
            self.actor_opt.step()
            metrics.update(a_metrics)

        self._soft_update_target()
        return metrics

    def _soft_update_target(self) -> None:
        """critic_target <- tau * critic + (1 - tau) * critic_target."""
        tau = self.config.tau
        with torch.no_grad():
            for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
                tp.data.copy_(tau * p.data + (1.0 - tau) * tp.data)

    # -----------------------------------------------------------------------
    # Inference
    # -----------------------------------------------------------------------

    @torch.no_grad()
    def select_action(self, states: torch.Tensor, deterministic: bool = True,
                      generator: torch.Generator | None = None) -> torch.Tensor:
        """States -> normalised releases (B,).  deterministic -> family mean; else a sample."""
        params = self.actor(states)
        if deterministic:
            return self.distribution.mean_action(params)
        return self.distribution.rsample(params, generator=generator)[0]

    @torch.no_grad()
    def sample_log_prob(self, states: torch.Tensor) -> torch.Tensor:
        """Sampled action log-prob (B,) — used for the entropy metric."""
        params = self.actor(states)
        return self.distribution.rsample(params)[1]

    # -----------------------------------------------------------------------
    # Persistence
    # -----------------------------------------------------------------------

    def save(self, path) -> None:
        torch.save(
            {
                "actor":         {k: v.detach().cpu() for k, v in self.actor.state_dict().items()},
                "critic":        {k: v.detach().cpu() for k, v in self.critic.state_dict().items()},
                "critic_target": {k: v.detach().cpu() for k, v in self.critic_target.state_dict().items()},
                "iq_config":     asdict(self.config),
                "bc_config":     self.bc_config_dict,
                "policy_type":   POLICY_TYPE,
                "policy_family": self.policy_family,
            },
            path,
        )

    @classmethod
    def from_checkpoint(cls, path, device: str | torch.device) -> "IQLearnAgent":
        """Rebuild an agent from a saved iq_agent.pt (for results)."""
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        iq_config = IQConfig(**ckpt["iq_config"])
        iq_config.device = str(device)

        bc_ckpt = {
            "state_dict":  ckpt["actor"],
            "config":      ckpt["bc_config"],
            "policy_type": POLICY_TYPE,
        }
        agent = cls(iq_config, bc_ckpt, device)
        agent.actor.load_state_dict(ckpt["actor"])
        agent.critic.load_state_dict(ckpt["critic"])
        agent.critic_target.load_state_dict(ckpt["critic_target"])
        agent.actor.eval()
        return agent
