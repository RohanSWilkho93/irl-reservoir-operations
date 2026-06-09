"""
iqlearn/agent.py
================
IQ-Learn agent for the categorical policy: builds and owns the networks,
optimisers, and loss, and exposes the train step.

Construction (from a tuned BC checkpoint)
-----------------------------------------
  * actor      : rebuilt from the BC config inside bc_policy.pt and warm-started
                 with the BC weights.  Put in eval() mode so its Dropout layers
                 are OFF — the categorical soft value must be exact/deterministic
                 given the state.  eval() does NOT freeze it: params keep
                 requires_grad=True and the actor trains normally.
  * bc_policy  : a second CategoricalPolicy from the same checkpoint, frozen
                 (eval + requires_grad=False) — the KL anchor target.
  * critic     : TwinCritic from the IQ config (trained from scratch, tunable).
  * critic_tgt : frozen Polyak copy of the critic for stable bootstrap targets.

Training
--------
  update(batch, update_actor):
      critic step  ->  (optional) actor step  ->  Polyak target sync
  update_actor=False is the critic warm-up phase (actor held at BC weights).

Shapes: B = batch, D = state_dim, K = n_bins.
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
from iqlearn.utils import distribution


POLICY_TYPE = "categorical"


# =============================================================================
# IQ config
# =============================================================================

@dataclass
class IQConfig:
    """
    All hyperparameters for one IQ-Learn run.

    Actor architecture and the bin grid are NOT here — they are inherited from
    the BC checkpoint.  state_dim is needed to build the critic and is taken
    from the same data/BC config so it matches the warm-started actor.
    """
    # shared with data / BC (critic input + action)
    state_dim:  int
    action_dim: int = 1

    # critic architecture (trained from scratch — tunable)
    critic_hidden_dim:      int = 256
    critic_n_hidden_layers: int = 3

    # IQ-Learn hyperparameters
    gamma:         float = 0.99
    tau:           float = 0.005
    alpha_entropy: float = 0.25
    alpha_reg:     float = 0.5
    lambda_bc:     float = 0.5
    lr_actor:      float = 3e-4
    lr_critic:     float = 3e-4

    # training loop (used by iq_tuning, carried here for reproducibility)
    batch_size:            int = 256
    critic_warm_up_epochs: int = 100   # MAX warm-up epochs (early stopping may cut short)
    n_epochs:              int = 250   # MAX joint   epochs (early stopping may cut short)
    seed:                  int = 42

    # LR schedule + early stopping (tuned by iq_tuning; defaults keep old ckpts loadable)
    scheduler_type:  str = "none"      # "cosine" | "plateau" | "none"
    warmup_patience: int = 20          # warm-up epochs w/o composite gain -> stop
    joint_patience:  int = 30          # joint   epochs w/o composite gain -> stop

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
                {"state_dict": ..., "config": <BCConfig asdict>, "policy_type": ...}
    device    : target device.
    """

    def __init__(self, iq_config: IQConfig, bc_ckpt: dict, device: str | torch.device):
        self.config = iq_config
        self.device = torch.device(device)

        # BC config (actor architecture + frozen bin grid) — duck-typed object
        # exposing the attributes build_policy_network reads.
        self.bc_config_dict = bc_ckpt["config"]
        bc_cfg = SimpleNamespace(**self.bc_config_dict)

        # ---- frozen bin grid as float32 device tensors (the carried contract) ----
        self.bin_means = torch.tensor(bc_cfg.bin_means, dtype=torch.float32, device=self.device)
        self.bin_edges = torch.tensor(bc_cfg.bin_edges, dtype=torch.float32, device=self.device)

        # ---- actor: warm-start + eval (dropout off) but trainable ----
        self.actor = build_policy_network(bc_cfg)
        self.actor.load_state_dict(bc_ckpt["state_dict"])
        self.actor.to(self.device)
        self.actor.eval()                      # dropout OFF; params still require grad

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
            actor         = self.actor,
            critic        = self.critic,
            critic_target = self.critic_target,
            bc_policy     = self.bc_policy,
            bin_means     = self.bin_means,
            gamma         = iq_config.gamma,
            alpha_entropy = iq_config.alpha_entropy,
            alpha_reg     = iq_config.alpha_reg,
            lambda_bc     = iq_config.lambda_bc,
            q_clip        = iq_config.q_clip,
            q_reg_coef    = iq_config.q_reg_coef,
        )

    # -----------------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------------

    def update(self, batch: Batch, update_actor: bool = True) -> dict[str, Any]:
        """
        One IQ-Learn step: critic update, optional actor update, target sync.

        update_actor=False during the critic warm-up phase (the actor is held
        at its BC weights; note critic_loss detaches the actor anyway, so no
        actor gradient is produced even if this were True during warm-up).
        """
        # ---- critic ----
        critic_loss, metrics = self.loss_fn.critic_loss(batch)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        clip_grad_norm_(self.critic.parameters(), self.config.grad_clip)
        self.critic_opt.step()

        # ---- actor ----
        if update_actor:
            actor_loss, a_metrics = self.loss_fn.actor_loss(batch)
            self.actor_opt.zero_grad(set_to_none=True)
            actor_loss.backward()
            clip_grad_norm_(self.actor.parameters(), self.config.grad_clip)
            self.actor_opt.step()
            metrics.update(a_metrics)

        # ---- Polyak target sync (every step, incl. warm-up) ----
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
    def select_action(
        self,
        states:        torch.Tensor,                 # (B, D) on self.device
        deterministic: bool = True,
        generator:     torch.Generator | None = None,
    ) -> torch.Tensor:
        """
        Map states to normalised releases (B,).

        deterministic=True  : expected value  sum_k p_k * bin_means[k]
        deterministic=False : two-level sample (bin ~ p, then dequantise) — used
                              for the Monte-Carlo rollout fans in generate_results.
        """
        logits = self.actor(states)
        if deterministic:
            return distribution.expected_value(logits, self.bin_means)
        return distribution.sample(logits, self.bin_edges, generator=generator)

    # -----------------------------------------------------------------------
    # Persistence
    # -----------------------------------------------------------------------

    def save(self, path) -> None:
        """Save weights + both configs (everything needed to reconstruct)."""
        torch.save(
            {
                "actor":         {k: v.detach().cpu() for k, v in self.actor.state_dict().items()},
                "critic":        {k: v.detach().cpu() for k, v in self.critic.state_dict().items()},
                "critic_target": {k: v.detach().cpu() for k, v in self.critic_target.state_dict().items()},
                "iq_config":     asdict(self.config),
                "bc_config":     self.bc_config_dict,
                "policy_type":   POLICY_TYPE,
            },
            path,
        )

    @classmethod
    def from_checkpoint(cls, path, device: str | torch.device) -> "IQLearnAgent":
        """
        Rebuild an agent from a saved iq_agent.pt (for generate_results).

        The actor is warm-started inside __init__ from bc_config + the saved
        actor weights; critic and target are then loaded explicitly.  (The
        bc_policy ends up holding the final actor weights rather than the
        original BC weights — irrelevant, since it is only used during training.)
        """
        ckpt = torch.load(path, map_location="cpu")

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