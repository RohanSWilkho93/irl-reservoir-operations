"""
iqlearn/loss.py
===============
IQ-Learn loss for a parametric (continuous-action) policy.

Unlike a finite categorical, a Beta / LogNormal / gated policy has a CONTINUOUS
action, so the soft value
    V(s) = E_{a~pi}[ Q(s, a) - alpha * log pi(a | s) ]
cannot be enumerated.  It is estimated by Monte-Carlo: draw `n_action_samples`
actions from the actor, evaluate min(Q1, Q2), and average  Q - alpha*log_prob.

Critic loss (inverse-soft-Q, chi^2-regularised) — standard IQ-Learn:
    r           = q_expert - gamma * V(s') * (1 - done)     # implied reward
    term1       = mean(r)                                    # E_expert[r]
    term2       = mean( V(s) - gamma * V(s') * (1 - done) )  # value telescoping
    term3       = (1 / (4 * alpha_reg)) * mean(r**2)         # chi^2 reward reg
    critic_loss = -term1 + term2 + term3 + q_reg
V(s') uses the TARGET critic under no_grad; (1 - done) masks the bootstrap
across year boundaries.

Actor loss (SAC, BC-anchored), maximised soft Q with the BC-anchor KL:
    actor_loss = -E[min(Q1,Q2)(s, a~pi)] + alpha * E[log pi] + lambda_bc * KL(pi || bc)

Gradient flow
-------------
  * critic loss : the policy is held fixed — actions/log-probs are sampled under
    no_grad, the gradient flows through Q only.
  * actor  loss : actions are reparameterised samples, so the gradient flows
    through the policy (into the action -> Q, and through log_prob and the KL).
    Only the actor optimiser steps, so the critic is unaffected.

Shapes: B = batch, n = n_action_samples.
"""

from __future__ import annotations

from typing import Any

import torch

from iqlearn.expert_buffer import Batch


class IQLearnLoss:
    """
    Holds the networks + IQ hyperparameters and computes the critic / actor
    losses.  The optimisation steps live in the agent.

    Parameters
    ----------
    actor, critic, critic_target : the online actor and twin critics (+ target).
    bc_policy        : frozen BC actor (KL anchor) or None if lambda_bc == 0.
    distribution     : the shared PolicyDistribution (sampling / KL math).
    n_action_samples : Monte-Carlo samples for the soft value.
    """

    def __init__(self, *, actor, critic, critic_target, bc_policy, distribution,
                 gamma: float, alpha_entropy: float, alpha_reg: float,
                 lambda_bc: float, n_action_samples: int = 10,
                 q_clip: float = 100.0, q_reg_coef: float = 1e-3):
        self.actor         = actor
        self.critic        = critic
        self.critic_target = critic_target
        self.bc_policy     = bc_policy
        self.distribution  = distribution

        self.gamma         = float(gamma)
        self.alpha_entropy = float(alpha_entropy)
        self.alpha_reg     = float(alpha_reg)
        self.lambda_bc     = float(lambda_bc)
        self.n             = int(n_action_samples)
        self.q_clip        = float(q_clip)
        self.q_reg_coef    = float(q_reg_coef)

        if self.bc_policy is not None:
            self.bc_policy.eval()

    # -----------------------------------------------------------------------
    # Monte-Carlo soft value (policy held fixed: actions/log-probs detached).
    # -----------------------------------------------------------------------

    def _soft_value(self, states: torch.Tensor, use_target: bool) -> torch.Tensor:
        """V(s) ~= mean_n[ min(Q1,Q2)(s, a_n) - alpha * log pi(a_n|s) ],  (B,)."""
        B, D = states.shape
        rep = states.unsqueeze(0).expand(self.n, B, D).reshape(self.n * B, D)
        with torch.no_grad():                                  # policy fixed for the critic update
            params = self.actor(rep)
            actions, log_probs = self.distribution.rsample(params)
        critic = self.critic_target if use_target else self.critic
        q1, q2 = critic(rep, actions)
        q = torch.min(q1, q2).clamp(-self.q_clip, self.q_clip)
        q = q.view(self.n, B)
        log_probs = log_probs.view(self.n, B)
        return (q - self.alpha_entropy * log_probs).mean(dim=0)

    # -----------------------------------------------------------------------
    # Critic loss
    # -----------------------------------------------------------------------

    def critic_loss(self, batch: Batch) -> tuple[torch.Tensor, dict[str, Any]]:
        states, actions = batch.states, batch.actions
        next_states, dones = batch.next_states, batch.dones

        q1e, q2e = self.critic(states, actions)
        q_expert = torch.min(q1e, q2e).clamp(-self.q_clip, self.q_clip)         # (B,)

        v_current = self._soft_value(states, use_target=False)                 # grad -> online critic
        with torch.no_grad():
            v_next = self._soft_value(next_states, use_target=True)            # target, no grad

        gv_next = self.gamma * v_next * (1.0 - dones)
        bellman_residual = q_expert - gv_next                                  # r
        temporal_diff    = v_current - gv_next

        term1 = bellman_residual.mean()
        term2 = temporal_diff.mean()
        term3 = (1.0 / (4.0 * self.alpha_reg)) * (bellman_residual ** 2).mean()
        q_reg = self.q_reg_coef * (q_expert ** 2).mean()

        loss = -term1 + term2 + term3 + q_reg
        metrics = {
            "critic_loss":   loss.item(),
            "term1_expert":  term1.item(),
            "term2_value":   term2.item(),
            "term3_chi2":    term3.item(),
            "q_reg":         q_reg.item(),
            "mean_q_expert": q_expert.mean().item(),
            "mean_v":        v_current.mean().item(),
        }
        return loss, metrics

    # -----------------------------------------------------------------------
    # Actor loss
    # -----------------------------------------------------------------------

    def actor_loss(self, batch: Batch) -> tuple[torch.Tensor, dict[str, Any]]:
        states = batch.states

        params = self.actor(states)
        actions, log_probs = self.distribution.rsample(params)                 # reparameterised
        q1, q2 = self.critic(states, actions)
        q_new = torch.min(q1, q2).clamp(-self.q_clip, self.q_clip)

        policy_loss = -q_new.mean() + self.alpha_entropy * log_probs.mean()

        bc_kl = torch.zeros((), device=states.device)
        if self.bc_policy is not None and self.lambda_bc > 0.0:
            with torch.no_grad():
                bc_params = self.bc_policy(states)
            bc_kl = self.distribution.kl(params, bc_params).clamp(min=0.0).mean()

        loss = (policy_loss + self.lambda_bc * bc_kl).clamp(-1000.0, 1000.0)

        metrics = {
            "actor_loss":  loss.item(),
            "policy_loss": policy_loss.item(),
            "mean_q_new":  q_new.mean().item(),
            "entropy":     float((-log_probs.mean()).item()),
            "bc_kl":       float(bc_kl.item()),
        }
        return loss, metrics
