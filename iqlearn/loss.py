"""
iqlearn/loss.py
===============
IQ-Learn loss for the quantile-binned categorical policy.

Because the policy is a finite categorical over K bins, every expectation over
actions is an EXACT sum over bins.  The critic's q_all_bins gives Q at all K bin means; 
distribution.soft_value combines it with the policy
to produce  V(s) = sum_k p_k Q(s, a_k) + alpha * H(p).

Critic loss (maximised expert advantage, χ²-regularised)
-------------------------------------------------------
    r            = q_expert - gamma * V(s') * (1 - done)     # implied reward
    term1        = mean(r)                                   # E_expert[r]
    term2        = mean( V(s) - gamma * V(s') * (1 - done) ) # value telescoping
    term3        = (1 / (4 * alpha_reg)) * mean(r**2)        # χ² reward reg
    critic_loss  = -term1 + term2 + term3 + q_reg

V(s') uses the TARGET critic (stable bootstrap) under no_grad; (1 - done) masks
the bootstrap across episode (year) boundaries.

Actor loss (SAC, BC-anchored)
-----------------------------
    actor_loss = -V(s) + lambda_bc * KL(actor || bc)

GRADIENT FLOW — the one thing to get right
-------------------------------------------
Both losses use the SAME soft value V = sum_k p_k Q_k + alpha H(p), but with
MIRROR-IMAGE detaching:

  * critic loss : differentiate Q, hold the policy fixed  -> detach the actor
                  logits, let grad flow through Q.   (see _soft_value)
  * actor  loss : differentiate the policy, hold Q fixed  -> detach Q, let grad
                  flow through the logits (p and H).  (computed inline below)

Reusing one call for both would zero a gradient (the actor would never train,
or the critic loss would leak into the policy), so the two are kept separate.

Shapes: B = batch, K = n_bins.  All per-sample quantities are (B,) — matching
critic.forward (squeezed) and distribution.soft_value — so nothing broadcasts
to (B, B).
"""

from __future__ import annotations

from typing import Any

import torch

from iqlearn.utils import distribution
from iqlearn.expert_buffer import Batch


class IQLearnLoss:
    """
    Holds references to the networks and the IQ hyperparameters, and computes
    the critic and actor losses.  The optimisation steps live in the agent.

    Parameters
    ----------
    actor         : CategoricalPolicy — forward(states) -> (B, K) raw logits.
    critic        : TwinCritic — online.
    critic_target : TwinCritic — lagged copy (Polyak-updated by the agent).
    bc_policy     : frozen CategoricalPolicy BC prior, or None if lambda_bc == 0.
    bin_means     : (K,) float32 tensor on the working device (frozen grid).
    gamma, alpha_entropy, alpha_reg, lambda_bc : IQ hyperparameters.
    q_clip        : symmetric clamp on Q values (numerical guard).
    q_reg_coef    : coefficient of the mild Q-magnitude penalty.
    """

    def __init__(
        self,
        *,
        actor,
        critic,
        critic_target,
        bc_policy,
        bin_means:     torch.Tensor,
        gamma:         float,
        alpha_entropy: float,
        alpha_reg:     float,
        lambda_bc:     float,
        q_clip:        float = 100.0,
        q_reg_coef:    float = 1e-3,
    ):
        self.actor         = actor
        self.critic        = critic
        self.critic_target = critic_target
        self.bc_policy     = bc_policy
        self.bin_means     = bin_means

        self.gamma         = float(gamma)
        self.alpha_entropy = float(alpha_entropy)
        self.alpha_reg     = float(alpha_reg)
        self.lambda_bc     = float(lambda_bc)
        self.q_clip        = float(q_clip)
        self.q_reg_coef    = float(q_reg_coef)

        if self.bc_policy is not None:
            self.bc_policy.eval()

    # -----------------------------------------------------------------------
    # Soft value for the CRITIC loss: grad through Q, policy held FIXED.
    # -----------------------------------------------------------------------

    def _soft_value(self, states: torch.Tensor, use_target: bool) -> torch.Tensor:
        """
        V(s) = sum_k p_k Q(s, a_k) + alpha * H(p),  (B,)

        Logits are detached (policy fixed during the critic update); Q carries
        the gradient when use_target is False (online critic).  When use_target
        is True, the target critic's params have requires_grad=False, so V(s')
        is gradient-free — and the caller additionally wraps it in no_grad.
        """
        with torch.no_grad():
            logits = self.actor(states)                       # (B, K) — detached
        critic = self.critic_target if use_target else self.critic
        q1b, q2b = critic.q_all_bins(states, self.bin_means)  # (B, K) each
        q_min = torch.min(q1b, q2b).clamp(-self.q_clip, self.q_clip)
        return distribution.soft_value(q_min, logits, self.alpha_entropy)

    # -----------------------------------------------------------------------
    # Critic loss
    # -----------------------------------------------------------------------

    def critic_loss(self, batch: Batch) -> tuple[torch.Tensor, dict[str, Any]]:
        states, actions = batch.states, batch.actions
        next_states, dones = batch.next_states, batch.dones

        # Expert Q at the expert's actual release.
        q1e, q2e = self.critic(states, actions)                          # (B,) each
        q_expert = torch.min(q1e, q2e).clamp(-self.q_clip, self.q_clip)   # (B,)

        # Current soft value: grad -> online critic.
        v_current = self._soft_value(states, use_target=False)           # (B,)

        # Bootstrap target: target critic, no gradient.
        with torch.no_grad():
            v_next = self._soft_value(next_states, use_target=True)       # (B,)

        mask = 1.0 - dones                                                # (B,)
        gv_next = self.gamma * v_next * mask                              # (B,)

        bellman_residual = q_expert - gv_next                            # r = Q - γV'
        temporal_diff    = v_current - gv_next                           # V - γV'

        term1 = bellman_residual.mean()                                  # E_expert[r]
        term2 = temporal_diff.mean()                                     # E[V - γV']
        term3 = (1.0 / (4.0 * self.alpha_reg)) * (bellman_residual ** 2).mean()
        q_reg = self.q_reg_coef * (q_expert ** 2).mean()

        loss = -term1 + term2 + term3 + q_reg

        metrics = {
            "critic_loss":         loss.item(),
            "term1_expert_reward": term1.item(),
            "term2_value":         term2.item(),
            "term3_chi2":          term3.item(),
            "q_reg":               q_reg.item(),
            "mean_q_expert":       q_expert.mean().item(),
            "mean_v_current":      v_current.mean().item(),
            "mean_v_next":         v_next.mean().item(),
        }
        return loss, metrics

    # -----------------------------------------------------------------------
    # Actor loss
    # -----------------------------------------------------------------------

    def actor_loss(self, batch: Batch) -> tuple[torch.Tensor, dict[str, Any]]:
        states = batch.states

        logits = self.actor(states)                                      # (B, K) — grad

        # Q held fixed: detach the critic.
        with torch.no_grad():
            q1b, q2b = self.critic.q_all_bins(states, self.bin_means)
            q_min = torch.min(q1b, q2b).clamp(-self.q_clip, self.q_clip)  # (B, K)

        # Maximise soft value -> minimise its negative. Grad flows via the
        # logits (through p and H), not through Q.
        v = distribution.soft_value(q_min, logits, self.alpha_entropy)    # (B,)
        policy_loss = -v.mean()

        # BC anchor: KL(actor || bc), valid because both share the same bins.
        bc_kl = torch.zeros((), device=states.device)
        if self.bc_policy is not None and self.lambda_bc > 0.0:
            with torch.no_grad():
                bc_logits = self.bc_policy(states)
            bc_kl = distribution.kl(logits, bc_logits).clamp(min=0.0).mean()

        loss = policy_loss + self.lambda_bc * bc_kl

        with torch.no_grad():
            ent = distribution.entropy(logits).mean().item()

        metrics = {
            "actor_loss":  loss.item(),
            "policy_loss": policy_loss.item(),
            "mean_v":      v.mean().item(),
            "entropy":     ent,
            "bc_kl":       float(bc_kl.item()),
        }
        return loss, metrics