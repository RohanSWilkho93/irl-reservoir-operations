"""
iqlearn/networks/critic.py
==========================
Twin soft-Q critic for IQ-Learn.

Two INDEPENDENT Q-networks Q1, Q2 each map  [state ⊕ action] -> scalar.
The action is the continuous normalised release (a single scalar in [0, 1]);
there is no separate context input because the month encoding is already folded
into the state vector by utils/data.py.  Taking min(Q1, Q2) downstream is the
clipped-double-Q trick that curbs the systematic over-estimation a single
bootstrapped critic develops.

Independence from the actor
---------------------------
Unlike the actor (whose architecture is frozen to match the BC policy for
weight transfer), the critic is trained from scratch, so its width/depth are
free to be tuned.

The bridge to the exact soft value
----------------------------------
`q_all_bins` evaluates Q1 and Q2 at EVERY bin's mean action for each state, in
a single batched pass, returning two (B, K) matrices.  Because the categorical
policy is finite, the soft value

    V(s) = sum_k p_k * min(Q1, Q2)(s, a_k) + alpha * H(p)

is then an exact weighted sum over those K columns (see
iqlearn/utils/distribution.soft_value) — no action sampling, no n_samples
hyperparameter.  `q_all_bins` is what makes that enumeration cheap.

Shapes: B = batch, D = state_dim, K = n_bins.
"""

from __future__ import annotations

import torch
import torch.nn as nn


# =============================================================================
# Twin critic
# =============================================================================

class TwinCritic(nn.Module):
    """
    Two independent Q-networks over [state ⊕ action].

    Parameters
    ----------
    state_dim       : dimensionality of the (month-encoded) state vector.
    hidden_dim      : hidden-layer width (critic-specific; tunable).
    n_hidden_layers : number of hidden layers (>= 1).
    action_dim      : action dimensionality (1 for scalar release).
    """

    def __init__(
        self,
        state_dim:       int,
        hidden_dim:      int,
        n_hidden_layers: int,
        action_dim:      int = 1,
    ):
        super().__init__()
        self.state_dim  = state_dim
        self.action_dim = action_dim
        input_dim       = state_dim + action_dim

        self.q1 = self._build_q(input_dim, hidden_dim, n_hidden_layers)
        self.q2 = self._build_q(input_dim, hidden_dim, n_hidden_layers)

    # ---- construction -----------------------------------------------------

    @staticmethod
    def _build_q(input_dim: int, hidden_dim: int, n_hidden_layers: int) -> nn.Sequential:
        """Build one Q-MLP: input -> [Linear, ReLU] x n_hidden -> Linear(1)."""
        layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(n_hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers += [nn.Linear(hidden_dim, 1)]
        return nn.Sequential(*layers)


    # ---- forward passes ---------------------------------------------------

    def forward(
        self,
        state:  torch.Tensor,    # (B, D)
        action: torch.Tensor,    # (B,) or (B, 1)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Q at a single action per state.

        Returns (q1, q2), each (B,) — squeezed to match the (B,) shape of
        distribution.soft_value, so the critic loss does all of its
        q_expert / v_next / dones arithmetic in (B,) with no risk of an
        accidental (B, 1) x (B,) -> (B, B) broadcast.  Take
        q_expert = torch.min(q1, q2) at the expert's release.
        """
        if action.dim() == 1:
            action = action.unsqueeze(-1)            # (B,) -> (B, 1)
        x = torch.cat([state, action], dim=-1)       # (B, D + 1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)

    def q_all_bins(
        self,
        state:     torch.Tensor,    # (B, D)
        bin_means: torch.Tensor,    # (K,)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Evaluate Q1 and Q2 at EVERY bin's mean action, for each state.

        Returns
        -------
        (q1, q2)  each (B, K), where entry [b, k] = Q(state_b, bin_means_k).

        The columns are aligned with the policy logits' bins, so the caller can
        do  q_min = torch.min(q1, q2)  and pass it straight to
        distribution.soft_value(q_min, logits, alpha).

        Mechanics
        ---------
        state     (B, D)  -> (B, K, D)  via expand (no copy)
        bin_means (K,)    -> (B, K, 1)  via expand
        concat            -> (B, K, D+1) -> (B*K, D+1) for one batched MLP pass
        reshape Q outputs -> (B, K)
        """
        B = state.shape[0]
        K = bin_means.shape[0]
        D = state.shape[1]

        state_rep  = state.unsqueeze(1).expand(B, K, D)        # (B, K, D)
        action_rep = bin_means.view(1, K, 1).expand(B, K, 1)   # (B, K, 1)
        x = torch.cat([state_rep, action_rep], dim=-1).reshape(B * K, D + 1)

        q1 = self.q1(x).view(B, K)
        q2 = self.q2(x).view(B, K)
        return q1, q2


# =============================================================================
# Factory  (mirrors iqlearn.networks.policy.build_policy_network)
# =============================================================================

def build_critic_network(config) -> TwinCritic:
    """
    Build a TwinCritic from an IQ config object.

    Reads (duck-typed, so any config exposing these attributes works):
        config.state_dim
        config.critic_hidden_dim
        config.critic_n_hidden_layers
        config.action_dim          (optional; defaults to 1)
    """
    return TwinCritic(
        state_dim       = config.state_dim,
        hidden_dim      = config.critic_hidden_dim,
        n_hidden_layers = config.critic_n_hidden_layers,
        action_dim      = getattr(config, "action_dim", 1),
    )