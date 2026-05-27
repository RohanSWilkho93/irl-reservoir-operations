"""
networks/iqlearn.py
===================
Network definitions for IQ-Learn (Inverse Q-Learning).

Contents
--------
IQCriticNetwork      -- twin Q-network Q(s, a).
build_iqlearn_networks -- factory: returns (critic, critic_target).

Actor not defined here
----------------------
IQ-Learn reuses the same policy networks as BC and AIRL:
BetaActor, LognormalActor, HardgatingActor, SoftgatingActor
from networks/policy.py.  They are loaded from the BC checkpoint
and optionally fine-tuned during joint IQ-Learn training.

State convention
----------------
Month encoding is PART OF THE STATE, not a separate context tensor.
When use_month_encoding is true, the state vector is:
    [storage_norm, inflow_norm, sin(2*pi*month/12), cos(2*pi*month/12)]
and state_dim = 4.  All policy networks and the critic take this full
state as a single tensor -- the same convention used in BC and AIRL.

IQ-Learn loss (summary)
-----------------------
term1 = E_expert[ Q(s,a) - gamma * V(s') ]   (Bellman residual)
term2 = E_data  [ V(s)   - gamma * V(s') ]   (temporal diff)
term3 = (1/(4*alpha)) * E_expert[ (Q - gamma*V_next)^2 ]  (chi2 reg)
Loss  = -term1 + term2 + term3   (+ actor KL penalty)

V(s) is the soft value, estimated by Monte Carlo over actions
sampled from the current policy.

Implicit reward at inference:
    r(s,a) ~ Q(s,a) - gamma * V(s')

Visualised as Q-function contours in iqlearn/generate_results.py.

Twin Q-network
--------------
Two independent Q-networks (self.q1, self.q2) with separate parameters.
Training loss uses min(Q1, Q2) -- pessimistic Q-estimate -- to mitigate
overestimation (same trick as TD3 / SAC).  A frozen Polyak-averaged target
network provides stable Bellman bootstrap targets.

Input dimensions (standard reservoir setup, use_month_encoding=True)
----------------------------------------------------------------------
state_dim  = 4   [storage_norm, inflow_norm, sin_month, cos_month]
action_dim = 1   [release_norm]
input total = 5

No spectral norm, no dropout
-----------------------------
- No spectral_norm : Q-values must represent return scale accurately.
  Spectral norm caps Lipschitz constant but distorts value magnitude.
- No dropout : Bellman bootstrapping is already noisy from single-sample
  value estimation.  Dropout adds further variance to targets.
  Regularisation comes from the twin-network structure and Polyak updates.

Weight initialisation
---------------------
All Linear layers (trunk and output head) use Xavier-uniform weights
and zero biases.  Output heads must NOT be zero-initialised: a zero
Q-function makes term3 = 0 at init, killing the chi2 gradient signal.

Checkpoint compatibility with iqlearn/generate_results.py
----------------------------------------------------------
generate_results.py is self-contained with its own CriticNetwork class.
For checkpoint loading to work, the state_dict key layout here MUST
match that file exactly.

Both use the same layout:
  self.q1 = nn.Sequential([Linear, ReLU, ..., Linear(hidden->1)])
  self.q2 = nn.Sequential([Linear, ReLU, ..., Linear(hidden->1)])

State-dict keys produced:
  q1.0.weight, q1.0.bias, q1.2.weight, ..., q1.N.weight, q1.N.bias
  q2.0.weight, q2.0.bias, q2.2.weight, ..., q2.N.weight, q2.N.bias

iqlearn/train.py saves critic.state_dict() under checkpoint key "critic".
generate_results.py loads it via critic.load_state_dict(ckpt["critic"]).
Any structural change here MUST be mirrored in generate_results.py.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import List, Tuple


# =============================================================================
# IQCriticNetwork  (twin Q-network)
# =============================================================================

class IQCriticNetwork(nn.Module):
    """
    Twin Q-network Q(s, a) for IQ-Learn.

    Two independent Q-networks (self.q1, self.q2).  Each is a full
    nn.Sequential that includes all hidden layers plus the scalar output
    head (hidden_dim -> 1).  Callers use min(Q1, Q2) as the pessimistic
    Q-estimate for both Bellman targets and actor gradient computation
    (SAC convention).  q1_only() is retained as a utility but is not
    called by the standard IQ-Learn actor update.

    State convention
    ----------------
    Month encoding is part of the state tensor.  With use_month_encoding=True
    the standard state is [storage_norm, inflow_norm, sin_month, cos_month]
    and state_dim = 4.  There is NO separate context argument -- this matches
    the convention used by all policy networks and the AIRL critic.

    Input convention
    ----------------
    forward() takes two separate tensors: state and action.
    They are concatenated internally:
        x = cat([state, action])  -- shape (batch, state_dim + action_dim)

    Attribute names
    ---------------
    self.q1 and self.q2 are nn.Sequential objects containing the full
    Q-network (trunk + output head).  These names MUST NOT be changed --
    the integer-indexed state_dict keys they produce must match
    generate_results.py's CriticNetwork.

    Config fields required
    ----------------------
    state_dim              : int -- full state dimension (typically 4 with month enc)
    action_dim             : int -- normalised action dimension (typically 1)
    critic_hidden_dim      : int -- width of every hidden layer (Optuna param)
    critic_n_hidden_layers : int -- number of hidden layers (Optuna param)
    """

    def __init__(self, config) -> None:
        super().__init__()
        input_dim = config.state_dim + config.action_dim

        self.state_dim  = config.state_dim
        self.action_dim = config.action_dim

        # Both Q-networks: trunk + output head in one Sequential.
        # Attribute names q1/q2 are load-bearing for checkpoint compatibility.
        self.q1 = self._build_q(
            input_dim,
            config.critic_hidden_dim,
            config.critic_n_hidden_layers,
        )
        self.q2 = self._build_q(
            input_dim,
            config.critic_hidden_dim,
            config.critic_n_hidden_layers,
        )

        self._init_weights()

    @staticmethod
    def _build_q(
        input_dim:       int,
        hidden_dim:      int,
        n_hidden_layers: int,
    ) -> nn.Sequential:
        """
        Build one Q-network as a complete nn.Sequential.

        Architecture:
            Linear(input_dim -> hidden_dim) -> ReLU
            [Linear(hidden_dim -> hidden_dim) -> ReLU] x (n_hidden_layers - 1)
            Linear(hidden_dim -> 1)   <-- scalar output head, included here

        The output head is included in the Sequential so that state_dict keys
        match generate_results.py's CriticNetwork._build_q exactly.

        Parameters
        ----------
        input_dim       : state_dim + action_dim
        hidden_dim      : width of every hidden layer
        n_hidden_layers : number of hidden Linear->ReLU blocks (minimum 1)
        """
        layers: List[nn.Module] = [
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        ]
        for _ in range(n_hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        # Scalar output head -- part of the Sequential, not a separate attribute.
        layers.append(nn.Linear(hidden_dim, 1))

        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        """
        Xavier-uniform init for all Linear layers in both Q-networks.

        Includes the output head (last Linear in each Sequential).
        Xavier is used everywhere -- NOT zeros -- because a zero Q-function
        makes term3 = (1/(4*alpha)) * (Q - gamma*V_next)^2 = 0 at init,
        which eliminates the gradient signal from the chi2 regularisation
        term and stalls early learning.  Small Xavier weights break this
        degeneracy while keeping initial Q magnitudes modest.
        Biases are initialised to zero throughout.
        """
        for net in (self.q1, self.q2):
            for m in net.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0.0)

    def forward(
        self,
        state:  torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Q-value estimates from both Q-networks.

        State contains month encoding when use_month_encoding is true:
            state = [storage_norm, inflow_norm, sin_month, cos_month]
        Month is NOT passed as a separate context argument.

        Parameters
        ----------
        state  : (batch, state_dim)  -- full state including month encoding
        action : (batch, action_dim) -- release in [0, 1].
                 If 1-D (batch,), unsqueezed to (batch, 1) automatically.

        Returns
        -------
        (Q1, Q2) : each shape (batch, 1)
            Use torch.min(Q1, Q2) for pessimistic Bellman targets and
            for actor gradient computation (SAC convention).
        """
        if action.dim() == 1:
            action = action.unsqueeze(-1)   # (batch,) -> (batch, 1)

        x = torch.cat([state, action], dim=-1)   # (batch, state_dim + action_dim)
        return self.q1(x), self.q2(x)

    def q1_only(
        self,
        state:  torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """
        Evaluate Q1 only, without evaluating Q2.

        Utility method retained for completeness.  The standard IQ-Learn
        actor update in iqlearn/core.py uses min(Q1, Q2) (SAC convention).
        q1_only() can be used in ablations or TD3-style experiments where
        only Q1 gradients should flow to the actor.

        Parameters
        ----------
        state, action : same shapes as forward().

        Returns
        -------
        torch.Tensor of shape (batch, 1)
        """
        if action.dim() == 1:
            action = action.unsqueeze(-1)

        x = torch.cat([state, action], dim=-1)
        return self.q1(x)


# =============================================================================
# Factory
# =============================================================================

def build_iqlearn_networks(config) -> Tuple[IQCriticNetwork, IQCriticNetwork]:
    """
    Instantiate the primary critic and its Polyak-averaged target network.

    Both are IQCriticNetwork instances with separate parameters.  At
    initialisation the target is an exact weight copy of the primary critic.

    During training the target is updated by soft (Polyak) update in
    IQLearnAgent._soft_update_target() (iqlearn/core.py):
        target_param = tau * param + (1 - tau) * target_param

    where tau = config.tau (e.g. 0.002) from configs/algorithms/iqlearn.yaml.

    The actor (policy) is NOT instantiated here.  It is loaded from the BC
    checkpoint in iqlearn/tune.py and iqlearn/train.py.

    Parameters
    ----------
    config : IQLearnConfig
        Must contain: state_dim, action_dim,
        critic_hidden_dim, critic_n_hidden_layers.

    Returns
    -------
    critic        : IQCriticNetwork -- primary critic, passed to optimizer.
    critic_target : IQCriticNetwork -- Polyak target, never passed to optimizer.
                    Call .to(device) on both after this function returns.

    Optimizer contract
    ------------------
    Only critic.parameters() goes to the Adam optimizer.
    critic_target.parameters() must NEVER be in any optimizer -- it is
    updated exclusively by Polyak averaging in IQLearnAgent.

    Usage in iqlearn/core.py:
        critic, critic_target = build_iqlearn_networks(cfg)
        critic        = critic.to(device)
        critic_target = critic_target.to(device)
        critic_opt    = torch.optim.Adam(
            critic.parameters(), lr=cfg.learning_rate_critic
        )
    """
    critic        = IQCriticNetwork(config)
    critic_target = IQCriticNetwork(config)

    # Copy weights so critic and target start from the same point.
    # Without this, the Polyak formula would blend random weights into the
    # target for hundreds of steps before it converges.
    critic_target.load_state_dict(critic.state_dict())

    # Hard-disable gradients on the target.  Belt-and-suspenders: core.py
    # wraps target Q calls in torch.no_grad() as well, but requires_grad_(False)
    # guarantees no accidental gradient flow even if that context is omitted.
    critic_target.requires_grad_(False)

    return critic, critic_target
