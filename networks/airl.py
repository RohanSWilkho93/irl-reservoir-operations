"""
networks/airl.py
================
Network definitions for Adversarial Inverse Reinforcement Learning (AIRL).

Four components are defined here:

    CriticNetwork       — value function V(s), used by PPO for advantage estimation.
    AIRLRewardNetwork   — reward component g(s, a), spectral-normalised for stability.
    AIRLShapingNetwork  — potential shaping Phi(s), state-only.
    AIRLDiscriminator   — wraps the reward/shaping nets and the policy to implement
                          the full AIRL discriminator and reward extraction.

All four are instantiated together by ``build_airl_networks(config, policy)``.

Architecture hyperparameters
----------------------------
Critic and discriminator networks use *separate* width/depth hyperparameters, both
tuned by Optuna through ``airl/tune.py``.  See ``configs/algorithms/airl.yaml`` for
the search space:

    critic_hidden_dim      — width of every hidden layer in CriticNetwork
    critic_n_hidden_layers — depth of CriticNetwork
    disc_hidden_dim        — width shared by AIRLRewardNetwork and AIRLShapingNetwork
    disc_n_hidden_layers   — depth shared by both discriminator sub-networks
    disc_dropout           — dropout probability shared by both discriminator sub-networks

These are stored in ``AIRLConfig`` (defined in ``airl/core.py``) and passed in as
``config``.  Policy weights are loaded from the best BC checkpoint and injected into
the discriminator via the constructor; they are NOT re-instantiated here.

AIRL discriminator mechanics
-----------------------------
Given current policy pi, the discriminator D(s, a, s') measures how expert-like a
transition is:

    f(s, a, s') = clamp(g(s,a) + gamma*Phi(s') - Phi(s), -20, 20)
    D(s, a, s') = sigmoid(f(s,a,s') - log_pi(a|s))
    r_AIRL       = clamp(logit(clamp(D, 0.01, 0.99)), -10, 10)

The potential Phi cancels out under the optimal reward, leaving g(s,a) as the
recovered reward.  ``extract_reward_function(s, a)`` returns g(s, a) directly,
which is used for reward contour plots in ``airl/generate_results.py``.

Training flow
-------------
    1. BC pretraining:  policy weights initialised from ``results/<reservoir>/
                        behavioral_cloning/best_model.pt`` before AIRL starts.
    2. Discriminator warmup:  policy is frozen; discriminator trained for
       ``warmup_iterations`` iterations on expert vs. rollout transitions.
    3. Adversarial loop:
         a. Collect rollout from environment using current policy.
         b. Update discriminator with gradient penalty (WGAN-GP style).
         c. PPO update using AIRL rewards + KL regularisation toward BC prior.

Spectral normalisation
-----------------------
``AIRLRewardNetwork`` wraps every ``nn.Linear`` with
``torch.nn.utils.spectral_norm``.  This enforces a Lipschitz constraint on the
reward function, which stabilises discriminator training without requiring
gradient clipping.  ``AIRLShapingNetwork`` and ``CriticNetwork`` do not use
spectral norm — their objective landscapes are better behaved.

Dropout
-------
    AIRLRewardNetwork  : Dropout(disc_dropout) between hidden layers.  Tuned by Optuna
                         via ``disc_dropout`` in ``configs/algorithms/airl.yaml``.
    AIRLShapingNetwork : Dropout(disc_dropout) between hidden layers.  Same value as
                         the reward network — both are tuned together.
    CriticNetwork      : No dropout.  Value estimates must be low-variance; dropout
                         destabilises GAE advantage computation.  Fixed at 0.

Weight initialisation
----------------------
All ``nn.Linear`` layers use Xavier uniform for weights and constant-zero biases.
Output heads are zero-initialised (weights and bias) so the network starts near
zero, which is appropriate for rewards and value estimates.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Avoid circular imports at runtime; type annotation only.
    from networks.policy import BetaActor, LognormalActor, HardgatingActor, SoftgatingActor


# =============================================================================
# Shared MLP builder
# =============================================================================

def _build_mlp(
    input_dim:       int,
    hidden_dim:      int,
    n_hidden_layers: int,
    dropout:         float = 0.0,
    spectral_norm:   bool  = False,
) -> nn.Sequential:
    """
    Build a shared MLP trunk: ``n_hidden_layers`` blocks of Linear -> ReLU -> Dropout.

    Parameters
    ----------
    input_dim        : Dimension of the first layer's input.
    hidden_dim       : Width of every hidden layer.
    n_hidden_layers  : Number of hidden Linear->ReLU->Dropout blocks.
    dropout          : Dropout probability after each ReLU.  0.0 = no dropout.
    spectral_norm    : If True, wrap every ``nn.Linear`` with
                       ``torch.nn.utils.spectral_norm``.

    Returns
    -------
    nn.Sequential ending at hidden_dim neurons.  The output head is NOT included;
    callers append it separately so each class can initialise it independently.
    """

    def _linear(in_f: int, out_f: int) -> nn.Linear:
        layer = nn.Linear(in_f, out_f)
        if spectral_norm:
            layer = torch.nn.utils.spectral_norm(layer)  # type: ignore[assignment]
        return layer

    layers: list[nn.Module] = [_linear(input_dim, hidden_dim), nn.ReLU()]
    if dropout > 0.0:
        layers.append(nn.Dropout(dropout))

    for _ in range(n_hidden_layers - 1):
        layers.append(_linear(hidden_dim, hidden_dim))
        layers.append(nn.ReLU())
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))

    return nn.Sequential(*layers)


# =============================================================================
# CriticNetwork  (value function)
# =============================================================================

class CriticNetwork(nn.Module):
    """
    Value function V(s) for PPO advantage estimation in AIRL.

    Maps a normalised state vector to a scalar value estimate.  No spectral
    normalisation and no dropout — the PPO critic benefits from low-variance
    estimates rather than the Lipschitz regularisation needed by the reward net.

    Architecture
    ------------
    ::

        Linear(state_dim -> hidden_dim) -> ReLU
        [Linear(hidden_dim -> hidden_dim) -> ReLU] x (n_hidden_layers - 1)
        Linear(hidden_dim -> 1)

    Hyperparameters
    ---------------
    ``hidden_dim`` and ``n_hidden_layers`` are tuned by Optuna via
    ``critic_hidden_dim`` and ``critic_n_hidden_layers`` in
    ``configs/algorithms/airl.yaml``.

    Config fields required
    ----------------------
    state_dim, critic_hidden_dim, critic_n_hidden_layers.
    """

    def __init__(self, state_dim: int, hidden_dim: int, n_hidden_layers: int) -> None:
        super().__init__()
        self.state_dim       = state_dim
        self.hidden_dim      = hidden_dim
        self.n_hidden_layers = n_hidden_layers

        self.trunk  = _build_mlp(
            input_dim       = state_dim,
            hidden_dim      = hidden_dim,
            n_hidden_layers = n_hidden_layers,
            dropout         = 0.0,
            spectral_norm   = False,
        )
        self.output = nn.Linear(hidden_dim, 1)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.trunk.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        state : torch.Tensor of shape (batch, state_dim)
            Normalised state including optional month encoding.

        Returns
        -------
        torch.Tensor of shape (batch, 1)
            Scalar value estimate V(s).
        """
        return self.output(self.trunk(state))


# =============================================================================
# AIRLRewardNetwork  (reward component g(s, a))
# =============================================================================

class AIRLRewardNetwork(nn.Module):
    """
    Reward network g(s, a) — the learned, transferable reward in AIRL.

    Takes a concatenated [state, action] vector and returns a scalar reward.
    Every ``nn.Linear`` layer is wrapped with ``spectral_norm`` to enforce a
    Lipschitz constraint, which stabilises discriminator training without
    gradient clipping.  ``Dropout(disc_dropout)`` is applied between hidden
    layers for regularisation.

    Architecture
    ------------
    ::

        SN-Linear(state_dim + action_dim -> hidden_dim) -> ReLU -> Dropout(p)
        [SN-Linear(hidden_dim -> hidden_dim) -> ReLU -> Dropout(p)] x (n - 1)
        SN-Linear(hidden_dim -> 1)

    SN = spectral_norm wrapper.  p = disc_dropout (Optuna hyperparameter).

    Hyperparameters
    ---------------
    ``hidden_dim``, ``n_hidden_layers``, and ``dropout`` correspond to
    ``disc_hidden_dim``, ``disc_n_hidden_layers``, and ``disc_dropout`` in
    ``configs/algorithms/airl.yaml``.  All three are shared with
    ``AIRLShapingNetwork``; tuning one tunes both.

    Config fields required
    ----------------------
    state_dim, action_dim, disc_hidden_dim, disc_n_hidden_layers, disc_dropout.
    """

    def __init__(self, input_dim: int, hidden_dim: int, n_hidden_layers: int, dropout: float) -> None:
        """
        Parameters
        ----------
        input_dim        : state_dim + action_dim.  Computed by the caller.
        hidden_dim       : Width of every hidden layer (disc_hidden_dim).
        n_hidden_layers  : Depth (disc_n_hidden_layers).
        dropout          : Dropout probability between hidden layers (disc_dropout).
                           Tuned by Optuna; 0.0 disables dropout entirely.
        """
        super().__init__()
        self.input_dim       = input_dim
        self.hidden_dim      = hidden_dim
        self.n_hidden_layers = n_hidden_layers
        self.dropout         = dropout

        self.trunk  = _build_mlp(
            input_dim       = input_dim,
            hidden_dim      = hidden_dim,
            n_hidden_layers = n_hidden_layers,
            dropout         = dropout,
            spectral_norm   = True,
        )
        # Output head also spectral-normalised for end-to-end Lipschitz constraint.
        _output = nn.Linear(hidden_dim, 1)
        self.output = torch.nn.utils.spectral_norm(_output)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.trunk.modules():
            if isinstance(m, nn.Linear):
                # spectral_norm modifies the layer in-place: the actual stored
                # Parameter is renamed to weight_orig; m.weight becomes a plain
                # tensor recomputed by the forward pre-hook at every call and
                # is overwritten immediately — writing to it has no lasting effect.
                w = getattr(m, 'weight_orig', m.weight)
                nn.init.xavier_uniform_(w)
                nn.init.constant_(m.bias, 0.0)
        # Output head is also spectral-normed — target weight_orig here too.
        # IMPORTANT: do NOT use zeros_ on a spectral-normed layer.  The spectral
        # norm hook computes sigma = ||W||_2 via power iteration: u_hat = normalize(W v).
        # If W is all-zeros, normalize(zeros) = zeros / 0 = NaN, which propagates
        # into every forward pass.  Xavier gives small random weights so sigma > 0
        # while keeping initial output magnitudes small (bias remains 0).
        nn.init.xavier_uniform_(self.output.weight_orig)   # type: ignore[attr-defined]
        nn.init.zeros_(self.output.bias)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        state  : torch.Tensor of shape (batch, state_dim)
            Normalised state including optional month encoding.
        action : torch.Tensor of shape (batch, action_dim)
            Normalised action (release in [0, 1] for Beta/gating policies).

        Returns
        -------
        torch.Tensor of shape (batch, 1)
            Raw (unbounded) scalar reward g(s, a).
        """
        if action.dim() == 1:       # (batch,) -> (batch, 1)
            action = action.unsqueeze(-1)
        x = torch.cat([state, action], dim=-1)
        return self.output(self.trunk(x))


# =============================================================================
# AIRLShapingNetwork  (potential shaping Phi(s))
# =============================================================================

class AIRLShapingNetwork(nn.Module):
    """
    Potential shaping function Phi(s) in AIRL.

    Maps a state vector to a scalar potential.  The shaping function is
    state-only (no action input) and cancels in the optimal solution, so the
    recovered reward g(s, a) is identifiable.

    No spectral normalisation is used here — the shaping network does not need
    the same Lipschitz constraint as the reward network.  ``Dropout(disc_dropout)``
    is applied between hidden layers for regularisation.

    Architecture
    ------------
    ::

        Linear(state_dim -> hidden_dim) -> ReLU -> Dropout(p)
        [Linear(hidden_dim -> hidden_dim) -> ReLU -> Dropout(p)] x (n - 1)
        Linear(hidden_dim -> 1)

    p = disc_dropout (Optuna hyperparameter).

    Hyperparameters
    ---------------
    Shares ``disc_hidden_dim``, ``disc_n_hidden_layers``, and ``disc_dropout``
    with ``AIRLRewardNetwork``; all three are tuned together by Optuna.

    Config fields required
    ----------------------
    state_dim, disc_hidden_dim, disc_n_hidden_layers, disc_dropout.
    """

    def __init__(self, state_dim: int, hidden_dim: int, n_hidden_layers: int, dropout: float) -> None:
        super().__init__()
        self.state_dim       = state_dim
        self.hidden_dim      = hidden_dim
        self.n_hidden_layers = n_hidden_layers
        self.dropout         = dropout

        self.trunk  = _build_mlp(
            input_dim       = state_dim,
            hidden_dim      = hidden_dim,
            n_hidden_layers = n_hidden_layers,
            dropout         = dropout,
            spectral_norm   = False,
        )
        self.output = nn.Linear(hidden_dim, 1)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.trunk.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        state : torch.Tensor of shape (batch, state_dim)
            Normalised state including optional month encoding.

        Returns
        -------
        torch.Tensor of shape (batch, 1)
            Scalar potential Phi(s).
        """
        return self.output(self.trunk(state))


# =============================================================================
# AIRLDiscriminator
# =============================================================================

class AIRLDiscriminator(nn.Module):
    """
    Full AIRL discriminator.

    Wraps ``AIRLRewardNetwork``, ``AIRLShapingNetwork``, and the current policy
    to implement the three-part AIRL discriminator formula:

    .. code-block:: text

        f(s, a, s') = clamp(g(s,a) + gamma*Phi(s') - Phi(s),  -20, 20)
        D(s, a, s') = sigmoid(f(s,a,s') - log_pi(a|s))
        r_AIRL      = clamp(logit(clamp(D, 0.01, 0.99)), -10, 10)

    The policy is injected at construction time and its parameters are treated
    as read-only inside this module — ``get_log_prob`` is always called under
    ``torch.no_grad()`` from both ``forward()`` and ``get_reward()``.

    Parameters
    ----------
    reward_net   : AIRLRewardNetwork
        Instantiated reward component g(s, a).
    shaping_net  : AIRLShapingNetwork
        Instantiated shaping component Phi(s).
    policy       : nn.Module
        Any of BetaActor / LognormalActor / HardgatingActor / SoftgatingActor
        from ``networks/policy.py``.  Must implement ``get_log_prob(state, action)``.
    gamma        : float
        Discount factor used in the shaping term.  Tuned by Optuna
        (``gamma`` in ``configs/algorithms/airl.yaml``).

    Policy interface
    ----------------
    The policy must expose ``get_log_prob(state, action) -> torch.Tensor`` of
    shape ``(batch, 1)``.  All four actor classes in ``networks/policy.py``
    implement this method directly.

    Methods
    -------
    compute_f(s, a, s_next)
        Raw AIRL advantage f before subtracting log pi.  Useful for debugging
        and discriminator loss / gradient penalty computation.
    forward(s, a, s_next)
        Discriminator probability D(s, a, s').
    get_reward(s, a, s_next)
        Clipped logit reward r_AIRL.  Called inside the rollout loop.
        Always runs under ``torch.no_grad()``.
    extract_reward_function(s, a)
        Raw reward network output g(s, a) without shaping or policy correction.
        Used for reward contour plots in ``airl/generate_results.py``.
        Always runs under ``torch.no_grad()``.
    """

    def __init__(
        self,
        reward_net:  AIRLRewardNetwork,
        shaping_net: AIRLShapingNetwork,
        policy:      nn.Module,
        gamma:       float,
    ) -> None:
        super().__init__()
        self.reward_net  = reward_net
        self.shaping_net = shaping_net
        self.policy      = policy
        # register_buffer ensures gamma moves with .to(device) and appears in
        # state_dict() for checkpoint reproducibility.
        self.register_buffer('gamma', torch.tensor(gamma, dtype=torch.float32))

    # ------------------------------------------------------------------
    # Core formula
    # ------------------------------------------------------------------

    def compute_f(
        self,
        state:      torch.Tensor,
        action:     torch.Tensor,
        state_next: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the raw AIRL advantage:
            f(s, a, s') = clamp(g(s,a) + gamma*Phi(s') - Phi(s), -20, 20)

        Parameters
        ----------
        state      : (batch, state_dim)  — current normalised state.
        action     : (batch, action_dim) — action taken.
        state_next : (batch, state_dim)  — next normalised state.

        Returns
        -------
        torch.Tensor of shape (batch, 1)
        """
        g          = self.reward_net(state, action)     # (batch, 1)
        phi_s      = self.shaping_net(state)            # (batch, 1)
        phi_s_next = self.shaping_net(state_next)       # (batch, 1)
        f = g + self.gamma * phi_s_next - phi_s
        # nan_to_num before clamp: torch.clamp passes NaN through unchanged
        # (NaN comparison is always false in C++).
        f = torch.nan_to_num(f, nan=0.0, posinf=20.0, neginf=-20.0)
        return torch.clamp(f, -20.0, 20.0)

    def forward(
        self,
        state:      torch.Tensor,
        action:     torch.Tensor,
        state_next: torch.Tensor,
    ) -> torch.Tensor:
        """
        Discriminator output D(s, a, s') in (0, 1).

        ``log_pi(a|s)`` is computed under ``torch.no_grad()`` so that
        discriminator gradients do not propagate into the policy.

        Parameters
        ----------
        state      : (batch, state_dim)
        action     : (batch, action_dim)
        state_next : (batch, state_dim)

        Returns
        -------
        torch.Tensor of shape (batch, 1)
            D(s, a, s') = sigmoid(f(s,a,s') - log_pi(a|s)).
        """
        f = self.compute_f(state, action, state_next)   # (batch, 1)

        with torch.no_grad():
            log_pi = self.policy.get_log_prob(state, action)  # (batch, 1) or (batch,)
            # Clamp before subtraction: raw log_pi for low-probability actions
            # (e.g. Beta near boundaries) can be -300+, driving sigmoid -> 1
            # regardless of f.  Matches reference bound (-20, 2).
            log_pi = torch.clamp(log_pi, -20.0, 2.0)
            # Defense-in-depth: torch.clamp does NOT sanitize NaN (NaN >= -20 is
            # false in C++, so NaN passes through unchanged).  Replace any residual
            # NaN with the lower bound so sigmoid(f - log_pi) stays well-defined.
            log_pi = torch.nan_to_num(log_pi, nan=-20.0)

        if log_pi.dim() == 1:
            log_pi = log_pi.unsqueeze(-1)

        return torch.sigmoid(f - log_pi)

    # ------------------------------------------------------------------
    # Reward extraction (no_grad — not used for training)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def get_reward(
        self,
        state:      torch.Tensor,
        action:     torch.Tensor,
        state_next: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract the AIRL reward signal for rollout labelling.

        Applies the full pipeline:
            D     = forward(s, a, s')
            r     = clamp(logit(clamp(D, 0.01, 0.99)), -10, 10)

        The inner clamp ensures logit is numerically stable; the outer clamp
        prevents extreme reward spikes from destabilising PPO updates.

        Parameters
        ----------
        state      : (batch, state_dim)
        action     : (batch, action_dim)
        state_next : (batch, state_dim)

        Returns
        -------
        torch.Tensor of shape (batch, 1)
            Clipped AIRL reward r(s, a, s').
        """
        D      = self.forward(state, action, state_next)
        D_safe = torch.clamp(D, 0.01, 0.99)
        reward = torch.log(D_safe) - torch.log(1.0 - D_safe)   # logit
        return torch.clamp(reward, -10.0, 10.0)

    @torch.no_grad()
    def extract_reward_function(
        self,
        state:  torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """
        Return the raw reward network output g(s, a) without shaping or
        policy correction.

        Used in ``airl/generate_results.py`` to plot reward contours over
        (storage, inflow, release) slices without needing a next-state.

        Parameters
        ----------
        state  : (batch, state_dim)
        action : (batch, action_dim)

        Returns
        -------
        torch.Tensor of shape (batch, 1)
        """
        return self.reward_net(state, action)


# =============================================================================
# Factory
# =============================================================================

def build_airl_networks(config, policy: nn.Module) -> AIRLDiscriminator:
    """
    Instantiate CriticNetwork, AIRLRewardNetwork, AIRLShapingNetwork, and
    AIRLDiscriminator from a single config object and return the discriminator.

    The critic is attached to the discriminator as ``discriminator.critic``
    so callers can access it through the same returned object rather than
    managing four separate handles.

    Parameters
    ----------
    config : AIRLConfig
        Must contain:
            state_dim              — int, from reservoir YAML + month encoding flag
            action_dim             — int, typically 1
            critic_hidden_dim      — int, Optuna hyperparameter
            critic_n_hidden_layers — int, Optuna hyperparameter
            disc_hidden_dim        — int, Optuna hyperparameter (shared by reward+shaping)
            disc_n_hidden_layers   — int, Optuna hyperparameter (shared by reward+shaping)
            disc_dropout           — float, Optuna hyperparameter (shared by reward+shaping)
            gamma                  — float, discount factor
    policy : nn.Module
        Pre-loaded policy network (any of the four types from networks/policy.py).
        Must implement ``get_log_prob(state, action) -> Tensor``.

    Returns
    -------
    AIRLDiscriminator
        With ``.critic`` attribute set to the instantiated ``CriticNetwork``.
        Call ``.to(device)`` on the returned object to move all sub-networks.

    Optimizer note
    --------------
    Because ``policy``, ``reward_net``, ``shaping_net``, and ``critic`` are all
    registered sub-modules, ``discriminator.parameters()`` returns ALL of their
    parameters.  ``airl/core.py`` MUST build separate optimizers using explicit
    parameter groups.  Example::

        disc_opt   = Adam(list(discriminator.reward_net.parameters())
                         + list(discriminator.shaping_net.parameters()), lr=lr_disc)
        critic_opt = Adam(discriminator.critic.parameters(),  lr=lr_critic)
        policy_opt = Adam(discriminator.policy.parameters(),  lr=lr_policy)

    Train/eval mode note
    --------------------
    ``discriminator.train()`` recursively sets the policy to train mode, activating
    its BC dropout during ``get_log_prob`` calls.  Set ``discriminator.policy.eval()``
    before discriminator updates and restore with ``discriminator.policy.train()``
    before PPO updates so that log-probability estimates are deterministic during
    discriminator training.

    Examples
    --------
    >>> discriminator = build_airl_networks(airl_config, policy)
    >>> discriminator = discriminator.to(device)
    >>> critic = discriminator.critic          # access critic through discriminator
    """
    critic = CriticNetwork(
        state_dim       = config.state_dim,
        hidden_dim      = config.critic_hidden_dim,
        n_hidden_layers = config.critic_n_hidden_layers,
    )

    input_dim_reward = config.state_dim + config.action_dim
    reward_net = AIRLRewardNetwork(
        input_dim       = input_dim_reward,
        hidden_dim      = config.disc_hidden_dim,
        n_hidden_layers = config.disc_n_hidden_layers,
        dropout         = config.disc_dropout,
    )

    shaping_net = AIRLShapingNetwork(
        state_dim       = config.state_dim,
        hidden_dim      = config.disc_hidden_dim,
        n_hidden_layers = config.disc_n_hidden_layers,
        dropout         = config.disc_dropout,
    )

    discriminator = AIRLDiscriminator(
        reward_net  = reward_net,
        shaping_net = shaping_net,
        policy      = policy,
        gamma       = config.gamma,
    )

    # Attach critic as a sub-module so .to(device) and .parameters() cover it.
    discriminator.critic = critic  # type: ignore[attr-defined]

    return discriminator
