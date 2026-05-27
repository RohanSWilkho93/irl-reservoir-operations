"""
iqlearn/core.py
===============
Core logic for IQ-Learn (Inverse Q-Learning) applied to reservoir operations.

Components
----------
IQLearnConfig   -- flat dataclass holding all hyperparameters for one trial.
ExpertBuffer    -- wraps a DataSplits Split into a random-sample buffer.
compute_composite_score -- composite validation score for Optuna + early stopping.
IQLearnAgent    -- batch-based IQ-Learn training: critic update, actor update,
                   Polyak target update, evaluation, and checkpoint I/O.

State convention
----------------
Month encoding is PART OF THE STATE -- same as BC and AIRL.
With use_month_encoding = True (the standard):
    state = [storage_norm, inflow_norm, sin(2*pi*month/12), cos(2*pi*month/12)]
    state_dim = 4
All tensors passed to the actor and critic carry this full 4D state.
There is NO separate context argument anywhere in this file.

IQ-Learn algorithm (offline / batch)
--------------------------------------
IQ-Learn recovers an implicit reward from expert data without a discriminator.
The reward is implicitly encoded in the learned Q-function:
    r(s,a) ~ Q(s,a) - gamma * V(s')

Critic loss (chi-squared regularized IRL):
    J(Q) = -E_D_exp[Q(s,a) - gamma*V(s')]           <- term1 (maximize)
           + E_D_exp[V(s) - gamma*V(s')]              <- term2 (temporal diff)
           + (1/(4*alpha_reg)) * E_D_exp[(Q(s,a) - gamma*V(s'))^2]  <- term3 (chi2 reg)

V(s) = alpha_ent * log E_{a~pi}[exp(Q(s,a)/alpha_ent)]
     ~= alpha_ent * (logsumexp over K MC samples - log K)

Actor loss:
    J(pi) = -E_s[min(Q1,Q2)(s, a_pi)]              <- maximize pessimistic Q (reparam)
            + alpha_ent * E_s[log pi(a|s)]          <- entropy penalty (SAC-style)
            + lambda_bc * E_s[KL(pi(.|s) || pi_bc(.|s))]  <- anchor to BC prior

Training flow
-------------
1. Load data via load_reservoir_data (DataSplits).
2. Build ExpertBuffer from train Split and val Split.
3. Load BC checkpoint; build policy with build_policy_network.
4. Build IQLearnAgent(config, policy, policy_type).
5. Run critic warm-up (actor frozen, only critic updated).
6. Run joint training: agent.update(batch) alternates critic + actor.
7. Validate every eval_interval with agent.evaluate(val_split).
8. Save best checkpoint via agent.save(path).

Optimizer contract
------------------
Two separate Adam optimizers:
    critic_opt -- IQCriticNetwork.parameters() only
    actor_opt  -- actor (policy) network parameters only

The frozen BC prior (bc_actor) and the target network (critic_target) are
NEVER passed to any optimizer.  critic_target is updated only via Polyak
averaging in _soft_update_target().

Checkpoint format (written by train.py, read by generate_results.py)
----------------------------------------------------------------------
    actor           : actor state_dict
    critic          : IQCriticNetwork state_dict
    critic_target   : IQCriticNetwork state_dict
    config          : dict (from dataclasses.asdict(IQLearnConfig))
    policy_type     : str (e.g. 'hardgating')
    best_epoch      : int
    best_val_score  : float
    reservoir       : str
"""

from __future__ import annotations

import copy
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ---------------------------------------------------------------------------
# Project root on sys.path (needed when iqlearn/core.py is imported directly).
# ---------------------------------------------------------------------------
import sys
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from networks.iqlearn import IQCriticNetwork, build_iqlearn_networks
from networks.policy  import build_policy_network
from utils.data       import Split
from utils.metrics    import nrmse, safe_pearsonr

# Number of Monte Carlo samples for soft-value estimation.
# 10 balances variance vs. computational cost for a 4D state space.
_MC_SAMPLES = 10


# =============================================================================
# IQLearnConfig
# =============================================================================

@dataclass
class IQLearnConfig:
    """
    All hyperparameters needed to build and run one IQ-Learn trial.

    BC architecture fields (actor_hidden_dim, actor_n_hidden_layers, dropout,
    and the distribution-specific fields) are carried forward from
    best_config.json so that build_policy_network reproduces the exact
    architecture that was pretrained by Behavioral Cloning.  These should NOT
    be tuned by Optuna in iqlearn/tune.py -- they are loaded verbatim.

    IQ-Learn-specific fields (critic_*, learning rates, gamma, tau, alpha_*,
    lambda_bc, and the training schedule) are tuned by Optuna in
    iqlearn/tune.py using the search space in configs/algorithms/iqlearn.yaml.

    State convention
    ----------------
    state_dim is the FULL state dimension, including month encoding.
    With use_month_encoding = True (standard): state_dim = 4.
    There is NO context_dim field -- month is embedded in the state.
    """

    # ------------------------------------------------------------------
    # Data -- always required, no sensible universal default
    # ------------------------------------------------------------------
    state_dim:              int   = 4       # full state: storage + inflow + sin + cos
    action_dim:             int   = 1

    # ------------------------------------------------------------------
    # BC policy architecture (loaded from best_config.json, NOT tuned here)
    # ------------------------------------------------------------------
    actor_hidden_dim:       int   = 512
    actor_n_hidden_layers:  int   = 5
    dropout:                float = 0.0     # typically 0.0 for BC hardgating

    # Beta / Hardgating / Softgating distribution parameters (from BC)
    alpha_min:              float = 1.0
    alpha_max:              float = 10.0
    beta_min:               float = 1.0
    beta_max:               float = 100.0

    # Lognormal-only parameters
    sigma_min:              float = 0.1
    log_epsilon:            float = 1.0

    # Hardgating / Softgating parameters
    zero_threshold:         float = 0.01
    mse_weight:             float = 10.0    # SoftgatingActor only
    gate_weight:            float = 5.0     # SoftgatingActor only

    # ------------------------------------------------------------------
    # Critic -- twin Q-network, trained from scratch (Optuna-tuned)
    # ------------------------------------------------------------------
    critic_hidden_dim:      int   = 128
    critic_n_hidden_layers: int   = 4

    # ------------------------------------------------------------------
    # Training schedule (Optuna-tuned)
    # ------------------------------------------------------------------
    batch_size:             int   = 128
    critic_warm_up_epochs:  int   = 200     # epochs with actor frozen
    n_epochs:               int   = 500     # total joint training epochs

    # ------------------------------------------------------------------
    # Learning rates (Optuna-tuned)
    # ------------------------------------------------------------------
    learning_rate_actor:    float = 4.324e-5   # small -- actor starts from BC
    learning_rate_critic:   float = 2.168e-4   # larger -- critic trains from scratch

    # ------------------------------------------------------------------
    # IQ-Learn loss coefficients (Optuna-tuned)
    # ------------------------------------------------------------------
    gamma:                  float = 0.90    # discount factor
    tau:                    float = 0.00195 # Polyak averaging rate for target
    alpha_entropy:          float = 0.01619 # entropy temperature for soft-value V(s)
    alpha_regularization:   float = 0.09192 # chi-squared regularization weight
    lambda_bc:              float = 0.01906 # KL penalty weight toward BC prior

    # ------------------------------------------------------------------
    # Logging / runtime
    # ------------------------------------------------------------------
    log_interval:           int   = 50
    eval_interval:          int   = 50
    device:                 str   = "cpu"
    seed:                   int   = 2048
    verbose:                bool  = False

    # ------------------------------------------------------------------
    # Legacy compatibility -- build_policy_network uses hidden_dim /
    # n_hidden_layers; these properties forward to the actor fields.
    # ------------------------------------------------------------------
    @property
    def hidden_dim(self) -> int:
        return self.actor_hidden_dim

    @property
    def n_hidden_layers(self) -> int:
        return self.actor_n_hidden_layers

    @classmethod
    def from_dict(cls, d: dict) -> "IQLearnConfig":
        """Reconstruct from a dict (e.g. checkpoint['config']); drops unknown keys."""
        valid = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in valid})


# =============================================================================
# ExpertBuffer
# =============================================================================

class ExpertBuffer:
    """
    Fixed-size buffer of expert demonstrations backed by a DataSplits Split.

    Wraps the normalized (state, action, next_state, done) arrays from one
    split and provides uniformly random minibatch sampling.

    Parameters
    ----------
    split : Split from load_reservoir_data -- states, actions, next_states,
            dones are already normalized to [0, 1] (or [-1, 1] for sin/cos).

    Usage
    -----
        buf = ExpertBuffer(data.train)
        batch = buf.sample(128, device)
        # batch["states"]      -- (128, state_dim) float32 tensor
        # batch["actions"]     -- (128, 1)          float32 tensor
        # batch["next_states"] -- (128, state_dim) float32 tensor
        # batch["dones"]       -- (128, 1)          float32 tensor
    """

    def __init__(self, split: Split) -> None:
        self.states      = split.states.astype(np.float32)               # (N, state_dim)
        self.actions     = split.actions.reshape(-1, 1).astype(np.float32) # (N, 1)
        self.next_states = split.next_states.astype(np.float32)           # (N, state_dim)
        self.dones       = split.dones.astype(np.float32).reshape(-1, 1)  # (N, 1)
        self.size        = len(self.states)

        # Importance weights: upweight transitions with meaningful (non-zero)
        # releases so the agent sees impactful decisions more frequently.
        # Thresholds and multipliers match the original tuning code.
        a_flat  = np.abs(self.actions.flatten())
        weights = np.ones(self.size, dtype=np.float64)
        weights[a_flat > 0.05] = 3.0
        weights[a_flat > 0.10] = 5.0
        weights[a_flat > 0.30] = 8.0
        self._probs = weights / weights.sum()   # normalised sampling distribution

    def sample(self, batch_size: int, device: torch.device) -> Dict[str, torch.Tensor]:
        """
        Draw batch_size transitions using importance-weighted sampling.

        Transitions with larger releases (> 0.05, 0.10, 0.30 of normalised
        capacity) are upweighted by factors 3×, 5×, 8× so that meaningful
        release decisions dominate the minibatch.  Zero-release timesteps
        (spill / drought) are seen less frequently, avoiding a degenerate
        policy that learns only the no-release action.

        Returns a dict of float32 tensors on ``device``.  If batch_size >
        self.size, sampling is done with replacement.
        """
        replace = batch_size > self.size
        idx     = np.random.choice(
            self.size, size=batch_size, replace=replace, p=self._probs
        )
        return {
            "states":      torch.tensor(self.states[idx],      device=device),
            "actions":     torch.tensor(self.actions[idx],     device=device),
            "next_states": torch.tensor(self.next_states[idx], device=device),
            "dones":       torch.tensor(self.dones[idx],       device=device),
        }

    def __len__(self) -> int:
        return self.size


# =============================================================================
# Composite validation score
# =============================================================================

def compute_composite_score(
    release_corr:  float,
    release_nrmse: float,
    iqlearn_loss:  Optional[float] = None,
) -> float:
    """
    Composite validation score (higher = better).  Used by Optuna and early stopping.

    Weights (sum to 1.0):
        60 % -- normalized release Pearson r    ([-1,1] -> [0,1])
        30 % -- normalized release nRMSE        (lower nrmse -> higher score)
        10 % -- IQ-Learn loss quality           (lower loss -> higher score)
                If iqlearn_loss is None or NaN, the 10 % shifts to release_corr.

    Parameters
    ----------
    release_corr  : Pearson correlation between predicted and expert release.
    release_nrmse : Normalized RMSE of predicted vs expert release.
    iqlearn_loss  : Optional average IQ-Learn critic loss on the val split.
                    Typical range [-2, 5]; lower is better.

    Returns
    -------
    float in approximately [0, 1].  Returns 0.0 on any NaN input.
    """
    # NaN guard
    if release_corr != release_corr or release_nrmse != release_nrmse:
        return 0.0

    corr_norm = (float(release_corr) + 1.0) / 2.0          # [-1, 1] -> [0, 1]
    nrmse_sc  = max(0.0, min(1.0, 1.0 - float(release_nrmse)))

    if iqlearn_loss is None or iqlearn_loss != iqlearn_loss:
        return 0.70 * corr_norm + 0.30 * nrmse_sc

    # Map IQ-Learn loss to [0, 1]: loss=3.0 -> 0.0, loss=-2.0 -> 1.0
    loss_sc = max(0.0, min(1.0, (3.0 - float(iqlearn_loss)) / 5.0))
    return 0.60 * corr_norm + 0.30 * nrmse_sc + 0.10 * loss_sc


# =============================================================================
# IQLearnAgent
# =============================================================================

class IQLearnAgent:
    """
    IQ-Learn training agent (offline, batch-based).

    Combines:
        - An actor (policy network, initialised from BC weights).
        - A frozen BC prior (deep copy of the actor at construction time).
        - A twin Q-network critic (IQCriticNetwork) trained from scratch.
        - A Polyak-averaged target critic for stable Bellman bootstrap.

    Training is purely offline: no environment rollouts are needed.
    The critic loss encodes the IRL objective; the actor loss maximizes
    the implicit learned Q-value while staying close to the BC prior.

    Parameters
    ----------
    config      : IQLearnConfig for this trial.
    policy      : Pre-loaded policy network (any type from networks/policy.py).
                  The agent clones its weights and stores a frozen copy as bc_actor.
    policy_type : String tag matching the class ('beta', 'lognormal', 'hardgating',
                  'softgating').  Verified against the actual class name.

    Attributes
    ----------
    actor        : trainable copy of the input policy (updated by actor_opt).
    bc_actor     : frozen deep copy; used for KL regularization only.
    critic       : IQCriticNetwork (primary twin Q-network, updated by critic_opt).
    critic_target: IQCriticNetwork (Polyak copy, never in any optimizer).
    critic_opt   : Adam optimizer for critic.parameters() only.
    actor_opt    : Adam optimizer for actor.parameters() only.
    training_stats : defaultdict(list) of per-update scalar stats.
    """

    def __init__(
        self,
        config:      IQLearnConfig,
        policy:      nn.Module,
        policy_type: str,
    ) -> None:
        self.cfg         = config
        self.device      = torch.device(config.device)
        self.policy_type = policy_type

        # ------------------------------------------------------------------
        # Type-safety check: verify supplied network matches declared type.
        # Catches mismatches early, before any weight loading.
        # ------------------------------------------------------------------
        inferred = self._infer_policy_type(policy)
        if inferred != policy_type:
            raise ValueError(
                f"policy_type mismatch: declared '{policy_type}' but the "
                f"supplied network '{type(policy).__name__}' maps to "
                f"'{inferred}'.\n"
                f"Build the policy with build_policy_network('{policy_type}', config)."
            )

        # ------------------------------------------------------------------
        # Actor: build fresh, load BC weights, move to device
        # ------------------------------------------------------------------
        self.actor = build_policy_network(policy_type, config).to(self.device)
        self.actor.load_state_dict(policy.state_dict())

        # Frozen BC prior -- used ONLY for KL regularization.
        # No optimizer ever touches this; requires_grad_(False) is belt-and-suspenders.
        self.bc_actor = copy.deepcopy(self.actor)
        self.bc_actor.eval()
        self.bc_actor.requires_grad_(False)

        # ------------------------------------------------------------------
        # Critic + target
        # ------------------------------------------------------------------
        self.critic, self.critic_target = build_iqlearn_networks(config)
        self.critic        = self.critic.to(self.device)
        self.critic_target = self.critic_target.to(self.device)

        # ------------------------------------------------------------------
        # Optimizers -- strict separation: one per network
        # ------------------------------------------------------------------
        self.critic_opt = optim.Adam(
            self.critic.parameters(), lr=config.learning_rate_critic
        )
        self.actor_opt = optim.Adam(
            self.actor.parameters(), lr=config.learning_rate_actor
        )

        # ------------------------------------------------------------------
        # Logging
        # ------------------------------------------------------------------
        self.training_stats: Dict[str, list] = defaultdict(list)

    # ------------------------------------------------------------------
    # Policy-type inference helper
    # ------------------------------------------------------------------

    @staticmethod
    def _infer_policy_type(policy: nn.Module) -> str:
        """
        Infer the policy type string from the network class name.

        Used to verify that the supplied network was built with the correct
        factory function.  Raises ValueError if the class name is not one
        of the four recognized types.
        """
        cls = type(policy).__name__.lower()
        if   "beta"      in cls and "hard" not in cls and "soft" not in cls:
            return "beta"
        elif "lognormal" in cls:
            return "lognormal"
        elif "hard"      in cls:
            return "hardgating"
        elif "soft"      in cls:
            return "softgating"
        raise ValueError(
            f"Cannot infer policy type from class name '{type(policy).__name__}'.\n"
            f"Pass a network built by build_policy_network()."
        )

    # ------------------------------------------------------------------
    # Soft value function V(s)
    # ------------------------------------------------------------------

    def _soft_value(
        self,
        states:     torch.Tensor,
        use_target: bool = False,
        n_mc:       int  = _MC_SAMPLES,
    ) -> torch.Tensor:
        """
        Estimate the soft value V(s) by Monte Carlo sampling over the actor.

        V(s) = alpha_ent * log E_{a~pi(.|s)}[exp(Q(s,a) / alpha_ent)]
             ~ alpha_ent * (logsumexp(Q_samples / alpha_ent) - log(n_mc))

        Parameters
        ----------
        states     : (batch, state_dim) float32 tensor.
        use_target : If True, evaluate Q on the target critic (no grad, stable);
                     If False, evaluate Q on the primary critic (with grad, for V_s).
        n_mc       : Number of Monte Carlo samples from the actor.

        Returns
        -------
        V : (batch, 1) float32 tensor.
            Always returned detached from the computation graph -- V is used as
            a Bellman target value and never directly differentiated.

        Implementation note
        -------------------
        Actor parameters are NOT differentiated here regardless of use_target.
        Actor grads only flow during the explicit actor_loss backward in update().
        """
        critic_fn = self.critic_target if use_target else self.critic
        alpha     = self.cfg.alpha_entropy
        q_list    = []

        with torch.no_grad():
            for _ in range(n_mc):
                out  = self.actor(states, deterministic=False)
                a    = out.action                              # (batch, action_dim)
                q1_t, q2_t = critic_fn(states, a)
                q_list.append(torch.min(q1_t, q2_t))         # (batch, 1)

        # q_stack: (n_mc, batch, 1)
        q_stack = torch.stack(q_list, dim=0)
        V = alpha * (
            torch.logsumexp(q_stack / alpha, dim=0) - np.log(n_mc)
        )   # (batch, 1)
        return V.detach()

    # ------------------------------------------------------------------
    # Critic loss
    # ------------------------------------------------------------------

    def compute_critic_loss(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute the IQ-Learn critic loss (chi-squared regularized IRL).

        J(Q) = -term1 + term2 + term3

        term1 = E_D_exp[ Q(s,a) - gamma * V_target(s') ]  <- Bellman residual
        term2 = E_D_exp[ V(s) - gamma * V_target(s') ]    <- temporal diff
        term3 = (1 / (4*alpha_reg)) * E_D_exp[ (Q - gamma*V_target(s'))^2 ]  <- chi2

        Q(s,a)       is computed from the primary critic -- grads flow through Q.
        V_target(s') is computed from the target critic under no_grad (stable target).
        V(s)         is computed from the primary critic under no_grad (also detached).
                     Gradient signal to the critic comes entirely through Q(s,a)
                     in term1 and term3; V(s) in term2 acts as a normalizer only.

        The actor is frozen during this step; actor weights do not receive grads.

        Parameters
        ----------
        batch : dict with keys "states", "actions", "next_states", "dones",
                each a float32 tensor on self.device.

        Returns
        -------
        loss : scalar torch.Tensor (with grad).
        """
        s     = batch["states"]        # (B, state_dim)
        a     = batch["actions"]       # (B, 1)
        ns    = batch["next_states"]   # (B, state_dim)
        dones = batch["dones"]         # (B, 1) -- 1.0 at terminal transitions

        # Q(s,a) from primary critic -- grads flow
        q1, q2 = self.critic(s, a)
        Q_e    = torch.min(q1, q2)   # (B, 1)

        # V_target(s') -- NO grad, stable target
        V_next = self._soft_value(ns, use_target=True)   # (B, 1) detached

        # V_primary(s) -- computed via primary critic (detached like target;
        # gradient signal to critic comes through Q_e in term1 / term3)
        V_s    = self._soft_value(s, use_target=False)   # (B, 1) detached

        # Done masking: V(s') = 0 at terminal transitions (no future reward)
        masked_V_next = V_next * (1.0 - dones)           # (B, 1)

        # IQ-Learn loss terms
        bellman = Q_e - self.cfg.gamma * masked_V_next   # (B, 1)

        term1 = bellman.mean()
        term2 = (V_s - self.cfg.gamma * masked_V_next).mean()
        term3 = (bellman ** 2).mean() / (4.0 * self.cfg.alpha_regularization)

        # Small L2 penalty on Q magnitude to prevent unbounded Q-values.
        # Coefficient 0.001 matches the original tuning code.
        q_reg = 0.001 * (Q_e ** 2).mean()

        return -term1 + term2 + term3 + q_reg

    # ------------------------------------------------------------------
    # Actor loss
    # ------------------------------------------------------------------

    def compute_actor_loss(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute the actor loss.

        J(pi) = -E_s[ min(Q1,Q2)(s, a_pi) ]            <- pessimistic Q (reparam)
                + alpha_ent * E_s[ log pi(a_pi | s) ]   <- entropy penalty
                + lambda_bc * E_s[ KL(pi(.|s) || pi_bc(.|s)) ]  <- BC anchor

        where a_pi ~ pi(.|s) is sampled via reparameterization so gradients
        flow back through a_pi to the actor distribution parameters.

        min(Q1,Q2) is used (SAC convention) rather than Q1-only (TD3 convention).
        Using the pessimistic estimate prevents overestimation bias in the actor
        gradient and is consistent with how V(s) is computed in the critic.

        The Q-function parameters do not receive gradient during this step --
        critic_opt is zeroed in update() before this backward.

        Entropy penalty
        ---------------
        + alpha_ent * E[log pi(a|s)] discourages policy collapse.  log_prob
        is evaluated at the sampled a_pi (reparameterized) so gradients flow
        both through a_pi and through the distribution parameters directly.
        The entropy term is detached from the Q-value pathway: we call
        get_log_prob separately from the Q forward pass.

        KL penalty
        ----------
        KL(pi || pi_bc) is approximated as E_{a~pi}[log pi(a|s) - log pi_bc(a|s)].
        a_pi is detached for both log-prob calls so the KL gradient comes only
        through the distribution parameters, not through the sampled action.
        The KL is clamped to >= 0 to prevent negative values from MC noise.

        Parameters
        ----------
        batch : dict with key "states" (float32 tensor).

        Returns
        -------
        loss : scalar torch.Tensor (with grad through actor only).
        """
        s = batch["states"]   # (B, state_dim)

        # Sample action from current actor (reparameterization)
        self.actor.train()
        out  = self.actor(s, deterministic=False)
        a_pi = out.action      # (B, 1)

        # min(Q1, Q2) -- pessimistic estimate, grad flows through a_pi to actor.
        # Critic weights do NOT get grad: critic_opt is zeroed in update().
        q1_pi, q2_pi = self.critic(s, a_pi)
        q_pi         = torch.min(q1_pi, q2_pi)   # (B, 1)

        # Entropy penalty: log pi(a_pi | s) with grad through actor params.
        # Evaluated at a_pi (not a fresh sample) to reuse the reparameterized draw.
        log_prob_pi = self.actor.get_log_prob(s, a_pi)   # (B, 1) -- grad flows

        actor_loss = -q_pi.mean() + self.cfg.alpha_entropy * log_prob_pi.mean()

        # KL penalty toward BC prior
        if self.cfg.lambda_bc > 0.0:
            # Detach a_pi so KL gradient comes only via distribution params,
            # not through the Q-value pathway.
            a_det = a_pi.detach()
            log_prob_pi_kl = self.actor.get_log_prob(s, a_det)    # (B, 1)
            with torch.no_grad():
                log_prob_bc = self.bc_actor.get_log_prob(s, a_det)  # (B, 1)
            kl = (log_prob_pi_kl - log_prob_bc).clamp(min=0.0).mean()
            actor_loss = actor_loss + self.cfg.lambda_bc * kl

        return actor_loss

    # ------------------------------------------------------------------
    # Polyak update
    # ------------------------------------------------------------------

    def _soft_update_target(self) -> None:
        """
        Update target critic via Polyak averaging.

            target_param = tau * param + (1 - tau) * target_param

        where tau = config.tau (e.g. 0.00195).  Called once per update() call.
        """
        tau = self.cfg.tau
        with torch.no_grad():
            for p, p_t in zip(
                self.critic.parameters(), self.critic_target.parameters()
            ):
                p_t.data.mul_(1.0 - tau)
                p_t.data.add_(p.data, alpha=tau)

    # ------------------------------------------------------------------
    # Joint update step
    # ------------------------------------------------------------------

    def update(
        self,
        batch:          Dict[str, torch.Tensor],
        update_actor:   bool = True,
    ) -> Dict[str, float]:
        """
        Perform one critic step and (optionally) one actor step.

        During the critic warm-up phase, pass update_actor=False to keep
        the actor frozen while the critic stabilizes.

        Parameters
        ----------
        batch        : dict of tensors (from ExpertBuffer.sample).
        update_actor : If False, only the critic is updated (warm-up phase).

        Returns
        -------
        dict of scalar stats for logging:
            critic_loss, actor_loss (0.0 if actor frozen), kl_loss (if applicable).
        """
        # ------------------------------------------------------------------
        # 1. Critic update
        # ------------------------------------------------------------------
        self.critic.train()
        self.critic_opt.zero_grad()
        self.actor_opt.zero_grad()   # prevent stale actor grads during critic backward

        critic_loss = self.compute_critic_loss(batch)
        critic_loss.backward()
        self.critic_opt.step()
        self.actor_opt.zero_grad()   # clear any actor grads accumulated via V(s)

        # ------------------------------------------------------------------
        # 2. Actor update (skipped during warm-up)
        # ------------------------------------------------------------------
        actor_loss_val = 0.0
        if update_actor:
            self.actor.train()
            self.actor_opt.zero_grad()
            self.critic_opt.zero_grad()  # critic should not be updated here

            actor_loss = self.compute_actor_loss(batch)
            actor_loss.backward()
            self.actor_opt.step()
            self.critic_opt.zero_grad()  # clear any critic grads from Q1 call

            actor_loss_val = float(actor_loss.item())

        # ------------------------------------------------------------------
        # 3. Polyak update of target critic
        # ------------------------------------------------------------------
        self._soft_update_target()

        # ------------------------------------------------------------------
        # 4. Log stats
        # ------------------------------------------------------------------
        stats = {
            "critic_loss": float(critic_loss.item()),
            "actor_loss":  actor_loss_val,
        }
        for k, v in stats.items():
            self.training_stats[k].append(v)
        return stats

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        split:  Split,
    ) -> Dict[str, float]:
        """
        Evaluate the deterministic policy on a DataSplits Split.

        Runs the actor in deterministic mode on each expert state and compares
        the predicted action to the expert action.  Computes release Pearson r
        and nRMSE on the NORMALIZED action space (consistent across reservoirs).

        No environment simulation is performed -- only batch inference.

        Parameters
        ----------
        split : Split from load_reservoir_data (val or test).

        Returns
        -------
        dict with keys: release_corr, release_nrmse, composite_score.
        """
        self.actor.eval()
        states_t = torch.tensor(split.states, dtype=torch.float32, device=self.device)

        chunk    = 2048   # process in chunks to avoid OOM on large splits
        preds    = []
        with torch.no_grad():
            for start in range(0, len(states_t), chunk):
                out = self.actor(states_t[start:start + chunk], deterministic=True)
                preds.append(out.action.cpu().numpy().flatten())

        pred_actions   = np.concatenate(preds, axis=0)
        expert_actions = split.actions.flatten()

        corr,  _ = safe_pearsonr(expert_actions, pred_actions)
        nrmse_v  = nrmse(expert_actions, pred_actions)
        score    = compute_composite_score(corr, nrmse_v)

        return {
            "release_corr":      float(corr),
            "release_nrmse":     float(nrmse_v),
            "composite_score":   score,
        }

    # ------------------------------------------------------------------
    # Full training loop
    # ------------------------------------------------------------------

    def train(
        self,
        train_buf:  ExpertBuffer,
        val_split:  Split,
        trial       = None,
    ) -> Dict:
        """
        Full IQ-Learn training loop with warm-up, validation, and early stopping.

        Phase 1 -- critic warm-up:
            For critic_warm_up_epochs, update only the critic (actor frozen).
            Lets the Q-function stabilize before joint training begins.

        Phase 2 -- joint training:
            Alternate critic + actor updates for n_epochs.
            Validate every eval_interval epochs.
            Track best val score and apply early stopping.

        Parameters
        ----------
        train_buf  : ExpertBuffer wrapping the training split.
        val_split  : Split for periodic validation.
        trial      : Optuna Trial for pruning (pass None in train.py).

        Returns
        -------
        dict with keys:
            best_val_score      : float
            best_epoch          : int
            training_stats      : dict of lists (per-update scalars)
        """
        if trial is not None:
            import optuna as _optuna

        best_score   = -float("inf")
        best_weights = None
        best_epoch   = 0

        # ------------------------------------------------------------------
        # Phase 1: critic warm-up
        # ------------------------------------------------------------------
        if self.cfg.verbose:
            print(f"  Critic warm-up: {self.cfg.critic_warm_up_epochs} epochs …")

        self.actor.eval()
        for epoch in range(self.cfg.critic_warm_up_epochs):
            batch = train_buf.sample(self.cfg.batch_size, self.device)
            self.update(batch, update_actor=False)

        # ------------------------------------------------------------------
        # Phase 2: joint training
        # ------------------------------------------------------------------
        if self.cfg.verbose:
            print(f"  Joint training: {self.cfg.n_epochs} epochs …")

        for epoch in range(self.cfg.n_epochs):
            batch = train_buf.sample(self.cfg.batch_size, self.device)
            stats = self.update(batch, update_actor=True)

            # ------------------------------------------------------------------
            # Validation
            # ------------------------------------------------------------------
            if epoch % self.cfg.eval_interval == 0:
                val_metrics = self.evaluate(val_split)
                score       = val_metrics["composite_score"]

                self.training_stats["val_score"].append(score)
                self.training_stats["val_release_corr"].append(val_metrics["release_corr"])
                self.training_stats["val_release_nrmse"].append(val_metrics["release_nrmse"])

                if score > best_score:
                    best_score   = score
                    best_epoch   = epoch
                    best_weights = {
                        "actor":         {k: v.clone() for k, v in self.actor.state_dict().items()},
                        "critic":        {k: v.clone() for k, v in self.critic.state_dict().items()},
                        "critic_target": {k: v.clone() for k, v in self.critic_target.state_dict().items()},
                    }

                # Optuna pruning
                if trial is not None:
                    trial.report(score, epoch + self.cfg.critic_warm_up_epochs)
                    if trial.should_prune():
                        raise _optuna.exceptions.TrialPruned()

                if self.cfg.verbose and epoch % self.cfg.log_interval == 0:
                    print(
                        f"  Epoch {epoch:5d}  "
                        f"score={score:.4f}  "
                        f"r={val_metrics['release_corr']:.4f}  "
                        f"nrmse={val_metrics['release_nrmse']:.4f}  "
                        f"c_loss={stats['critic_loss']:.4f}  "
                        f"a_loss={stats['actor_loss']:.4f}"
                    )

        # Restore best weights
        if best_weights is not None:
            self.actor.load_state_dict(best_weights["actor"])
            self.critic.load_state_dict(best_weights["critic"])
            self.critic_target.load_state_dict(best_weights["critic_target"])

        return {
            "best_val_score":  best_score,
            "best_epoch":      best_epoch,
            "training_stats":  dict(self.training_stats),
        }

    # ------------------------------------------------------------------
    # Train / eval mode helpers
    # ------------------------------------------------------------------

    def train_mode(self) -> None:
        """Set actor and critic to training mode."""
        self.actor.train()
        self.critic.train()

    def eval_mode(self) -> None:
        """Set actor and critic to evaluation mode."""
        self.actor.eval()
        self.critic.eval()

    # ------------------------------------------------------------------
    # Checkpoint I/O
    # ------------------------------------------------------------------

    def save(
        self,
        path:           str | Path,
        best_epoch:     int   = 0,
        best_val_score: float = 0.0,
        reservoir:      str   = "",
    ) -> None:
        """
        Save actor, critic, and critic_target weights plus metadata.

        The checkpoint format is the canonical format expected by
        iqlearn/generate_results.py:

            actor           : actor state_dict
            critic          : IQCriticNetwork state_dict
            critic_target   : IQCriticNetwork state_dict
            config          : dict (from dataclasses.asdict(IQLearnConfig))
            policy_type     : str
            best_epoch      : int
            best_val_score  : float
            reservoir       : str
            training_stats  : dict of lists
        """
        torch.save(
            {
                "actor":          self.actor.state_dict(),
                "critic":         self.critic.state_dict(),
                "critic_target":  self.critic_target.state_dict(),
                "config":         asdict(self.cfg),
                "policy_type":    self.policy_type,
                "best_epoch":     best_epoch,
                "best_val_score": best_val_score,
                "reservoir":      reservoir,
                "training_stats": dict(self.training_stats),
            },
            path,
        )
