"""
iqlearn/distributions/base.py
=============================
The pluggable policy-distribution strategy.

The pipeline (BC tuning, IQ-Learn loss, results) never references a concrete
distribution.  It calls only the methods on this interface, and the concrete
family (Beta / LogNormal / HardGating / SoftGating) is selected at runtime from
the data (see iqlearn.distributions.detect_family_pair).

A `PolicyDistribution` is *stateless math + the family's hyperparameters*.  The
learnable parameters live in the policy network (encoder + the per-family heads
this class constructs via `make_heads`).  Given the encoder features, the policy
calls `params_from_features` to obtain the (clamped) distribution parameters;
everything else (sampling, deterministic decode, the BC negative log-likelihood,
and the BC-anchor KL) is computed from those parameters.

Conventions
-----------
* state already contains the (optional) sin/cos month encoding — there is no
  separate `context` input (utils/data.py folds it into the state vector).
* action_dim == 1 (a scalar normalised release in [0, 1]).
* params : dict[str, Tensor], each (B, 1).
* sample / mean_action return action (B,);  sample also returns log_prob (B,).
* nll returns a scalar;  kl returns (B,).

Serialisation
-------------
`to_config()` returns the family hyperparameters as a plain dict; together with
the family name it is stored in bc_best_config.json, so IQ-Learn and results can
rebuild the identical distribution with `make_distribution(family, params)`.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PolicyDistribution:
    """Abstract strategy.  Subclasses implement the family-specific math."""

    name: str = "base"

    def __init__(self, **hp):
        # Family hyperparameters (bounds, thresholds, epsilons) — serialised.
        self.hp = hp

    # ---- network heads -----------------------------------------------------

    def make_heads(self, hidden_dim: int, action_dim: int = 1) -> nn.ModuleDict:
        """Build the per-family parameter heads (the policy owns them)."""
        raise NotImplementedError

    def init_heads(self, heads: nn.ModuleDict) -> None:
        """Optional family-specific bias/weight initialisation (default: no-op)."""
        return None

    def params_from_features(self, features: torch.Tensor,
                             heads: nn.ModuleDict) -> dict:
        """Map encoder features -> clamped distribution parameters (dict of (B,1))."""
        raise NotImplementedError

    # ---- actions -----------------------------------------------------------

    def mean_action(self, params: dict) -> torch.Tensor:
        """Deterministic point prediction (B,) — used for decode / scoring."""
        raise NotImplementedError

    def rsample(self, params: dict,
                generator: torch.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Stochastic action and its log-prob: (action (B,), log_prob (B,)).

        Reparameterised where possible (Beta / Normal) so the actor loss can
        backprop dQ/da.  `generator` is accepted for API compatibility with the
        Monte-Carlo rollout fans; torch's distribution sampling advances the
        global RNG, so successive rollouts still differ.
        """
        raise NotImplementedError

    # ---- losses ------------------------------------------------------------

    def nll(self, params: dict, expert_action: torch.Tensor) -> torch.Tensor:
        """BC negative log-likelihood of expert_action (B,) under `params`."""
        raise NotImplementedError

    def kl(self, params: dict, ref_params: dict) -> torch.Tensor:
        """KL(self(params) || self(ref_params)) per sample (B,).  BC anchor term."""
        raise NotImplementedError

    # ---- on-policy quantities (used by AIRL / PPO) -------------------------

    def log_prob(self, params: dict, action: torch.Tensor) -> torch.Tensor:
        """Log-likelihood log pi(a|s) of a GIVEN action (B,) — for the PPO ratio."""
        raise NotImplementedError

    def entropy(self, params: dict) -> torch.Tensor:
        """Per-sample policy entropy (B,) — the PPO entropy bonus."""
        raise NotImplementedError

    # ---- serialisation -----------------------------------------------------

    def to_config(self) -> dict:
        return dict(self.hp)
