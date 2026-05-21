"""
networks/policy.py
==================
Policy network definitions for all four output distribution types used in
Behavioral Cloning, AIRL, and IQ-Learn.

All four networks share the same encoder (MLP with ReLU + Dropout activations).
The output heads differ by distribution family. All forward() calls return a
PolicyOutput dataclass with a consistent interface — unused fields are None.

State representation
--------------------
Month is treated as a state variable, not a separate context input. Whether
month encoding is applied is controlled by `use_month_encoding` in the
reservoir config (configs/reservoirs/<name>.yaml). The data pipeline
(utils/data.py) conditionally appends sin(2π·month/12) and cos(2π·month/12):

    state_dim = len(columns.state) + (2 if use_month_encoding else 0)

For the standard two-variable experiments in the paper (use_month_encoding: true):
    columns.state = [storage, net_inflow]  →  state_dim = 4

For augmented experiments, users extend columns.state in the reservoir config
(configs/reservoirs/<name>.yaml) with additional CSV columns, and state_dim
grows accordingly — no changes to this file are needed.

Policy types
------------
beta        : Beta(α, β) over normalised release in (0, 1).
              For reservoirs where release is always > 0.
              α ∈ [alpha_min, alpha_max], β ∈ [beta_min, beta_max].
              alpha_min = beta_min = 1.0 (fixed). alpha_max and beta_max
              are Optuna hyperparameters (behavioral_cloning.yaml → beta).

lognormal   : LogNormal(μ, σ) — alternative for always-positive releases.
              Action recovered as exp(sample) − log_epsilon, clamped ≥ 0.
              sigma_min and log_epsilon are Optuna hyperparameters
              (behavioral_cloning.yaml → lognormal).

hardgating  : Bernoulli gate × Beta — for reservoirs with zero-release periods.
              Gate is a hard Bernoulli sample (0 or 1) during training;
              continuous gate probability during inference (deterministic=True).
              alpha_max and beta_max are Optuna hyperparameters
              (behavioral_cloning.yaml → hardgating).

softgating  : Continuous gate × Beta — same architecture as hardgating but the
              gate is always the continuous probability (fully differentiable).
              alpha_max and beta_max are Optuna hyperparameters
              (behavioral_cloning.yaml → softgating).

Dropout
-------
All networks apply Dropout after every hidden layer. The dropout rate is an
Optuna hyperparameter (behavioral_cloning.yaml → shared: dropout). It is
carried forward into AIRL and IQ-Learn alongside hidden_dim and n_hidden_layers
so that pretrained weights transfer cleanly.

User choice
-----------
The policy type is set per reservoir in configs/reservoirs/<name>.yaml under
the key `policy_network`. To override at runtime without editing the config,
pass --policy_network <type> to behavioral_cloning/tune.py or train.py.

build_policy_network(policy_type, config) validates the string and raises a
descriptive ValueError listing valid options if an unknown type is given.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional

# Valid policy type strings — must match values in configs/reservoirs/<name>.yaml.
VALID_POLICY_TYPES = ("beta", "lognormal", "hardgating", "softgating")


# =============================================================================
# Unified return type
# =============================================================================

@dataclass
class PolicyOutput:
    """
    Unified output from every policy network's forward() call.

    All four networks return this dataclass. Fields unused by a given network
    are None, so callers can always do output.action, output.gate_prob, etc.

    Fields
    ------
    action     : Sampled (stochastic) or mode (deterministic) action tensor.
                 Shape: (batch, action_dim).
    alpha      : Beta α parameter. Set by BetaActor, HardgatingActor,
                 SoftgatingActor. None for LognormalActor.
    beta       : Beta β parameter. Same networks as alpha. None for Lognormal.
    mu         : Log-space mean. Set by LognormalActor only.
    sigma      : Log-space std.  Set by LognormalActor only.
    gate_prob  : Gate probability in [0.01, 0.99]. Set by HardgatingActor and
                 SoftgatingActor. None for BetaActor and LognormalActor.
    """
    action:    torch.Tensor
    alpha:     Optional[torch.Tensor] = None   # Beta / Hardgating / Softgating
    beta:      Optional[torch.Tensor] = None   # Beta / Hardgating / Softgating
    mu:        Optional[torch.Tensor] = None   # Lognormal only
    sigma:     Optional[torch.Tensor] = None   # Lognormal only
    gate_prob: Optional[torch.Tensor] = None   # Hardgating / Softgating only


# =============================================================================
# Shared encoder builder
# =============================================================================

def _build_encoder(
    input_dim:      int,
    hidden_dim:     int,
    n_hidden_layers: int,
    dropout:        float = 0.0,
) -> nn.Sequential:
    """
    Shared MLP encoder used by all four policy networks.

    Architecture per layer: Linear → ReLU → Dropout.
    Output size: hidden_dim.

    Parameters
    ----------
    input_dim        : state_dim (= len(columns.state) + 2 for month sin/cos).
    hidden_dim       : Width of every hidden layer. Optuna hyperparameter.
    n_hidden_layers  : Number of hidden layers. Optuna hyperparameter.
    dropout          : Dropout probability after each ReLU. Optuna hyperparameter.
                       0.0 means no dropout (Dropout is still added but is a no-op).
    """
    layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
    for _ in range(n_hidden_layers - 1):
        layers.extend([
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        ])
    return nn.Sequential(*layers)


# =============================================================================
# Beta actor
# =============================================================================

class BetaActor(nn.Module):
    """
    Policy network for reservoirs where release is always > 0.

    Outputs a Beta(α, β) distribution over normalised release ∈ (0, 1).
    The concentration parameters are bounded:
        α ∈ [config.alpha_min, config.alpha_max]
        β ∈ [config.beta_min,  config.beta_max]

    alpha_min = beta_min = 1.0 (fixed in BCConfig; ensures a valid Beta).
    alpha_max and beta_max are Optuna hyperparameters drawn from
    configs/algorithms/behavioral_cloning.yaml → beta.
    Larger upper bounds allow the distribution to become more peaked
    (near-deterministic), which is important for run-of-river reservoirs.

    Config fields required
    ----------------------
    state_dim, action_dim, hidden_dim, n_hidden_layers, dropout,
    alpha_min, alpha_max, beta_min, beta_max.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder    = _build_encoder(
            config.state_dim, config.hidden_dim, config.n_hidden_layers, config.dropout
        )
        self.alpha_head = nn.Linear(config.hidden_dim, config.action_dim)
        self.beta_head  = nn.Linear(config.hidden_dim, config.action_dim)
        self._init_weights()

    def _init_weights(self):
        for m in self.encoder:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)
        # softplus(0.7) ≈ 1.1 → moderate concentration at init
        nn.init.zeros_(self.alpha_head.weight)
        nn.init.constant_(self.alpha_head.bias, 0.7)
        # softplus(2.3) ≈ 2.4 → mild right-skew at init
        nn.init.zeros_(self.beta_head.weight)
        nn.init.constant_(self.beta_head.bias, 2.3)

    def forward(
        self,
        state:         torch.Tensor,
        deterministic: bool = False,
    ) -> PolicyOutput:
        """
        Parameters
        ----------
        state         : Normalised state including month encoding.
                        Shape (batch, state_dim).
                        Columns: [storage_norm, inflow_norm, sin_month, cos_month, ...]
        deterministic : If True, return mode α/(α+β) instead of a sample.

        Returns
        -------
        PolicyOutput with action, alpha, beta set; mu, sigma, gate_prob = None.
        """
        feat = self.encoder(state)

        alpha = torch.clamp(
            F.softplus(self.alpha_head(feat)) + self.config.alpha_min,
            self.config.alpha_min,
            self.config.alpha_max,
        )
        beta = torch.clamp(
            F.softplus(self.beta_head(feat)) + self.config.beta_min,
            self.config.beta_min,
            self.config.beta_max,
        )
        # Safety net: clamp does not sanitize NaN (e.g. from NaN encoder input).
        # Replace any residual NaN with the minimum valid concentration.
        alpha = torch.nan_to_num(alpha, nan=self.config.alpha_min)
        beta  = torch.nan_to_num(beta,  nan=self.config.beta_min)

        if deterministic:
            action = alpha / (alpha + beta)
        else:
            dist   = torch.distributions.Beta(alpha, beta)
            action = torch.clamp(dist.rsample(), 1e-6, 1.0 - 1e-6)

        return PolicyOutput(action=action, alpha=alpha, beta=beta)


# =============================================================================
# Lognormal actor
# =============================================================================

class LognormalActor(nn.Module):
    """
    Policy network for reservoirs where release is always > 0 (alternative to Beta).

    Predicts log-space Gaussian parameters (μ, σ); recovers the release as:
        action = exp(sample) − log_epsilon,  clamped ≥ 0.

    sigma_min prevents σ from collapsing to zero (degenerate point mass).
    log_epsilon is a stability offset inside the log transform; tuned by
    Optuna from configs/algorithms/behavioral_cloning.yaml → lognormal.

    Config fields required
    ----------------------
    state_dim, action_dim, hidden_dim, n_hidden_layers, dropout,
    sigma_min, log_epsilon.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder    = _build_encoder(
            config.state_dim, config.hidden_dim, config.n_hidden_layers, config.dropout
        )
        self.mu_head    = nn.Linear(config.hidden_dim, config.action_dim)
        self.sigma_head = nn.Linear(config.hidden_dim, config.action_dim)
        self._init_weights()

    def _init_weights(self):
        for m in self.encoder:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)
        # log(50 + 1) ≈ 3.9: reasonable initial log-release
        nn.init.xavier_uniform_(self.mu_head.weight)
        nn.init.constant_(self.mu_head.bias, 3.9)
        # softplus(0) + sigma_min: moderate uncertainty at init
        nn.init.zeros_(self.sigma_head.weight)
        nn.init.constant_(self.sigma_head.bias, 0.0)

    def forward(
        self,
        state:         torch.Tensor,
        deterministic: bool = False,
    ) -> PolicyOutput:
        """
        Parameters
        ----------
        state         : Normalised state including month encoding.
                        Shape (batch, state_dim).
        deterministic : If True, return exp(μ) − log_epsilon (no sampling).

        Returns
        -------
        PolicyOutput with action, mu, sigma set; alpha, beta, gate_prob = None.
        """
        feat  = self.encoder(state)

        mu    = self.mu_head(feat)                                          # unbounded
        sigma = F.softplus(self.sigma_head(feat)) + self.config.sigma_min  # strictly > 0

        if deterministic:
            log_action = mu
        else:
            dist       = torch.distributions.Normal(mu, sigma)
            log_action = dist.rsample()

        action = torch.clamp(
            torch.exp(log_action) - self.config.log_epsilon, min=0.0
        )

        return PolicyOutput(action=action, mu=mu, sigma=sigma)


# =============================================================================
# Hardgating actor
# =============================================================================

class HardgatingActor(nn.Module):
    """
    Two-stage policy for reservoirs with zero-release periods.

    Stage 1 — Bernoulli gate:  should a release occur at all?
    Stage 2 — Beta component:  if yes, how much (normalised)?

    α ∈ [config.alpha_min, config.alpha_max],
    β ∈ [config.beta_min,  config.beta_max].
    alpha_min = beta_min = 1.0 (fixed). alpha_max and beta_max are Optuna
    hyperparameters (behavioral_cloning.yaml → hardgating).

    The gate supervision threshold (zero_threshold) is a training hyperparameter
    used by BCTrainer to label observations as zero/nonzero; it does not appear
    inside this network.

    Gate behaviour
    --------------
    Training   (deterministic=False): gate ∈ {0, 1} via Bernoulli sampling.
    Inference  (deterministic=True):  gate = gate_prob ∈ (0, 1) (expected action).

    Config fields required
    ----------------------
    state_dim, action_dim, hidden_dim, n_hidden_layers, dropout,
    alpha_min, alpha_max, beta_min, beta_max.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder    = _build_encoder(
            config.state_dim, config.hidden_dim, config.n_hidden_layers, config.dropout
        )
        self.gate_head  = nn.Linear(config.hidden_dim, 1)
        self.alpha_head = nn.Linear(config.hidden_dim, config.action_dim)
        self.beta_head  = nn.Linear(config.hidden_dim, config.action_dim)
        self._init_weights()

    def _init_weights(self):
        for m in self.encoder:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)
        # sigmoid(−1.4) ≈ 0.20 → sparse gate at initialisation
        nn.init.zeros_(self.gate_head.weight)
        nn.init.constant_(self.gate_head.bias, -1.4)
        nn.init.zeros_(self.alpha_head.weight)
        nn.init.constant_(self.alpha_head.bias, 2.0)
        nn.init.zeros_(self.beta_head.weight)
        nn.init.constant_(self.beta_head.bias, 8.0)

    def forward(
        self,
        state:         torch.Tensor,
        deterministic: bool = False,
    ) -> PolicyOutput:
        """
        Parameters
        ----------
        state         : Normalised state including month encoding.
                        Shape (batch, state_dim).
        deterministic : If True, uses gate_prob and Beta mode (no sampling).

        Returns
        -------
        PolicyOutput with action, alpha, beta, gate_prob set; mu, sigma = None.
        """
        feat = self.encoder(state)

        gate_prob = torch.clamp(torch.sigmoid(self.gate_head(feat)), 0.01, 0.99)

        if deterministic:
            gate = gate_prob
        else:
            gate = torch.distributions.Bernoulli(probs=gate_prob).sample()

        alpha = torch.clamp(
            F.softplus(self.alpha_head(feat)) + self.config.alpha_min,
            self.config.alpha_min, self.config.alpha_max,
        )
        beta = torch.clamp(
            F.softplus(self.beta_head(feat)) + self.config.beta_min,
            self.config.beta_min, self.config.beta_max,
        )
        alpha = torch.nan_to_num(alpha, nan=self.config.alpha_min)
        beta  = torch.nan_to_num(beta,  nan=self.config.beta_min)

        if deterministic:
            continuous = alpha / (alpha + beta)
        else:
            continuous = torch.clamp(
                torch.distributions.Beta(alpha, beta).rsample(), 0.01, 0.99
            )

        action = gate * continuous
        return PolicyOutput(action=action, alpha=alpha, beta=beta, gate_prob=gate_prob)


# =============================================================================
# Softgating actor
# =============================================================================

class SoftgatingActor(nn.Module):
    """
    Two-stage policy for reservoirs with zero-release periods.

    Identical architecture to HardgatingActor, but the gate is always the
    continuous probability — no Bernoulli sampling at any point. This makes
    every gradient step flow through the gate head, at the cost of the network
    never producing an exact zero action. Sparsity is induced via the loss
    function (mse_weight / gate_weight in BCTrainer), not by hard gating.

    α ∈ [config.alpha_min, config.alpha_max],
    β ∈ [config.beta_min,  config.beta_max].
    alpha_max and beta_max are Optuna hyperparameters
    (behavioral_cloning.yaml → softgating).

    Config fields required
    ----------------------
    state_dim, action_dim, hidden_dim, n_hidden_layers, dropout,
    alpha_min, alpha_max, beta_min, beta_max.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.encoder    = _build_encoder(
            config.state_dim, config.hidden_dim, config.n_hidden_layers, config.dropout
        )
        self.gate_head  = nn.Linear(config.hidden_dim, 1)
        self.alpha_head = nn.Linear(config.hidden_dim, config.action_dim)
        self.beta_head  = nn.Linear(config.hidden_dim, config.action_dim)
        self._init_weights()

    def _init_weights(self):
        for m in self.encoder:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0.0)
        nn.init.zeros_(self.gate_head.weight)
        nn.init.constant_(self.gate_head.bias, -1.4)
        nn.init.zeros_(self.alpha_head.weight)
        nn.init.constant_(self.alpha_head.bias, 2.0)
        nn.init.zeros_(self.beta_head.weight)
        nn.init.constant_(self.beta_head.bias, 8.0)

    def forward(
        self,
        state:         torch.Tensor,
        deterministic: bool = False,
    ) -> PolicyOutput:
        """
        Parameters
        ----------
        state         : Normalised state including month encoding.
                        Shape (batch, state_dim).
        deterministic : If True, uses Beta mode instead of sample.
                        Gate is always continuous regardless of this flag.

        Returns
        -------
        PolicyOutput with action, alpha, beta, gate_prob set; mu, sigma = None.
        """
        feat = self.encoder(state)

        gate_prob = torch.clamp(torch.sigmoid(self.gate_head(feat)), 0.01, 0.99)
        gate      = gate_prob   # Soft gate: always continuous, never sampled

        alpha = torch.clamp(
            F.softplus(self.alpha_head(feat)) + self.config.alpha_min,
            self.config.alpha_min, self.config.alpha_max,
        )
        beta = torch.clamp(
            F.softplus(self.beta_head(feat)) + self.config.beta_min,
            self.config.beta_min, self.config.beta_max,
        )
        alpha = torch.nan_to_num(alpha, nan=self.config.alpha_min)
        beta  = torch.nan_to_num(beta,  nan=self.config.beta_min)

        if deterministic:
            continuous = alpha / (alpha + beta)
        else:
            continuous = torch.clamp(
                torch.distributions.Beta(alpha, beta).rsample(), 0.01, 0.99
            )

        action = gate * continuous
        return PolicyOutput(action=action, alpha=alpha, beta=beta, gate_prob=gate_prob)


# =============================================================================
# Factory
# =============================================================================

def build_policy_network(policy_type: str, config) -> nn.Module:
    """
    Instantiate the correct policy network from a type string.

    The type string is read from configs/reservoirs/<name>.yaml (policy_network
    key) by tune.py and train.py. It can be overridden at runtime by passing
    --policy_network <type> on the command line without editing any config file.

    Parameters
    ----------
    policy_type : str
        One of: "beta", "lognormal", "hardgating", "softgating".
        Case-insensitive; leading/trailing whitespace is stripped.
    config : BCConfig
        Must contain the fields required by the chosen network (see individual
        class docstrings). Fields not used by the chosen network are ignored.
        config.state_dim must equal len(columns.state) + 2 and is computed
        in tune.py from the reservoir YAML — not hardcoded.

    Returns
    -------
    nn.Module
        Instantiated (untrained) policy network on CPU. Call .to(device) after.

    Raises
    ------
    ValueError
        If policy_type is not one of the four recognised options.

    Examples
    --------
    >>> net = build_policy_network("beta", config)       # from reservoir YAML
    >>> net = build_policy_network("lognormal", config)  # CLI override
    """
    _REGISTRY = {
        "beta":       BetaActor,
        "lognormal":  LognormalActor,
        "hardgating": HardgatingActor,
        "softgating": SoftgatingActor,
    }
    key = policy_type.lower().strip()
    if key not in _REGISTRY:
        raise ValueError(
            f"Unknown policy_type '{policy_type}'. "
            f"Valid options: {list(_REGISTRY.keys())}. "
            f"Set policy_network in configs/reservoirs/<name>.yaml "
            f"or pass --policy_network <type> on the command line."
        )
    return _REGISTRY[key](config)
