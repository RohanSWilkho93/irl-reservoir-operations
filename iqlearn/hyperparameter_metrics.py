"""
iqlearn/hyperparameter_metrics.py
=================================
Composite score + validity check for IQ-Learn hyperparameter tuning.

The Optuna objective maximises a weighted composite of five terms:

  reward quality (env-free, on TRAIN expert transitions)
    expert_advantage    : do expert actions get higher Q than random actions?
    q_smoothness        : is Q temporally smooth over consecutive expert states?
  behavioural fidelity (CLOSED-LOOP rollout, on VAL)
    prediction_fidelity : roll the policy through mass balance and score r + nRMSE
                          for BOTH storage and release vs the observed val series.
  robustness (env-free, on TRAIN)
    entropy             : mean policy entropy, normalised to [0, 1] by log K.
    action_diversity    : std / mean of deterministic actions (collapse detector).

Only prediction_fidelity uses the simulator (iqlearn.environment.ReservoirRollout);
the rest are env-free.  All functions return plain floats and run under no_grad.

Shapes: B = batch, K = n_bins, T = #val steps.
"""

from __future__ import annotations

import math
from typing import Any

import torch

from utils.metrics import rmse, safe_pearsonr
from iqlearn.utils import distribution


# =============================================================================
# Env-free metrics (TRAIN)
# =============================================================================

@torch.no_grad()
def expert_advantage(agent, states: torch.Tensor, actions: torch.Tensor) -> float:
    """(q_expert - q_random) / (|q_expert| + |q_random| + eps), clamped to [0, 1]."""
    q_expert = torch.min(*agent.critic(states, actions)).mean()
    rand = torch.rand_like(actions)
    q_random = torch.min(*agent.critic(states, rand)).mean()
    denom = q_expert.abs() + q_random.abs() + 1e-6
    return float(((q_expert - q_random) / denom).clamp(0.0, 1.0).item())


@torch.no_grad()
def q_smoothness(agent, states: torch.Tensor, actions: torch.Tensor) -> float:
    """1 / (1 + mean|ΔQ|) over CONSECUTIVE expert states, in (0, 1]."""
    q = torch.min(*agent.critic(states, actions))
    dq = (q[1:] - q[:-1]).abs().mean()
    return float((1.0 / (1.0 + dq)).item())


@torch.no_grad()
def policy_entropy(agent, states: torch.Tensor) -> float:
    """Mean categorical entropy normalised to [0, 1] by log K."""
    logits = agent.actor(states)
    k = logits.shape[-1]
    h = distribution.entropy(logits).mean()
    return float((h / math.log(k)).clamp(0.0, 1.0).item())


@torch.no_grad()
def action_diversity(agent, states: torch.Tensor) -> float:
    """std / mean of deterministic actions across states, in [0, 1]. Detects collapse."""
    acts = agent.select_action(states, deterministic=True)
    div = acts.std() / (acts.abs().mean() + 1e-6)
    return float(div.clamp(0.0, 1.0).item())


# =============================================================================
# Closed-loop fidelity (VAL, via mass balance)
# =============================================================================

def _nrmse(obs, sim) -> float:
    """rmse / observed-range (scale-free). 1.0 (worst) if the range is ~0."""
    rng = float(obs.max() - obs.min())
    if rng < 1e-8:
        return 1.0
    return float(rmse(obs, sim) / rng)


@torch.no_grad()
def prediction_fidelity(agent, env) -> tuple[float, dict[str, float]]:
    """
    Roll the policy through the mass-balance simulator and score storage AND
    release.  For each: fidelity = (pearson_r + (1 - nRMSE)) / 2; the returned
    scalar is their mean.  Both r and nRMSE are scale-free, so engineering-unit
    trajectories score the same as normalised ones.
    """
    traj = env.rollout(agent)

    rel_r, _ = safe_pearsonr(traj["obs_release"], traj["sim_release"])
    sto_r, _ = safe_pearsonr(traj["obs_storage"], traj["sim_storage"])
    rel_nrmse = _nrmse(traj["obs_release"], traj["sim_release"])
    sto_nrmse = _nrmse(traj["obs_storage"], traj["sim_storage"])

    rel_fid = (rel_r + (1.0 - rel_nrmse)) / 2.0
    sto_fid = (sto_r + (1.0 - sto_nrmse)) / 2.0
    fidelity = (rel_fid + sto_fid) / 2.0

    sub = {
        "release_r":        float(rel_r),
        "storage_r":        float(sto_r),
        "release_nrmse":    float(rel_nrmse),
        "storage_nrmse":    float(sto_nrmse),
        "release_fidelity": float(rel_fid),
        "storage_fidelity": float(sto_fid),
    }
    return float(fidelity), sub


# =============================================================================
# Composite score
# =============================================================================

_SCORED_KEYS = (
    "expert_advantage",
    "q_smoothness",
    "prediction_fidelity",
    "entropy",
    "action_diversity",
)


def composite_score(
    agent,
    buffer,
    env,
    weights:   dict[str, float],
    *,
    n_samples: int = 1000,
) -> tuple[float, dict[str, Any]]:
    """
    Weighted composite.  Env-free terms sample TRAIN transitions from `buffer`;
    prediction_fidelity rolls the policy through `env` (the VAL simulator).

    Returns (score, metrics); metrics also carries the fidelity sub-components
    (release_r, storage_r, ...) for logging and "composite_score".
    """
    n = min(n_samples, buffer.size)
    idx = torch.randint(0, buffer.size, (n,), device=buffer.device)
    s_rand, a_rand = buffer.states[idx], buffer.actions[idx]   # advantage/entropy/diversity
    s_ord,  a_ord  = buffer.states[:n],  buffer.actions[:n]    # chronological for smoothness

    pf, pf_sub = prediction_fidelity(agent, env)

    m: dict[str, Any] = {
        "expert_advantage":    expert_advantage(agent, s_rand, a_rand),
        "q_smoothness":        q_smoothness(agent, s_ord, a_ord),
        "prediction_fidelity": pf,
        "entropy":             policy_entropy(agent, s_rand),
        "action_diversity":    action_diversity(agent, s_rand),
    }

    # Non-finite -> 0 so one bad term can't NaN the score.
    for key in _SCORED_KEYS:
        if not math.isfinite(m[key]):
            m[key] = 0.0

    score = sum(weights.get(key, 0.0) * m[key] for key in _SCORED_KEYS)

    m.update(pf_sub)                 # logged, not weighted
    m["composite_score"] = score
    return score, m


# =============================================================================
# Validity guard (env-free)
# =============================================================================

@torch.no_grad()
def is_valid_solution(agent, buffer, *, n_samples: int = 500) -> tuple[bool, str]:
    """
    Reject degenerate solutions (NaN/Inf or exploded Q, collapsed policy).
    Relaxed on purpose — the composite score does fine-grained ranking; this
    only catches outright failures so they score ~0 instead of crashing.
    """
    n = min(n_samples, buffer.size)
    idx = torch.randint(0, buffer.size, (n,), device=buffer.device)
    q = torch.min(*agent.critic(buffer.states[idx], buffer.actions[idx]))

    if not torch.isfinite(q).all():
        return False, "Q contains NaN or Inf"
    q_std = q.std().item()
    if q_std > 1e4:
        return False, f"Q exploded (std={q_std:.1f})"

    pred = agent.select_action(buffer.states[idx], deterministic=True)
    if not torch.isfinite(pred).all():
        return False, "Predicted actions contain NaN or Inf"
    if pred.std().item() < 1e-6:
        return False, "Policy collapsed to a constant action"

    return True, f"valid (Q std={q_std:.2f})"