"""
iqlearn/utils/distribution.py
=============================
Math for the quantile-binned categorical policy head.

The policy network outputs K raw logits per state.  softmax(logits) is a
per-state distribution over K bins that partition the normalised release axis
[0, 1]; bin k is summarised by its empirical mean release ``bin_means[k]`` and
bounded by ``bin_edges[k] .. bin_edges[k + 1]``.

These helpers turn raw logits (+ the frozen bin grid) into the quantities the
IQ-Learn losses, metrics, and rollouts need.  Everything is EXACT: because the
action space is discretised into K bins, expectations over the policy are SUMS
over bins, not Monte-Carlo estimates.  No sampling is needed for training; the
only sampling helper (`sample`) is for stochastic rollouts at evaluation time.

All functions are stateless and torch-only.  The frozen grid is supplied by the
caller as tensors (from bc_best_config.json):

    bin_means : (K,)    empirical mean release within each bin
    bin_edges : (K + 1,) strictly increasing bin boundaries on [0, 1]

Shapes use B = batch, K = n_bins.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Probabilities
# ---------------------------------------------------------------------------

def log_probs(logits: torch.Tensor) -> torch.Tensor:
    """log softmax over bins.  (B, K) -> (B, K).  Numerically stable."""
    return F.log_softmax(logits, dim=-1)


def probs(logits: torch.Tensor) -> torch.Tensor:
    """softmax over bins.  (B, K) -> (B, K)."""
    return F.softmax(logits, dim=-1)


# ---------------------------------------------------------------------------
# Entropy and expected value
# ---------------------------------------------------------------------------

def entropy(logits: torch.Tensor) -> torch.Tensor:
    """
    Categorical entropy  H(p) = -sum_k p_k log p_k.   (B, K) -> (B,)

    Closed form — this is the entropy bonus inside the soft value, computed
    exactly.  softmax keeps every p_k > 0, so p * log p is finite everywhere.
    """
    logp = F.log_softmax(logits, dim=-1)
    p    = logp.exp()
    return -(p * logp).sum(dim=-1)


def expected_value(logits: torch.Tensor, bin_means: torch.Tensor) -> torch.Tensor:
    """
    Deterministic point prediction  E[a | s] = sum_k p_k * bin_means[k].
    (B, K), (K,) -> (B,)

    Identical to the BC head's inference rule, so IQ action-fidelity scoring
    matches BC scoring.  Uses EMPIRICAL bin means (not bin centres), which is
    more faithful on the skewed release distribution.
    """
    return probs(logits) @ bin_means


# ---------------------------------------------------------------------------
# Soft value  (the core IQ-Learn primitive)
# ---------------------------------------------------------------------------

def soft_value(q_per_bin: torch.Tensor, logits: torch.Tensor,
               alpha: float) -> torch.Tensor:
    """
    Exact soft value
        V(s) = E_{a~pi}[ Q(s, a) - alpha * log pi(a | s) ]
             = sum_k p_k Q(s, a_k)  +  alpha * H(p)
    (B, K), (B, K), float -> (B,)

    Parameters
    ----------
    q_per_bin : (B, K)  critic Q for each state evaluated at every bin's mean
                        action a_k = bin_means[k].
    logits    : (B, K)  raw policy logits for the same states.
    alpha     :         entropy temperature (alpha_entropy).

    Because the policy is a finite categorical, the expectation over actions is
    an exact weighted sum over bins — no sampling, no n_samples hyperparameter,
    no Monte-Carlo variance.  This replaces the Beta version's sampled estimate.
    """
    logp       = F.log_softmax(logits, dim=-1)
    p          = logp.exp()
    expected_q = (p * q_per_bin).sum(dim=-1)     # sum_k p_k Q_k
    ent        = -(p * logp).sum(dim=-1)         # H(p)
    return expected_q + alpha * ent


# ---------------------------------------------------------------------------
# KL divergence  (BC anchor term)
# ---------------------------------------------------------------------------

def kl(logits_p: torch.Tensor, logits_q: torch.Tensor) -> torch.Tensor:
    """
    KL(p || q) = sum_k p_k (log p_k - log q_k).   (B, K), (B, K) -> (B,)

    Closed form.  Used for the BC-anchor term  lambda_bc * KL(actor || bc).
    Valid ONLY because the actor and the BC prior share the SAME bin grid — the
    IQ actor is warm-started from bc_policy.pt and inherits its bins, so the two
    categoricals are defined over identical support.
    """
    logp = F.log_softmax(logits_p, dim=-1)
    logq = F.log_softmax(logits_q, dim=-1)
    p    = logp.exp()
    return (p * (logp - logq)).sum(dim=-1)


# ---------------------------------------------------------------------------
# Bin assignment  (torch mirror of bc_binning.assign_bins)
# ---------------------------------------------------------------------------

def assign_bins(actions: torch.Tensor, bin_edges: torch.Tensor) -> torch.Tensor:
    """
    Map continuous normalised actions to bin indices.  (B,), (K+1,) -> (B,) long

    Mirrors iqlearn.utils.bc_binning.assign_bins (searchsorted side='right'
    minus 1, clamped to [0, K-1]) but stays on-device in torch.  Out-of-range
    val/test actions (possible since bounds come from training only) clamp to
    the edge bins.
    """
    k   = bin_edges.numel() - 1
    idx = torch.searchsorted(bin_edges, actions, right=True) - 1
    return idx.clamp_(0, k - 1)


def action_log_prob(logits: torch.Tensor, actions: torch.Tensor,
                    bin_edges: torch.Tensor) -> torch.Tensor:
    """
    Log-likelihood the policy assigns to each continuous action, via its bin:
        log pi(a | s) = log p_{bin(a)}
    (B, K), (B,), (K+1,) -> (B,)

    Diagnostic / optional NLL term; the exact IQ losses do not require it
    (the soft value and entropy are computed by enumeration, not from a sampled
    expert log-prob).
    """
    logp = F.log_softmax(logits, dim=-1)                # (B, K)
    idx  = assign_bins(actions, bin_edges)              # (B,)
    return logp.gather(1, idx.unsqueeze(1)).squeeze(1)  # (B,)


# ---------------------------------------------------------------------------
# Sampling  (evaluation-time rollouts only)
# ---------------------------------------------------------------------------

def sample(logits: torch.Tensor, bin_edges: torch.Tensor,
           generator: torch.Generator | None = None) -> torch.Tensor:
    """
    Stochastic action via two-level sampling:
        1. pick a bin   k ~ Categorical(p)
        2. dequantise   a ~ Uniform(edge_k, edge_{k+1})
    (B, K), (K+1,) -> (B,)   continuous release on [0, 1]

    The within-bin jitter recovers a continuous action from the discrete head.
    Used ONLY for evaluation-time rollouts (e.g. the Monte-Carlo fans in
    generate_results); training never samples — all policy expectations are
    enumerated exactly.
    """
    p   = F.softmax(logits, dim=-1)                                           # (B, K)
    idx = torch.multinomial(p, num_samples=1, generator=generator).squeeze(1) # (B,)
    lo  = bin_edges[idx]                                                       # (B,)
    hi  = bin_edges[idx + 1]                                                   # (B,)
    u   = torch.rand(idx.shape, device=logits.device, generator=generator)
    return lo + u * (hi - lo)