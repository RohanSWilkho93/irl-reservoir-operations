"""
iqlearn/expert_buffer.py
========================
Expert transition buffer for IQ-Learn.

utils/data.py already builds the full transition tuple per split
    states (N, D), actions (N,), next_states (N, D), dones (N,)
so this buffer is deliberately thin: it moves the TRAIN split onto the target
device once, precomputes importance-sampling weights, and serves minibatches.

Importance sampling (zero-inflation fix)
---------------------------------------
Reservoir release is heavily zero-inflated — most transitions sit near zero
release.  Uniform sampling would let that mass dominate the gradient and the
critic would learn "expert releases ~ 0 everywhere."  We upweight transitions
with larger (normalised) release so the rarer high-release behaviour is seen
often enough to shape Q there.

NOTE: this is an *uncorrected* reweighting — a deliberate emphasis, not
statistically-unbiased importance sampling.  The loss does NOT divide by the
weights, so the effective expert distribution is tilted toward higher releases
on purpose (matching the vanilla pipeline).  Set importance_sample=False at
sample time to recover uniform sampling.

Shapes: N = #transitions, B = batch, D = state_dim.
"""

from __future__ import annotations

from typing import NamedTuple, Sequence

import torch

from utils.data import Split


# =============================================================================
# Batch container
# =============================================================================

class Batch(NamedTuple):
    """One minibatch of expert transitions, all on the buffer's device."""
    states:      torch.Tensor   # (B, D)
    actions:     torch.Tensor   # (B,)    normalised release in [0, 1]
    next_states: torch.Tensor   # (B, D)
    dones:       torch.Tensor   # (B,)    float {0.0, 1.0}


# =============================================================================
# Expert buffer
# =============================================================================

class ExpertBuffer:
    """
    Device-resident expert transitions + importance-sampled minibatches.

    Parameters
    ----------
    split         : utils.data.Split — the TRAIN split (its states/actions/
                    next_states/dones are read).
    device        : target device string or torch.device.
    is_thresholds : ascending normalised-release cut points.
    is_weights    : sampling weight applied above each threshold (same length
                    as is_thresholds; later/higher thresholds overwrite lower).
    base_weight   : weight for transitions below the first threshold.

    The defaults reproduce the vanilla scheme: releases above 0.05 / 0.10 / 0.30
    (normalised) get weights 3 / 5 / 8; everything else gets 1.
    """

    def __init__(
        self,
        split:         Split,
        device:        str | torch.device,
        *,
        is_thresholds: Sequence[float] = (0.05, 0.10, 0.30),
        is_weights:    Sequence[float] = (3.0, 5.0, 8.0),
        base_weight:   float = 1.0,
    ):
        if len(is_thresholds) != len(is_weights):
            raise ValueError("is_thresholds and is_weights must have equal length.")

        self.device = torch.device(device)

        # ---- transitions -> device tensors (one-time copy) ----
        self.states      = torch.as_tensor(split.states,      dtype=torch.float32, device=self.device)
        self.actions     = torch.as_tensor(split.actions,     dtype=torch.float32, device=self.device)
        self.next_states = torch.as_tensor(split.next_states, dtype=torch.float32, device=self.device)
        # bool -> float so the loss's (1 - dones) mask is clean float arithmetic
        self.dones       = torch.as_tensor(split.dones,       dtype=torch.float32, device=self.device)

        self.size      = self.states.shape[0]
        self.state_dim = self.states.shape[1]

        # ---- precompute importance-sampling weights (actions never change) ----
        self._is_weights = self._build_is_weights(is_thresholds, is_weights, base_weight)

    # ---- importance weights ----------------------------------------------

    def _build_is_weights(
        self,
        thresholds:  Sequence[float],
        weights:     Sequence[float],
        base_weight: float,
    ) -> torch.Tensor:
        """
        Per-transition (unnormalised) sampling weight.  Cascading assignment:
        with ascending thresholds the highest matched threshold wins, so a
        transition is weighted by the band its release falls into.
        torch.multinomial normalises internally, so no need to normalise here.
        """
        w = torch.full((self.size,), float(base_weight),
                       dtype=torch.float32, device=self.device)
        for thr, wt in zip(thresholds, weights):
            w[self.actions > thr] = float(wt)
        return w

    # ---- sampling ---------------------------------------------------------

    def sample(
        self,
        batch_size:        int,
        importance_sample: bool = True,
        generator:         torch.Generator | None = None,
    ) -> Batch:
        """
        Draw a minibatch of transitions (with replacement).

        importance_sample=True : draw ~ precomputed high-release weights.
        importance_sample=False: draw uniformly.

        A `generator` (if given) must live on the buffer's device.
        """
        if importance_sample:
            idx = torch.multinomial(
                self._is_weights, batch_size, replacement=True, generator=generator
            )
        else:
            idx = torch.randint(
                0, self.size, (batch_size,), device=self.device, generator=generator
            )

        return Batch(
            states      = self.states[idx],
            actions     = self.actions[idx],
            next_states = self.next_states[idx],
            dones       = self.dones[idx],
        )

    def __len__(self) -> int:
        return self.size