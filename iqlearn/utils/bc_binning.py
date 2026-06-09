"""
bc_binning.py (For Behavioral Cloning before AIRL and IQLearn)
=============
Quantile / log bin construction for the categorical Behavioral Cloning head.

Pure NumPy — no torch — so it can be imported by both the trainer and the
metrics without dragging in the network module.  Used by:
    - tune.py     : build_bins() to freeze the grid; assign_bins() to label
                    training actions for the cross-entropy target.
    - metrics.py  : assign_bins() / bin_means for distributional metrics.

The categorical head discretises normalised release (in [0, 1]) into K bins.
This module turns the pooled TRAINING releases plus (n_bins, binning) into:
    edges     : (K+1,) strictly increasing bin boundaries
    bin_means : (K,)   empirical mean release within each bin
Both are frozen into bc_best_config.json so inference reconstructs the exact
grid the model trained on.

Conventions
-----------
- Bins are LEFT-INCLUSIVE: bin k covers [edges[k], edges[k+1]); the top bin is
  closed on the right via clamping.
- assign_bins CLAMPS out-of-range values to the first / last bin.  Training
  releases are in [0, 1] by construction, but val/test extremes can fall
  outside (they are never clipped upstream), so they still receive a valid
  label and never raise.
- build_bins ALWAYS returns exactly n_bins bins (edges length n_bins+1), even
  under tied values such as a zero-inflation spike.  Tied / non-increasing
  quantile edges are nudged apart by a tiny epsilon to preserve K; bins that
  end up with no training data take the bin centre as their mean.

Support-ceiling note
--------------------
The grid is bounded by the training release range, so the head cannot predict a
release above the largest training release.  
"""

from __future__ import annotations

import numpy as np

# Minimum bin width used to keep edges strictly increasing under tied values.
_EPS = 1e-6


# ---------------------------------------------------------------------------
# Bin assignment
# ---------------------------------------------------------------------------

def assign_bins(values, edges) -> np.ndarray:
    """
    Map continuous values to integer bin indices in [0, K-1].

    Parameters
    ----------
    values : array-like, shape (N,)
        Values to bin (normalised releases).
    edges  : array-like, shape (K+1,)
        Strictly increasing bin boundaries from build_bins().

    Returns
    -------
    np.ndarray, shape (N,), dtype int64
        Bin index per value.  value <= edges[0] -> 0; value >= edges[-1] ->
        K-1; out-of-range values are clamped (never raises).
    """
    values = np.asarray(values, dtype=np.float64).ravel()
    edges  = np.asarray(edges,  dtype=np.float64).ravel()
    n_bins = len(edges) - 1

    # side="right": a value lands in the bin whose left edge it meets or exceeds.
    idx = np.searchsorted(edges, values, side="right") - 1
    return np.clip(idx, 0, n_bins - 1).astype(np.int64)


# ---------------------------------------------------------------------------
# Edge construction (per strategy)
# ---------------------------------------------------------------------------

def _quantile_edges(actions: np.ndarray, n_bins: int) -> np.ndarray:
    """K+1 edges at evenly spaced quantiles — each bin holds ~equal count."""
    qs = np.linspace(0.0, 1.0, n_bins + 1)
    return np.quantile(actions, qs)


def _log_edges(actions: np.ndarray, n_bins: int) -> np.ndarray:
    """
    K+1 log-spaced edges, giving fine resolution at small releases.

    The smallest positive value anchors the log spacing; the first edge is then
    lowered to the data minimum so zeros / near-zeros fall into bin 0.
    """
    amin = float(actions.min())
    amax = float(actions.max())
    pos  = actions[actions > 0.0]
    lo   = float(pos.min()) if pos.size else (amax / 1e3)

    if not (lo < amax):                       # degenerate: all equal / single value
        lo = (amax / 1e3) if amax > 0 else 1.0

    edges = np.geomspace(lo, amax, n_bins + 1)
    edges[0] = amin                           # bin 0 absorbs the minimum (possibly 0)
    return edges


def _enforce_strictly_increasing(edges: np.ndarray) -> np.ndarray:
    """
    Nudge tied / non-increasing edges apart by _EPS.

    Preserves the number of edges, so build_bins always returns exactly K bins
    even when a mass point (e.g. a zero spike) collapses several quantile edges
    onto the same value.
    """
    edges = np.array(edges, dtype=np.float64)
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + _EPS
    return edges


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def build_bins(actions, n_bins: int, binning: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Build the frozen bin grid for the categorical head.

    Parameters
    ----------
    actions : array-like, shape (N,)
        Pooled normalised TRAINING releases (in [0, 1] by construction).
    n_bins  : int
        Number of bins K (>= 1).  Equals the network's output-layer width.
    binning : str
        "quantile" (equal-count edges) or "log" (log-spaced edges).

    Returns
    -------
    edges     : np.ndarray (K+1,)  strictly increasing bin boundaries.
    bin_means : np.ndarray (K,)    empirical mean release per bin; the bin
                                   centre is used for any bin with no training
                                   data (e.g. epsilon-nudged bins beside a spike).
    """
    actions = np.asarray(actions, dtype=np.float64).ravel()
    if actions.size == 0:
        raise ValueError("build_bins received an empty actions array.")
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}.")

    if binning == "quantile":
        edges = _quantile_edges(actions, n_bins)
    elif binning == "log":
        edges = _log_edges(actions, n_bins)
    else:
        raise ValueError(
            f"Unknown binning strategy: {binning!r}. Expected 'quantile' or 'log'."
        )

    edges = _enforce_strictly_increasing(edges)

    # Empirical mean per bin; bin centre where a bin has no training data.
    idx       = assign_bins(actions, edges)
    counts    = np.bincount(idx, minlength=n_bins).astype(np.float64)
    sums      = np.bincount(idx, weights=actions, minlength=n_bins)
    centers   = 0.5 * (edges[:-1] + edges[1:])
    bin_means = np.where(counts > 0, sums / np.maximum(counts, 1.0), centers)

    return edges, bin_means