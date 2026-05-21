"""
utils/metrics.py
================
Primitive metric functions shared across all algorithm modules.

Each algorithm's tune.py imports these to build its own composite score.
Each algorithm's generate_results.py imports these to report final test metrics.

Usage context
-------------
rmse        : Called on NORMALIZED [0, 1] values during Optuna tuning.
              Because data is already in [0, 1], RMSE is already scale-free
              and equivalent to nRMSE — no further normalization needed.

nrmse       : Called on DENORMALIZED (original engineering units) values
              when reporting final test results. Dividing by the observed
              range makes numbers comparable across reservoirs of vastly
              different scales (e.g., Garrison vs. Dexter storage).

safe_pearsonr : Scale-independent — used identically during tuning and
                evaluation. Handles constant arrays, NaN, and inf safely.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr
from typing import Tuple


def safe_pearsonr(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    """
    Pearson correlation with full edge-case handling.

    Returns (0.0, 1.0) — correlation of zero, p-value of 1 — whenever
    the result would be undefined or numerically unreliable.

    Parameters
    ----------
    x, y : array-like
        Arrays to correlate. Must have the same length.

    Returns
    -------
    corr : float   Pearson r in [-1, 1]
    p    : float   Two-tailed p-value
    """
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)

    if np.std(x) < 1e-10 or np.std(y) < 1e-10:
        return 0.0, 1.0
    if np.any(np.isnan(x)) or np.any(np.isnan(y)):
        return 0.0, 1.0
    if np.any(np.isinf(x)) or np.any(np.isinf(y)):
        return 0.0, 1.0

    try:
        corr, p = pearsonr(x, y)
        return (0.0, 1.0) if np.isnan(corr) else (float(corr), float(p))
    except Exception:
        return 0.0, 1.0


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Root Mean Squared Error.

    Intended for NORMALIZED [0, 1] values during Optuna tuning.
    On normalized data this is already scale-free (equivalent to nRMSE).

    Parameters
    ----------
    y_true, y_pred : array-like
        Ground truth and predicted values.

    Returns
    -------
    float   RMSE >= 0
    """
    y_true = np.asarray(y_true, dtype=float).flatten()
    y_pred = np.asarray(y_pred, dtype=float).flatten()
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def nrmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Normalized Root Mean Squared Error — normalized by observed range.

    Intended for DENORMALIZED (original engineering units) values when
    reporting final test results. Dividing by (max - min) of the observed
    series makes the metric comparable across reservoirs with different
    physical scales.

    Returns 0.0 when the observed range is effectively zero (constant series).

    Parameters
    ----------
    y_true : array-like   Observed values in original engineering units.
    y_pred : array-like   Predicted values in original engineering units.

    Returns
    -------
    float   nRMSE >= 0  (lower is better)
    """
    y_true = np.asarray(y_true, dtype=float).flatten()
    y_pred = np.asarray(y_pred, dtype=float).flatten()

    obs_range = y_true.max() - y_true.min()
    if obs_range < 1e-8:
        return 0.0

    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)) / obs_range)
