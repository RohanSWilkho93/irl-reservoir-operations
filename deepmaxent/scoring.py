"""
deepmaxent/scoring.py
=====================
Metrics + the composite "unified score" the Optuna objective maximises.

The unified score (validation split) combines the IRL objective (state-visitation
frequency difference) with closed-loop behavioural fidelity:

    score = 0.50  * (1 - SVF_diff_norm)
          + 0.125 * (release_corr+1)/2   + 0.125 * (storage_corr+1)/2
          + 0.125 * (1 - release_nRMSE/2) + 0.125 * (1 - storage_nRMSE/2)

NOTE on nRMSE: Deep MaxEnt scores nRMSE = RMSE / std(observed) (NOT divided by
range), so we keep a local std-based `compute_nrmse` to match the Paper-1 metric
exactly rather than reuse utils.metrics.nrmse (which divides by range).
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

from utils.metrics import safe_pearsonr, rmse as compute_rmse   # identical definitions


def compute_nrmse(y_true, y_pred) -> float:
    std_true = np.std(y_true)
    return np.inf if std_true < 1e-10 else compute_rmse(y_true, y_pred) / std_true


def compute_mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def svf_metrics(mu_expert: np.ndarray, mu_learned: np.ndarray) -> Tuple[float, float, float]:
    """L1 difference, overlap mass, and overlap percentage of two SVF tensors."""
    diff = float(np.abs(mu_expert - mu_learned).sum())
    overlap = float(np.minimum(mu_expert, mu_learned).sum())
    overlap_pct = 100.0 * overlap / (mu_expert.sum() + 1e-8)
    return diff, overlap, overlap_pct


def compute_unified_score(svf_diff, release_corr, storage_corr,
                          release_nrmse, storage_nrmse,
                          svf_diff_bounds=(0, 1000),
                          w_svf=0.50, w_release_corr=0.125, w_storage_corr=0.125,
                          w_release_nrmse=0.125, w_storage_nrmse=0.125) -> Tuple[float, Dict]:
    svf_min, svf_max = svf_diff_bounds
    svf_normalized = 1.0 - (np.clip(svf_diff, svf_min, svf_max) - svf_min) / (svf_max - svf_min + 1e-8)
    rc = (release_corr + 1.0) / 2.0
    sc = (storage_corr + 1.0) / 2.0
    rn = 1.0 - np.clip(release_nrmse, 0, 2.0) / 2.0
    sn = 1.0 - np.clip(storage_nrmse, 0, 2.0) / 2.0
    score = (w_svf * svf_normalized + w_release_corr * rc + w_storage_corr * sc
             + w_release_nrmse * rn + w_storage_nrmse * sn)
    return float(score), {
        "svf_normalized": float(svf_normalized),
        "release_corr_normalized": float(rc), "storage_corr_normalized": float(sc),
        "release_nrmse_normalized": float(rn), "storage_nrmse_normalized": float(sn),
    }
