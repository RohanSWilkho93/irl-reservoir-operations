"""
airl/scoring.py
===============
The AIRL validation objective (Optuna maximises this):

  score = 0.50 * disc_balance
        + 0.125 * (release_corr+1)/2  + 0.125 * (1 - release_nRMSE)
        + 0.125 * (storage_corr+1)/2  + 0.125 * (1 - storage_nRMSE)

`disc_balance = max(0, 1 - |expert_acc-0.5| - |policy_acc-0.5|)` rewards a
discriminator at the adversarial equilibrium (it can no longer separate expert
from policy).  The corr/nRMSE terms come from a closed-loop rollout.  nRMSE is
range-based (utils.metrics.nrmse), matching the IQ-Learn results convention.
"""

from __future__ import annotations

import math
from typing import Dict

import numpy as np

from utils.metrics import safe_pearsonr, nrmse


def composite_score(release_corr, storage_corr, release_nrmse, storage_nrmse,
                    expert_acc, policy_acc) -> float:
    vals = [release_corr, storage_corr, release_nrmse, storage_nrmse, expert_acc, policy_acc]
    if any((v is None or (isinstance(v, float) and math.isnan(v))) for v in vals):
        return 0.0
    rc = (release_corr + 1) / 2
    sc = (storage_corr + 1) / 2
    rn = max(0.0, min(1.0, 1 - release_nrmse))
    sn = max(0.0, min(1.0, 1 - storage_nrmse))
    disc = max(0.0, 1 - abs(expert_acc - 0.5) - abs(policy_acc - 0.5))
    return float(0.50 * disc + 0.125 * rc + 0.125 * rn + 0.125 * sc + 0.125 * sn)


def rollout_fidelity(traj: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Corr + nRMSE of a closed-loop rollout (iqlearn ReservoirRollout output)."""
    rc, _ = safe_pearsonr(traj["obs_release"], traj["sim_release"])
    sc, _ = safe_pearsonr(traj["obs_storage"], traj["sim_storage"])
    return {"release_corr": float(rc), "storage_corr": float(sc),
            "release_nrmse": float(nrmse(traj["obs_release"], traj["sim_release"])),
            "storage_nrmse": float(nrmse(traj["obs_storage"], traj["sim_storage"]))}
