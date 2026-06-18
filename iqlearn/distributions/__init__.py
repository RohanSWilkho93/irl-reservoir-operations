"""
iqlearn/distributions
======================
Registry + data-driven family selection for the four parametric policy families.

The user never picks a family.  `detect_family_pair` inspects the expert release
record and returns the candidate pair, enforcing the Paper-1 pairing:

  * release has zero days  ->  zero-inflated families  (HardGating, SoftGating)
  * release is continuous  ->  continuous families     (Beta, LogNormal)

BC tuning trains BOTH families in the returned pair and keeps the better policy;
that single winner warm-starts IQ-Learn.
"""

from __future__ import annotations

import numpy as np

from .base import PolicyDistribution
from .beta import BetaDistribution
from .lognormal import LogNormalDistribution
from .hardgating import HardGatingDistribution
from .softgating import SoftGatingDistribution

_REGISTRY = {
    "beta":       BetaDistribution,
    "lognormal":  LogNormalDistribution,
    "hardgating": HardGatingDistribution,
    "softgating": SoftGatingDistribution,
}

CONTINUOUS_FAMILIES = ("beta", "lognormal")
ZERO_INFLATED_FAMILIES = ("hardgating", "softgating")


def make_distribution(family: str, params: dict | None = None) -> PolicyDistribution:
    """Construct a distribution by name with its (optional) hyperparameters."""
    if family not in _REGISTRY:
        raise ValueError(f"Unknown policy family '{family}'. "
                         f"Choose from {sorted(_REGISTRY)}.")
    return _REGISTRY[family](**(params or {}))


def detect_family_pair(
    raw_actions,
    *,
    zero_frac_threshold: float = 0.01,
    zero_release_eps: float | None = None,
) -> tuple[str, str]:
    """
    Choose the candidate family pair from the expert release record.

    Parameters
    ----------
    raw_actions : array of observed releases in ENGINEERING units (train split).
    zero_frac_threshold : a reservoir is treated as zero-inflated when more than
        this fraction of days are at (near-)zero release.  Default 1%.
    zero_release_eps : a release at or below this counts as zero.  If None, uses
        max(1e-6, 1e-4 * max_release) so a single rounding artefact does not flip
        the decision.

    Returns
    -------
    (family_a, family_b) — the two families BC will tune and compare.
    """
    a = np.asarray(raw_actions, dtype=np.float64).ravel()
    if a.size == 0:
        return CONTINUOUS_FAMILIES
    if zero_release_eps is None:
        zero_release_eps = max(1e-6, 1e-4 * float(np.nanmax(a)))
    zero_frac = float(np.mean(a <= zero_release_eps))
    return ZERO_INFLATED_FAMILIES if zero_frac > zero_frac_threshold else CONTINUOUS_FAMILIES


__all__ = [
    "PolicyDistribution", "BetaDistribution", "LogNormalDistribution",
    "HardGatingDistribution", "SoftGatingDistribution",
    "make_distribution", "detect_family_pair",
    "CONTINUOUS_FAMILIES", "ZERO_INFLATED_FAMILIES",
]
