"""
airl/bc_tuning.py
=================
AIRL's own Stage-1 Behavioral-Cloning tuner.  It is AIRL-specific (its own run
folder results/<reservoir>/airl/<run_id>/ and its own `bc_tuning:` config block),
but reuses the proven BC internals from iqlearn.bc_tuning so the BC math is shared
and bug-for-bug identical: data-driven family pair, Optuna over both, keep the
better, save bc_policy.pt + bc_best_config.json.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from iqlearn.bc_tuning import _tune_one_family, _save_run_args
from iqlearn.distributions import detect_family_pair
from iqlearn.utils.runs import _resolve_run_id

POLICY_TYPE = "parametric"


def run_airl_bc_tuning(*, reservoir, res_cfg, res_cfg_path, algo_cfg, data,
                       device_str, run_id=None, cli_args=None) -> dict:
    """Detect the family pair, tune both, keep the winner — into the AIRL run folder."""
    bc = algo_cfg["bc_tuning"]
    n_jobs = (bc["runtime"]["num_workers"] if bc["runtime"]["num_workers"] is not None
              else bc["optuna"]["n_jobs"])
    n_trials = bc["optuna"]["n_trials"]

    sel = bc.get("selection", {}) or {}
    families = detect_family_pair(
        data.train.raw_actions,
        zero_frac_threshold=float(sel.get("zero_frac_threshold", 0.01)),
        zero_release_eps=sel.get("zero_release_eps", None))
    _eps = sel.get("zero_release_eps")
    if _eps is None:
        _eps = max(1e-6, 1e-4 * float(np.max(data.train.raw_actions)))
    zero_frac = float(np.mean(np.asarray(data.train.raw_actions) <= _eps))
    print(f"  state_dim={data.state_dim}  train={len(data.train.states)}  val={len(data.val.states)}  "
          f"test={len(data.test.states)}  device={device_str}")
    print(f"  zero-release fraction (train) = {zero_frac:.3%}  ->  candidate families: {families}")

    base_dir = _ROOT / "results" / reservoir / "airl"
    run_id, folder = _resolve_run_id(base_dir, run_id)
    folder.mkdir(parents=True, exist_ok=True)
    print(f"\nRun folder : {folder.name}  (run_id={run_id})")

    best_score, best_cfg, best_state, best_family = -float("inf"), None, None, None
    for family in families:
        score, cfg, state = _tune_one_family(family, algo_cfg, data, device_str, n_trials, n_jobs, reservoir)
        if state is not None and score > best_score:
            best_score, best_cfg, best_state, best_family = score, cfg, state, family
    if best_cfg is None:
        sys.exit("All BC trials failed for every candidate family.")

    print(f"\n  ✓ Winning family: {best_family}  (val score {best_score:.4f})")
    best_config_path = folder / "bc_best_config.json"
    with open(best_config_path, "w") as f:
        json.dump({"reservoir": reservoir, "policy_type": POLICY_TYPE, "policy_family": best_family,
                   "candidate_families": list(families), "best_score": best_score,
                   "config": asdict(best_cfg)}, f, indent=2)
    policy_path = folder / "bc_policy.pt"
    torch.save({"state_dict": best_state, "config": asdict(best_cfg),
                "policy_type": POLICY_TYPE, "policy_family": best_family}, policy_path)
    print(f"Best config saved → {best_config_path}\nPretrained policy saved → {policy_path}\n")
    _save_run_args(folder, reservoir, run_id, cli_args, families, best_family)

    return {"run_folder": folder, "run_id": run_id, "best_config": best_cfg,
            "best_score": best_score, "policy_path": policy_path,
            "policy_family": best_family, "candidate_families": list(families)}
