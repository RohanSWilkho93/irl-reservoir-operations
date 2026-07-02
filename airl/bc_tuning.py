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

    # ---- distributed (local shared-journal) options ----
    storage = getattr(cli_args, "storage", None)
    role = getattr(cli_args, "role", None) or "full"
    sname_prefix = getattr(cli_args, "study_name", None)
    if storage is not None and run_id is None:
        sys.exit("ERROR: --storage requires an explicit --run_id so every worker + the "
                 "finalize step target the same run folder.")

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
        sname = f"{sname_prefix}_{family}" if sname_prefix else f"{reservoir}_airlbc_{family}"
        score, cfg, state = _tune_one_family(family, algo_cfg, data, device_str, n_trials, n_jobs, reservoir,
                                             storage=storage, study_name=sname, role=role)
        if state is not None and score > best_score:
            best_score, best_cfg, best_state, best_family = score, cfg, state, family

    if role == "worker":
        print(f"\n  [worker] AIRL BC trials contributed for {families}; exiting (no policy saved).")
        return {"run_folder": folder, "run_id": run_id, "role": "worker",
                "candidate_families": list(families)}

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


# =============================================================================
# CLI / standalone (for the multi-process distributed-tuning workflow)
# =============================================================================

def _parse_args():
    import argparse
    p = argparse.ArgumentParser(description="AIRL Stage-1 BC tuning (data-driven family selection).")
    p.add_argument("--reservoir", required=True)
    p.add_argument("--data_path", default=None)
    p.add_argument("--date_column", default=None)
    p.add_argument("--state_variables", nargs="+", default=None)
    p.add_argument("--use_month_encoding", type=lambda s: s.lower() == "true", default=None)
    p.add_argument("--split_train", type=int, default=None)
    p.add_argument("--split_val", type=int, default=None)
    p.add_argument("--split_test", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--n_trials", type=int, default=None)
    p.add_argument("--run_id", type=int, default=None)
    p.add_argument("--storage", default=None, help="Local JournalFileStorage path; requires --run_id.")
    p.add_argument("--study_name", default=None, help="Shared study-name prefix (default: <reservoir>_airlbc).")
    p.add_argument("--role", choices=["full", "worker", "finalize"], default="full")
    p.add_argument("--save-config", dest="save_config", action="store_true",
                   help="Persist CLI overrides back into the YAML config files "
                        "(default: overrides apply to this run only).")
    return p.parse_args()


def main():
    import yaml
    from utils.data import load_reservoir_data
    from iqlearn.utils.runs import _resolve_device
    from iqlearn.bc_tuning import _apply_cli_overrides
    args = _parse_args()
    res_cfg_path = _ROOT / "configs" / "reservoirs" / f"{args.reservoir}.yaml"
    algo_cfg_path = _ROOT / "configs" / "algorithms" / "airl.yaml"
    if not res_cfg_path.exists():
        sys.exit(f"Reservoir config not found: {res_cfg_path}")
    res_cfg = yaml.safe_load(open(res_cfg_path))
    algo_cfg = yaml.safe_load(open(algo_cfg_path))
    _apply_cli_overrides(args, res_cfg, algo_cfg, res_cfg_path, algo_cfg_path,
                         save_config=args.save_config)
    device_str = _resolve_device(algo_cfg["bc_tuning"]["runtime"]["device"])
    data = load_reservoir_data(res_cfg, res_cfg_path)
    run_airl_bc_tuning(reservoir=args.reservoir, res_cfg=res_cfg, res_cfg_path=res_cfg_path,
                       algo_cfg=algo_cfg, data=data, device_str=device_str,
                       run_id=args.run_id, cli_args=args)


if __name__ == "__main__":
    main()
