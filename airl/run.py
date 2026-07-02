"""
airl/run.py
===========
End-to-end AIRL driver, on a single data load and a single shared run folder
results/<reservoir>/airl/<run_id>/ :

    BC tuning (save best policy)  ->  discriminator warm-up + joint adversarial
    tuning (save best agent)      ->  result figures

Stage 1 is AIRL's own BC tuner (data-driven family auto-selection); Stage 2
warm-starts the actor from bc_policy.pt and trains the AIRL discriminator + PPO.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import yaml

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils.data import load_reservoir_data
from iqlearn.utils.runs import _resolve_device, _writeback_yaml
from airl.bc_tuning import run_airl_bc_tuning
from airl.airl_tuning import run_airl_tuning
from airl.results import run_generate_results


def _parse_args():
    p = argparse.ArgumentParser(
        description="End-to-end AIRL: BC tuning -> adversarial tuning -> results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--reservoir", required=True)
    # shared data load
    p.add_argument("--data_path", default=None)
    p.add_argument("--date_column", default=None)
    p.add_argument("--state_variables", nargs="+", default=None)
    p.add_argument("--use_month_encoding", type=lambda s: s.lower() == "true", default=None)
    p.add_argument("--split_train", type=int, default=None)
    p.add_argument("--split_val", type=int, default=None)
    p.add_argument("--split_test", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--run_id", type=int, default=None)
    # per-stage budgets
    p.add_argument("--bc_n_trials", type=int, default=None)
    p.add_argument("--bc_n_jobs", type=int, default=None)
    p.add_argument("--airl_n_trials", type=int, default=None)
    p.add_argument("--airl_n_jobs", type=int, default=None)
    # IQ/AIRL physics (mass balance)
    p.add_argument("--storage_variable", default=None)
    p.add_argument("--inflow_variable", default=None)
    p.add_argument("--max_storage", type=float, default=None)
    p.add_argument("--min_storage", type=float, default=None)
    p.add_argument("--max_release", type=float, default=None)
    p.add_argument("--min_release", type=float, default=None)
    p.add_argument("--seconds_per_day", type=float, default=None)
    p.add_argument("--volume_factor", type=float, default=None)
    p.add_argument("--save-config", dest="save_config", action="store_true",
                   help="Persist CLI overrides back into the YAML config files "
                        "(default: overrides apply to this run only).")
    return p.parse_args()


def _apply_overrides(a, res_cfg, algo_cfg, res_cfg_path=None, algo_cfg_path=None,
                     save_config: bool = False):
    """Apply CLI overrides in-memory (this run only).  With save_config=True the
    same changes are also written back to the YAML files (comment-preserving)."""
    res_u, algo_u = {}, {}

    def r_top(key, val):
        res_cfg[key] = val; res_u[key] = val

    def r_col(key, val):
        res_cfg.setdefault("columns", {})[key] = val
        res_u.setdefault("columns", {})[key] = val

    if a.data_path is not None: r_top("data_path", a.data_path)
    if a.date_column is not None: r_col("date", a.date_column)
    if a.state_variables is not None: r_col("state", a.state_variables)
    if a.use_month_encoding is not None: r_col("use_month_encoding", a.use_month_encoding)
    for k, key in (("split_train", "train"), ("split_val", "val"), ("split_test", "test")):
        if getattr(a, k) is not None:
            res_cfg.setdefault("split", {})[key] = getattr(a, k)
            res_u.setdefault("split", {})[key] = getattr(a, k)
    for k in ("storage_variable", "inflow_variable"):
        v = getattr(a, k)
        if v is not None: r_col(k.replace("_variable", ""), v)
    mb = {k: getattr(a, k) for k in ("max_storage", "min_storage", "max_release", "min_release",
                                     "seconds_per_day", "volume_factor") if getattr(a, k) is not None}
    if mb:
        res_cfg.setdefault("reservoir", {}).setdefault("mass_balance", {}).update(mb)
        res_u.setdefault("reservoir", {})["mass_balance"] = mb

    dev = a.device
    if dev is not None:
        algo_cfg["bc_tuning"]["runtime"]["device"] = dev
        algo_cfg["airl_tuning"]["runtime"]["device"] = dev
        algo_u.setdefault("bc_tuning", {}).setdefault("runtime", {})["device"] = dev
        algo_u.setdefault("airl_tuning", {}).setdefault("runtime", {})["device"] = dev
    if a.bc_n_trials is not None:
        algo_cfg["bc_tuning"]["optuna"]["n_trials"] = a.bc_n_trials
        algo_u.setdefault("bc_tuning", {}).setdefault("optuna", {})["n_trials"] = a.bc_n_trials
    if a.bc_n_jobs is not None:
        algo_cfg["bc_tuning"]["runtime"]["num_workers"] = a.bc_n_jobs
        algo_u.setdefault("bc_tuning", {}).setdefault("runtime", {})["num_workers"] = a.bc_n_jobs
    if a.airl_n_trials is not None:
        algo_cfg["airl_tuning"]["optuna"]["n_trials"] = a.airl_n_trials
        algo_u.setdefault("airl_tuning", {}).setdefault("optuna", {})["n_trials"] = a.airl_n_trials
    if a.airl_n_jobs is not None:
        algo_cfg["airl_tuning"]["runtime"]["num_workers"] = a.airl_n_jobs
        algo_u.setdefault("airl_tuning", {}).setdefault("runtime", {})["num_workers"] = a.airl_n_jobs

    if save_config:
        if res_u and res_cfg_path is not None: _writeback_yaml(res_cfg_path, res_u)
        if algo_u and algo_cfg_path is not None: _writeback_yaml(algo_cfg_path, algo_u)


def main():
    a = _parse_args()
    res_cfg_path = _ROOT / "configs" / "reservoirs" / f"{a.reservoir}.yaml"
    algo_cfg_path = _ROOT / "configs" / "algorithms" / "airl.yaml"
    if not res_cfg_path.exists(): sys.exit(f"Reservoir config not found: {res_cfg_path}")
    if not algo_cfg_path.exists(): sys.exit(f"Algorithm config not found: {algo_cfg_path}")
    res_cfg = yaml.safe_load(open(res_cfg_path)); algo_cfg = yaml.safe_load(open(algo_cfg_path))
    _apply_overrides(a, res_cfg, algo_cfg, res_cfg_path, algo_cfg_path, save_config=a.save_config)

    device_str = _resolve_device(a.device if a.device is not None
                                 else algo_cfg["bc_tuning"]["runtime"]["device"])
    print(f"\nLoading data for reservoir '{a.reservoir}' …")
    data = load_reservoir_data(res_cfg, res_cfg_path)

    bc_args = SimpleNamespace(data_path=a.data_path, date_column=a.date_column,
                              state_variables=a.state_variables, use_month_encoding=a.use_month_encoding,
                              split_train=a.split_train, split_val=a.split_val, split_test=a.split_test,
                              device=a.device, num_workers=a.bc_n_jobs, n_trials=a.bc_n_trials)
    airl_args = SimpleNamespace(device=a.device, num_workers=a.airl_n_jobs, n_trials=a.airl_n_trials,
                                storage_variable=a.storage_variable, inflow_variable=a.inflow_variable,
                                max_storage=a.max_storage, min_storage=a.min_storage,
                                max_release=a.max_release, min_release=a.min_release,
                                seconds_per_day=a.seconds_per_day, volume_factor=a.volume_factor)

    print("\n" + "=" * 72); print("  STAGE 1/3  —  Behavioral Cloning"); print("=" * 72)
    bc = run_airl_bc_tuning(reservoir=a.reservoir, res_cfg=res_cfg, res_cfg_path=res_cfg_path,
                            algo_cfg=algo_cfg, data=data, device_str=device_str,
                            run_id=a.run_id, cli_args=bc_args)
    run_id = bc["run_id"]

    print("\n" + "=" * 72); print(f"  STAGE 2/3  —  AIRL (run_id={run_id})"); print("=" * 72)
    airl = run_airl_tuning(reservoir=a.reservoir, res_cfg=res_cfg, algo_cfg=algo_cfg, data=data,
                           device_str=device_str, run_id=run_id, cli_args=airl_args)

    print("\n" + "=" * 72); print(f"  STAGE 3/3  —  Results (figures)  (run_id={run_id})"); print("=" * 72)
    figures_dir = None
    try:
        figures_dir = run_generate_results(reservoir=a.reservoir, res_cfg=res_cfg, res_cfg_path=res_cfg_path,
                                           algo_cfg=algo_cfg, data=data, device_str=device_str,
                                           run_id=run_id)["figures_dir"]
    except Exception as exc:
        print(f"  WARNING: figure generation failed ({exc}).")
        print(f"  Re-run standalone:  python airl/results.py --reservoir {a.reservoir} --run_id {run_id}")

    print("\n" + "=" * 72)
    print(f"  PIPELINE COMPLETE  —  run_id={run_id}   folder={bc['run_folder']}")
    print(f"  Family : {bc['policy_family']}  (candidates: {', '.join(bc['candidate_families'])})")
    print(f"  BC   : score={bc['best_score']:.4f}  → bc_policy.pt")
    print(f"  AIRL : composite={airl['best_score']:.4f}  → airl_agent.pt   "
          f"(test rel_corr={airl['test_metrics']['release_corr']:.3f})")
    if figures_dir is not None:
        print(f"  Figures → {figures_dir}")
    print("=" * 72)


if __name__ == "__main__":
    main()
