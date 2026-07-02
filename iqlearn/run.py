"""
iqlearn/run.py
==============
End-to-end driver for the IQ-Learn pipeline:

    fresh Behavioral Cloning tuning  ->  IQ-Learn tuning  ->  result figures

on a SINGLE data load and a SINGLE shared run folder
(results/<reservoir>/iqlearn/<run_id>/).  Every invocation runs BC from
scratch and then warm-starts IQ from the BC policy it just produced.

Why a driver (vs. running the two scripts back-to-back)
-------------------------------------------------------
  * Data is loaded ONCE and handed to both stages -> identical splits /
    bounds / normalizer / state_dim, so IQ's state_dim drift guard can never
    trip on its own pipeline.
  * One run_id is resolved by BC (which creates the folder) and reused by IQ,
    so bc_policy.pt and iq_agent.pt always land in the same folder.

CLI
---
Shared (feed the one data load):
    --reservoir --data_path --date_column --state_variables --use_month_encoding
    --split_train --split_val --split_test --device --run_id
Per-stage Optuna budgets (BC and IQ are independent searches):
    --bc_n_trials --bc_n_jobs   --iq_n_trials --iq_n_jobs
IQ-only physics (mass balance):
    --storage_variable --inflow_variable
    --max_storage --min_storage --max_release --min_release
    --seconds_per_day --volume_factor

The superset args are split into two stage-native namespaces so each
run_*_tuning() / _apply_cli_overrides() sees a cli_args identical in shape to
what its own _parse_args() would produce.

NOTE on module names
--------------------
This imports the BC entry from `iqlearn.bc_tuning`.  If your BC file lives at a
different module path (e.g. `iqlearn.tune`), change the import below to match.
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
from iqlearn.utils.runs import _resolve_device
from iqlearn.bc_tuning import run_bc_tuning, _apply_cli_overrides as _bc_apply_overrides
from iqlearn.iq_tuning import run_iq_tuning, _apply_cli_overrides as _iq_apply_overrides
from iqlearn.results import run_generate_results


# =============================================================================
# CLI
# =============================================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="End-to-end IQ-Learn pipeline: fresh BC tuning, then IQ tuning.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument("--reservoir", required=True,
                   help="Reservoir name — must match configs/reservoirs/<name>.yaml.")

    # --- shared: feed the single data load (written to the reservoir config) ---
    p.add_argument("--data_path", default=None, help="Override data_path.")
    p.add_argument("--date_column", default=None, help="Override columns.date.")
    p.add_argument("--state_variables", nargs="+", default=None, help="Override columns.state.")
    p.add_argument("--use_month_encoding", type=lambda s: s.lower() == "true", default=None,
                   help="true|false. Override columns.use_month_encoding.")
    p.add_argument("--split_train", type=int, default=None, help="Override split.train (years).")
    p.add_argument("--split_val",   type=int, default=None, help="Override split.val (years).")
    p.add_argument("--split_test",  type=int, default=None, help="Override split.test (years).")
    p.add_argument("--device", default=None, help="auto | cpu | cuda | cuda:N | mps (both stages).")
    p.add_argument("--run_id", type=int, default=None,
                   help="Shared run folder id. If omitted, BC auto-increments it.")

    # --- per-stage Optuna budgets ---
    p.add_argument("--bc_n_trials", type=int, default=None, help="BC Optuna trials (else config).")
    p.add_argument("--bc_n_jobs",   type=int, default=None, help="BC parallel workers (else config).")
    p.add_argument("--iq_n_trials", type=int, default=None, help="IQ Optuna trials (else config).")
    p.add_argument("--iq_n_jobs",   type=int, default=None, help="IQ parallel workers (else config).")

    # --- config persistence ---
    p.add_argument("--save-config", dest="save_config", action="store_true",
                   help="Persist CLI overrides back into the YAML config files "
                        "(default: overrides apply to this run only).")

    # --- IQ-only physics (mass balance); CLI > config > data ---
    p.add_argument("--storage_variable", default=None, help="State column that is storage.")
    p.add_argument("--inflow_variable",  default=None, help="State column that is inflow.")
    p.add_argument("--max_storage", type=float, default=None)
    p.add_argument("--min_storage", type=float, default=None)
    p.add_argument("--max_release", type=float, default=None)
    p.add_argument("--min_release", type=float, default=None)
    p.add_argument("--seconds_per_day", type=float, default=None)
    p.add_argument("--volume_factor",   type=float, default=None)

    return p.parse_args()


def _bc_namespace(a: argparse.Namespace) -> SimpleNamespace:
    """Stage-native args for bc_tuning (attribute names match BC's _parse_args)."""
    return SimpleNamespace(
        reservoir=a.reservoir,
        data_path=a.data_path,
        date_column=a.date_column,
        state_variables=a.state_variables,
        use_month_encoding=a.use_month_encoding,
        split_train=a.split_train,
        split_val=a.split_val,
        split_test=a.split_test,
        device=a.device,
        num_workers=a.bc_n_jobs,     # BC parallelism comes from runtime.num_workers
        n_trials=a.bc_n_trials,
        run_id=a.run_id,
    )


def _iq_namespace(a: argparse.Namespace) -> SimpleNamespace:
    """Stage-native args for iq_tuning (attribute names match IQ's _parse_args)."""
    return SimpleNamespace(
        reservoir=a.reservoir,
        run_id=a.run_id,
        device=a.device,
        num_workers=a.iq_n_jobs,     # IQ parallelism comes from runtime.num_workers
        n_trials=a.iq_n_trials,
        storage_variable=a.storage_variable,
        inflow_variable=a.inflow_variable,
        max_storage=a.max_storage,
        min_storage=a.min_storage,
        max_release=a.max_release,
        min_release=a.min_release,
        seconds_per_day=a.seconds_per_day,
        volume_factor=a.volume_factor,
    )


# =============================================================================
# Driver
# =============================================================================

def main() -> None:
    a = _parse_args()

    res_cfg_path  = _ROOT / "configs" / "reservoirs" / f"{a.reservoir}.yaml"
    algo_cfg_path = _ROOT / "configs" / "algorithms" / "iqlearn.yaml"
    if not res_cfg_path.exists():
        sys.exit(f"Reservoir config not found: {res_cfg_path}")
    if not algo_cfg_path.exists():
        sys.exit(f"Algorithm config not found: {algo_cfg_path}")

    res_cfg  = yaml.safe_load(open(res_cfg_path))
    algo_cfg = yaml.safe_load(open(algo_cfg_path))

    bc_args = _bc_namespace(a)
    iq_args = _iq_namespace(a)

    # Apply BOTH stages' CLI overrides in-memory (this run only).  With
    # --save-config they are ALSO written back to the YAML (comment-preserving).
    # BC touches data/split + bc_tuning; IQ touches physics + iq_tuning —
    # disjoint keys, so order is irrelevant.
    _bc_apply_overrides(bc_args, res_cfg, algo_cfg, res_cfg_path, algo_cfg_path,
                        save_config=a.save_config)
    _iq_apply_overrides(iq_args, res_cfg, algo_cfg, res_cfg_path, algo_cfg_path,
                        save_config=a.save_config)

    # Resolve device once; both stages receive the same resolved string.
    device_str = _resolve_device(
        a.device if a.device is not None else algo_cfg["bc_tuning"]["runtime"]["device"]
    )

    # Load data ONCE; hand the same object to both stages.
    print(f"\nLoading data for reservoir '{a.reservoir}' \u2026")
    data = load_reservoir_data(res_cfg, res_cfg_path)

    # ---- Stage 1/2: Behavioral Cloning (resolves + creates the run folder) ----
    print("\n" + "=" * 72)
    print("  STAGE 1/3  \u2014  Behavioral Cloning")
    print("=" * 72)
    bc_result = run_bc_tuning(
        reservoir=a.reservoir, res_cfg=res_cfg, res_cfg_path=res_cfg_path,
        algo_cfg=algo_cfg, data=data, device_str=device_str,
        run_id=a.run_id, cli_args=bc_args,
    )
    run_id = int(bc_result["run_folder"].name)   # single source of the shared run_id

    # ---- Stage 2/2: IQ-Learn (same folder, warm-start from bc_policy.pt) ----
    print("\n" + "=" * 72)
    print(f"  STAGE 2/3  \u2014  IQ-Learn   (run_id={run_id})")
    print("=" * 72)
    iq_result = run_iq_tuning(
        reservoir=a.reservoir, res_cfg=res_cfg, algo_cfg=algo_cfg,
        data=data, device_str=device_str, run_id=run_id, cli_args=iq_args,
    )

    # ---- Stage 3/3: results (figures) — the agent is already saved, so a
    # plotting error must never sink the run; warn and point to the standalone. ----
    print("\n" + "=" * 72)
    print(f"  STAGE 3/3  \u2014  Results (figures)   (run_id={run_id})")
    print("=" * 72)
    figures_dir = None
    try:
        results_out = run_generate_results(
            reservoir=a.reservoir, res_cfg=res_cfg, res_cfg_path=res_cfg_path,
            algo_cfg=algo_cfg, data=data, device_str=device_str, run_id=run_id,
        )
        figures_dir = results_out["figures_dir"]
    except Exception as exc:                  # never lose a trained agent to a figure error
        print(f"  WARNING: figure generation failed ({exc}).")
        print(f"  Re-run standalone:  python iqlearn/results.py "
              f"--reservoir {a.reservoir} --run_id {run_id}")

    # ---- Summary ----
    print("\n" + "=" * 72)
    print(f"  PIPELINE COMPLETE  \u2014  run_id={run_id}   folder={bc_result['run_folder']}")
    print(f"  Family : {bc_result['policy_family']}  "
          f"(candidates: {', '.join(bc_result['candidate_families'])})")
    print(f"  BC : score={bc_result['best_score']:.4f}   \u2192 {bc_result['policy_path'].name}")
    print(f"  IQ : score={iq_result['best_score']:.4f}   \u2192 {iq_result['agent_path'].name}")
    print(f"  Results \u2192 {bc_result['run_folder'] / 'figures'}")
    if figures_dir is not None:
        print(f"  Figures            \u2192 {figures_dir}")
    print("=" * 72)


if __name__ == "__main__":
    main()