"""
deepmaxent/run.py
=================
End-to-end driver for Deep MaxEnt IRL:

    HP tuning (Optuna)  ->  retrain + save best model  ->  result figures

Single-stage (no behavioral-cloning warm-start), on ONE data load and ONE shared
run folder results/<reservoir>/deepmaxent/<run_id>/.

CLI
---
    python deepmaxent/run.py --reservoir englebright --device cpu \
        --n_trials 500 --num_workers 4

Figure generation is defensive: a plotting error never discards the trained
model — the run folder is kept and the standalone results command is printed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from deepmaxent.data import load_raw_reservoir_data
from deepmaxent.tuning import run_deepmaxent_tuning, _apply_cli_overrides
from deepmaxent.results import run_generate_results
from iqlearn.utils.runs import _resolve_device


def _parse_args():
    p = argparse.ArgumentParser(
        description="End-to-end Deep MaxEnt IRL: tune -> save best -> results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--reservoir", required=True)
    p.add_argument("--data_path", default=None)
    p.add_argument("--split_train", type=int, default=None)
    p.add_argument("--split_val", type=int, default=None)
    p.add_argument("--split_test", type=int, default=None)
    p.add_argument("--device", default=None, help="auto | cpu | cuda | cuda:N")
    p.add_argument("--num_workers", type=int, default=None, help="parallel Optuna workers")
    p.add_argument("--n_trials", type=int, default=None)
    p.add_argument("--run_id", type=int, default=None)
    p.add_argument("--save-config", dest="save_config", action="store_true",
                   help="Persist CLI overrides back into the YAML config files "
                        "(default: overrides apply to this run only).")
    return p.parse_args()


def main():
    a = _parse_args()
    res_cfg_path = _ROOT / "configs" / "reservoirs" / f"{a.reservoir}.yaml"
    algo_cfg_path = _ROOT / "configs" / "algorithms" / "deepmaxent.yaml"
    if not res_cfg_path.exists():
        sys.exit(f"Reservoir config not found: {res_cfg_path}")
    if not algo_cfg_path.exists():
        sys.exit(f"Algorithm config not found: {algo_cfg_path}")
    res_cfg = yaml.safe_load(open(res_cfg_path))
    algo_cfg = yaml.safe_load(open(algo_cfg_path))

    _apply_cli_overrides(a, res_cfg, algo_cfg, res_cfg_path, algo_cfg_path,
                         save_config=a.save_config)
    device_str = _resolve_device(a.device if a.device is not None
                                 else algo_cfg["deepmaxent"]["runtime"]["device"])

    print(f"\nLoading data for reservoir '{a.reservoir}' …")
    data = load_raw_reservoir_data(res_cfg, res_cfg_path)

    print("\n" + "=" * 72)
    print("  STAGE 1/2  —  Deep MaxEnt IRL tuning + best-model save")
    print("=" * 72)
    out = run_deepmaxent_tuning(
        reservoir=a.reservoir, res_cfg=res_cfg, res_cfg_path=res_cfg_path,
        algo_cfg=algo_cfg, data=data, device_str=device_str, run_id=a.run_id, cli_args=a)
    run_id = out["run_id"]

    print("\n" + "=" * 72)
    print(f"  STAGE 2/2  —  Results (figures)   (run_id={run_id})")
    print("=" * 72)
    figures_dir = None
    try:
        figures_dir = run_generate_results(
            reservoir=a.reservoir, res_cfg=res_cfg, res_cfg_path=res_cfg_path,
            algo_cfg=algo_cfg, data=data, device_str=device_str, run_id=run_id)["figures_dir"]
    except Exception as exc:                       # never lose a trained model to a figure error
        print(f"  WARNING: figure generation failed ({exc}).")
        print(f"  Re-run standalone:  python deepmaxent/results.py --reservoir {a.reservoir} --run_id {run_id}")

    print("\n" + "=" * 72)
    print(f"  PIPELINE COMPLETE  —  run_id={run_id}   folder={out['run_folder']}")
    print(f"  Best val unified score : {out['best_score']:.4f}")
    print(f"  Test : release_corr={out['test_metrics']['release_corr']:.3f}  "
          f"storage_corr={out['test_metrics']['storage_corr']:.3f}  "
          f"unified={out['test_metrics']['unified_score']:.3f}")
    if figures_dir is not None:
        print(f"  Figures → {figures_dir}")
    print("=" * 72)


if __name__ == "__main__":
    main()
