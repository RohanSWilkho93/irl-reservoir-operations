"""
iqlearn/tune.py
===============
Optuna hyperparameter search for IQ-Learn (Inverse Q-Learning).

Run this AFTER behavioral_cloning/train.py.  The BC checkpoint supplies a
pre-trained policy whose weights are loaded at the start of every trial.  Only
the IQ-Learn-specific components (critic network, learning rates, IRL loss
coefficients) are tuned here.  The BC policy architecture is held fixed.

Usage
-----
# Minimal -- auto-selects the highest-numbered completed BC run
python iqlearn/tune.py --reservoir conchas --policy_type hardgating

# Specify an exact BC run_id
python iqlearn/tune.py --reservoir conchas --policy_type hardgating --bc_run_id 1

# Override device and worker count
python iqlearn/tune.py --reservoir conchas --policy_type hardgating --bc_run_id 1 \\
    --device cuda --num_workers 8

# Full first-run override (values written back to iqlearn.yaml)
python iqlearn/tune.py --reservoir conchas --policy_type hardgating --bc_run_id 1 \\
    --device auto --num_workers 20 --n_trials 2000 --run_id 1

Search space
------------
All IQ-Learn-specific hyperparameters are drawn from
configs/algorithms/iqlearn.yaml (search_space section).

Spec syntax (matches iqlearn.yaml format):
    [a, b, c]                      -> suggest_categorical
    {low: x, high: y}              -> suggest_float (uniform)
    {low: x, high: y, log: true}   -> suggest_float (log-uniform)
    {low: x, high: y, type: int}   -> suggest_int

Validation metric
-----------------
Composite score (see iqlearn/core.py):
    70 % release Pearson r  +  30 % release nRMSE score
Optuna maximises this score.  Pruning fires when a trial falls below
the median of completed trials (MedianPruner, n_warmup_steps=5).

Outputs (per run_id)
--------------------
results/<reservoir>/iqlearn/<run_id>_<policy_type>/
    best_config.json  -- all hyperparameters for the best trial
    run_args.json     -- CLI arguments and metadata
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import torch
import yaml

# ---------------------------------------------------------------------------
# Project root on sys.path so sibling packages resolve correctly.
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from behavioral_cloning.tune import (
    _bc_default,
    _resolve_device,
    _writeback_yaml,
    _resolve_run_id,
)
from networks.policy import build_policy_network
from utils.data      import load_reservoir_data
from iqlearn.core    import IQLearnConfig, IQLearnAgent, ExpertBuffer


# =============================================================================
# Search-space sampling helper  (identical spec to airl/tune.py)
# =============================================================================

def _sample_param(trial: optuna.Trial, name: str, spec: Any) -> Any:
    """
    Sample one hyperparameter using the spec from configs/algorithms/iqlearn.yaml.

    Spec formats
    ------------
    [a, b, c]                      -> suggest_categorical(name, [a,b,c])
    {low: x, high: y}              -> suggest_float(name, x, y)
    {low: x, high: y, log: true}   -> suggest_float(name, x, y, log=True)
    {low: x, high: y, type: int}   -> suggest_int(name, x, y)
    """
    if isinstance(spec, list):
        return trial.suggest_categorical(name, spec)

    if isinstance(spec, dict):
        low  = spec["low"]
        high = spec["high"]
        if spec.get("type") == "int":
            return trial.suggest_int(name, int(low), int(high))
        return trial.suggest_float(
            name, float(low), float(high),
            log=bool(spec.get("log", False)),
        )

    raise ValueError(
        f"Unrecognised search_space spec for {name!r}: {spec!r}\n"
        f"Expected a list or a dict with {{low, high[, log][, type]}}."
    )


# =============================================================================
# BC run locator  (mirrors airl/tune.py exactly)
# =============================================================================

def _find_bc_run(
    bc_base_dir: Path,
    reservoir:   str,
    policy_type: str,
    bc_run_id:   int | None,
) -> Path:
    """
    Locate the BC run folder for (reservoir, policy_type).

    Scans bc_base_dir for folders named <int>_<policy_type> containing
    both best_config.json and model.pt (completed run).

    If bc_run_id is None, auto-selects the highest-numbered completed run.
    Exits with a clear error on any validation failure.
    """
    import re as _re
    _PATTERN = _re.compile(r"^(\d+)_(.+)$")

    if not bc_base_dir.exists():
        sys.exit(
            f"\nERROR: No BC results directory found for reservoir {reservoir!r}.\n"
            f"  Expected: {bc_base_dir}\n"
            f"\n  Fix:\n"
            f"    python behavioral_cloning/tune.py  --reservoir {reservoir}\n"
            f"    python behavioral_cloning/train.py --reservoir {reservoir} --run_id <id>\n"
        )

    completed: list[tuple[int, Path]] = []
    all_runs:  dict[int, str]         = {}

    for d in sorted(bc_base_dir.iterdir()):
        m = _PATTERN.match(d.name)
        if not (d.is_dir() and m):
            continue
        rid, ptype = int(m.group(1)), m.group(2)
        all_runs[rid] = ptype
        if ptype == policy_type and (d / "best_config.json").exists() and (d / "model.pt").exists():
            completed.append((rid, d))

    if not completed:
        avail = ", ".join(sorted(f"{r}_{t}" for r, t in all_runs.items())) or "none"
        sys.exit(
            f"\nERROR: No completed BC run found for:\n"
            f"  reservoir   = {reservoir!r}\n"
            f"  policy_type = {policy_type!r}\n"
            f"\n  A completed run requires both best_config.json and model.pt.\n"
            f"\n  Fix:\n"
            f"    python behavioral_cloning/tune.py  --reservoir {reservoir}\n"
            f"    python behavioral_cloning/train.py --reservoir {reservoir} --run_id <id>\n"
            f"\n  BC runs available for this reservoir: {avail}\n"
        )

    if bc_run_id is None:
        chosen_id, chosen_dir = max(completed, key=lambda x: x[0])
        print(f"\n  --bc_run_id not specified; auto-selected BC run {chosen_id}_{policy_type}")
        return chosen_dir

    for rid, d in completed:
        if rid == bc_run_id:
            return d

    # Specific run requested but not found -- give a precise error
    if bc_run_id in all_runs:
        if all_runs[bc_run_id] == policy_type:
            bc_folder = bc_base_dir / f"{bc_run_id}_{policy_type}"
            missing = [f for f in ("best_config.json", "model.pt") if not (bc_folder / f).exists()]
            sys.exit(
                f"\nERROR: BC run {bc_run_id}_{policy_type} is incomplete -- "
                f"missing: {', '.join(missing)}\n"
                f"  Run behavioral_cloning/train.py first:\n"
                f"    python behavioral_cloning/train.py "
                f"--reservoir {reservoir} --policy_type {policy_type} --run_id {bc_run_id}\n"
                f"  Completed {policy_type!r} runs: {sorted(r for r, _ in completed)}\n"
            )
        else:
            sys.exit(
                f"\nERROR: BC run {bc_run_id} has policy_type={all_runs[bc_run_id]!r}, "
                f"not {policy_type!r}.\n"
                f"  Completed {policy_type!r} runs: {sorted(r for r, _ in completed)}\n"
            )

    sys.exit(
        f"\nERROR: BC run_id={bc_run_id} not found for policy_type={policy_type!r} in:\n"
        f"  {bc_base_dir}\n"
        f"  Completed {policy_type!r} runs: {sorted(r for r, _ in completed)}\n"
    )


# =============================================================================
# Objective factory
# =============================================================================

def _make_iqlearn_objective(
    algo_cfg:      dict,
    data,
    bc_state_dict: dict,
    bc_cfg_dict:   dict,
    policy_type:   str,
    device_str:    str,
):
    """
    Return a closure that Optuna calls on every trial.

    Heavy objects (DataSplits, BC state_dict) are captured once in the
    closure.  Each trial creates its own independent IQLearnAgent.

    Name mapping
    ------------
    iqlearn.yaml           IQLearnConfig field
    ---------------------- ---------------------
    lr_actor            -> learning_rate_actor
    lr_critic           -> learning_rate_critic
    bc_cfg "hidden_dim" -> actor_hidden_dim
    bc_cfg "n_hidden_layers" -> actor_n_hidden_layers
    """
    search_space = algo_cfg["search_space"]

    def objective(trial: optuna.Trial) -> float:

        # 1. Sample IQ-Learn hyperparameters from yaml search space
        sampled = {
            name: _sample_param(trial, name, spec)
            for name, spec in search_space.items()
        }

        # 2. Build IQLearnConfig
        #    BC architecture fields are fixed; IQ-Learn fields come from Optuna.
        config = IQLearnConfig(
            state_dim  = data.state_dim,
            action_dim = 1,

            # BC policy architecture -- fixed, NOT tuned
            actor_hidden_dim      = int(bc_cfg_dict.get("hidden_dim",      _bc_default("hidden_dim"))),
            actor_n_hidden_layers = int(bc_cfg_dict.get("n_hidden_layers",  _bc_default("n_hidden_layers"))),
            dropout               = float(bc_cfg_dict.get("dropout",        _bc_default("dropout"))),
            alpha_min             = float(bc_cfg_dict.get("alpha_min",      _bc_default("alpha_min"))),
            alpha_max             = float(bc_cfg_dict.get("alpha_max",      _bc_default("alpha_max"))),
            beta_min              = float(bc_cfg_dict.get("beta_min",       _bc_default("beta_min"))),
            beta_max              = float(bc_cfg_dict.get("beta_max",       _bc_default("beta_max"))),
            sigma_min             = float(bc_cfg_dict.get("sigma_min",      _bc_default("sigma_min"))),
            log_epsilon           = float(bc_cfg_dict.get("log_epsilon",    _bc_default("log_epsilon"))),
            zero_threshold        = float(bc_cfg_dict.get("zero_threshold", _bc_default("zero_threshold"))),
            mse_weight            = float(bc_cfg_dict.get("mse_weight",     _bc_default("mse_weight"))),
            gate_weight           = float(bc_cfg_dict.get("gate_weight",    _bc_default("gate_weight"))),

            # Critic network (Optuna-tuned)
            critic_hidden_dim      = int(sampled["critic_hidden_dim"]),
            critic_n_hidden_layers = int(sampled["critic_n_hidden_layers"]),

            # Training schedule (Optuna-tuned)
            critic_warm_up_epochs = int(sampled["critic_warm_up_epochs"]),
            n_epochs              = int(sampled["n_epochs"]),

            # Learning rates -- yaml names differ from config field names
            learning_rate_actor  = float(sampled["lr_actor"]),
            learning_rate_critic = float(sampled["lr_critic"]),

            # IQ-Learn loss coefficients (Optuna-tuned)
            gamma                = float(sampled["gamma"]),
            tau                  = float(sampled["tau"]),
            alpha_entropy        = float(sampled["alpha_entropy"]),
            alpha_regularization = float(sampled["alpha_regularization"]),
            lambda_bc            = float(sampled["lambda_bc"]),

            # Runtime -- use trial.number as seed for per-trial diversity
            device  = device_str,
            seed    = trial.number,
            verbose = False,
        )

        # 3. Per-trial reproducibility seed
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)

        # 4. Build actor with cloned BC weights (independent per trial)
        bc_policy = build_policy_network(policy_type, config).to(torch.device(device_str))
        bc_policy.load_state_dict({k: v.clone() for k, v in bc_state_dict.items()})

        # 5. Expert buffer + agent
        train_buf = ExpertBuffer(data.train)
        agent     = IQLearnAgent(config, bc_policy, policy_type)

        # 6. Train -- trial enables Optuna pruning via trial.report() in core.py
        try:
            result = agent.train(train_buf, data.val, trial=trial)
        except optuna.exceptions.TrialPruned:
            raise
        except Exception as exc:
            print(f"  [trial {trial.number}] exception: {exc}")
            return -float("inf")
        finally:
            del agent, bc_policy, train_buf
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        best_score = float(result["best_val_score"])
        return best_score if np.isfinite(best_score) else -float("inf")

    return objective


# =============================================================================
# CLI argument parsing
# =============================================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="IQ-Learn hyperparameter tuning with Optuna.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "--reservoir", required=True,
        help="Reservoir name -- must match configs/reservoirs/<name>.yaml.",
    )
    p.add_argument(
        "--policy_type", required=True,
        choices=["beta", "lognormal", "hardgating", "softgating"],
        help=(
            "Policy distribution type.  Must match the BC run being loaded.  "
            "Used to locate results/<reservoir>/behavioral_cloning/*_<policy_type>/."
        ),
    )
    p.add_argument(
        "--bc_run_id", type=int, default=None,
        help=(
            "Integer run_id of the BC run to load weights from.  "
            "If omitted, the highest-numbered completed run for --policy_type "
            "is selected automatically."
        ),
    )
    p.add_argument(
        "--device", default=None,
        help="Override runtime.device in iqlearn.yaml: auto | cpu | cuda | cuda:N | mps.",
    )
    p.add_argument(
        "--num_workers", type=int, default=None,
        help="Override runtime.num_workers (parallel Optuna jobs).",
    )
    p.add_argument(
        "--n_trials", type=int, default=None,
        help="Override optuna.n_trials in iqlearn.yaml.",
    )
    p.add_argument(
        "--run_id", type=int, default=None,
        help=(
            "Integer run identifier for the results folder.  "
            "Auto-increments if omitted.  Folder: <run_id>_<policy_type>."
        ),
    )
    p.add_argument(
        "--use_month_encoding",
        type=lambda s: s.lower() == "true",
        default=None,
        help="Override columns.use_month_encoding in the reservoir YAML (true|false).",
    )

    return p.parse_args()


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args = _parse_args()

    res_cfg_path  = _ROOT / "configs" / "reservoirs" / f"{args.reservoir}.yaml"
    algo_cfg_path = _ROOT / "configs" / "algorithms" / "iqlearn.yaml"

    for path, label in [(res_cfg_path, "Reservoir"), (algo_cfg_path, "Algorithm")]:
        if not path.exists():
            sys.exit(f"\nERROR: {label} config not found: {path}\n")

    with open(res_cfg_path,  "r") as f:
        res_cfg = yaml.safe_load(f)
    with open(algo_cfg_path, "r") as f:
        algo_cfg = yaml.safe_load(f)

    # Apply CLI overrides and write back to YAML for reproducibility
    algo_updates: dict = {}
    if args.device is not None:
        algo_cfg["runtime"]["device"] = args.device
        algo_updates.setdefault("runtime", {})["device"] = args.device
    if args.num_workers is not None:
        algo_cfg["runtime"]["num_workers"] = args.num_workers
        algo_updates.setdefault("runtime", {})["num_workers"] = args.num_workers
    if args.n_trials is not None:
        algo_cfg["optuna"]["n_trials"] = args.n_trials
        algo_updates.setdefault("optuna", {})["n_trials"] = args.n_trials
    if algo_updates:
        _writeback_yaml(algo_cfg_path, algo_updates)

    # Apply reservoir-level CLI overrides and write back
    res_updates: dict = {}
    if args.use_month_encoding is not None:
        res_cfg["columns"]["use_month_encoding"] = args.use_month_encoding
        res_updates.setdefault("columns", {})["use_month_encoding"] = args.use_month_encoding
    if res_updates:
        _writeback_yaml(res_cfg_path, res_updates)

    # Resolve runtime settings
    device_str = _resolve_device(algo_cfg["runtime"]["device"])
    n_jobs = (
        algo_cfg["runtime"]["num_workers"]
        if algo_cfg["runtime"]["num_workers"] is not None
        else algo_cfg["optuna"]["n_jobs"]
    )
    n_trials = algo_cfg["optuna"]["n_trials"]

    # Locate and validate BC checkpoint
    bc_base_dir = _ROOT / "results" / args.reservoir / "behavioral_cloning"
    bc_run_dir  = _find_bc_run(
        bc_base_dir = bc_base_dir,
        reservoir   = args.reservoir,
        policy_type = args.policy_type,
        bc_run_id   = args.bc_run_id,
    )

    with open(bc_run_dir / "best_config.json", "r") as f:
        bc_saved = json.load(f)

    if bc_saved.get("reservoir") != args.reservoir:
        sys.exit(
            f"\nERROR: BC checkpoint reservoir mismatch.\n"
            f"  CLI --reservoir = {args.reservoir!r}\n"
            f"  JSON reservoir  = {bc_saved.get('reservoir')!r}\n"
        )
    if str(bc_saved.get("policy_type")) != args.policy_type:
        sys.exit(
            f"\nERROR: BC checkpoint policy_type mismatch.\n"
            f"  CLI --policy_type = {args.policy_type!r}\n"
            f"  JSON policy_type  = {bc_saved.get('policy_type')!r}\n"
        )

    policy_type = args.policy_type
    bc_cfg_dict = bc_saved["config"]

    print(f"\nLoaded BC checkpoint : {bc_run_dir.name}")
    print(f"  Reservoir   : {args.reservoir}")
    print(f"  Policy type : {policy_type}")
    print(f"  BC val score: {bc_saved['best_score']:.4f}")

    # BC weights stay on CPU; objective clones and moves per trial
    bc_ckpt       = torch.load(bc_run_dir / "model.pt", map_location="cpu", weights_only=False)
    bc_state_dict = bc_ckpt["model_state_dict"]

    # Load data
    print(f"\nLoading data for reservoir {args.reservoir!r} ...")
    data = load_reservoir_data(res_cfg, res_cfg_path)
    print(
        f"  state_dim  = {data.state_dim}\n"
        f"  train rows = {len(data.train.states)}\n"
        f"  val rows   = {len(data.val.states)}\n"
        f"  test rows  = {len(data.test.states)}\n"
        f"  device     = {device_str}"
    )

    # Guard: state_dim mismatch between BC checkpoint and current data config
    bc_state_dim = int(bc_cfg_dict["state_dim"])
    if data.state_dim != bc_state_dim:
        n_raw       = len(list(res_cfg["columns"]["state"]))
        bc_month    = (bc_state_dim == n_raw + 2)
        cur_month   = bool(res_cfg["columns"].get("use_month_encoding", True))
        sys.exit(
            f"\nERROR: state_dim mismatch -- BC checkpoint and current YAML are incompatible.\n"
            f"  BC checkpoint ({bc_run_dir.name})  state_dim = {bc_state_dim} "
            f"(use_month_encoding={'true' if bc_month else 'false'})\n"
            f"  Current YAML                        state_dim = {data.state_dim} "
            f"(use_month_encoding={'true' if cur_month else 'false'})\n\n"
            f"  Fix: set use_month_encoding: {'true' if bc_month else 'false'} "
            f"in configs/reservoirs/{args.reservoir}.yaml\n"
        )

    # Output directory
    iqlearn_base = _ROOT / "results" / args.reservoir / "iqlearn"
    run_id, results_dir = _resolve_run_id(iqlearn_base, policy_type, args.run_id)
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nRun folder : {results_dir.name}  (run_id={run_id})")

    # Optuna study
    study = optuna.create_study(
        direction  = "maximize",
        sampler    = optuna.samplers.TPESampler(seed=42),
        pruner     = optuna.pruners.MedianPruner(n_warmup_steps=5),
        study_name = f"{args.reservoir}_iqlearn",
    )

    objective = _make_iqlearn_objective(
        algo_cfg      = algo_cfg,
        data          = data,
        bc_state_dict = bc_state_dict,
        bc_cfg_dict   = bc_cfg_dict,
        policy_type   = policy_type,
        device_str    = device_str,
    )

    print(f"\nStarting Optuna search: {n_trials} trials, {n_jobs} job(s) ...\n")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)

    # Guard: at least one trial must have completed
    completed_trials = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
    ]
    if not completed_trials:
        sys.exit(
            "\nAll trials failed or were pruned.\n"
            "Check device availability, BC checkpoint integrity, "
            "and search space bounds in iqlearn.yaml.\n"
        )

    # Reconstruct best IQLearnConfig from best trial params
    best = study.best_trial
    p    = best.params

    best_config = IQLearnConfig(
        state_dim  = data.state_dim,
        action_dim = 1,

        actor_hidden_dim      = int(bc_cfg_dict.get("hidden_dim",      _bc_default("hidden_dim"))),
        actor_n_hidden_layers = int(bc_cfg_dict.get("n_hidden_layers",  _bc_default("n_hidden_layers"))),
        dropout               = float(bc_cfg_dict.get("dropout",        _bc_default("dropout"))),
        alpha_min             = float(bc_cfg_dict.get("alpha_min",      _bc_default("alpha_min"))),
        alpha_max             = float(bc_cfg_dict.get("alpha_max",      _bc_default("alpha_max"))),
        beta_min              = float(bc_cfg_dict.get("beta_min",       _bc_default("beta_min"))),
        beta_max              = float(bc_cfg_dict.get("beta_max",       _bc_default("beta_max"))),
        sigma_min             = float(bc_cfg_dict.get("sigma_min",      _bc_default("sigma_min"))),
        log_epsilon           = float(bc_cfg_dict.get("log_epsilon",    _bc_default("log_epsilon"))),
        zero_threshold        = float(bc_cfg_dict.get("zero_threshold", _bc_default("zero_threshold"))),
        mse_weight            = float(bc_cfg_dict.get("mse_weight",     _bc_default("mse_weight"))),
        gate_weight           = float(bc_cfg_dict.get("gate_weight",    _bc_default("gate_weight"))),

        critic_hidden_dim      = int(p["critic_hidden_dim"]),
        critic_n_hidden_layers = int(p["critic_n_hidden_layers"]),
        critic_warm_up_epochs  = int(p["critic_warm_up_epochs"]),
        n_epochs               = int(p["n_epochs"]),
        learning_rate_actor    = float(p["lr_actor"]),
        learning_rate_critic   = float(p["lr_critic"]),
        gamma                  = float(p["gamma"]),
        tau                    = float(p["tau"]),
        alpha_entropy          = float(p["alpha_entropy"]),
        alpha_regularization   = float(p["alpha_regularization"]),
        lambda_bc              = float(p["lambda_bc"]),

        device  = device_str,
        seed    = 42,
        verbose = False,
    )

    # Save best_config.json
    save_dict = {
        "reservoir":     args.reservoir,
        "policy_type":   policy_type,
        "best_score":    best.value,
        "trial_number":  best.number,
        "bc_run_id":     args.bc_run_id,
        "bc_run_folder": bc_run_dir.name,
        "config":        asdict(best_config),
    }
    out_path = results_dir / "best_config.json"
    with open(out_path, "w") as f:
        json.dump(save_dict, f, indent=2)
    print(f"\nBest trial #{best.number}  score = {best.value:.4f}")
    print(f"Best config saved -> {out_path}\n")

    # Save run_args.json
    run_args_path = results_dir / "run_args.json"
    run_args: dict = {}
    if run_args_path.exists():
        with open(run_args_path, "r") as f:
            run_args = json.load(f)
    run_args["tune"] = {
        "reservoir":     args.reservoir,
        "bc_run_id":     args.bc_run_id,
        "bc_run_folder": bc_run_dir.name,
        "run_id":        run_id,
        "folder":        results_dir.name,
        "policy_type":   policy_type,
        "device":        args.device,
        "device_used":   device_str,
        "num_workers":   args.num_workers,
        "n_trials":      args.n_trials,
        "timestamp":     datetime.now().isoformat(timespec="seconds"),
    }
    with open(run_args_path, "w") as f:
        json.dump(run_args, f, indent=2)
    print(f"Run args saved  -> {run_args_path}\n")


if __name__ == "__main__":
    main()
