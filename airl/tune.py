"""
airl/tune.py
============
Optuna hyperparameter search for Adversarial Inverse Reinforcement Learning.

Run this AFTER behavioral_cloning/train.py.  The BC checkpoint supplies a
pre-trained policy whose weights are loaded at the start of every trial.  Only
the AIRL-specific components (critic, discriminator networks, learning rates,
PPO schedule, KL coefficient) are tuned here.  The BC policy architecture is
held fixed.

Usage
-----
# Minimal — auto-selects the highest-numbered completed BC run for the given policy_type
python airl/tune.py --reservoir garrison --policy_type beta

# Specify an exact BC run_id
python airl/tune.py --reservoir stockton --policy_type lognormal --bc_run_id 2

# Override device and worker count
python airl/tune.py --reservoir stockton --policy_type beta --bc_run_id 2 \\
    --device cuda --num_workers 8

# Full first-run override (values are written back to configs for reproducibility)
python airl/tune.py --reservoir garrison --policy_type beta --bc_run_id 1 \\
    --device auto --num_workers 10 --n_trials 500 --run_id 1

Search space
------------
All AIRL-specific hyperparameters are drawn from
configs/algorithms/airl.yaml (search_space section).

Spec syntax (matches airl.yaml format):
    [a, b, c]                      → suggest_categorical
    {low: x, high: y}              → suggest_float (uniform)
    {low: x, high: y, log: true}   → suggest_float (log-uniform)
    {low: x, high: y, type: int}   → suggest_int

Validation metric
-----------------
Composite score (see airl/core.py for formula):
    50 % discriminator balance + 25 % release (corr + 1-nrmse) + 25 % storage (corr + 1-nrmse)
Optuna maximises this score.  Pruning fires when a trial's intermediate score
falls below the median of completed trials.

Outputs (per run_id)
--------------------
results/<reservoir>/airl/<run_id>_<policy_type>/
    best_config.json  — all hyperparameters for the best trial
    run_args.json     — CLI arguments and metadata
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
from airl.core       import (
    AIRLConfig,
    AIRLAgent,
    ReservoirEnvironment,
    _load_raw_splits,
)


# =============================================================================
# Search-space sampling helper
# =============================================================================

def _sample_param(trial: optuna.Trial, name: str, spec: Any) -> Any:
    """
    Sample one hyperparameter from an Optuna trial using the spec from
    configs/algorithms/airl.yaml.

    Spec formats
    ------------
    [a, b, c]                      → suggest_categorical(name, [a,b,c])
    {low: x, high: y}              → suggest_float(name, x, y)
    {low: x, high: y, log: true}   → suggest_float(name, x, y, log=True)
    {low: x, high: y, type: int}   → suggest_int(name, x, y)
    """
    if isinstance(spec, list):
        return trial.suggest_categorical(name, spec)

    if isinstance(spec, dict):
        low  = spec["low"]
        high = spec["high"]
        if spec.get("type") == "int":
            return trial.suggest_int(name, int(low), int(high))
        else:
            return trial.suggest_float(name, float(low), float(high),
                                       log=bool(spec.get("log", False)))

    raise ValueError(
        f"Unrecognised search_space spec for '{name}': {spec!r}\n"
        f"Expected a list or a dict with {{low, high[, log][, type]}}."
    )


# =============================================================================
# BC run locator
# =============================================================================

def _find_bc_run(
    bc_base_dir: Path,
    reservoir:   str,
    policy_type: str,
    bc_run_id:   int | None,
) -> Path:
    """
    Locate the BC run folder for (reservoir, policy_type).

    Scans ``bc_base_dir`` for folders named ``<int>_<policy_type>`` that
    contain both ``best_config.json`` and ``model.pt`` (i.e. a completed run).

    Parameters
    ----------
    bc_base_dir : Path to results/<reservoir>/behavioral_cloning/
    reservoir   : Reservoir name (used only for error messages).
    policy_type : Required policy distribution type ("beta", "lognormal",
                  "hardgating", or "softgating").
    bc_run_id   : If provided, validate that the specific run exists and
                  matches ``policy_type``.  If None, auto-selects the
                  highest-numbered completed run for ``policy_type``.

    Exits
    -----
    Exits with a clear, actionable error if:
      • No BC results directory exists for this reservoir.
      • No completed BC run exists for (reservoir, policy_type).
      • ``bc_run_id`` is given but doesn't match ``policy_type``.
      • ``bc_run_id`` is given but has no best_config.json / model.pt.
    """
    import re as _re
    _PATTERN = _re.compile(r"^(\d+)_(.+)$")

    # ------------------------------------------------------------------
    # Guard: results directory must exist
    # ------------------------------------------------------------------
    if not bc_base_dir.exists():
        sys.exit(
            f"\nERROR: No BC results directory found for reservoir '{reservoir}'.\n"
            f"  Expected: {bc_base_dir}\n"
            f"\n  Fix:\n"
            f"    python behavioral_cloning/tune.py  --reservoir {reservoir}\n"
            f"    python behavioral_cloning/train.py --reservoir {reservoir} --run_id <id>\n"
        )

    # ------------------------------------------------------------------
    # Scan for all runs, separating by policy_type and completion status
    # ------------------------------------------------------------------
    completed: list[tuple[int, Path]] = []   # (run_id, path) for target policy_type
    all_runs:  dict[int, str]         = {}   # run_id → policy_type (any type)

    for d in sorted(bc_base_dir.iterdir()):
        m = _PATTERN.match(d.name)
        if not (d.is_dir() and m):
            continue
        rid   = int(m.group(1))
        ptype = m.group(2)
        all_runs[rid] = ptype
        if ptype == policy_type:
            has_cfg   = (d / "best_config.json").exists()
            has_model = (d / "model.pt").exists()
            if has_cfg and has_model:
                completed.append((rid, d))

    # ------------------------------------------------------------------
    # Guard: at least one completed run must exist for this policy_type
    # ------------------------------------------------------------------
    if not completed:
        avail = sorted(f"{rid}_{pt}" for rid, pt in all_runs.items())
        avail_str = ", ".join(avail) if avail else "none"
        sys.exit(
            f"\nERROR: No completed BC tuning found for:\n"
            f"  reservoir   = '{reservoir}'\n"
            f"  policy_type = '{policy_type}'\n"
            f"\n  A completed run requires both best_config.json and model.pt.\n"
            f"\n  Fix:\n"
            f"    python behavioral_cloning/tune.py  --reservoir {reservoir}\n"
            f"    python behavioral_cloning/train.py --reservoir {reservoir} --run_id <id>\n"
            f"\n  BC runs currently available for this reservoir: {avail_str}\n"
        )

    # ------------------------------------------------------------------
    # If no specific run requested, auto-select highest-numbered completed run
    # ------------------------------------------------------------------
    if bc_run_id is None:
        chosen_id, chosen_dir = max(completed, key=lambda x: x[0])
        print(
            f"\n  --bc_run_id not specified; "
            f"auto-selected BC run {chosen_id}_{policy_type}"
        )
        return chosen_dir

    # ------------------------------------------------------------------
    # Specific run requested — validate it exists and matches policy_type
    # ------------------------------------------------------------------
    for rid, d in completed:
        if rid == bc_run_id:
            return d

    # Three distinct failure modes for a specific bc_run_id:
    if bc_run_id in all_runs:
        if all_runs[bc_run_id] == policy_type:
            # Folder exists with correct policy_type but is missing model.pt
            # (tune.py ran but behavioral_cloning/train.py was never run).
            bc_folder = bc_base_dir / f"{bc_run_id}_{policy_type}"
            missing = []
            if not (bc_folder / "best_config.json").exists():
                missing.append("best_config.json")
            if not (bc_folder / "model.pt").exists():
                missing.append("model.pt")
            sys.exit(
                f"\nERROR: BC run {bc_run_id}_{policy_type} is incomplete — "
                f"missing: {', '.join(missing)}\n"
                f"  Run behavioral_cloning/train.py first:\n"
                f"    python behavioral_cloning/train.py "
                f"--reservoir {reservoir} "
                f"--policy_type {policy_type} "
                f"--run_id {bc_run_id}\n"
                f"  Completed '{policy_type}' runs: "
                f"{sorted(r for r, _ in completed)}\n"
            )
        else:
            # Folder exists but belongs to a different policy_type
            sys.exit(
                f"\nERROR: BC run {bc_run_id} exists but has "
                f"policy_type='{all_runs[bc_run_id]}', not '{policy_type}'.\n"
                f"  Pass --bc_run_id matching a '{policy_type}' run, "
                f"or omit --bc_run_id to auto-select.\n"
                f"  Completed '{policy_type}' runs: "
                f"{sorted(r for r, _ in completed)}\n"
            )

    # bc_run_id doesn't exist at all
    sys.exit(
        f"\nERROR: BC run_id={bc_run_id} not found for "
        f"policy_type='{policy_type}' in:\n"
        f"  {bc_base_dir}\n"
        f"  Completed '{policy_type}' runs: "
        f"{sorted(r for r, _ in completed)}\n"
    )


# =============================================================================
# Objective factory
# =============================================================================

def _make_airl_objective(
    algo_cfg:     dict,
    data,                       # DataSplits from load_reservoir_data()
    raw_splits:   tuple,        # (train_df, val_df, test_df) from _load_raw_splits()
    res_cfg:      dict,
    bc_state_dict: dict,        # BC policy state_dict (loaded once, shared read-only)
    bc_cfg_dict:  dict,         # BCConfig fields from best_config.json["config"]
    policy_type:  str,
    device_str:   str,
):
    """
    Return a closure that Optuna calls on every trial.

    All heavy objects (DataSplits, raw DataFrames, BC state_dict) are captured
    once.  Each trial creates its own independent AIRLAgent and environments.
    """
    search_space = algo_cfg["search_space"]
    train_df, val_df, _ = raw_splits

    def objective(trial: optuna.Trial) -> float:
        # ------------------------------------------------------------------
        # 1. Sample AIRL hyperparameters
        # ------------------------------------------------------------------
        sampled = {
            name: _sample_param(trial, name, spec)
            for name, spec in search_space.items()
        }

        # ------------------------------------------------------------------
        # 2. Build AIRLConfig
        #    BC architecture fields come from bc_cfg_dict (not tuned here).
        #    AIRL-specific fields come from Optuna.
        # ------------------------------------------------------------------
        config = AIRLConfig(
            # Data dimensions
            state_dim  = data.state_dim,
            action_dim = 1,

            # BC policy architecture (fixed from best_config.json)
            hidden_dim      = int(bc_cfg_dict.get("hidden_dim",      _bc_default("hidden_dim"))),
            n_hidden_layers = int(bc_cfg_dict.get("n_hidden_layers",  _bc_default("n_hidden_layers"))),
            dropout         = float(bc_cfg_dict.get("dropout",        _bc_default("dropout"))),
            alpha_min       = float(bc_cfg_dict.get("alpha_min",      _bc_default("alpha_min"))),
            alpha_max       = float(bc_cfg_dict.get("alpha_max",      _bc_default("alpha_max"))),
            beta_min        = float(bc_cfg_dict.get("beta_min",       _bc_default("beta_min"))),
            beta_max        = float(bc_cfg_dict.get("beta_max",       _bc_default("beta_max"))),
            sigma_min       = float(bc_cfg_dict.get("sigma_min",      _bc_default("sigma_min"))),
            log_epsilon     = float(bc_cfg_dict.get("log_epsilon",    _bc_default("log_epsilon"))),
            zero_threshold  = float(bc_cfg_dict.get("zero_threshold", _bc_default("zero_threshold"))),
            mse_weight      = float(bc_cfg_dict.get("mse_weight",     _bc_default("mse_weight"))),
            gate_weight     = float(bc_cfg_dict.get("gate_weight",    _bc_default("gate_weight"))),

            # Critic (Optuna-tuned)
            critic_hidden_dim      = int(sampled["critic_hidden_dim"]),
            critic_n_hidden_layers = int(sampled["critic_n_hidden_layers"]),

            # Discriminator (Optuna-tuned)
            disc_hidden_dim      = int(sampled["disc_hidden_dim"]),
            disc_n_hidden_layers = int(sampled["disc_n_hidden_layers"]),
            disc_dropout         = float(sampled["disc_dropout"]),

            # Learning rates (Optuna-tuned)
            lr_policy        = float(sampled["lr_policy"]),
            lr_critic        = float(sampled["lr_critic"]),
            lr_discriminator = float(sampled["lr_discriminator"]),

            # Discriminator training (Optuna-tuned)
            disc_updates            = int(sampled["disc_updates"]),
            warmup_disc_updates     = int(sampled["warmup_disc_updates"]),
            gradient_penalty_coef   = float(sampled["gradient_penalty_coef"]),
            label_smoothing_epsilon = float(sampled["label_smoothing_epsilon"]),

            # PPO (Optuna-tuned)
            gamma        = float(sampled["gamma"]),
            gae_lambda   = float(sampled["gae_lambda"]),
            clip_epsilon  = float(sampled["clip_epsilon"]),
            entropy_coef  = float(sampled["entropy_coef"]),
            ppo_epochs    = int(sampled["ppo_epochs"]),

            # KL regularisation (Optuna-tuned)
            kl_regularization_coef = float(sampled["kl_regularization_coef"]),

            # Training schedule (Optuna-tuned)
            warmup_iterations       = int(sampled["warmup_iterations"]),
            num_iterations          = int(sampled["num_iterations"]),
            steps_per_iteration     = int(sampled["steps_per_iteration"]),
            batch_size              = int(sampled["batch_size"]),
            early_stopping_patience = int(sampled["early_stopping_patience"]),

            # Runtime
            device  = device_str,
            seed    = trial.number,  # different seed per trial for diversity
            verbose = False,
        )

        # ------------------------------------------------------------------
        # 3. Reproducibility seed
        # ------------------------------------------------------------------
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)

        # ------------------------------------------------------------------
        # 4. Reconstruct BC policy for this trial (independent copy)
        # ------------------------------------------------------------------
        # Build a BCConfig-compatible object so build_policy_network works.
        # AIRLConfig shares all BC fields with BCConfig, so we can pass it
        # directly — build_policy_network only reads the relevant fields.
        bc_policy = build_policy_network(policy_type, config).to(
            torch.device(device_str)
        )
        bc_policy.load_state_dict(
            {k: v.clone() for k, v in bc_state_dict.items()}
        )

        # ------------------------------------------------------------------
        # 5. Build environments
        # ------------------------------------------------------------------
        train_env = ReservoirEnvironment(train_df, config, data.normalizer, res_cfg)
        val_env   = ReservoirEnvironment(val_df,   config, data.normalizer, res_cfg)

        # ------------------------------------------------------------------
        # 6. Instantiate agent and load expert data
        # ------------------------------------------------------------------
        agent = AIRLAgent(config, bc_policy, policy_type)
        agent.add_expert_from_split(data.train)

        # ------------------------------------------------------------------
        # 7. Discriminator warmup
        # ------------------------------------------------------------------
        agent.warmup_discriminator(train_env, config.warmup_iterations)

        # ------------------------------------------------------------------
        # 8. Main adversarial training loop
        # ------------------------------------------------------------------
        result = agent.train(train_env, val_env, data.val, trial=trial)

        best_score = float(result["best_val_score"])

        # ------------------------------------------------------------------
        # 9. Cleanup — free GPU/CPU memory before next trial
        # ------------------------------------------------------------------
        del agent, bc_policy, train_env, val_env
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if not np.isfinite(best_score):
            return -float("inf")

        return best_score

    return objective


# =============================================================================
# CLI argument parsing
# =============================================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="AIRL hyperparameter tuning with Optuna.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required
    p.add_argument(
        "--reservoir", required=True,
        help="Reservoir name — must match configs/reservoirs/<name>.yaml.",
    )
    p.add_argument(
        "--policy_type", required=True,
        choices=["beta", "lognormal", "hardgating", "softgating"],
        help=(
            "Policy distribution type.  Must match the BC run being loaded.  "
            "Used to locate results/<reservoir>/behavioral_cloning/*_<policy_type>/ "
            "and to validate the checkpoint before training starts."
        ),
    )
    p.add_argument(
        "--bc_run_id", type=int, default=None,
        help=(
            "Integer run_id of the BC run to load weights from.  "
            "Matches the folder <bc_run_id>_<policy_type> under "
            "results/<reservoir>/behavioral_cloning/.  "
            "If omitted, the highest-numbered completed run for "
            "--policy_type is selected automatically."
        ),
    )

    # Algorithm config overrides
    p.add_argument(
        "--device", default=None,
        help=(
            "Override runtime.device in airl.yaml: "
            "auto | cpu | cuda | cuda:N | mps.  "
            "'auto' selects GPU if available, otherwise CPU."
        ),
    )
    p.add_argument(
        "--num_workers", type=int, default=None,
        help="Override runtime.num_workers (parallel Optuna jobs).",
    )
    p.add_argument(
        "--n_trials", type=int, default=None,
        help="Override optuna.n_trials in airl.yaml.",
    )
    p.add_argument(
        "--run_id", type=int, default=None,
        help=(
            "Integer run identifier for the AIRL results folder.  "
            "If omitted, auto-increments from existing folders in "
            "results/<reservoir>/airl/.  "
            "Folder name: <run_id>_<policy_type>."
        ),
    )

    return p.parse_args()


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args = _parse_args()

    res_cfg_path  = _ROOT / "configs" / "reservoirs" / f"{args.reservoir}.yaml"
    algo_cfg_path = _ROOT / "configs" / "algorithms" / "airl.yaml"

    # ------------------------------------------------------------------
    # Validate config files exist
    # ------------------------------------------------------------------
    if not res_cfg_path.exists():
        sys.exit(
            f"\nERROR: Reservoir config not found: {res_cfg_path}\n"
            f"  Available: configs/reservoirs/*.yaml\n"
        )
    if not algo_cfg_path.exists():
        sys.exit(
            f"\nERROR: Algorithm config not found: {algo_cfg_path}\n"
            f"  Expected: configs/algorithms/airl.yaml\n"
        )

    # ------------------------------------------------------------------
    # Load configs
    # ------------------------------------------------------------------
    with open(res_cfg_path,  "r") as f:
        res_cfg = yaml.safe_load(f)
    with open(algo_cfg_path, "r") as f:
        algo_cfg = yaml.safe_load(f)

    # ------------------------------------------------------------------
    # Apply CLI overrides and write back to YAML for reproducibility
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Resolve runtime settings
    # ------------------------------------------------------------------
    device_str = _resolve_device(algo_cfg["runtime"]["device"])
    n_jobs     = (
        algo_cfg["runtime"]["num_workers"]
        if algo_cfg["runtime"]["num_workers"] is not None
        else algo_cfg["optuna"]["n_jobs"]
    )
    n_trials = algo_cfg["optuna"]["n_trials"]

    # ------------------------------------------------------------------
    # Locate and load BC checkpoint
    # Validates that BC tuning has been completed for (reservoir, policy_type)
    # before any expensive data loading or Optuna setup happens.
    # ------------------------------------------------------------------
    bc_base_dir = _ROOT / "results" / args.reservoir / "behavioral_cloning"
    bc_run_dir  = _find_bc_run(
        bc_base_dir = bc_base_dir,
        reservoir   = args.reservoir,
        policy_type = args.policy_type,
        bc_run_id   = args.bc_run_id,
    )

    bc_config_path = bc_run_dir / "best_config.json"
    bc_model_path  = bc_run_dir / "model.pt"

    with open(bc_config_path, "r") as f:
        bc_saved = json.load(f)

    # Cross-validate JSON fields against CLI args — catches any file-system
    # inconsistency that _find_bc_run could not detect from folder names alone.
    if bc_saved.get("reservoir") != args.reservoir:
        sys.exit(
            f"\nERROR: BC checkpoint reservoir mismatch.\n"
            f"  CLI --reservoir  = '{args.reservoir}'\n"
            f"  JSON reservoir   = '{bc_saved.get('reservoir')}'\n"
            f"  Checkpoint: {bc_config_path}\n"
        )
    if str(bc_saved.get("policy_type")) != args.policy_type:
        sys.exit(
            f"\nERROR: BC checkpoint policy_type mismatch.\n"
            f"  CLI --policy_type  = '{args.policy_type}'\n"
            f"  JSON policy_type   = '{bc_saved.get('policy_type')}'\n"
            f"  Checkpoint: {bc_config_path}\n"
        )

    policy_type = args.policy_type   # confirmed against both folder name and JSON
    bc_cfg_dict = bc_saved["config"]

    print(f"\nLoaded BC checkpoint : {bc_run_dir.name}")
    print(f"  Reservoir   : {args.reservoir}")
    print(f"  Policy type : {policy_type}")
    print(f"  BC val score: {bc_saved['best_score']:.4f}")

    # Load BC weights — keep on CPU; the objective will move to device per trial
    bc_ckpt = torch.load(bc_model_path, map_location="cpu")
    bc_state_dict = bc_ckpt["model_state_dict"]

    # ------------------------------------------------------------------
    # Load and split data
    # ------------------------------------------------------------------
    print(f"\nLoading data for reservoir '{args.reservoir}' …")
    data       = load_reservoir_data(res_cfg, res_cfg_path)
    raw_splits = _load_raw_splits(res_cfg, res_cfg_path)

    print(
        f"  state_dim    = {data.state_dim}\n"
        f"  train rows   = {len(data.train.states)}\n"
        f"  val rows     = {len(data.val.states)}\n"
        f"  test rows    = {len(data.test.states)}\n"
        f"  device       = {device_str}"
    )

    # ------------------------------------------------------------------
    # Guard: state_dim from current YAML must match the BC checkpoint.
    # Catches the common mistake of toggling use_month_encoding in the
    # YAML after BC training has already run.
    # ------------------------------------------------------------------
    bc_state_dim   = int(bc_cfg_dict["state_dim"])
    if data.state_dim != bc_state_dim:
        n_raw_cols     = len(list(res_cfg["columns"]["state"]))
        bc_used_month  = (bc_state_dim == n_raw_cols + 2)
        cur_uses_month = bool(res_cfg["columns"].get("use_month_encoding", True))
        sys.exit(
            f"\nERROR: state_dim mismatch — BC checkpoint and current YAML are incompatible.\n"
            f"  BC checkpoint  ({bc_run_dir.name})  state_dim = {bc_state_dim}"
            f"  (use_month_encoding={'true' if bc_used_month else 'false'})\n"
            f"  Current YAML                        state_dim = {data.state_dim}"
            f"  (use_month_encoding={'true' if cur_uses_month else 'false'})\n\n"
            f"  Fix — update configs/reservoirs/{args.reservoir}.yaml:\n"
            f"    use_month_encoding: {'true' if bc_used_month else 'false'}\n\n"
            f"  Alternatively, retrain BC with use_month_encoding="
            f"{'true' if cur_uses_month else 'false'} to match the current YAML,\n"
            f"  then pass that run's --bc_run_id.\n"
        )

    # ------------------------------------------------------------------
    # Output directory
    # ------------------------------------------------------------------
    airl_base_dir = _ROOT / "results" / args.reservoir / "airl"
    run_id, results_dir = _resolve_run_id(airl_base_dir, policy_type, args.run_id)
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nRun folder : {results_dir.name}  (run_id={run_id})")

    # ------------------------------------------------------------------
    # Optuna study
    # ------------------------------------------------------------------
    study = optuna.create_study(
        direction  = "maximize",
        sampler    = optuna.samplers.TPESampler(seed=42),
        pruner     = optuna.pruners.MedianPruner(n_warmup_steps=5),
        study_name = f"{args.reservoir}_airl",
    )

    objective = _make_airl_objective(
        algo_cfg      = algo_cfg,
        data          = data,
        raw_splits    = raw_splits,
        res_cfg       = res_cfg,
        bc_state_dict = bc_state_dict,
        bc_cfg_dict   = bc_cfg_dict,
        policy_type   = policy_type,
        device_str    = device_str,
    )

    print(f"\nStarting Optuna search: {n_trials} trials, {n_jobs} job(s) …\n")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)

    # ------------------------------------------------------------------
    # Guard: ensure at least one trial completed
    # ------------------------------------------------------------------
    completed = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
    ]
    if not completed:
        sys.exit(
            "\nAll trials failed or were pruned.\n"
            "Check device availability, BC checkpoint, and search space bounds.\n"
        )

    # ------------------------------------------------------------------
    # Reconstruct best AIRLConfig from trial params
    # ------------------------------------------------------------------
    best = study.best_trial
    p    = best.params

    best_config = AIRLConfig(
        state_dim  = data.state_dim,
        action_dim = 1,

        # BC architecture — carried forward unchanged
        hidden_dim      = int(bc_cfg_dict.get("hidden_dim",      _bc_default("hidden_dim"))),
        n_hidden_layers = int(bc_cfg_dict.get("n_hidden_layers",  _bc_default("n_hidden_layers"))),
        dropout         = float(bc_cfg_dict.get("dropout",        _bc_default("dropout"))),
        alpha_min       = float(bc_cfg_dict.get("alpha_min",      _bc_default("alpha_min"))),
        alpha_max       = float(bc_cfg_dict.get("alpha_max",      _bc_default("alpha_max"))),
        beta_min        = float(bc_cfg_dict.get("beta_min",       _bc_default("beta_min"))),
        beta_max        = float(bc_cfg_dict.get("beta_max",       _bc_default("beta_max"))),
        sigma_min       = float(bc_cfg_dict.get("sigma_min",      _bc_default("sigma_min"))),
        log_epsilon     = float(bc_cfg_dict.get("log_epsilon",    _bc_default("log_epsilon"))),
        zero_threshold  = float(bc_cfg_dict.get("zero_threshold", _bc_default("zero_threshold"))),
        mse_weight      = float(bc_cfg_dict.get("mse_weight",     _bc_default("mse_weight"))),
        gate_weight     = float(bc_cfg_dict.get("gate_weight",    _bc_default("gate_weight"))),

        # AIRL fields from best trial
        critic_hidden_dim      = int(p["critic_hidden_dim"]),
        critic_n_hidden_layers = int(p["critic_n_hidden_layers"]),
        disc_hidden_dim        = int(p["disc_hidden_dim"]),
        disc_n_hidden_layers   = int(p["disc_n_hidden_layers"]),
        disc_dropout           = float(p["disc_dropout"]),
        lr_policy              = float(p["lr_policy"]),
        lr_critic              = float(p["lr_critic"]),
        lr_discriminator       = float(p["lr_discriminator"]),
        disc_updates           = int(p["disc_updates"]),
        warmup_disc_updates    = int(p["warmup_disc_updates"]),
        gradient_penalty_coef  = float(p["gradient_penalty_coef"]),
        label_smoothing_epsilon= float(p["label_smoothing_epsilon"]),
        gamma                  = float(p["gamma"]),
        gae_lambda             = float(p["gae_lambda"]),
        clip_epsilon           = float(p["clip_epsilon"]),
        entropy_coef           = float(p["entropy_coef"]),
        ppo_epochs             = int(p["ppo_epochs"]),
        kl_regularization_coef = float(p["kl_regularization_coef"]),
        warmup_iterations      = int(p["warmup_iterations"]),
        num_iterations         = int(p["num_iterations"]),
        steps_per_iteration    = int(p["steps_per_iteration"]),
        batch_size             = int(p["batch_size"]),
        early_stopping_patience= int(p["early_stopping_patience"]),

        device  = device_str,
        seed    = 42,
        verbose = False,
    )

    # ------------------------------------------------------------------
    # Save best_config.json
    # ------------------------------------------------------------------
    save_dict = {
        "reservoir":    args.reservoir,
        "policy_type":  policy_type,
        "best_score":   best.value,
        "trial_number": best.number,
        "bc_run_id":    args.bc_run_id,
        "bc_run_folder":bc_run_dir.name,
        "config":       asdict(best_config),
    }

    out_path = results_dir / "best_config.json"
    with open(out_path, "w") as f:
        json.dump(save_dict, f, indent=2)

    print(f"\nBest trial #{best.number}  score = {best.value:.4f}")
    print(f"Best config saved → {out_path}\n")

    # ------------------------------------------------------------------
    # Save run_args.json
    # ------------------------------------------------------------------
    run_args_path = results_dir / "run_args.json"
    run_args: dict = {}
    if run_args_path.exists():
        with open(run_args_path, "r") as f:
            run_args = json.load(f)

    run_args["tune"] = {
        "reservoir":   args.reservoir,
        "bc_run_id":   args.bc_run_id,
        "bc_run_folder": bc_run_dir.name,
        "run_id":      run_id,
        "folder":      results_dir.name,
        "policy_type": policy_type,
        "device":      args.device,
        "device_used": device_str,
        "num_workers": args.num_workers,
        "n_trials":    args.n_trials,
        "timestamp":   datetime.now().isoformat(timespec="seconds"),
    }

    with open(run_args_path, "w") as f:
        json.dump(run_args, f, indent=2)
    print(f"Run args saved  → {run_args_path}\n")


if __name__ == "__main__":
    main()
