"""
iqlearn/train.py
================
Final model training for IQ-Learn (Inverse Q-Learning).

MUST be run AFTER iqlearn/tune.py.  Reads all hyperparameters from
results/<reservoir>/iqlearn/<run_id>_<policy_type>/best_config.json, loads the
corresponding BC checkpoint to initialise the actor, and runs a single full
IQ-Learn training run with those parameters.

If best_config.json is not found, the script exits with a clear error and
instructions.

Training strategy
-----------------
IQ-Learn is fully batch-based (no environment rollouts).

Phase 1 -- critic warm-up (critic_warm_up_epochs):
    Only the critic is updated; actor weights are frozen.
    Lets the Q-function stabilize before joint optimisation begins.

Phase 2 -- joint training (n_epochs):
    Alternates critic and actor updates.
    Validates every eval_interval epochs on the VAL split.
    Early stopping on the composite score
    (release / storage correlation + NRMSE).

The TEST split is never touched here -- it is reserved for
iqlearn/generate_results.py.

Outputs (written into the same run folder as tune.py)
------------------------------------------------------
results/<reservoir>/iqlearn/<run_id>_<policy_type>/
    model.pt        -- actor, critic, critic_target weights + metadata.
    train_log.json  -- per-eval training history and val metrics.
    run_args.json   -- updated with this run's CLI arguments and metadata.

Usage
-----
# Standard -- reads everything from best_config.json
python iqlearn/train.py --reservoir conchas --policy_type hardgating --run_id 1

# Override device
python iqlearn/train.py --reservoir conchas --policy_type hardgating --run_id 1 \\
    --device cpu

# Override seed (ensemble member with different initialisation)
python iqlearn/train.py --reservoir conchas --policy_type hardgating --run_id 1 \\
    --seed 7
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml

# ---------------------------------------------------------------------------
# Project root on sys.path so sibling packages resolve correctly.
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from behavioral_cloning.tune import _resolve_device
from networks.policy         import build_policy_network
from utils.data              import load_reservoir_data
from utils.runs              import _find_run_folder
from iqlearn.core            import IQLearnConfig, IQLearnAgent, ExpertBuffer


# =============================================================================
# Best-config loader
# =============================================================================

def _load_best_config(results_dir: Path, reservoir: str, policy_type: str) -> dict:
    """
    Load and validate best_config.json produced by iqlearn/tune.py.

    Performs five checks before returning:
      1. File exists (tune.py was run).
      2. File is valid JSON (not corrupted).
      3. Required top-level keys are present.
      4. Required sub-keys inside "config" are present.
      5. Saved reservoir and policy_type match the CLI arguments.

    Warns (does not exit) if best_score is very low.

    Parameters
    ----------
    results_dir : Path to results/<reservoir>/iqlearn/<run_id>_<policy_type>/
    reservoir   : Reservoir name from the CLI.
    policy_type : Policy type from the CLI.

    Returns
    -------
    dict  Parsed contents of best_config.json.
    """
    path = results_dir / "best_config.json"

    # ------------------------------------------------------------------
    # 1. File existence
    # ------------------------------------------------------------------
    if not path.exists():
        sys.exit(
            f"\nERROR: best_config.json not found.\n"
            f"  Expected: {path}\n\n"
            f"  iqlearn/tune.py must be run before iqlearn/train.py.  Run:\n"
            f"    python iqlearn/tune.py --reservoir {reservoir} "
            f"--policy_type {policy_type}\n"
        )

    # ------------------------------------------------------------------
    # 2. Valid JSON
    # ------------------------------------------------------------------
    try:
        with open(path, "r") as f:
            saved = json.load(f)
    except json.JSONDecodeError as e:
        sys.exit(
            f"\nERROR: best_config.json is not valid JSON.\n"
            f"  Error: {e}\n"
            f"  File:  {path}\n\n"
            f"  The file may be corrupted.  Re-run tune.py to regenerate it.\n"
        )
    except OSError as e:
        sys.exit(f"\nERROR: Cannot read best_config.json.\n  {e}\n")

    # ------------------------------------------------------------------
    # 3. Required top-level keys
    # ------------------------------------------------------------------
    required_top = {"reservoir", "policy_type", "best_score", "bc_run_folder", "config"}
    missing_top  = required_top - set(saved.keys())
    if missing_top:
        sys.exit(
            f"\nERROR: best_config.json is missing required top-level keys: "
            f"{sorted(missing_top)}\n"
            f"  File: {path}\n\n"
            f"  The file may be from an older version of tune.py.  "
            f"Re-run tune.py to regenerate it.\n"
        )

    # ------------------------------------------------------------------
    # 4. Required sub-keys inside "config"
    # ------------------------------------------------------------------
    required_cfg = {
        "state_dim", "action_dim",
        "actor_hidden_dim", "actor_n_hidden_layers", "dropout",
        "critic_hidden_dim", "critic_n_hidden_layers",
        "critic_warm_up_epochs", "n_epochs", "batch_size",
        "learning_rate_actor", "learning_rate_critic",
        "gamma", "tau", "alpha_entropy", "alpha_regularization", "lambda_bc",
        "device", "seed",
    }
    cfg_dict    = saved["config"]
    missing_cfg = required_cfg - set(cfg_dict.keys())
    if missing_cfg:
        sys.exit(
            f"\nERROR: best_config.json['config'] is missing keys: "
            f"{sorted(missing_cfg)}\n"
            f"  File: {path}\n\n"
            f"  The file may be from an older version of tune.py.  "
            f"Re-run tune.py to regenerate it.\n"
        )

    # ------------------------------------------------------------------
    # 5. Reservoir and policy_type match
    # ------------------------------------------------------------------
    if saved["reservoir"] != reservoir:
        sys.exit(
            f"\nERROR: best_config.json was tuned for reservoir "
            f"'{saved['reservoir']}', but you are training for '{reservoir}'.\n\n"
            f"  These must match.  Run tune.py for this reservoir first:\n"
            f"    python iqlearn/tune.py --reservoir {reservoir} "
            f"--policy_type {policy_type}\n"
        )
    if str(saved["policy_type"]) != policy_type:
        sys.exit(
            f"\nERROR: best_config.json has policy_type='{saved['policy_type']}' "
            f"but --policy_type='{policy_type}' was requested.\n\n"
            f"  Pass --policy_type {saved['policy_type']} to match this run, or "
            f"run tune.py with --policy_type {policy_type} to create a new run.\n"
        )

    # ------------------------------------------------------------------
    # Warning: suspiciously low validation score
    # ------------------------------------------------------------------
    best_score = saved["best_score"]
    if best_score < 0.1:
        print(
            f"\nWARNING: Tuning achieved a very low validation score "
            f"({best_score:.4f}).\n"
            f"  The trained model may not generalise well.\n"
            f"  Consider re-running tune.py with more trials or a wider "
            f"search space.\n",
            file=sys.stderr,
        )

    return saved


# =============================================================================
# CLI argument parsing
# =============================================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Train the best IQ-Learn policy for a reservoir.  "
            "Requires iqlearn/tune.py to have been run first."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--reservoir", required=True,
        help="Reservoir name -- must match configs/reservoirs/<name>.yaml.",
    )
    p.add_argument(
        "--policy_type", default=None,
        choices=["beta", "lognormal", "hardgating", "softgating"],
        help=(
            "Policy distribution type.  If omitted, inferred automatically "
            "from the run folder name (e.g. '1_hardgating' → hardgating).  "
            "Only needed to disambiguate if the folder name is unexpected."
        ),
    )
    p.add_argument(
        "--run_id", type=int, required=True,
        help=(
            "Integer run identifier matching the folder created by tune.py "
            "(e.g. 1 for folder '1_hardgating')."
        ),
    )
    p.add_argument(
        "--device", default=None,
        help=(
            "Override the device stored in best_config.json.  "
            "Useful when moving from a GPU cluster to a local machine.  "
            "Options: auto | cpu | cuda | cuda:N | mps."
        ),
    )
    p.add_argument(
        "--seed", type=int, default=None,
        help=(
            "Override the seed stored in best_config.json.  "
            "Use to train an ensemble member with a different initialisation."
        ),
    )
    p.add_argument(
        "--verbose", action="store_true",
        help="Print per-epoch progress during training.",
    )
    return p.parse_args()


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args = _parse_args()

    res_cfg_path  = _ROOT / "configs" / "reservoirs" / f"{args.reservoir}.yaml"
    iqlearn_base  = _ROOT / "results" / args.reservoir / "iqlearn"

    # ------------------------------------------------------------------
    # Validate reservoir config exists
    # ------------------------------------------------------------------
    if not res_cfg_path.exists():
        sys.exit(
            f"\nERROR: Reservoir config not found: {res_cfg_path}\n"
            f"  Available: configs/reservoirs/*.yaml\n"
        )

    with open(res_cfg_path, "r") as f:
        res_cfg = yaml.safe_load(f)

    # ------------------------------------------------------------------
    # Locate the IQ-Learn run folder (e.g. results/<res>/iqlearn/1_hardgating/)
    # ------------------------------------------------------------------
    results_dir = _find_run_folder(iqlearn_base, args.run_id)

    # ------------------------------------------------------------------
    # Infer policy_type from folder name if not supplied on CLI.
    # Folder convention: <run_id>_<policy_type>  e.g. 1_hardgating
    # ------------------------------------------------------------------
    if args.policy_type is None:
        parts = results_dir.name.split("_", 1)
        if len(parts) != 2 or not parts[1]:
            sys.exit(
                f"\nERROR: Cannot infer policy_type from folder name "
                f"'{results_dir.name}'.\n"
                f"  Pass --policy_type explicitly.\n"
            )
        args.policy_type = parts[1]
        print(f"Policy type inferred from folder: {args.policy_type}")

    # ------------------------------------------------------------------
    # Load and validate best_config.json
    # ------------------------------------------------------------------
    saved = _load_best_config(results_dir, args.reservoir, args.policy_type)

    print(f"\nLoaded best_config.json")
    print(f"  Reservoir    : {saved['reservoir']}")
    print(f"  Policy type  : {saved['policy_type']}")
    print(f"  Best score   : {saved['best_score']:.4f}  "
          f"(trial #{saved.get('trial_number', '?')})")
    print(f"  BC run folder: {saved['bc_run_folder']}")

    policy_type = args.policy_type
    cfg_dict    = saved["config"]

    # ------------------------------------------------------------------
    # Reconstruct IQLearnConfig from JSON.
    # IQLearnConfig.from_dict drops any unknown keys gracefully.
    # ------------------------------------------------------------------
    config = IQLearnConfig.from_dict(cfg_dict)

    # ------------------------------------------------------------------
    # Apply CLI overrides
    # ------------------------------------------------------------------
    raw_device = args.device if args.device is not None else config.device
    resolved   = _resolve_device(raw_device)

    if resolved.startswith("cuda") and not torch.cuda.is_available():
        print(
            f"\nWARNING: Requested device '{resolved}' but CUDA is not available.  "
            f"Falling back to CPU.\n",
            file=sys.stderr,
        )
        resolved = "cpu"
    elif resolved == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        print(
            f"\nWARNING: Requested device 'mps' but MPS is not available.  "
            f"Falling back to CPU.\n",
            file=sys.stderr,
        )
        resolved = "cpu"

    config.device  = resolved
    config.verbose = args.verbose

    if args.seed is not None:
        print(f"\nSeed overridden to {args.seed} (was {cfg_dict['seed']}).")
        config.seed = args.seed

    print(f"\nDevice  : {config.device}")
    print(f"Seed    : {config.seed}")

    # ------------------------------------------------------------------
    # Reproducibility
    # ------------------------------------------------------------------
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    # ------------------------------------------------------------------
    # Locate and load BC checkpoint.
    # bc_run_folder is the folder name, e.g. "1_hardgating".
    # ------------------------------------------------------------------
    bc_model_path = (
        _ROOT / "results" / args.reservoir
        / "behavioral_cloning" / saved["bc_run_folder"] / "model.pt"
    )
    if not bc_model_path.exists():
        sys.exit(
            f"\nERROR: BC model checkpoint not found.\n"
            f"  Expected: {bc_model_path}\n\n"
            f"  Run behavioral_cloning/train.py for this run first:\n"
            f"    python behavioral_cloning/train.py --reservoir {args.reservoir} "
            f"--run_id {saved.get('bc_run_id', '?')}\n"
        )

    bc_ckpt       = torch.load(bc_model_path, map_location="cpu", weights_only=False)
    bc_state_dict = bc_ckpt["model_state_dict"]
    print(f"\nLoaded BC checkpoint : {saved['bc_run_folder']}")

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print(f"\nLoading data for reservoir '{args.reservoir}' ...")
    data = load_reservoir_data(res_cfg, res_cfg_path)

    print(
        f"  state_dim  = {data.state_dim}\n"
        f"  train rows = {len(data.train.states)}\n"
        f"  val rows   = {len(data.val.states)}\n"
        f"  test rows  = {len(data.test.states)}"
    )

    # ------------------------------------------------------------------
    # Validate state_dim consistency
    # ------------------------------------------------------------------
    saved_dim = cfg_dict["state_dim"]
    if saved_dim != data.state_dim:
        sys.exit(
            f"\nERROR: state_dim mismatch.\n"
            f"  best_config.json : state_dim = {saved_dim}\n"
            f"  Current data     : state_dim = {data.state_dim}\n\n"
            f"  The state variables or month-encoding setting may have changed "
            f"since tuning.\n"
            f"  Re-run iqlearn/tune.py to obtain a matching best_config.json.\n"
        )

    # ------------------------------------------------------------------
    # Build actor from BC architecture and load BC weights.
    # build_policy_network reads config.hidden_dim and config.n_hidden_layers,
    # which are forwarded from config.actor_hidden_dim / actor_n_hidden_layers
    # via the properties defined in IQLearnConfig.
    # ------------------------------------------------------------------
    device    = torch.device(config.device)
    bc_policy = build_policy_network(policy_type, config).to(device)
    bc_policy.load_state_dict(
        {k: v.clone() for k, v in bc_state_dict.items()}
    )
    print(f"BC policy loaded and moved to {config.device}.")

    # ------------------------------------------------------------------
    # Build agent
    # ------------------------------------------------------------------
    agent = IQLearnAgent(config, bc_policy, policy_type)

    # ------------------------------------------------------------------
    # Build expert buffer from training split
    # ------------------------------------------------------------------
    train_buf = ExpertBuffer(data.train)
    print(f"\nExpert buffer populated: {train_buf.size} transitions.")

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    total_epochs = config.critic_warm_up_epochs + config.n_epochs
    print(
        f"\nStarting IQ-Learn training:\n"
        f"  critic warm-up : {config.critic_warm_up_epochs} epochs\n"
        f"  joint training : {config.n_epochs} epochs\n"
        f"  total          : {total_epochs} epochs\n"
    )

    result = agent.train(
        train_buf = train_buf,
        val_split = data.val,
        trial     = None,   # no Optuna pruning in final training
    )

    best_val_score = float(result["best_val_score"])
    best_epoch     = int(result["best_epoch"])
    training_stats = result["training_stats"]

    print(f"\nTraining complete.")
    print(f"  Best val score  : {best_val_score:.4f}")
    print(f"  Best epoch      : {best_epoch + config.critic_warm_up_epochs} "
          f"(joint epoch {best_epoch})")

    # ------------------------------------------------------------------
    # Save model checkpoint
    # ------------------------------------------------------------------
    model_path = results_dir / "model.pt"
    agent.save(
        model_path,
        best_epoch     = best_epoch,
        best_val_score = best_val_score,
        reservoir      = args.reservoir,
    )
    print(f"\nModel saved  -> {model_path}")

    # ------------------------------------------------------------------
    # Save training log
    # ------------------------------------------------------------------
    train_log = {
        "reservoir":     args.reservoir,
        "policy_type":   policy_type,
        "run_id":        args.run_id,
        "bc_run_folder": saved["bc_run_folder"],
        "device":        config.device,
        "seed":          config.seed,
        "best_val_score": round(best_val_score, 6),
        "best_epoch":    best_epoch,
        "training_stats": {
            k: [round(float(v), 6) for v in vals]
            for k, vals in training_stats.items()
        },
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }

    log_path = results_dir / "train_log.json"
    with open(log_path, "w") as f:
        json.dump(train_log, f, indent=2)
    print(f"Train log saved -> {log_path}")

    # ------------------------------------------------------------------
    # Update run_args.json
    # ------------------------------------------------------------------
    run_args_path = results_dir / "run_args.json"
    run_args: dict = {}
    if run_args_path.exists():
        with open(run_args_path, "r") as f:
            run_args = json.load(f)

    run_args["train"] = {
        "reservoir":      args.reservoir,
        "policy_type":    policy_type,
        "run_id":         args.run_id,
        "bc_run_folder":  saved["bc_run_folder"],
        "device_cli":     args.device,
        "device_used":    config.device,
        "seed_cli":       args.seed,
        "seed_used":      config.seed,
        "verbose":        args.verbose,
        "best_val_score": round(best_val_score, 6),
        "best_epoch":     best_epoch,
        "timestamp":      datetime.now().isoformat(timespec="seconds"),
    }

    with open(run_args_path, "w") as f:
        json.dump(run_args, f, indent=2)
    print(f"Run args updated -> {run_args_path}\n")


if __name__ == "__main__":
    main()
