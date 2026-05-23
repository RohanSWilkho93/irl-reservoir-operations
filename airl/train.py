"""
airl/train.py
=============
Final model training for Adversarial Inverse Reinforcement Learning.

MUST be run AFTER airl/tune.py.  Reads all hyperparameters from
results/<reservoir>/airl/<run_id>_<policy_type>/best_config.json, loads the
corresponding BC checkpoint to initialise the policy, and runs a single full
adversarial training run with those parameters.

If best_config.json is not found, the script exits with a clear error and
instructions.

Training strategy
-----------------
Trains on the TRAINING split.  Validates on the VAL split every
``eval_interval`` iterations.  Early stopping on the composite score
(discriminator balance + release / storage correlation + NRMSE).
The TEST split is never touched here — it is reserved for
airl/generate_results.py.

Outputs (written into the same run folder as tune.py)
------------------------------------------------------
results/<reservoir>/airl/<run_id>_<policy_type>/
    model.pt           — policy, critic, discriminator, and BC-prior weights.
    train_log.json     — per-iteration training history and val metrics.
    run_args.json      — updated with this run's CLI arguments and metadata.

Usage
-----
# Standard — reads everything from best_config.json
python airl/train.py --reservoir garrison --policy_type beta --run_id 1

# Override device
python airl/train.py --reservoir garrison --policy_type beta --run_id 1 \\
    --device cpu

# Override seed (ensemble member with different initialisation)
python airl/train.py --reservoir garrison --policy_type beta --run_id 1 \\
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
from airl.core               import (
    AIRLConfig,
    AIRLAgent,
    ReservoirEnvironment,
    _load_raw_splits,
)


# =============================================================================
# Best-config loader
# =============================================================================

def _load_best_config(results_dir: Path, reservoir: str, policy_type: str) -> dict:
    """
    Load and validate best_config.json produced by airl/tune.py.

    Performs five checks before returning:
      1. File exists (tune.py was run).
      2. File is valid JSON (not corrupted).
      3. Required top-level keys are present.
      4. Required sub-keys inside "config" are present.
      5. Saved reservoir and policy_type match the CLI arguments.

    Warns (does not exit) if best_score is very low.

    Parameters
    ----------
    results_dir  : Path to results/<reservoir>/airl/<run_id>_<policy_type>/
    reservoir    : Reservoir name from the CLI.
    policy_type  : Policy type from the CLI.

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
            f"  airl/tune.py must be run before airl/train.py.  Run:\n"
            f"    python airl/tune.py --reservoir {reservoir} "
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
    required_top = {"reservoir", "policy_type", "best_score",
                    "bc_run_folder", "config"}
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
        "hidden_dim", "n_hidden_layers", "dropout",
        "critic_hidden_dim", "critic_n_hidden_layers",
        "disc_hidden_dim", "disc_n_hidden_layers", "disc_dropout",
        "lr_policy", "lr_critic", "lr_discriminator",
        "disc_updates", "warmup_disc_updates",
        "gradient_penalty_coef", "label_smoothing_epsilon",
        "gamma", "clip_epsilon", "entropy_coef", "ppo_epochs",
        "kl_regularization_coef",
        "warmup_iterations", "num_iterations",
        "steps_per_iteration", "batch_size",
        "early_stopping_patience",
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
            f"    python airl/tune.py --reservoir {reservoir} "
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
            "Train the best AIRL policy for a reservoir.  "
            "Requires airl/tune.py to have been run first."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--reservoir", required=True,
        help="Reservoir name — must match configs/reservoirs/<name>.yaml.",
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
            "(e.g. 1 for folder '1_beta')."
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
        help="Print per-iteration progress during training.",
    )
    return p.parse_args()


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args = _parse_args()

    res_cfg_path = _ROOT / "configs" / "reservoirs" / f"{args.reservoir}.yaml"
    airl_base    = _ROOT / "results" / args.reservoir / "airl"

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
    # Locate the AIRL run folder (e.g. results/<res>/airl/1_beta/)
    # ------------------------------------------------------------------
    results_dir = _find_run_folder(airl_base, args.run_id)

    # ------------------------------------------------------------------
    # Infer policy_type from folder name if not supplied on CLI
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
    # Locate and load BC checkpoint
    # (bc_run_folder is the folder name, e.g. "1_beta")
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

    bc_ckpt = torch.load(bc_model_path, map_location="cpu", weights_only=False)
    bc_state_dict = bc_ckpt["model_state_dict"]
    print(f"\nLoaded BC checkpoint : {saved['bc_run_folder']}")

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print(f"\nLoading data for reservoir '{args.reservoir}' …")
    data       = load_reservoir_data(res_cfg, res_cfg_path)
    raw_splits = _load_raw_splits(res_cfg, res_cfg_path)
    train_df, val_df, _ = raw_splits

    print(
        f"  state_dim    = {data.state_dim}\n"
        f"  train rows   = {len(data.train.states)}\n"
        f"  val rows     = {len(data.val.states)}\n"
        f"  test rows    = {len(data.test.states)}"
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
            f"  The state variables or use_month_encoding may have changed "
            f"since tuning.\n"
            f"  Re-run airl/tune.py to obtain a matching best_config.json.\n"
        )

    # ------------------------------------------------------------------
    # Reconstruct AIRLConfig from JSON
    # ------------------------------------------------------------------
    config = AIRLConfig(
        # Data dimensions
        state_dim  = cfg_dict["state_dim"],
        action_dim = cfg_dict.get("action_dim", 1),

        # BC policy architecture (fixed from BC tuning)
        hidden_dim      = int(cfg_dict["hidden_dim"]),
        n_hidden_layers = int(cfg_dict["n_hidden_layers"]),
        dropout         = float(cfg_dict["dropout"]),
        alpha_min       = float(cfg_dict.get("alpha_min",      1.0)),
        alpha_max       = float(cfg_dict.get("alpha_max",      50.0)),
        beta_min        = float(cfg_dict.get("beta_min",       1.0)),
        beta_max        = float(cfg_dict.get("beta_max",       50.0)),
        sigma_min       = float(cfg_dict.get("sigma_min",      0.1)),
        log_epsilon     = float(cfg_dict.get("log_epsilon",    1.0)),
        zero_threshold  = float(cfg_dict.get("zero_threshold", 0.01)),
        mse_weight      = float(cfg_dict.get("mse_weight",     10.0)),
        gate_weight     = float(cfg_dict.get("gate_weight",    5.0)),

        # Critic
        critic_hidden_dim      = int(cfg_dict["critic_hidden_dim"]),
        critic_n_hidden_layers = int(cfg_dict["critic_n_hidden_layers"]),

        # Discriminator
        disc_hidden_dim      = int(cfg_dict["disc_hidden_dim"]),
        disc_n_hidden_layers = int(cfg_dict["disc_n_hidden_layers"]),
        disc_dropout         = float(cfg_dict["disc_dropout"]),

        # Learning rates
        lr_policy        = float(cfg_dict["lr_policy"]),
        lr_critic        = float(cfg_dict["lr_critic"]),
        lr_discriminator = float(cfg_dict["lr_discriminator"]),

        # Discriminator training
        disc_updates            = int(cfg_dict["disc_updates"]),
        warmup_disc_updates     = int(cfg_dict["warmup_disc_updates"]),
        gradient_penalty_coef   = float(cfg_dict["gradient_penalty_coef"]),
        label_smoothing_epsilon = float(cfg_dict["label_smoothing_epsilon"]),

        # PPO
        gamma        = float(cfg_dict["gamma"]),
        gae_lambda   = float(cfg_dict.get("gae_lambda",   0.95)),
        clip_epsilon = float(cfg_dict["clip_epsilon"]),
        entropy_coef = float(cfg_dict["entropy_coef"]),
        ppo_epochs   = int(cfg_dict["ppo_epochs"]),

        # KL regularisation
        kl_regularization_coef = float(cfg_dict["kl_regularization_coef"]),

        # Training schedule
        warmup_iterations       = int(cfg_dict["warmup_iterations"]),
        num_iterations          = int(cfg_dict["num_iterations"]),
        steps_per_iteration     = int(cfg_dict["steps_per_iteration"]),
        batch_size              = int(cfg_dict["batch_size"]),
        early_stopping_patience = int(cfg_dict["early_stopping_patience"]),

        # Replay / trajectory (fall back to dataclass defaults if absent)
        expert_buffer_size   = int(cfg_dict.get("expert_buffer_size",   60_000)),
        policy_buffer_size   = int(cfg_dict.get("policy_buffer_size",  120_000)),
        trajectory_years     = int(cfg_dict.get("trajectory_years",         1)),
        num_expert_traj      = int(cfg_dict.get("num_expert_traj",        100)),
        align_to_year_start  = bool(cfg_dict.get("align_to_year_start",  True)),
        end_at_year_boundary = bool(cfg_dict.get("end_at_year_boundary", True)),
        eval_interval        = int(cfg_dict.get("eval_interval",          10)),
        max_grad_norm        = float(cfg_dict.get("max_grad_norm",        0.5)),

        # Runtime
        device  = cfg_dict["device"],
        seed    = int(cfg_dict["seed"]),
        verbose = args.verbose,
    )

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

    config.device = resolved

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
    # Build BC policy and load weights
    # ------------------------------------------------------------------
    device = torch.device(config.device)
    bc_policy = build_policy_network(policy_type, config).to(device)
    bc_policy.load_state_dict(
        {k: v.clone() for k, v in bc_state_dict.items()}
    )
    print(f"\nBC policy loaded and moved to {config.device}.")

    # ------------------------------------------------------------------
    # Build environments
    # ------------------------------------------------------------------
    train_env = ReservoirEnvironment(train_df, config, data.normalizer, res_cfg)
    val_env   = ReservoirEnvironment(val_df,   config, data.normalizer, res_cfg)

    # ------------------------------------------------------------------
    # Instantiate agent and load expert data
    # ------------------------------------------------------------------
    agent = AIRLAgent(config, bc_policy, policy_type)
    agent.add_expert_from_split(data.train)
    print(f"Expert buffer populated: {len(agent.expert_buffer)} transitions.")

    # ------------------------------------------------------------------
    # Discriminator warmup
    # ------------------------------------------------------------------
    # warmup_discriminator prints its own header when verbose=True; only
    # print here when verbose=False so the message is never shown twice.
    if not config.verbose:
        print(f"\nWarming up discriminator for {config.warmup_iterations} iterations …")
    agent.warmup_discriminator(train_env, config.warmup_iterations)

    # ------------------------------------------------------------------
    # Main adversarial training
    # ------------------------------------------------------------------
    print(f"\nStarting AIRL training: {config.num_iterations} iterations …\n")
    result = agent.train(
        train_env = train_env,
        val_env   = val_env,
        val_split = data.val,
        trial     = None,       # no Optuna pruning in final training
    )

    best_score          = float(result["best_val_score"])
    iterations_done     = int(result["iterations_completed"])
    training_stats      = result["training_stats"]

    print(f"\nTraining complete.")
    print(f"  Best val score       : {best_score:.4f}")
    print(f"  Iterations completed : {iterations_done}")

    # ------------------------------------------------------------------
    # Save model checkpoint
    # ------------------------------------------------------------------
    model_path = results_dir / "model.pt"
    agent.save(model_path)
    print(f"\nModel saved  → {model_path}")

    # ------------------------------------------------------------------
    # Save training log
    # ------------------------------------------------------------------
    train_log = {
        "reservoir":           args.reservoir,
        "policy_type":         policy_type,
        "run_id":              args.run_id,
        "bc_run_folder":       saved["bc_run_folder"],
        "device":              config.device,
        "seed":                config.seed,
        "best_val_score":      round(best_score, 6),
        "iterations_completed":iterations_done,
        "training_stats":      {k: [round(float(v), 6) for v in vals]
                                for k, vals in training_stats.items()},
        "timestamp":           datetime.now().isoformat(timespec="seconds"),
    }

    log_path = results_dir / "train_log.json"
    with open(log_path, "w") as f:
        json.dump(train_log, f, indent=2)
    print(f"Train log saved → {log_path}")

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
        "best_val_score": round(best_score, 6),
        "timestamp":      datetime.now().isoformat(timespec="seconds"),
    }

    with open(run_args_path, "w") as f:
        json.dump(run_args, f, indent=2)
    print(f"Run args updated \u2192 {run_args_path}\n")


if __name__ == "__main__":
    main()
