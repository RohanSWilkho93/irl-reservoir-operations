"""
behavioral_cloning/tune.py
==========================
Optuna hyperparameter search for Behavioral Cloning.

Run this BEFORE airl/tune.py and iqlearn/tune.py.  The best configuration
is saved to results/<reservoir>/behavioral_cloning/best_config.json and is
loaded automatically by the downstream algorithm tuning scripts to fix the
policy architecture and transfer pre-trained weights.

Usage
-----
# Minimal — reservoir config supplies everything else
python behavioral_cloning/tune.py --reservoir garrison

# Override policy network and device
python behavioral_cloning/tune.py --reservoir stockton \\
    --policy_network hardgating --device cuda

# Full override on first run (values are written back to configs for
# reproducibility; subsequent runs need only --reservoir)
python behavioral_cloning/tune.py --reservoir garrison \\
    --data_path data/Garrison.csv \\
    --state_variables storage net_inflow \\
    --use_month_encoding true \\
    --policy_network beta \\
    --split_train 14 --split_val 1 --split_test 3 \\
    --device auto --num_workers 16

Search space
------------
All hyperparameters are drawn from configs/algorithms/behavioral_cloning.yaml.
Shared parameters (architecture, optimizer, scheduler, early stopping) apply
to every policy type. Network-specific parameters are sampled only when the
corresponding policy type is active for that reservoir.

Validation metric
-----------------
score = (release_pearson_r + (1 - release_rmse)) / 2

Both terms are in [0, 1] on normalised release values.  Optuna maximises
this composite score.  RMSE on normalised [0, 1] data is scale-free and
equivalent to nRMSE (observed range ≈ 1), so no further normalisation is
applied during tuning.  The nrmse() function is reserved for final paper
reporting on raw engineering-unit values via generate_results.py.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from dataclasses import dataclass, asdict, fields as dc_fields
from datetime import datetime
from pathlib import Path

import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import yaml

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so sibling packages resolve correctly.
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils.data    import load_reservoir_data
from utils.metrics import rmse, safe_pearsonr
from networks.policy import build_policy_network, VALID_POLICY_TYPES


# =============================================================================
# BCConfig — flat container for all hyperparameters of a single trial
# =============================================================================

@dataclass
class BCConfig:
    """
    All hyperparameters needed to build and train a policy network.

    Shared fields (all policy types)
    ---------------------------------
    state_dim, action_dim, hidden_dim, n_hidden_layers, dropout,
    lr, epochs, batch_size, scheduler_type, early_stopping_patience,
    seed, device.

    Beta / Hardgating / Softgating
    ---------------------------------
    alpha_min, alpha_max, beta_min, beta_max.

    Lognormal
    ---------
    sigma_min, log_epsilon.

    Hardgating / Softgating
    -----------------------
    zero_threshold.

    Softgating only
    ---------------
    mse_weight, gate_weight.

    Fields not used by the active policy type retain their defaults and are
    ignored by the loss function.  They are still serialised to best_config.json
    so the file is self-contained.
    """
    # Architecture — must be supplied (no sensible universal default)
    state_dim:       int
    action_dim:      int   = 1

    # Shared — Optuna hyperparameters
    hidden_dim:               int   = 128
    n_hidden_layers:          int   = 3
    dropout:                  float = 0.1
    lr:                       float = 1e-3
    epochs:                   int   = 300
    batch_size:               int   = 128
    scheduler_type:           str   = "cosine"
    early_stopping_patience:  int   = 30
    seed:                     int   = 42
    device:                   str   = "cpu"

    # Beta / Hardgating / Softgating
    alpha_min: float = 1.0
    alpha_max: float = 50.0
    beta_min:  float = 1.0
    beta_max:  float = 50.0

    # Lognormal
    sigma_min:   float = 0.1
    log_epsilon: float = 1.0

    # Hardgating / Softgating
    zero_threshold: float = 0.01

    # Softgating only
    mse_weight:  float = 10.0
    gate_weight: float = 5.0


def _bc_default(field_name: str):
    """Return the default value of a BCConfig field by name."""
    for f in dc_fields(BCConfig):
        if f.name == field_name:
            return f.default
    raise KeyError(field_name)


# =============================================================================
# Loss functions
# =============================================================================

def _compute_loss(
    output,
    expert_actions: torch.Tensor,
    policy_type:    str,
    config:         BCConfig,
) -> torch.Tensor:
    """
    Dispatch to the correct supervised loss for the active policy type.

    Parameters
    ----------
    output        : PolicyOutput from model.forward(state, deterministic=False).
    expert_actions: Normalised expert release. Shape (batch,) or (batch, 1).
    policy_type   : One of "beta", "lognormal", "hardgating", "softgating".
    config        : BCConfig for the current trial (supplies network-specific
                    thresholds and weights).

    Returns
    -------
    Scalar loss tensor.
    """
    a = expert_actions.view(-1, 1)   # (batch, 1) — broadcast-safe with heads

    if policy_type == "beta":
        a_clamped = a.clamp(1e-6, 1.0 - 1e-6)
        dist = torch.distributions.Beta(output.alpha, output.beta)
        return -dist.log_prob(a_clamped).mean()

    elif policy_type == "lognormal":
        log_a = torch.log(a + config.log_epsilon)
        dist  = torch.distributions.Normal(output.mu, output.sigma)
        return -dist.log_prob(log_a).mean()

    elif policy_type == "hardgating":
        gate_labels = (a > config.zero_threshold).float()
        gate_loss   = F.binary_cross_entropy(output.gate_prob, gate_labels)

        nonzero = gate_labels.squeeze(1).bool()   # (batch,) mask
        if nonzero.sum() > 0:
            a_nz  = a[nonzero].clamp(1e-6, 1.0 - 1e-6)
            dist  = torch.distributions.Beta(
                output.alpha[nonzero], output.beta[nonzero]
            )
            beta_loss = -dist.log_prob(a_nz).mean()
        else:
            beta_loss = torch.zeros(1, device=a.device)

        return gate_loss + beta_loss

    elif policy_type == "softgating":
        gate_labels = (a > config.zero_threshold).float()
        gate_loss   = F.binary_cross_entropy(output.gate_prob, gate_labels)
        mse_loss    = F.mse_loss(output.action, a)
        return config.mse_weight * mse_loss + config.gate_weight * gate_loss

    else:
        raise ValueError(f"Unknown policy_type: {policy_type!r}")


# =============================================================================
# Training and validation
# =============================================================================

def _train_and_validate(
    config:      BCConfig,
    data,
    policy_type: str,
    trial:       optuna.Trial | None = None,
) -> float:
    """
    Train a policy network for one hyperparameter configuration and return
    the validation score.

    Validation score
    ----------------
    score = (release_pearson_r + (1 - release_rmse)) / 2

    Computed on normalised [0, 1] release values in deterministic inference
    mode.  Both terms are in [0, 1]; the composite score is maximised.

    Parameters
    ----------
    config      : BCConfig for this trial.
    data        : DataSplits from load_reservoir_data().
    policy_type : Active policy type string.
    trial       : Optuna trial (for pruning).  None during final training.

    Returns
    -------
    float  Validation score in [-1, 1].  Returns -inf for degenerate trials
           (NaN / inf loss) so Optuna deprioritises them.
    """
    # ---- Reproducibility ----
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    device = torch.device(config.device)

    # ---- Network and optimizer ----
    model     = build_policy_network(policy_type, config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    if config.scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=50, T_mult=1
        )
    elif config.scheduler_type == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", patience=5, factor=0.5
        )
    else:
        scheduler = None

    # ---- Training DataLoader ----
    train_states  = torch.tensor(data.train.states,  dtype=torch.float32)
    train_actions = torch.tensor(data.train.actions, dtype=torch.float32)
    train_dl = DataLoader(
        TensorDataset(train_states, train_actions),
        batch_size = config.batch_size,
        shuffle    = True,
        drop_last  = False,
    )

    # ---- Validation data (full split, loaded once) ----
    val_states_t   = torch.tensor(data.val.states, dtype=torch.float32).to(device)
    val_actions_np = data.val.actions   # numpy (N,) — stays on CPU for metrics

    # ---- Training loop ----
    best_score   = -float("inf")
    patience_ctr = 0
    best_weights = None

    for epoch in range(config.epochs):
        model.train()
        for states_b, actions_b in train_dl:
            states_b  = states_b.to(device)
            actions_b = actions_b.to(device)

            optimizer.zero_grad()
            output = model(states_b, deterministic=False)
            loss   = _compute_loss(output, actions_b, policy_type, config)

            if not torch.isfinite(loss):
                return -float("inf")   # NaN/inf — skip this trial

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        if scheduler is not None and config.scheduler_type == "cosine":
            scheduler.step()

        # ---- Validation ----
        model.eval()
        with torch.no_grad():
            val_out  = model(val_states_t, deterministic=True)
            val_pred = val_out.action.squeeze(1).cpu().numpy()   # (N,)

        val_corr, _ = safe_pearsonr(val_actions_np, val_pred)
        val_rmse_v  = rmse(val_actions_np, val_pred)
        score       = (val_corr + (1.0 - val_rmse_v)) / 2.0

        if scheduler is not None and config.scheduler_type == "plateau":
            scheduler.step(score)

        # ---- Early stopping ----
        if score > best_score + 1e-6:
            best_score   = score
            patience_ctr = 0
            best_weights = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_ctr += 1
            if patience_ctr >= config.early_stopping_patience:
                break

        # ---- Optuna pruning ----
        if trial is not None:
            trial.report(score, epoch)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

    # ---- Final evaluation with best weights ----
    if best_weights is not None:
        model.load_state_dict(best_weights)

    model.eval()
    with torch.no_grad():
        val_out  = model(val_states_t, deterministic=True)
        val_pred = val_out.action.squeeze(1).cpu().numpy()

    val_corr, _ = safe_pearsonr(val_actions_np, val_pred)
    val_rmse_v  = rmse(val_actions_np, val_pred)
    return (val_corr + (1.0 - val_rmse_v)) / 2.0


# =============================================================================
# Optuna objective factory
# =============================================================================

def _make_objective(algo_cfg: dict, data, policy_type: str, device: str):
    """
    Return a closure that Optuna calls on every trial.

    Captures algo_cfg (search space), data, policy_type, and device from the
    enclosing scope.  The returned callable matches Optuna's expected signature:
        objective(trial: optuna.Trial) -> float
    """
    shared = algo_cfg["shared"]
    net_sp = algo_cfg.get(policy_type, {})   # may be empty for edge cases

    def objective(trial: optuna.Trial) -> float:
        # ---- Shared hyperparameters ----
        seed            = trial.suggest_categorical("seed",             shared["seed"])
        hidden_dim      = trial.suggest_categorical("hidden_dim",       shared["hidden_dim"])
        n_hidden_layers = trial.suggest_categorical("n_hidden_layers",  shared["n_hidden_layers"])
        dropout         = trial.suggest_categorical("dropout",          shared["dropout"])
        epochs          = trial.suggest_categorical("epochs",           shared["epochs"])
        batch_size      = trial.suggest_categorical("batch_size",       shared["batch_size"])
        scheduler_type  = trial.suggest_categorical("scheduler_type",   shared["scheduler_type"])
        early_stop_pat  = trial.suggest_categorical(
            "early_stopping_patience", shared["early_stopping_patience"]
        )

        lr_sp = shared["lr"]
        lr    = trial.suggest_float(
            "lr", lr_sp["low"], lr_sp["high"], log=lr_sp.get("log", False)
        )

        config = BCConfig(
            state_dim               = data.state_dim,
            action_dim              = 1,
            hidden_dim              = hidden_dim,
            n_hidden_layers         = n_hidden_layers,
            dropout                 = dropout,
            lr                      = lr,
            epochs                  = epochs,
            batch_size              = batch_size,
            scheduler_type          = scheduler_type,
            early_stopping_patience = early_stop_pat,
            seed                    = seed,
            device                  = device,
        )

        # ---- Network-specific hyperparameters ----
        if policy_type == "beta":
            config.alpha_max = trial.suggest_categorical("alpha_max", net_sp["alpha_max"])
            config.beta_max  = trial.suggest_categorical("beta_max",  net_sp["beta_max"])

        elif policy_type == "lognormal":
            sm_sp = net_sp["sigma_min"]
            le_sp = net_sp["log_epsilon"]
            config.sigma_min   = trial.suggest_float(
                "sigma_min",   sm_sp["low"], sm_sp["high"], log=sm_sp.get("log", False)
            )
            config.log_epsilon = trial.suggest_float(
                "log_epsilon", le_sp["low"], le_sp["high"], log=le_sp.get("log", False)
            )

        elif policy_type == "hardgating":
            config.zero_threshold = trial.suggest_categorical(
                "zero_threshold", net_sp["zero_threshold"]
            )
            config.alpha_max = trial.suggest_categorical("alpha_max", net_sp["alpha_max"])
            config.beta_max  = trial.suggest_categorical("beta_max",  net_sp["beta_max"])

        elif policy_type == "softgating":
            config.zero_threshold = trial.suggest_categorical(
                "zero_threshold", net_sp["zero_threshold"]
            )
            config.alpha_max = trial.suggest_categorical("alpha_max", net_sp["alpha_max"])
            config.beta_max  = trial.suggest_categorical("beta_max",  net_sp["beta_max"])
            mw_sp = net_sp["mse_weight"]
            gw_sp = net_sp["gate_weight"]
            config.mse_weight  = trial.suggest_float(
                "mse_weight",  mw_sp["low"], mw_sp["high"], log=mw_sp.get("log", False)
            )
            config.gate_weight = trial.suggest_float(
                "gate_weight", gw_sp["low"], gw_sp["high"], log=gw_sp.get("log", False)
            )

        return _train_and_validate(config, data, policy_type, trial=trial)

    return objective


# =============================================================================
# Config helpers
# =============================================================================

def _resolve_device(device_str: str) -> str:
    """Resolve 'auto' to the actual available device string."""
    if device_str == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_str


def _deep_update(target: dict, source: dict) -> None:
    """
    Recursively update target with values from source.

    Descends into nested dicts instead of replacing them wholesale.
    Compatible with ruamel.yaml CommentedMap so YAML comments survive
    the write-back.
    """
    for key, val in source.items():
        if key in target and hasattr(target[key], "items") and hasattr(val, "items"):
            _deep_update(target[key], val)
        else:
            target[key] = val


def _writeback_yaml(path: Path, updates: dict) -> None:
    """
    Merge updates into an existing YAML file and write it back.

    Uses ruamel.yaml (comment-preserving) if installed; falls back to plain
    PyYAML with a warning.

    Parameters
    ----------
    path    : Path to the YAML file to update.
    updates : Dict of key → value pairs to merge (can be nested).
    """
    try:
        from ruamel.yaml import YAML
        ryaml = YAML()
        ryaml.preserve_quotes = True
        ryaml.best_width = 4096
        with open(path, "r") as f:
            doc = ryaml.load(f)
        _deep_update(doc, updates)
        with open(path, "w") as f:
            ryaml.dump(doc, f)
    except ImportError:
        import warnings
        warnings.warn(
            "ruamel.yaml not installed — writing YAML with plain PyYAML. "
            "YAML comments will be lost. Install ruamel.yaml to preserve them.",
            UserWarning,
            stacklevel=2,
        )
        with open(path, "r") as f:
            doc = yaml.safe_load(f)
        _deep_update(doc, updates)
        with open(path, "w") as f:
            yaml.dump(doc, f, default_flow_style=False, sort_keys=False)


# =============================================================================
# Run-folder helpers
# =============================================================================

def _resolve_run_id(
    base_dir:    Path,
    policy_type: str,
    run_id_arg:  int | None,
) -> tuple[int, Path]:
    """
    Resolve the integer run_id and return the corresponding run folder Path.

    If run_id_arg is None, scans base_dir for existing folders matching
    '<int>_*', finds the highest integer prefix, and returns highest + 1.
    If no folders exist yet, starts at 1.

    If run_id_arg is explicitly provided, uses it directly and warns if the
    folder already exists (contents will be overwritten).

    Parameters
    ----------
    base_dir    : Path to results/<reservoir>/behavioral_cloning/
    policy_type : Resolved policy type string (e.g. "beta").
    run_id_arg  : Value of --run_id from the CLI, or None.

    Returns
    -------
    (run_id, run_folder)  run_id is the integer; run_folder is the full Path.
    """
    pattern = re.compile(r"^(\d+)_(.+)$")
    existing_ids: list[int] = []

    if base_dir.exists():
        for d in base_dir.iterdir():
            m = pattern.match(d.name)
            if d.is_dir() and m:
                existing_ids.append(int(m.group(1)))

    if run_id_arg is None:
        run_id = max(existing_ids, default=0) + 1
    else:
        run_id = run_id_arg

    folder_name = f"{run_id}_{policy_type}"
    run_folder  = base_dir / folder_name

    if run_folder.exists() and run_id_arg is not None:
        print(
            f"\nWARNING: Run folder already exists: {run_folder}\n"
            f"  Contents will be overwritten.\n",
            file=sys.stderr,
        )

    return run_id, run_folder


# =============================================================================
# CLI argument parsing
# =============================================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Behavioral Cloning hyperparameter tuning with Optuna.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required
    p.add_argument(
        "--reservoir", required=True,
        help="Reservoir name — must match configs/reservoirs/<name>.yaml.",
    )

    # Reservoir config overrides
    p.add_argument("--data_path",   default=None,
                   help="Override data_path in the reservoir config.")
    p.add_argument("--date_column", default=None,
                   help="Override columns.date in the reservoir config.")
    p.add_argument("--state_variables", nargs="+", default=None,
                   help="Override columns.state in the reservoir config.")
    p.add_argument(
        "--use_month_encoding",
        type=lambda s: s.lower() == "true",
        default=None,
        help="Override columns.use_month_encoding (true|false).",
    )
    p.add_argument(
        "--policy_network", choices=list(VALID_POLICY_TYPES), default=None,
        help="Override policy_network in the reservoir config.",
    )
    p.add_argument("--split_train", type=int, default=None,
                   help="Override split.train in the reservoir config.")
    p.add_argument("--split_val",   type=int, default=None,
                   help="Override split.val in the reservoir config.")
    p.add_argument("--split_test",  type=int, default=None,
                   help="Override split.test in the reservoir config.")

    # Algorithm config overrides
    p.add_argument(
        "--device", default=None,
        help=(
            "Override runtime.device: auto | cpu | cuda | cuda:N | mps. "
            "'auto' selects GPU if available, otherwise CPU."
        ),
    )
    p.add_argument(
        "--num_workers", type=int, default=None,
        help="Override runtime.num_workers (parallel Optuna jobs).",
    )
    p.add_argument(
        "--n_trials", type=int, default=None,
        help="Override optuna.n_trials in the algorithm config.",
    )
    p.add_argument(
        "--run_id", type=int, default=None,
        help=(
            "Integer run identifier.  If omitted, auto-increments from "
            "existing runs in results/<reservoir>/behavioral_cloning/. "
            "Folder name will be <run_id>_<policy_type>."
        ),
    )

    return p.parse_args()


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args = _parse_args()

    res_cfg_path  = _ROOT / "configs" / "reservoirs" / f"{args.reservoir}.yaml"
    algo_cfg_path = _ROOT / "configs" / "algorithms" / "behavioral_cloning.yaml"

    if not res_cfg_path.exists():
        sys.exit(
            f"Reservoir config not found: {res_cfg_path}\n"
            f"Available reservoirs: configs/reservoirs/*.yaml"
        )

    # ---- Load configs ----
    with open(res_cfg_path,  "r") as f:
        res_cfg = yaml.safe_load(f)
    with open(algo_cfg_path, "r") as f:
        algo_cfg = yaml.safe_load(f)

    # ---- Apply CLI overrides and collect changes for write-back ----
    res_updates  = {}
    algo_updates = {}

    if args.data_path is not None:
        res_cfg["data_path"] = args.data_path
        res_updates["data_path"] = args.data_path

    if args.date_column is not None:
        res_cfg["columns"]["date"] = args.date_column
        res_updates.setdefault("columns", {})["date"] = args.date_column

    if args.state_variables is not None:
        res_cfg["columns"]["state"] = args.state_variables
        res_updates.setdefault("columns", {})["state"] = args.state_variables

    if args.use_month_encoding is not None:
        res_cfg["columns"]["use_month_encoding"] = args.use_month_encoding
        res_updates.setdefault("columns", {})["use_month_encoding"] = args.use_month_encoding

    if args.policy_network is not None:
        res_cfg["policy_network"] = args.policy_network
        res_updates["policy_network"] = args.policy_network

    if args.split_train is not None:
        res_cfg.setdefault("split", {})["train"] = args.split_train
        res_updates.setdefault("split", {})["train"] = args.split_train

    if args.split_val is not None:
        res_cfg.setdefault("split", {})["val"] = args.split_val
        res_updates.setdefault("split", {})["val"] = args.split_val

    if args.split_test is not None:
        res_cfg.setdefault("split", {})["test"] = args.split_test
        res_updates.setdefault("split", {})["test"] = args.split_test

    if args.device is not None:
        algo_cfg["runtime"]["device"] = args.device
        algo_updates.setdefault("runtime", {})["device"] = args.device

    if args.num_workers is not None:
        algo_cfg["runtime"]["num_workers"] = args.num_workers
        algo_updates.setdefault("runtime", {})["num_workers"] = args.num_workers

    if args.n_trials is not None:
        algo_cfg["optuna"]["n_trials"] = args.n_trials
        algo_updates.setdefault("optuna", {})["n_trials"] = args.n_trials

    if res_updates:
        _writeback_yaml(res_cfg_path, res_updates)
    if algo_updates:
        _writeback_yaml(algo_cfg_path, algo_updates)

    # ---- Resolve runtime settings ----
    device_str = _resolve_device(algo_cfg["runtime"]["device"])
    n_jobs = (
        algo_cfg["runtime"]["num_workers"]
        if algo_cfg["runtime"]["num_workers"] is not None
        else algo_cfg["optuna"]["n_jobs"]
    )

    policy_type = str(res_cfg["policy_network"]).lower().strip()
    if policy_type not in VALID_POLICY_TYPES:
        sys.exit(
            f"Unknown policy_network '{policy_type}' in {res_cfg_path}.\n"
            f"Valid options: {list(VALID_POLICY_TYPES)}"
        )

    # ---- Load and split data ----
    # Bounds are auto-computed from the training split and written back to
    # res_cfg_path inside load_reservoir_data() — this must happen after the
    # CLI write-back above so both sets of changes land in the same file.
    print(f"\nLoading data for reservoir '{args.reservoir}' …")
    data = load_reservoir_data(res_cfg, res_cfg_path)
    print(
        f"  state_dim    = {data.state_dim}\n"
        f"  train rows   = {len(data.train.states)}\n"
        f"  val rows     = {len(data.val.states)}\n"
        f"  test rows    = {len(data.test.states)}\n"
        f"  policy type  = {policy_type}\n"
        f"  device       = {device_str}"
    )

    # ---- Output directory ----
    bc_base_dir = _ROOT / "results" / args.reservoir / "behavioral_cloning"
    run_id, results_dir = _resolve_run_id(bc_base_dir, policy_type, args.run_id)
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nRun folder : {results_dir.name}  (run_id={run_id})")

    # ---- Optuna study ----
    study = optuna.create_study(
        direction  = "maximize",
        sampler    = optuna.samplers.TPESampler(seed=42),
        pruner     = optuna.pruners.MedianPruner(n_warmup_steps=20),
        study_name = f"{args.reservoir}_behavioral_cloning",
    )

    objective = _make_objective(algo_cfg, data, policy_type, device_str)

    n_trials = algo_cfg["optuna"]["n_trials"]
    print(f"\nStarting Optuna search: {n_trials} trials, {n_jobs} job(s) …\n")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)

    # ---- Guard: ensure at least one trial completed ----
    completed = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE
    ]
    if not completed:
        sys.exit(
            "All trials failed or were pruned. "
            "Check your loss function, data, and search space."
        )

    # ---- Reconstruct best BCConfig from trial params ----
    best = study.best_trial
    p   = best.params

    best_config = BCConfig(
        state_dim               = data.state_dim,
        action_dim              = 1,
        hidden_dim              = p["hidden_dim"],
        n_hidden_layers         = p["n_hidden_layers"],
        dropout                 = p["dropout"],
        lr                      = p["lr"],
        epochs                  = p["epochs"],
        batch_size              = p["batch_size"],
        scheduler_type          = p["scheduler_type"],
        early_stopping_patience = p["early_stopping_patience"],
        seed                    = p["seed"],
        device                  = device_str,
        # Network-specific — fall back to BCConfig defaults if not sampled
        alpha_max       = p.get("alpha_max",       _bc_default("alpha_max")),
        beta_max        = p.get("beta_max",        _bc_default("beta_max")),
        sigma_min       = p.get("sigma_min",       _bc_default("sigma_min")),
        log_epsilon     = p.get("log_epsilon",     _bc_default("log_epsilon")),
        zero_threshold  = p.get("zero_threshold",  _bc_default("zero_threshold")),
        mse_weight      = p.get("mse_weight",      _bc_default("mse_weight")),
        gate_weight     = p.get("gate_weight",     _bc_default("gate_weight")),
    )

    # ---- Save best_config.json ----
    save_dict = {
        "reservoir":   args.reservoir,
        "policy_type": policy_type,
        "best_score":  best.value,
        "trial_number": best.number,
        "config":      asdict(best_config),
    }

    out_path = results_dir / "best_config.json"
    with open(out_path, "w") as f:
        json.dump(save_dict, f, indent=2)

    print(f"Best trial #{best.number}  score = {best.value:.4f}")
    print(f"Best config saved → {out_path}\n")

    # ---- Save run_args.json ----
    run_args_path = results_dir / "run_args.json"
    run_args = {
        "tune": {
            "reservoir":          args.reservoir,
            "run_id":             run_id,
            "folder":             results_dir.name,
            "policy_network":     policy_type,
            "data_path":          args.data_path,
            "date_column":        args.date_column,
            "state_variables":    args.state_variables,
            "use_month_encoding": args.use_month_encoding,
            "split_train":        args.split_train,
            "split_val":          args.split_val,
            "split_test":         args.split_test,
            "device":             args.device,
            "n_trials":           args.n_trials,
            "num_workers":        args.num_workers,
            "timestamp":          datetime.now().isoformat(timespec="seconds"),
        }
    }
    with open(run_args_path, "w") as f:
        json.dump(run_args, f, indent=2)
    print(f"Run args saved  → {run_args_path}\n")


if __name__ == "__main__":
    main()
