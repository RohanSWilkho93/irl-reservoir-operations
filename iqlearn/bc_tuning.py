"""
behavioral_cloning/tune.py
==========================
Optuna hyperparameter search for Behavioral Cloning.

The best configuration is saved to results/<reservoir>/iqlearn/<run_id>/bc_best_config.json and 
is loaded automatically by IQLearn tuning scripts to fix the
policy architecture and transfer pre-trained weights.

Policy head
-----------
A single non-parametric head: a quantile-binned categorical distribution over
normalised release.  Skew, bounded support, zero-inflation, and multimodality
are all captured by the predicted per-state histogram, so `policy_type` is no
longer a hyperparameter.

The network outputs K logits (one per bin).  Training minimises cross-entropy
against the expert action's bin (optionally ordinal-smoothed).  The bin edges
and empirical bin means are computed once per trial from the pooled TRAINING
releases, frozen, and — for the winning trial — serialised into
bc_best_config.json so the saved model is fully reproducible at inference time.

Search space
------------
All hyperparameters are drawn from configs/algorithms/iqlearn.yaml:
shared backbone (architecture, optimizer, scheduler, early stopping) plus the
categorical block (n_bins, binning, ordinal_smoothing).

Validation metric
-----------------
score = (release_pearson_r + (1 - release_rmse)) / 2

Computed on the categorical head's EXPECTED VALUE (sum_k p_k * bin_mean_k) in
deterministic inference mode, on normalised [0, 1] release values.  Both terms
are in [0, 1]; Optuna maximises the composite.  RMSE on normalised data is
scale-free (observed range ~= 1), equivalent to nRMSE.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from dataclasses import dataclass, asdict, field
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
from iqlearn.networks.policy import build_policy_network
from iqlearn.utils.bc_binning import build_bins, assign_bins
from iqlearn.utils.runs import _resolve_device, _writeback_yaml, _resolve_run_id


# Single head type — kept as a constant for run-folder naming and metadata.
POLICY_TYPE = "categorical"


# =============================================================================
# BCConfig — flat container for all hyperparameters of a single trial
# =============================================================================

@dataclass
class BCConfig:
    """
    All hyperparameters needed to build and train the categorical policy.

    Shared backbone
    ---------------
    state_dim, action_dim, hidden_dim, n_hidden_layers, dropout, lr, epochs,
    batch_size, scheduler_type, early_stopping_patience, seed, device.

    Categorical head
    ----------------
    n_bins            : number of quantile bins K (output-layer width).
    binning           : edge-placement strategy ("quantile" | "log").
    ordinal_smoothing : Gaussian soft-label width in BIN UNITS (0.0 = hard).

    Frozen, data-derived (NOT Optuna-sampled)
    -----------------------------------------
    bin_edges : length n_bins + 1, computed from pooled training releases.
    bin_means : length n_bins, empirical mean release within each bin.

    bin_edges / bin_means are set after the bins are built for the trial and
    are serialised into bc_best_config.json so inference reconstructs the exact
    grid the model was trained on (guards against any future change to the
    binning code silently producing different edges).
    """
    # Architecture — must be supplied (no sensible universal default)
    state_dim:       int
    action_dim:      int   = 1

    # Shared backbone — Optuna hyperparameters
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

    # Categorical head — Optuna hyperparameters
    n_bins:            int   = 50
    binning:           str   = "quantile"
    ordinal_smoothing: float = 0.0

    # Frozen, data-derived (set after build_bins; serialised for reproducibility)
    bin_edges: list = field(default_factory=list)
    bin_means: list = field(default_factory=list)


# =============================================================================
# Categorical loss
# =============================================================================

def _ordinal_soft_targets(
    bin_idx:   torch.Tensor,   # (B,) long — expert action's bin index
    n_bins:    int,
    smoothing: float,
    device:    torch.device,
) -> torch.Tensor:
    """
    Build per-sample target distributions over bins.

    smoothing <= 0 : hard one-hot targets (plain cross-entropy).
    smoothing  > 0 : a Gaussian of width `smoothing` (in bin units) centred on
                     the true bin, normalised to sum to 1.  This penalises a
                     near-miss bin less than a far-miss bin, respecting the fact
                     that bins lie on an ordered release axis, and regularises
                     toward smooth predicted histograms.

    Returns
    -------
    (B, n_bins) target probability distributions.
    """
    if smoothing <= 0.0:
        return F.one_hot(bin_idx, num_classes=n_bins).float()

    positions = torch.arange(n_bins, device=device).float().unsqueeze(0)  # (1, K)
    centers   = bin_idx.float().unsqueeze(1)                              # (B, 1)
    sq_dist   = (positions - centers) ** 2                               # (B, K)
    weights   = torch.exp(-0.5 * sq_dist / (smoothing ** 2))
    return weights / weights.sum(dim=1, keepdim=True)


def _compute_loss(logits: torch.Tensor, target_dist: torch.Tensor) -> torch.Tensor:
    """
    Cross-entropy between predicted bin logits and the target distribution.

    Equivalent to KL(target || softmax(logits)) up to a target-only constant,
    so it reduces to standard cross-entropy when targets are one-hot.

    Parameters
    ----------
    logits      : (B, K) raw bin scores from the network.
    target_dist : (B, K) target probability distribution over bins.

    Returns
    -------
    Scalar loss tensor.
    """
    log_probs = F.log_softmax(logits, dim=1)
    return -(target_dist * log_probs).sum(dim=1).mean()


# =============================================================================
# Training and validation
# =============================================================================

def _train_and_validate(
    config: BCConfig,
    data,
    trial:  optuna.Trial | None = None,
    return_model=False) -> float:
    """
    Train the categorical policy for one configuration and return the
    validation score.

    Validation score
    ----------------
    score = (release_pearson_r + (1 - release_rmse)) / 2

    Computed on the head's EXPECTED VALUE  (sum_k p_k * bin_mean_k)  on
    normalised [0, 1] release values.  Both terms are in [0, 1]; the composite
    is maximised.

    Parameters
    ----------
    config : BCConfig for this trial (bin_edges / bin_means already frozen).
    data   : DataSplits from load_reservoir_data().
    trial  : Optuna trial (for pruning).  None during final training.

    Returns
    -------
    float  Validation score.  Returns -inf for degenerate trials (NaN / inf
           loss) so Optuna deprioritises them.
    """
    # ---- Reproducibility ----
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    device = torch.device(config.device)

    # ---- Frozen bin artifacts for this trial ----
    edges     = np.asarray(config.bin_edges, dtype=np.float64)              # (K+1,)
    bin_means = torch.tensor(config.bin_means, dtype=torch.float32,
                             device=device)                                # (K,)

    # ---- Network and optimizer ----
    model     = build_policy_network(config).to(device)   # forward(states) -> (B, K) logits
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
    # Bin indices are precomputed ONCE — edges are fixed within a trial, so
    # there is no need to re-bin every batch.  Training releases are in [0, 1]
    # by construction (normalised with training bounds), so they always fall
    # inside the grid; assign_bins still clamps defensively.
    train_states  = torch.tensor(data.train.states, dtype=torch.float32)
    train_bin_idx = torch.tensor(
        assign_bins(data.train.actions, edges), dtype=torch.long
    )
    train_dl = DataLoader(
        TensorDataset(train_states, train_bin_idx),
        batch_size = config.batch_size,
        shuffle    = True,
        drop_last  = False,
    )

    # ---- Validation data (full split, loaded once) ----
    val_states_t   = torch.tensor(data.val.states, dtype=torch.float32).to(device)
    val_actions_np = data.val.actions   # numpy (N,) — stays on CPU for metrics

    def _expected_value(states_t: torch.Tensor) -> np.ndarray:
        """Deterministic point prediction: sum_k p_k * bin_mean_k."""
        logits = model(states_t)                       # (N, K)
        probs  = F.softmax(logits, dim=1)              # (N, K)
        return (probs @ bin_means).cpu().numpy()       # (N,)

    # ---- Training loop ----
    best_score   = -float("inf")
    patience_ctr = 0
    best_weights = None

    for epoch in range(config.epochs):
        model.train()
        for states_b, bins_b in train_dl:
            states_b = states_b.to(device)
            bins_b   = bins_b.to(device)

            optimizer.zero_grad()
            logits = model(states_b)                                   # (B, K)
            target = _ordinal_soft_targets(
                bins_b, config.n_bins, config.ordinal_smoothing, device
            )
            loss = _compute_loss(logits, target)

            if not torch.isfinite(loss):
                return (-float("inf"), None) if return_model else -float("inf")

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        if scheduler is not None and config.scheduler_type == "cosine":
            scheduler.step()

        # ---- Validation ----
        model.eval()
        with torch.no_grad():
            val_pred = _expected_value(val_states_t)

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
        val_pred = _expected_value(val_states_t)

    val_corr, _ = safe_pearsonr(val_actions_np, val_pred)
    val_rmse_v  = rmse(val_actions_np, val_pred)
    # return (val_corr + (1.0 - val_rmse_v)) / 2.0

    score = (val_corr + (1.0 - val_rmse_v)) / 2.0
    if return_model:
        return score, model
    return score


# =============================================================================
# Optuna objective factory
# =============================================================================

def _make_objective(algo_cfg: dict, data, device: str):
    """
    Return a closure that Optuna calls on every trial.

    Captures algo_cfg (search space), data, and device from the enclosing
    scope.  The returned callable matches Optuna's expected signature:
        objective(trial: optuna.Trial) -> float
    """
    shared = algo_cfg["bc_tuning"]["shared"]
    cat    = algo_cfg["bc_tuning"]["categorical"]

    def objective(trial: optuna.Trial) -> float:
        # ---- Shared backbone hyperparameters ----
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

        # ---- Categorical head hyperparameters ----
        n_bins            = trial.suggest_categorical("n_bins",            cat["n_bins"])
        binning           = trial.suggest_categorical("binning",           cat["binning"])
        ordinal_smoothing = trial.suggest_categorical("ordinal_smoothing", cat["ordinal_smoothing"])

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
            n_bins                  = n_bins,
            binning                 = binning,
            ordinal_smoothing       = ordinal_smoothing,
        )

        # ---- Freeze the bin grid for this trial ----
        # Edges depend on (n_bins, binning), so they are rebuilt per trial from
        # the pooled training releases.  Cheap; no mid-search storage needed.
        edges, bin_means = build_bins(data.train.actions, n_bins, binning)
        config.bin_edges = [float(x) for x in edges]
        config.bin_means = [float(x) for x in bin_means]

        return _train_and_validate(config, data, trial=trial)

    return objective


# =============================================================================
# Config helpers
# =============================================================================

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


# =============================================================================
# CLI argument parsing
# =============================================================================

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="IQLearn Behavioral Cloning hyperparameter tuning with Optuna.",
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
                   help="Date column in the Reservoir CSV file. Override columns.date in the reservoir config.")
    p.add_argument("--state_variables", nargs="+", default=None,
                   help="State Variable columns in the Reservoir CSV file. Override columns.state in the reservoir config.")
    p.add_argument(
        "--use_month_encoding",
        type=lambda s: s.lower() == "true",
        default=None,
        help="True/False. Override columns.use_month_encoding (true|false).",
    )
    p.add_argument("--split_train", type=int, default=None,
                   help="Number of years in training split. Override split.train in the reservoir config.")
    p.add_argument("--split_val",   type=int, default=None,
                   help="Number of years in validation split. Override split.val in the reservoir config.")
    p.add_argument("--split_test",  type=int, default=None,
                   help="Number of years in testing split. Override split.test in the reservoir config.")

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
            "existing runs in results/<reservoir>/iqlearn/. "
            "Folder name will be <run_id>."
        ),
    )

    return p.parse_args()


# =============================================================================
# Apply CLI Overrides --- User entered hyperparameters
# =============================================================================

def _apply_cli_overrides(args, res_cfg, algo_cfg, res_cfg_path, algo_cfg_path) -> None:
    """Apply CLI overrides in-memory and persist them. Reservoir overrides sit
    at the reservoir config's top level; algorithm overrides sit UNDER the
    bc_tuning block of iqlearn.yaml, so the write-back dict mirrors that."""
    res_updates, algo_updates = {}, {}

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
        algo_cfg["bc_tuning"]["runtime"]["device"] = args.device
        algo_updates.setdefault("bc_tuning", {}).setdefault("runtime", {})["device"] = args.device

    if args.num_workers is not None:
        algo_cfg["bc_tuning"]["runtime"]["num_workers"] = args.num_workers
        algo_updates.setdefault("bc_tuning", {}).setdefault("runtime", {})["num_workers"] = args.num_workers

    if args.n_trials is not None:
        algo_cfg["bc_tuning"]["optuna"]["n_trials"] = args.n_trials
        algo_updates.setdefault("bc_tuning", {}).setdefault("optuna",  {})["n_trials"]    = args.n_trials

    if res_updates:
        _writeback_yaml(res_cfg_path, res_updates)
    if algo_updates:
        _writeback_yaml(algo_cfg_path, algo_updates)

# =============================================================================
# Saving run_args.json
# =============================================================================

def _save_run_args(results_dir, reservoir, run_id, cli_args) -> None:
    g = lambda name: getattr(cli_args, name, None) if cli_args else None
    run_args = {"tune": {
        "reservoir": reservoir, "run_id": run_id, "folder": results_dir.name,
        "policy_network": POLICY_TYPE,
        "data_path": g("data_path"), "date_column": g("date_column"),
        "state_variables": g("state_variables"), "use_month_encoding": g("use_month_encoding"),
        "split_train": g("split_train"), "split_val": g("split_val"), "split_test": g("split_test"),
        "device": g("device"), "n_trials": g("n_trials"), "num_workers": g("num_workers"),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }}
    with open(results_dir / "run_args.json", "w") as f:
        json.dump(run_args, f, indent=2)

# =============================================================================
# Run Behavioral Cloning Hyperparameter Tuning
# =============================================================================

def run_bc_tuning(*, reservoir, res_cfg, res_cfg_path, algo_cfg,
                  data, device_str, run_id=None, cli_args=None) -> dict:
    """Run the BC Optuna study, persist artifacts, return handles for downstream stages."""
    bc       = algo_cfg["bc_tuning"]
    n_jobs   = (bc["runtime"]["num_workers"]
                if bc["runtime"]["num_workers"] is not None
                else bc["optuna"]["n_jobs"])
    n_trials = bc["optuna"]["n_trials"]

    print(f"  state_dim={data.state_dim}  train={len(data.train.states)}  "
          f"val={len(data.val.states)}  test={len(data.test.states)}  "
          f"head={POLICY_TYPE}  device={device_str}")

    bc_base_dir = _ROOT / "results" / reservoir / "iqlearn"
    run_id, results_dir = _resolve_run_id(bc_base_dir, run_id)
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nRun folder : {results_dir.name}  (run_id={run_id})")

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=20),
        study_name=f"{reservoir}_behavioral_cloning",
    )
    objective = _make_objective(algo_cfg, data, device_str)

    print(f"\nStarting Optuna search: {n_trials} trials, {n_jobs} job(s) …\n")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs)

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        sys.exit("All trials failed or were pruned. Check loss, data, and search space.")

    best = study.best_trial
    p    = best.params
    bc_best_config = BCConfig(
        state_dim=data.state_dim, action_dim=1,
        hidden_dim=p["hidden_dim"], n_hidden_layers=p["n_hidden_layers"],
        dropout=p["dropout"], lr=p["lr"], epochs=p["epochs"],
        batch_size=p["batch_size"], scheduler_type=p["scheduler_type"],
        early_stopping_patience=p["early_stopping_patience"], seed=p["seed"],
        device=device_str, n_bins=p["n_bins"], binning=p["binning"],
        ordinal_smoothing=p["ordinal_smoothing"],
    )
    edges, bin_means = build_bins(data.train.actions, bc_best_config.n_bins, bc_best_config.binning)
    bc_best_config.bin_edges = [float(x) for x in edges]
    bc_best_config.bin_means = [float(x) for x in bin_means]

    best_config_path = results_dir / "bc_best_config.json"
    with open(best_config_path, "w") as f:
        json.dump({"reservoir": reservoir, "policy_type": POLICY_TYPE,
                   "best_score": best.value, "trial_number": best.number,
                   "config": asdict(bc_best_config)}, f, indent=2)
    print(f"Best trial #{best.number}  score={best.value:.4f}  "
          f"n_bins={bc_best_config.n_bins} binning={bc_best_config.binning}")
    print(f"Best config saved → {best_config_path}\n")

    _, bc_model = _train_and_validate(bc_best_config, data, trial=None, return_model=True)

    policy_path = results_dir / "bc_policy.pt"
    torch.save(
        {"state_dict":  {k: v.cpu() for k, v in bc_model.state_dict().items()},
         "config":      asdict(bc_best_config),
         "policy_type": POLICY_TYPE},
        policy_path,
    )
    print(f"Pretrained policy saved → {policy_path}\n")

    _save_run_args(results_dir, reservoir, run_id, cli_args)

    return {
        "run_folder":       results_dir,
        "best_config":      bc_best_config,
        "best_config_path": best_config_path,
        "best_score":       best.value,
        "policy_path":    policy_path
    }

# =============================================================================
# Main
# =============================================================================

def main():

    args = _parse_args()
    res_cfg_path  = _ROOT / "configs" / "reservoirs" / f"{args.reservoir}.yaml"
    algo_cfg_path = _ROOT / "configs" / "algorithms" / "iqlearn.yaml"

    if not res_cfg_path.exists():
        sys.exit(f"Reservoir config not found: {res_cfg_path}")
    res_cfg  = yaml.safe_load(open(res_cfg_path))
    algo_cfg = yaml.safe_load(open(algo_cfg_path))

    _apply_cli_overrides(args, res_cfg, algo_cfg, res_cfg_path, algo_cfg_path)

    device_str = _resolve_device(algo_cfg["bc_tuning"]["runtime"]["device"])

    print(f"\nLoading data for reservoir '{args.reservoir}' …")

    data = load_reservoir_data(res_cfg, res_cfg_path)

    run_bc_tuning(reservoir=args.reservoir, res_cfg=res_cfg, res_cfg_path=res_cfg_path,
                  algo_cfg=algo_cfg, data=data, device_str=device_str,
                  run_id=args.run_id, cli_args=args)

if __name__ == "__main__":
    main()