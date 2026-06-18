"""
iqlearn/iq_tuning.py
====================
Optuna hyperparameter search for IQ-Learn, warm-started from a BC policy.

Pipeline position
-----------------
Reads bc_policy.pt from results/<reservoir>/iqlearn/<run_id>/ (produced by
bc_tuning), tunes the IQ-Learn critic + actor refinement, and writes
iq_agent.pt / iq_best_config.json / iq_run_args.json into the SAME folder.
`run.py` calls run_iq_tuning() with data + the shared run_id; the standalone
main() below locates an existing BC run by --run_id (or the latest).

Per-trial training (the heart, in _train_and_score)
---------------------------------------------------
  seed (from BC) -> build agent (warm-started actor + fresh critic)
  PHASE 1  critic warm-up   (actor frozen): critic-only scheduler; early-stop on
           the Q-driven sub-score (the only part of the composite that moves while
           the actor is frozen -> equivalent to "stop when Q quality plateaus",
           and avoids the redundant constant rollout every warm-up epoch).
  RESET    critic LR back to lr_critic, build FRESH joint schedulers so the critic
           re-adapts to the now-moving actor at full LR.
  PHASE 2  joint            (actor+critic): critic + actor schedulers; per-epoch
           composite scoring -> Optuna report (joint-epoch step) + pruning +
           early-stop + best-snapshot.  Best snapshot restored at the end.

One scheduler_type drives three fresh instances: critic-warmup, critic-joint,
actor-joint.  The actor scheduler steps ONLY in the joint phase (the actor takes
no gradient steps during warm-up).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import optuna
import torch
import yaml

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils.data import load_reservoir_data
from iqlearn.utils.runs import _resolve_device, _writeback_yaml, _find_run_folder
from iqlearn.expert_buffer import ExpertBuffer
from iqlearn.environment import MassBalanceConfig, ReservoirRollout
from iqlearn.agent import IQConfig, IQLearnAgent, POLICY_TYPE
from iqlearn.hyperparameter_metrics import (
    composite_score, is_valid_solution, expert_advantage, q_smoothness,
)

_INVALID_SCORE = 0.01   # score assigned to invalid / failed trials


# =============================================================================
# Reproducibility
# =============================================================================

def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# Schedulers (mirror bc_tuning's three options)
# =============================================================================

def _build_scheduler(optimizer, scheduler_type: str):
    """cosine -> CosineAnnealingWarmRestarts; plateau -> ReduceLROnPlateau(max); none -> None."""
    if scheduler_type == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=1)
    if scheduler_type == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=5, factor=0.5)
    return None


def _step_scheduler(sched, score: float, scheduler_type: str) -> None:
    if sched is None:
        return
    if scheduler_type == "plateau":
        sched.step(score)        # higher composite is better (mode="max")
    else:
        sched.step()


# =============================================================================
# Search space  ->  IQConfig
# =============================================================================

def _sample_params(trial, iq_block: dict) -> dict:
    tr, cr, ql = iq_block["training"], iq_block["critic"], iq_block["iqlearn"]

    def f(name, spec):   # continuous {low, high, log}
        return trial.suggest_float(name, spec["low"], spec["high"], log=spec.get("log", False))

    return {
        "batch_size":             trial.suggest_categorical("batch_size", tr["batch_size"]),
        "critic_warm_up_epochs":  trial.suggest_categorical("critic_warm_up_epochs", tr["critic_warm_up_epochs"]),
        "n_epochs":               trial.suggest_categorical("n_epochs", tr["n_epochs"]),
        "warmup_patience":        trial.suggest_categorical("warmup_patience", tr["warmup_patience"]),
        "joint_patience":         trial.suggest_categorical("joint_patience", tr["joint_patience"]),
        "scheduler_type":         trial.suggest_categorical("scheduler_type", tr["scheduler_type"]),
        "critic_hidden_dim":      trial.suggest_categorical("critic_hidden_dim", cr["hidden_dim"]),
        "critic_n_hidden_layers": trial.suggest_categorical("critic_n_hidden_layers", cr["n_hidden_layers"]),
        "gamma":                  trial.suggest_categorical("gamma", ql["gamma"]),
        "lr_actor":               f("lr_actor",      ql["lr_actor"]),
        "lr_critic":              f("lr_critic",     ql["lr_critic"]),
        "tau":                    f("tau",           ql["tau"]),
        "alpha_entropy":          f("alpha_entropy", ql["alpha_entropy"]),
        "alpha_reg":              f("alpha_reg",     ql["alpha_reg"]),
        "lambda_bc":              f("lambda_bc",     ql["lambda_bc"]),
    }


def _iq_config_from_params(p: dict, *, state_dim: int, seed: int, device: str) -> IQConfig:
    return IQConfig(
        state_dim=state_dim, action_dim=1,
        critic_hidden_dim=p["critic_hidden_dim"], critic_n_hidden_layers=p["critic_n_hidden_layers"],
        gamma=p["gamma"], tau=p["tau"], alpha_entropy=p["alpha_entropy"], alpha_reg=p["alpha_reg"],
        lambda_bc=p["lambda_bc"], lr_actor=p["lr_actor"], lr_critic=p["lr_critic"],
        batch_size=p["batch_size"], critic_warm_up_epochs=p["critic_warm_up_epochs"],
        n_epochs=p["n_epochs"], seed=seed, device=device,
        scheduler_type=p["scheduler_type"],
        warmup_patience=p["warmup_patience"], joint_patience=p["joint_patience"],
    )


# =============================================================================
# Mass-balance resolution (CLI > config > data) and normalization bounds
# =============================================================================

def _col_bounds(data, col: str) -> tuple[float, float]:
    """
    Train (lo, hi) for a column from data.bounds, defensively handling either
    a {'min','max'} mapping or a (min, max) pair.
    """
    b = data.bounds[col]
    if isinstance(b, dict):
        return float(b["min"]), float(b["max"])
    return float(b[0]), float(b[1])


def _resolve_mass_balance(res_cfg: dict, data, cli_args) -> MassBalanceConfig:
    cols = res_cfg.get("columns", {}) or {}
    g = (lambda n: getattr(cli_args, n, None)) if cli_args is not None else (lambda n: None)

    storage_col = g("storage_variable") or cols.get("storage")
    inflow_col  = g("inflow_variable")  or cols.get("inflow")
    action_col  = cols.get("action")
    if not storage_col or not inflow_col:
        sys.exit("ERROR: columns.storage and columns.inflow must be set in the reservoir "
                 "config (or passed via --storage_variable / --inflow_variable).")
    if not action_col:
        sys.exit("ERROR: columns.action must be set in the reservoir config.")

    mb_cfg = (res_cfg.get("reservoir", {}) or {}).get("mass_balance", {}) or {}

    def pick(*vals, default):
        for v in vals:
            if v is not None:
                return v
        return default

    def bound(cli_name, cfg_key, col, which):
        v = g(cli_name)
        if v is not None:
            return float(v)
        cv = mb_cfg.get(cfg_key)
        if cv is not None:
            return float(cv)
        lo, hi = _col_bounds(data, col)
        return hi if which == "max" else lo

    mb = MassBalanceConfig(
        storage_col=str(storage_col), inflow_col=str(inflow_col), action_col=str(action_col),
        seconds_per_day=float(pick(g("seconds_per_day"), mb_cfg.get("seconds_per_day"), default=86400.0)),
        volume_factor=float(pick(g("volume_factor"), mb_cfg.get("volume_factor"), default=1.0e6)),
        max_storage=bound("max_storage", "max_storage", storage_col, "max"),
        min_storage=bound("min_storage", "min_storage", storage_col, "min"),
        max_release=bound("max_release", "max_release", action_col, "max"),
        min_release=bound("min_release", "min_release", action_col, "min"),
    )
    mb.validate()
    return mb


# =============================================================================
# Per-trial training + scoring
# =============================================================================

def _snapshot(agent) -> dict:
    sd = lambda m: {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
    return {"actor": sd(agent.actor), "critic": sd(agent.critic), "critic_target": sd(agent.critic_target)}


def _restore(agent, snap: dict) -> None:
    agent.actor.load_state_dict(snap["actor"])
    agent.critic.load_state_dict(snap["critic"])
    agent.critic_target.load_state_dict(snap["critic_target"])


def _warmup_score(agent, buffer, weights: dict, *, n_samples: int = 1000) -> float:
    """
    Q-driven sub-score = w_adv * expert_advantage + w_smooth * q_smoothness.

    During warm-up the actor is frozen (= BC), so entropy / action_diversity /
    prediction_fidelity are constant; only these two terms move.  Monitoring
    them is equivalent to monitoring the full composite for the stop decision,
    and skips the (constant, expensive) rollout each warm-up epoch.
    """
    n = min(n_samples, buffer.size)
    idx = torch.randint(0, buffer.size, (n,), device=buffer.device)
    adv = expert_advantage(agent, buffer.states[idx], buffer.actions[idx])
    smo = q_smoothness(agent, buffer.states[:n], buffer.actions[:n])
    return weights.get("expert_advantage", 0.0) * adv + weights.get("q_smoothness", 0.0) * smo


def _train_and_score(
    iq_config: IQConfig, bc_ckpt: dict, buffer, env, weights: dict, device: str, *,
    eval_interval: int, min_delta: float, trial=None, return_agent: bool = False,
):
    # Seed BEFORE building the agent: critic Xavier init consumes RNG, so every
    # trial shares an identical critic init (fair HP comparison) + reproducibility.
    _seed_everything(iq_config.seed)
    agent = IQLearnAgent(iq_config, bc_ckpt, device)
    sched_type = iq_config.scheduler_type

    # ---- PHASE 1: critic warm-up (actor frozen) ----
    warm_sched = _build_scheduler(agent.critic_opt, sched_type)
    best_warm, warm_stall = -math.inf, 0
    W = iq_config.critic_warm_up_epochs
    for epoch in range(1, W + 1):
        agent.update(buffer.sample(iq_config.batch_size, importance_sample=True), update_actor=False)
        if epoch % eval_interval != 0 and epoch != W:
            continue
        score = _warmup_score(agent, buffer, weights)
        _step_scheduler(warm_sched, score, sched_type)
        if score > best_warm + min_delta:
            best_warm, warm_stall = score, 0
        else:
            warm_stall += 1
            if warm_stall >= iq_config.warmup_patience:
                break

    # ---- RESET critic LR -> lr_critic; fresh schedulers for the joint phase ----
    for group in agent.critic_opt.param_groups:
        group["lr"] = iq_config.lr_critic
        group.pop("initial_lr", None)        # force the fresh scheduler to re-seed base = lr_critic
    crit_sched  = _build_scheduler(agent.critic_opt, sched_type)
    actor_sched = _build_scheduler(agent.actor_opt,  sched_type)

    # ---- PHASE 2: joint training (actor + critic) ----
    best_score, joint_stall, best_snap, joint_eval = -math.inf, 0, None, 0
    N = iq_config.n_epochs
    for epoch in range(1, N + 1):
        agent.update(buffer.sample(iq_config.batch_size, importance_sample=True), update_actor=True)
        if epoch % eval_interval != 0 and epoch != N:
            continue

        joint_eval += 1
        valid, _ = is_valid_solution(agent, buffer)
        score = composite_score(agent, buffer, env, weights)[0] if valid else _INVALID_SCORE

        _step_scheduler(crit_sched,  score, sched_type)
        _step_scheduler(actor_sched, score, sched_type)   # actor scheduler steps ONLY here

        if trial is not None:
            trial.report(score, joint_eval)               # joint-progress step (the pruning clock)
            if trial.should_prune():
                raise optuna.TrialPruned()

        if score > best_score + min_delta:
            best_score, joint_stall = score, 0
            if return_agent:
                best_snap = _snapshot(agent)
        else:
            joint_stall += 1
            if joint_stall >= iq_config.joint_patience:
                break

    # Guard: no joint eval happened (n_epochs < eval_interval) -> evaluate once.
    if best_score == -math.inf:
        best_score = composite_score(agent, buffer, env, weights)[0]
        if return_agent:
            best_snap = _snapshot(agent)

    if return_agent:
        if best_snap is not None:
            _restore(agent, best_snap)
        return best_score, agent
    return best_score


def _make_objective(iq_block, bc_ckpt, buffer, env, weights, state_dim, seed, device,
                    eval_interval, min_delta):
    def objective(trial):
        cfg = _iq_config_from_params(
            _sample_params(trial, iq_block), state_dim=state_dim, seed=seed, device=device,
        )
        try:
            return _train_and_score(cfg, bc_ckpt, buffer, env, weights, device,
                                    eval_interval=eval_interval, min_delta=min_delta, trial=trial)
        except optuna.TrialPruned:
            raise
        except Exception as exc:                          # noqa: BLE001 — robustness over purity
            print(f"  trial {trial.number} failed ({type(exc).__name__}: {exc}); scoring "
                  f"{_INVALID_SCORE}", file=sys.stderr)
            return _INVALID_SCORE
    return objective


# =============================================================================
# Orchestration
# =============================================================================

def _save_iq_run_args(run_folder: Path, reservoir: str, run_id: int, cli_args, mb: MassBalanceConfig) -> None:
    g = (lambda n: getattr(cli_args, n, None)) if cli_args is not None else (lambda n: None)
    payload = {"iq_tune": {
        "reservoir": reservoir, "run_id": run_id, "folder": run_folder.name,
        "policy_type": POLICY_TYPE,
        "device": g("device"), "n_trials": g("n_trials"), "num_workers": g("num_workers"),
        "mass_balance": asdict(mb),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }}
    with open(run_folder / "iq_run_args.json", "w") as f:
        json.dump(payload, f, indent=2)


def run_iq_tuning(*, reservoir: str, res_cfg: dict, algo_cfg: dict, data, device_str: str,
                  run_id: int | None, cli_args=None) -> dict:
    """Run the IQ-Learn Optuna study against a pre-loaded `data` and return artifacts."""
    iq = algo_cfg["iq_tuning"]
    weights = iq["scoring"]
    opt = iq.get("optuna", {}) or {}
    rt  = iq.get("runtime", {}) or {}
    n_trials = int(getattr(cli_args, "n_trials", None) or opt.get("n_trials", 100))
    # parallelism: runtime.num_workers takes precedence over optuna.n_jobs (mirrors bc_tuning)
    n_jobs = int(rt["num_workers"]) if rt.get("num_workers") is not None else int(opt.get("n_jobs", 1))
    eval_interval = max(1, int(opt.get("eval_interval", 1)))
    min_delta = float((iq.get("early_stopping", {}) or {}).get("min_delta", 0.0))
    pruner_cfg = opt.get("pruner", {}) or {}

    # ---- locate the (existing) run folder + bc_policy.pt ----
    base_dir = _ROOT / "results" / reservoir / "iqlearn"
    if run_id is None:
        ids = ([int(d.name) for d in base_dir.iterdir() if d.is_dir() and d.name.isdigit()]
               if base_dir.exists() else [])
        if not ids:
            sys.exit(f"ERROR: no run folder under {base_dir}. Run BC tuning first, or pass --run_id.")
        run_id = max(ids)
    run_folder = _find_run_folder(base_dir, run_id)
    bc_policy_path = run_folder / "bc_policy.pt"
    if not bc_policy_path.exists():
        sys.exit(f"ERROR: bc_policy.pt not found in {run_folder}. Run BC tuning for run_id={run_id} first.")

    # ---- load BC checkpoint once; seed + drift guard ----
    bc_ckpt = torch.load(bc_policy_path, map_location="cpu", weights_only=False)
    bc_config = bc_ckpt["config"]
    seed = int(bc_config["seed"])
    policy_family = bc_config.get("policy_family", "beta")
    if data.state_dim != bc_config["state_dim"]:
        sys.exit(f"ERROR: state_dim mismatch — data={data.state_dim} vs bc_policy="
                 f"{bc_config['state_dim']}. The IQ data load must match the BC run "
                 f"(same --state_variables / --use_month_encoding).")

    # ---- physics + env + buffer ----
    mb = _resolve_mass_balance(res_cfg, data, cli_args)
    norm_bounds = {
        mb.storage_col: _col_bounds(data, mb.storage_col),
        mb.inflow_col:  _col_bounds(data, mb.inflow_col),
        mb.action_col:  _col_bounds(data, mb.action_col),
    }
    env = ReservoirRollout(data.val, data.state_cols, mb, norm_bounds, device_str)
    buffer = ExpertBuffer(data.train, device_str)

    print(f"\nIQ-Learn tuning  |  reservoir={reservoir}  run_id={run_id}  "
          f"family={policy_family}  seed={seed}  trials={n_trials}  device={device_str}")
    print(f"  warm-start: {bc_policy_path.name}   state_dim={data.state_dim}   "
          f"train={buffer.size} transitions   val={env.T} steps")

    # ---- Optuna study ----
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=int(pruner_cfg.get("n_startup_trials", 5)),
            n_warmup_steps=int(pruner_cfg.get("n_warmup_steps", 0)),
        ),
    )
    study.optimize(
        _make_objective(iq, bc_ckpt, buffer, env, weights, data.state_dim, seed,
                        device_str, eval_interval, min_delta),
        n_trials=n_trials, n_jobs=n_jobs,
    )

    # ---- retrain the winning config (single-threaded, isolated) and save ----
    best = study.best_trial
    best_cfg = _iq_config_from_params(best.params, state_dim=data.state_dim, seed=seed, device=device_str)
    best_cfg.policy_family = policy_family
    best_score, agent = _train_and_score(best_cfg, bc_ckpt, buffer, env, weights, device_str,
                                          eval_interval=eval_interval, min_delta=min_delta,
                                          trial=None, return_agent=True)
    agent_path = run_folder / "iq_agent.pt"
    agent.save(agent_path)

    best_config_path = run_folder / "iq_best_config.json"
    with open(best_config_path, "w") as f:
        json.dump({
            "best_params": best.params,
            "search_best_score": float(best.value),
            "retrained_best_score": float(best_score),
            "seed": seed,
            "mass_balance": asdict(mb),
            "iq_config": asdict(best_cfg),
        }, f, indent=2)
    _save_iq_run_args(run_folder, reservoir, run_id, cli_args, mb)

    print(f"\nIQ best (search) {best.value:.4f}  |  retrained {best_score:.4f}")
    print(f"IQ agent  -> {agent_path}")
    print(f"IQ config -> {best_config_path}")

    return {"run_folder": run_folder, "best_config": best_cfg,
            "best_config_path": best_config_path, "best_score": best_score,
            "agent_path": agent_path}


# =============================================================================
# CLI / standalone entry
# =============================================================================

def _parse_args():
    p = argparse.ArgumentParser(
        description="IQ-Learn hyperparameter tuning, warm-started from a BC policy.")
    p.add_argument("--reservoir", required=True)
    p.add_argument("--run_id", type=int, default=None,
                   help="Run folder under results/<reservoir>/iqlearn/ containing bc_policy.pt "
                        "(default: latest).")
    p.add_argument("--device", default=None)
    p.add_argument("--num_workers", type=int, default=None)
    p.add_argument("--n_trials", type=int, default=None)
    # physics / mass-balance overrides (CLI > config > data)
    p.add_argument("--storage_variable", default=None)
    p.add_argument("--inflow_variable", default=None)
    p.add_argument("--max_storage", type=float, default=None)
    p.add_argument("--min_storage", type=float, default=None)
    p.add_argument("--max_release", type=float, default=None)
    p.add_argument("--min_release", type=float, default=None)
    p.add_argument("--seconds_per_day", type=float, default=None)
    p.add_argument("--volume_factor", type=float, default=None)
    return p.parse_args()


def _apply_cli_overrides(args, res_cfg: dict, algo_cfg: dict, res_cfg_path: Path, algo_cfg_path: Path) -> None:
    """Mutate in-memory configs with provided CLI values; write only the changed keys back."""
    g = lambda n: getattr(args, n, None)

    # --- reservoir: physics roles + mass-balance overrides ---
    res_updates: dict = {}
    cols_upd = {k: g(v) for k, v in (("storage", "storage_variable"), ("inflow", "inflow_variable"))
                if g(v) is not None}
    if cols_upd:
        res_cfg.setdefault("columns", {}).update(cols_upd)
        res_updates["columns"] = cols_upd
    mb_upd = {k: g(k) for k in ("max_storage", "min_storage", "max_release", "min_release",
                                "seconds_per_day", "volume_factor") if g(k) is not None}
    if mb_upd:
        res_cfg.setdefault("reservoir", {}).setdefault("mass_balance", {}).update(mb_upd)
        res_updates["reservoir"] = {"mass_balance": mb_upd}

    # --- algorithm: iq runtime / optuna overrides ---
    iq = algo_cfg.setdefault("iq_tuning", {})
    rt_upd = {k: g(v) for k, v in (("device", "device"), ("num_workers", "num_workers"))
              if g(v) is not None}
    op_upd = {"n_trials": g("n_trials")} if g("n_trials") is not None else {}
    if rt_upd:
        iq.setdefault("runtime", {}).update(rt_upd)
    if op_upd:
        iq.setdefault("optuna", {}).update(op_upd)
    algo_sub: dict = {}
    if rt_upd:
        algo_sub["runtime"] = rt_upd
    if op_upd:
        algo_sub["optuna"] = op_upd

    if res_updates:
        _writeback_yaml(res_cfg_path, res_updates)
    if algo_sub:
        _writeback_yaml(algo_cfg_path, {"iq_tuning": algo_sub})


def main():
    args = _parse_args()
    res_cfg_path  = _ROOT / "configs" / "reservoirs" / f"{args.reservoir}.yaml"
    algo_cfg_path = _ROOT / "configs" / "algorithms" / "iqlearn.yaml"
    if not res_cfg_path.exists():
        sys.exit(f"Reservoir config not found: {res_cfg_path}")
    if not algo_cfg_path.exists():
        sys.exit(f"Algorithm config not found: {algo_cfg_path}")

    with open(res_cfg_path) as f:
        res_cfg = yaml.safe_load(f)
    with open(algo_cfg_path) as f:
        algo_cfg = yaml.safe_load(f)

    _apply_cli_overrides(args, res_cfg, algo_cfg, res_cfg_path, algo_cfg_path)
    device_str = _resolve_device(algo_cfg["iq_tuning"]["runtime"]["device"])

    print(f"Loading data for reservoir '{args.reservoir}' \u2026")
    data = load_reservoir_data(res_cfg, res_cfg_path)

    run_iq_tuning(reservoir=args.reservoir, res_cfg=res_cfg, algo_cfg=algo_cfg,
                  data=data, device_str=device_str, run_id=args.run_id, cli_args=args)


if __name__ == "__main__":
    main()