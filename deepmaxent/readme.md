# Deep MaxEnt IRL for Reservoir Operations

Discretized-grid **Maximum-Entropy** inverse RL (Ziebart-style) that decodes a
reservoir operating policy — and its implied reward — from the observed record.
Unlike IQ-Learn (actor–critic, continuous action), this method bins the
state/action space, solves a small MDP exactly, and fits a reward network by
matching expert vs. policy **state-visitation frequencies**.

One command runs the whole study (single stage — no behavioral-cloning warm-start):

```
HP tuning (Optuna)  ─►  retrain + save best model  ─►  results / figures
        best_score                reward_net.pt              figures/*.png
```

---

## 1. Method (what the math does)

- **Discretize** storage / release / inflow into bins (`storage_step`,
  `release_step`, `inflow_step`). State = (storage-bin × inflow-bin); month is an
  exogenous 12-step cyclic context; action = release-bin.
- **Dynamics**: inflow follows a data-estimated Markov chain (Laplace-smoothed);
  storage is deterministic mass balance `S' = clip(S + fvf·(inflow − release))`,
  `fvf = seconds_per_day / volume_factor`.
- **Reward**: a small MLP over 5 features
  `[norm storage, norm release, sin(month), cos(month), norm inflow] → R`.
- **Forward**: entropy-regularized soft value iteration (temperature `tau`,
  discount `gamma`, cyclic 12-month horizon) → softmax policy `Pi`.
- **Learn**: MaxEnt gradient `grad = μ_E − μ_L` (expert vs. policy state-visitation
  frequencies); the reward net steps on `loss = −(R · grad)`. Early-stop on
  validation SVF-L1.
- **Objective (Optuna, maximised)** = the **validation unified score**:
  `0.50·(1−SVF_diff_norm) + 0.125·release_corr + 0.125·storage_corr + 0.125·(1−release_nRMSE) + 0.125·(1−storage_nRMSE)`.

---

## 2. Folder layout

```
deepmaxent/
├── run.py        # end-to-end driver: tune ─► save best ─► results
├── tuning.py     # Optuna search + retrain/save the single best model
├── trainer.py    # MaxEntTrainer: soft VI, SVF, MaxEnt gradient, MC simulate
├── mdp.py        # discretization, trajectories, inflow Markov chain, transition matrix
├── networks.py   # RewardNet (5-feature MLP) + frozen feature-normalization stats
├── data.py       # raw (engineering-unit) loader + chronological year split
├── scoring.py    # unified score, SVF metrics, std-based nRMSE
└── results.py    # MC fans, reward-structure maps, reward SHAP (auto-run; standalone too)
```

Shared with the rest of the repo: `configs/reservoirs/<name>.yaml` (columns,
split, mass_balance), `utils/metrics.py`, and `iqlearn/utils/runs.py` (run-folder
/ device / YAML write-back). The algorithm config is `configs/algorithms/deepmaxent.yaml`.

---

## 3. Installation

Python **≥ 3.10**, from the repository root:

```bash
pip install -r requirements.txt
```

`shap` is only needed for the reward-SHAP figures — `results.py` skips them with a warning if it
is absent. Run everything **from the repository root** (paths resolve relative to it).

---

## 4. Running

```bash
python deepmaxent/run.py --reservoir englebright --device cpu \
    --n_trials 500 --num_workers 4
# figures auto-generate; re-render an existing run with:
python deepmaxent/results.py --reservoir englebright          # defaults to latest run
```

| Flag | Meaning |
|---|---|
| `--reservoir` | matches `configs/reservoirs/<name>.yaml` |
| `--data_path`, `--split_train/val/test` | feed/override the data load (years) |
| `--device` | `auto \| cpu \| cuda` (value iteration is CPU/numpy; only the reward net uses the device) |
| `--n_trials`, `--num_workers` | Optuna budget / parallel workers |
| `--run_id` | reuse a folder id; omitted → auto-increment |
| `--save-config` | persist the above overrides back into the YAML (default: this run only) |

> **Compute note.** Each trial builds a **dense** transition matrix
> `P` of shape `(n_states, 12, n_release, n_states)` and runs value iteration, so
> tuning is heavier than IQ-Learn. Two guards in `deepmaxent.yaml → guards` prune
> oversized discretizations: `max_states` and `max_transition_elems` (the real
> memory bottleneck — fine release/inflow steps can imply multi-GB `P`). Pruned
> trials are logged, never silently dropped.

---

## 5. Outputs (`results/<reservoir>/deepmaxent/<run_id>/`)

| File | Contents |
|---|---|
| `best_config.json` | winning hyperparameters + grid sizes + best score |
| `reward_net.pt` | reward-net weights **+ feature-normalization stats + config** (needed to re-query / SHAP the reward) |
| `policy_Pi.npy`, `reward_table_R.npy` | softmax policy and reward table over the grid |
| `s_space.npy`, `r_space.npy`, `i_space.npy` | discretization grids |
| `metrics.json`, `run_args.json` | val/test SVF + corr + nRMSE + unified score; provenance |
| `figures/mc_fan_test.png`, `figures/mc_fan_full.png` | **Monte-Carlo rollout fans** (median + 25–75% IQR) of storage and release |
| `figures/reward_maps.png` | **reward structure**: 12 monthly storage×release reward contours, expert obs overlaid |
| `figures/shap_reward_overall.png` | **reward-only SHAP**, combined |
| `figures/shap_reward_monthly.png` | reward-only SHAP per month (sin/cos rows dropped) — only when `use_month_encoding` |

---

## 6. Re-generating / customizing results

The pipeline already produces the figures (Stage 2). Run `results.py` directly only to re-render
an existing run or change what is plotted — it reloads `reward_net.pt` + the saved spaces/policy,
so it needs nothing beyond the run folder:

```bash
python deepmaxent/results.py --reservoir englebright          # defaults to the latest run
```

| Flag | Default | Meaning |
|---|---|---|
| `--run_id` | latest | which run folder to visualize |
| `--device` | cpu | inference device for the reward net |
| `--n_mc` | 50 | Monte-Carlo rollouts for the fans |
| `--shap_n_background`, `--shap_nsamples`, `--shap_max_explain` | 100 / 100 / 300 | KernelSHAP budget |
| `--seed` | from `best_config.json` | RNG seed |

See the top-level [README](../README.md) for the project overview.
