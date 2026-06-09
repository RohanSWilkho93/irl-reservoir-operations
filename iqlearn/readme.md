# IQ-Learn for Reservoir Operations

Inverse reinforcement learning that **decodes reservoir operating policy from observed records**.
A categorical-policy / continuous-action twin-critic agent is trained with the
[IQ-Learn](https://arxiv.org/abs/2106.12142) objective (Garg et al., 2021) to imitate a real
operator, then interrogated with closed-loop simulation and SHAP to explain *what* it learned.

The pipeline runs in two stages on a **single data load** and a **single shared run folder**:

```
Behavioral Cloning (warm-start)  ─►  IQ-Learn tuning  ─►  results / figures
        bc_policy.pt                     iq_agent.pt            figures/*.png
```

---

## 1. Folder layout

```
<repo_root>/
├── iqlearn/
│   ├── run.py                  # end-to-end driver: BC ─► IQ
│   ├── bc_tuning.py            # Stage 1: behavioral-cloning tuning (Optuna)
│   ├── iq_tuning.py            # Stage 2: IQ-Learn tuning (Optuna)
│   ├── agent.py                # IQLearnAgent (actor + twin critic + target)
│   ├── loss.py                 # categorical IQ-Learn loss
│   ├── environment.py          # mass-balance closed-loop rollout
│   ├── expert_buffer.py        # in-memory expert transition buffer
│   ├── hyperparameter_metrics.py  # trial-scoring (rollout fidelity)
│   ├── results.py              # post-training figures (auto-run by run.py; also standalone)
│   ├── networks/
│   │   ├── policy.py           # CategoricalPolicy (state ─► bin logits)
│   │   └── critic.py           # TwinCritic  Q(state, release)
│   └── utils/
│       ├── bc_binning.py       # release discretization (bin edges/means)
│       ├── distribution.py     # categorical-action math (probs, sample, soft-value)
│       └── runs.py             # run-folder / device / YAML write-back plumbing
├── utils/
│   ├── data.py                 # load_reservoir_data: read, normalize, chronological split
│   └── metrics.py              # RMSE, Pearson, … (shared)
├── configs/
│   ├── algorithms/iqlearn.yaml # bc_tuning + iq_tuning blocks (search spaces, runtime, scoring)
│   └── reservoirs/<name>.yaml  # columns, physics roles, mass_balance, split, bounds
├── data/
│   └── <name>.csv              # daily storage / inflow / release record
└── results/
    └── <name>/iqlearn/<run_id>/   # one folder per run (created by the BC stage)
```

> Run everything **from the repository root** — paths like `data/<reservoir>.csv` and `configs/…`
> are resolved relative to it (`run.py` and `results.py` add the repo root to `sys.path`).

---

## 2. What each file contains

**Drivers & stages**

- **`run.py`** — End-to-end driver. Loads data once, runs BC (which resolves and *creates*
  the run folder), then warm-starts IQ from `bc_policy.pt` in the **same** folder so splits,
  bounds, normalizer, and `state_dim` are guaranteed identical. Splits the CLI superset into
  BC- and IQ-native namespaces and writes config overrides back (comment-preserving).
- **`bc_tuning.py`** — Stage 1. Supervised fit of the categorical policy to the operator's
  *observed* release (binned), tuned by Optuna. Output: `bc_policy.pt` (the IQ warm-start).
  Entry point: `run_bc_tuning(...)`.
- **`iq_tuning.py`** — Stage 2. Loads `bc_policy.pt` and trains the agent with the IQ-Learn
  objective; each Optuna trial is scored by **closed-loop rollout fidelity** on the validation
  split. Resolves the physical mass balance (**CLI > config > data**). Saves `iq_agent.pt`,
  `iq_best_config.json` (best hyperparameters **+ resolved mass-balance + seed**), and
  `iq_run_args.json`. Entry point: `run_iq_tuning(...)`.

**Model & learning**

- **`agent.py`** — `IQLearnAgent`: actor (categorical policy) + twin critic + target critic.
  `select_action` returns the normalized release — *deterministic* = expected value
  `Σ pₖ·bin_meanₖ`, *stochastic* = two-level sample (with optional `generator`).
  `from_checkpoint` / `save` / `load`; `IQConfig` holds all hyperparameters.
- **`loss.py`** — The categorical IQ-Learn loss: the inverse-soft-Q objective over expert and
  policy transitions, using the critic's per-bin Q and the policy's soft value.
- **`networks/policy.py`** — `CategoricalPolicy`: MLP mapping state → **K raw logits** over the
  discretized release bins (not softmaxed). `build_policy_network(config)`.
- **`networks/critic.py`** — `TwinCritic`: two Q-heads. `forward(state, action) → (q1, q2)` for a
  **continuous** normalized release scalar; `q_all_bins(state, bin_means) → (B, K)` per-bin Q.
  `build_critic_network(config)`.
- **`expert_buffer.py`** — In-memory buffer of observed transitions
  `(state, action, next_state, done)` that feeds the IQ-Learn loss.

**Simulation, scoring & math**

- **`environment.py`** — `ReservoirRollout`: closed-loop simulator. Steps the policy through a
  data trajectory, propagating storage by mass balance
  `S₊₁ = S + (inflow − release)·(seconds_per_day / volume_factor)` with clamp/spill on physical
  bounds (inflow and month read from data). `rollout()` returns simulated/observed storage and
  release in **engineering units**; supports deterministic or stochastic draws. `MassBalanceConfig`.
- **`hyperparameter_metrics.py`** — Trial-scoring: Pearson *r* and normalized RMSE of simulated
  vs. observed storage/release, combined per the config's `scoring` weights into the Optuna objective.
- **`utils/distribution.py`** — Categorical-action math: probabilities/log-probs, entropy, KL,
  sampling (with `generator`), `expected_value` (`Σ pₖ·bin_meanₖ`), `soft_value` (log-sum-exp of
  per-bin Q with the policy and entropy), bin assignment.
- **`utils/bc_binning.py`** — Builds the release discretization (bin edges/means) used to convert
  continuous releases ↔ categorical bins.
- **`utils/runs.py`** — Run-folder and device plumbing: device resolution, run-id pattern and
  folder lookup, deep config update, and comment-preserving YAML write-back.

**Shared (repo-root `utils/`)**

- **`data.py`** — `load_reservoir_data`: reads the CSV, builds the (optional) `sin_month/cos_month`
  encoding, min-max normalizes, **chronologically** splits into train/val/test by years, and writes
  `split` / `bounds` / `data_path` back to the reservoir config. Exposes `Split` and `DataSplits`
  (`states`, `actions`, `raw_actions`, `dates`, `bounds`, `state_cols`, …).
- **`metrics.py`** — Shared numeric helpers (RMSE, safe Pearson correlation, …).

**Visualization**

- **`results.py`** — Post-training figures. **Invoked automatically by `run.py` as Stage 3/3**
  after IQ (and also runnable standalone — see §6). Reproduces the *exact* splits and physics from
  the config + `iq_best_config.json`, then renders the Reward (Q) contours, the Monte-Carlo rollout
  fans, and policy/critic SHAP. Standalone CLI **and** importable `run_generate_results(...)`.

---

## 3. Installation

Python **≥ 3.10** (uses `X | None` typing). Install:

```bash
pip install torch numpy pandas optuna pyyaml matplotlib shap
```

`shap` is only needed for the SHAP figures — `results.py` degrades gracefully (skips them with a
warning) if it is absent.

---

## 4. Configuration

**`configs/algorithms/iqlearn.yaml`** — two top-level blocks, `bc_tuning:` and `iq_tuning:`, each with:

- `model:` — network architecture defaults,
- `runtime:` — `device` (`cpu | cuda | …`) and `num_workers`,
- `optuna:` / `n_trials:` — search budget,
- `search_space:` — hyperparameter ranges Optuna samples,
- `scoring:` (IQ) — weights combining *r* and nRMSE into the trial objective.

**`configs/reservoirs/<name>.yaml`** — per-reservoir:

- `columns:` — `date`, `state` (list), `action`, `use_month_encoding`, and the physics roles
  `storage` / `inflow` (each must be a member of `state`),
- `reservoir.bounds:` — normalization min/max **written by `data.py`** (do not hand-edit),
- `reservoir.mass_balance:` — `seconds_per_day`, `volume_factor`, and physical
  `max/min_storage`, `max/min_release` (`null` → fall back to `bounds`),
- `data_path:` and `split:` (`train` / `val` / `test` in **years**).

CLI flags on `run.py` override config values and are **written back** to the reservoir YAML, so a
later standalone `results.py` reproduces the same run.

---

## 5. Running the pipeline (train ─► figures)

From the repository root:

```bash
python iqlearn/run.py \
  --reservoir <reservoir> \
  --data_path data/<reservoir>.csv \
  --state_variables storage net_inflow \
  --use_month_encoding true \
  --split_train 14 --split_val 1 --split_test 3 \
  --device cuda \
  --bc_n_trials 1000 --bc_n_jobs 8 \
  --iq_n_trials 300  --iq_n_jobs 8
```

Useful flags (full list via `--help`):

| Flag | Meaning |
|---|---|
| `--reservoir` | **required**; matches `configs/reservoirs/<name>.yaml` |
| `--data_path`, `--date_column`, `--state_variables`, `--use_month_encoding` | feed/override the single data load |
| `--split_train/val/test` | split sizes in **years** |
| `--device` | `auto \| cpu \| cuda \| cuda:N \| mps` (both stages) |
| `--run_id` | reuse a folder id; omitted → BC auto-increments |
| `--bc_n_trials/-_n_jobs`, `--iq_n_trials/-_n_jobs` | per-stage Optuna budgets |
| `--storage_variable`, `--inflow_variable` | which state columns are the physics roles |
| `--max/min_storage`, `--max/min_release`, `--seconds_per_day`, `--volume_factor` | IQ-only physical mass-balance overrides |

> `run.py` imports the BC entry point from `iqlearn.bc_tuning`. If your BC file is named
> differently, update that import.

When the pipeline finishes, `run.py` automatically runs Stage 3/3 — `results.py` on the run it just
produced — so the `figures/` are generated for you (see §7). Figure generation is defensive: if it
fails (e.g. `shap`/`matplotlib` missing), the trained agent is still saved and `run.py` prints the
standalone command to retry. Run `results.py` yourself only to **re-render** or **customize** (§6).

---

## 6. Re-generating / customizing results

The pipeline already produces the figures (Stage 3/3 of `run.py`). Run `results.py` directly only to
**re-render** an existing run or **change** what is plotted — e.g. a different `--run_id`, more
Monte-Carlo rollouts, or a different SHAP split. It defaults to the **latest** run folder:

```bash
python iqlearn/results.py --reservoir <reservoir>
```

Knobs (all optional):

| Flag | Default | Meaning |
|---|---|---|
| `--run_id` | latest | which run folder to visualize |
| `--device` | `cpu` | inference device |
| `--n_mc` | 200 | Monte-Carlo rollouts for the fans |
| `--grid_size` | 120 | Q-contour grid resolution |
| `--contour_max_inflows` | 80 | observed inflows averaged per contour panel |
| `--shap_split` | `all` | split SHAP explains: `train \| val \| test \| all` |
| `--shap_n_background`, `--shap_nsamples` | 100 | KernelSHAP background / coalition samples |
| `--shap_max_explain` | 300 | rows explained per SHAP run (`≤0` = all) |
| `--seed` | from `iq_best_config.json` | RNG seed |

It loads `iq_agent.pt` and reuses the **exact** training physics and seed from
`iq_best_config.json`, and reproduces the identical splits from the config — so it needs nothing
from the pipeline run beyond the run folder.

---

## 7. Outputs

Everything lands in `results/<reservoir>/iqlearn/<run_id>/`:

| File | Produced by | Contents |
|---|---|---|
| `bc_policy.pt` | BC | warm-start policy weights |
| `iq_agent.pt` | IQ | final agent (actor + twin critic + target) |
| `iq_best_config.json` | IQ | best hyperparameters + **resolved mass-balance** + seed |
| `iq_run_args.json` | IQ | the run's arguments (provenance) |
| `figures/` | `results.py` | all plots below |

**Figures** (`figures/`):

- **`q_contours.png`** — the **Reward contour** figure. The learned twin-critic value
  `Q(s, a) = min(Q₁, Q₂)` over a storage × release grid, presented as the reward landscape, with
  **all** historical observations overlaid (magenta). One panel per calendar month when month
  encoding is on, else a single panel.
- **`mc_fan_test.png`** / **`mc_fan_full.png`** — Monte-Carlo rollout fans (storage top, release
  bottom): observed vs. median simulated with a 25–75% IQR band, titled with Pearson *r* and nRMSE.
  `test` = **Test data**, `full` = **All data** (train+val+test, one continuous closed-loop run).
- **`shap_policy_overall.png`** / **`shap_critic_overall.png`** — global mean|SHAP| feature
  importance. Policy features = state `[storage, inflow (+ sin/cos)]` (explains expected release);
  critic features = state **+ release** (explains `min(Q₁, Q₂)`, a function of state *and* action).
- **`shap_policy_monthly.png`** / **`shap_critic_monthly.png`** *(month encoding only)* — per-month
  importance heatmaps with the seasonal `sin/cos` rows dropped and the rest renormalized to 100% per
  month (policy rows `Storage, Inflow`; critic rows `Storage, Inflow, Release`).

---

## 8. Method notes

- **Categorical policy, continuous-action critic.** The policy outputs a distribution over
  discretized release bins; the critic scores a *continuous* normalized release. The deterministic
  action is the bin-probability-weighted expected release.
- **Physics in the loop.** Scoring and all simulations roll the policy out closed-loop through the
  mass balance, so trials are ranked on operational fidelity, not one-step accuracy.
- **Figure data scopes** (in `results.py`): contours overlay **all** data; SHAP uses `--shap_split`
  (default **all**); rollout fans are produced for **test** and **all** data.
- **"Reward" labeling.** The contour plots `min(Q₁, Q₂)` and labels it *Reward* — a deliberate,
  consistent presentation choice for the recovered value/reward landscape.

---

## 9. Quick start

```bash
# 1. train (BC ─► IQ)
python iqlearn/run.py --reservoir <reservoir> --data_path data/<reservoir>.csv \
  --state_variables storage net_inflow --use_month_encoding true \
  --split_train 14 --split_val 1 --split_test 3 --device cuda \
  --bc_n_trials 1000 --bc_n_jobs 8 --iq_n_trials 3000 --iq_n_jobs 8

# 2. figures auto-generate at the end of step 1.
#    Re-render or customize an existing run only if you want to:
python iqlearn/results.py --reservoir <reservoir>

# 3. open results/<reservoir>/iqlearn/<run_id>/figures/
```