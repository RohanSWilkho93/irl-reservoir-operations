# IQ-Learn for Reservoir Operations

Inverse reinforcement learning that **decodes a reservoir operating policy from
the observed release record**.  A continuous-action twin-critic agent is trained
with the [IQ-Learn](https://arxiv.org/abs/2106.12142) objective (Garg et al.,
2021) to imitate a real operator, then interrogated with closed-loop simulation,
a learned reward (Q) landscape, and SHAP.

The pipeline runs in three stages on a **single data load** and a **single
shared run folder**:

```
Behavioral Cloning (warm-start)  ─►  IQ-Learn tuning  ─►  results / figures
        bc_policy.pt                     iq_agent.pt            figures/*.png
```

---

## 1. The policy family is chosen for you

You never pick a distribution.  Behavioral Cloning inspects the expert release
record and tunes the matching **candidate pair**, then keeps the better policy —
which is exactly the Paper-1 pairing, enforced automatically:

| Release record | Candidate families tuned | Why |
|---|---|---|
| has zero-release days (> 1% of days) | **HardGating**, **SoftGating** | zero-inflated: a Bernoulli gate (release / no-release) × a Beta amount |
| continuous (no zero days) | **Beta**, **LogNormal** | a single positive-release distribution |

BC runs a full Optuna search for **both** families in the pair, retrains each
winner, and saves **only the better** one as `bc_policy.pt` (+ `bc_best_config.json`,
which records the winning `policy_family`).  IQ-Learn then warm-starts from that
single policy and inherits its family automatically — no flag required.

The four families:

| Family | Heads | Action | BC loss | BC-anchor KL |
|---|---|---|---|---|
| `beta` | α, β | reparameterised Beta on [0,1] | Beta NLL | closed-form Beta KL |
| `lognormal` | μ, σ | `exp(N(μ,σ)) − ε`, log-space | Gaussian NLL in log-space | closed-form Gaussian KL |
| `hardgating` | gate, α, β | gate (0/1) × Beta → **exact zeros** | zero-inflated NLL | Bernoulli gate KL + gate·Beta KL |
| `softgating` | gate, α, β | gate-prob × Beta → smooth toward zero | zero-inflated NLL | Bernoulli gate KL + gate·Beta KL |

The selection rule is configurable (`configs/algorithms/iqlearn.yaml → bc_tuning.selection`):

```yaml
selection:
  zero_frac_threshold: 0.01    # > this fraction of (near-)zero days => zero-inflated pair
  zero_release_eps: null       # release <= eps counts as zero; null => max(1e-6, 1e-4*max)
```

---

## 2. Folder layout

```
<repo_root>/
├── iqlearn/
│   ├── run.py                  # end-to-end driver: detect pair ► BC ► IQ ► results
│   ├── bc_tuning.py            # Stage 1: tune BOTH candidate families, keep the winner (Optuna)
│   ├── iq_tuning.py            # Stage 2: IQ-Learn tuning, warm-started from bc_policy.pt (Optuna)
│   ├── agent.py                # IQLearnAgent (actor + twin critic + target)
│   ├── loss.py                 # IQ-Learn loss with a Monte-Carlo soft value
│   ├── environment.py          # mass-balance closed-loop rollout
│   ├── expert_buffer.py        # in-memory expert transition buffer (importance-sampled)
│   ├── hyperparameter_metrics.py  # composite trial score (reward quality + rollout fidelity)
│   ├── results.py              # post-training figures (auto-run by run.py; also standalone)
│   ├── networks/
│   │   ├── policy.py           # ParametricPolicy: shared MLP backbone + the family's heads
│   │   └── critic.py           # TwinCritic  Q(state, release)
│   ├── distributions/          # the pluggable policy families + the data-driven selector
│   │   ├── base.py             #   PolicyDistribution interface
│   │   ├── beta.py / lognormal.py / _gating.py / hardgating.py / softgating.py
│   │   └── __init__.py         #   make_distribution(...) + detect_family_pair(...)
│   └── utils/
│       └── runs.py             # run-folder / device / comment-preserving YAML write-back
├── utils/{data.py, metrics.py} # shared loaders + metrics (used by every algorithm)
├── configs/
│   ├── algorithms/iqlearn.yaml # bc_tuning (shared + per-family search spaces) + iq_tuning
│   └── reservoirs/<name>.yaml  # columns, physics roles, mass_balance, split, bounds
├── data/<name>.csv             # daily storage / inflow / release record
└── results/<name>/iqlearn/<run_id>/   # one folder per run (created by the BC stage)
```

> Run everything **from the repository root** — `data/<name>.csv` and `configs/…`
> resolve relative to it (`run.py` / `results.py` add the repo root to `sys.path`).

---

## 3. What each file contains

**Drivers & stages**

- **`run.py`** — End-to-end driver. Loads data once, detects the candidate family
  pair, runs BC (which resolves and *creates* the run folder), then warm-starts IQ
  from `bc_policy.pt` in the **same** folder so splits, bounds, normalizer, and
  `state_dim` are guaranteed identical. Stage 3/3 renders the figures.
- **`bc_tuning.py`** — Stage 1. `detect_family_pair` from the release record →
  tune BOTH families with Optuna → retrain each winner → keep the better one as
  `bc_policy.pt`. Validation metric: `(release_r + (1 − release_rmse)) / 2` on the
  family's deterministic action. Entry point: `run_bc_tuning(...)`.
- **`iq_tuning.py`** — Stage 2. Loads `bc_policy.pt`, reads its `policy_family`,
  rebuilds the matching actor, and trains the agent with the IQ-Learn objective;
  each trial is scored by **closed-loop rollout fidelity + reward quality** on the
  validation split. Resolves the physical mass balance (**CLI > config > data**).
  Saves `iq_agent.pt`, `iq_best_config.json`, `iq_run_args.json`. Entry: `run_iq_tuning(...)`.

**Model & learning**

- **`distributions/`** — the strategy package. `base.PolicyDistribution` declares
  the interface (`make_heads`, `params_from_features`, `mean_action`, `rsample`,
  `nll`, `kl`); each family implements it; `make_distribution(family, params)` and
  `detect_family_pair(raw_actions, …)` are the public entry points.
- **`networks/policy.py`** — `ParametricPolicy`: shared MLP backbone whose features
  feed the chosen family's parameter heads. `forward(states) → params dict`.
- **`networks/critic.py`** — `TwinCritic`: two Q-heads over `[state ⊕ release]`;
  `forward(state, action) → (q1, q2)`. Distribution-independent (tuned by Optuna).
- **`agent.py`** — `IQLearnAgent`: actor (warm-started, family inherited from BC) +
  twin critic + target. `select_action` returns the family mean (deterministic) or a
  sample. `save` / `from_checkpoint`; `IQConfig` holds the IQ hyperparameters.
- **`loss.py`** — IQ-Learn loss. Because the action is continuous, the soft value
  `V(s)=E_π[Q − α·log π]` is a **Monte-Carlo** estimate over `n_action_samples` draws;
  the actor loss adds `λ_bc · KL(actor ‖ BC)` via the family's closed-form KL.
- **`expert_buffer.py`** — device-resident expert transitions with high-release
  importance sampling (a deliberate emphasis to counter zero-inflation).

**Simulation, scoring & math**

- **`environment.py`** — `ReservoirRollout`: closed-loop simulator stepping the policy
  through the mass balance `S₊₁ = S + (inflow − release)·(seconds_per_day/volume_factor)`
  with clamp/spill on physical bounds.
- **`hyperparameter_metrics.py`** — the composite Optuna objective: expert advantage +
  Q-smoothness (reward quality) + closed-loop prediction fidelity + entropy +
  action-diversity (robustness).
- **`utils/runs.py`** — device resolution, run-id allocation/lookup, comment-preserving
  YAML write-back, and a UTF-8 stdout shim (Windows-safe console output).

**Shared (repo-root `utils/`)** — `data.py` (`load_reservoir_data`: read, sin/cos month
encoding, train-only min-max normalize, chronological year split, bounds write-back) and
`metrics.py` (RMSE, safe Pearson).

**Visualization** — `results.py` renders the reward (Q) contours, Monte-Carlo rollout fans
(test + full), and policy/critic SHAP. Family-agnostic (it only calls `agent.critic` and
`agent.select_action`). Auto-run as Stage 3/3 and runnable standalone.

---

## 4. Installation

Python **≥ 3.10**. From the repository root:

```bash
pip install -r requirements.txt
```

`shap` is only needed for the SHAP figures — `results.py` degrades gracefully (skips them
with a warning) if it is absent.

---

## 5. Running the pipeline

```bash
python iqlearn/run.py \
  --reservoir conchas \
  --data_path data/conchas.csv \
  --state_variables storage net_inflow \
  --use_month_encoding true \
  --split_train 14 --split_val 1 --split_test 3 \
  --device cuda \
  --bc_n_trials 1000 --bc_n_jobs 8 \
  --iq_n_trials 2000 --iq_n_jobs 8
```

| Flag | Meaning |
|---|---|
| `--reservoir` | **required**; matches `configs/reservoirs/<name>.yaml` |
| `--data_path`, `--date_column`, `--state_variables`, `--use_month_encoding` | feed/override the single data load |
| `--split_train/val/test` | split sizes in **years** |
| `--device` | `auto \| cpu \| cuda \| cuda:N \| mps` (both stages) |
| `--run_id` | reuse a folder id; omitted → BC auto-increments |
| `--bc_n_trials/-_n_jobs`, `--iq_n_trials/-_n_jobs` | per-stage Optuna budgets (BC trials are **per family**) |
| `--storage_variable`, `--inflow_variable` | which state columns are the physics roles |
| `--max/min_storage`, `--max/min_release`, `--seconds_per_day`, `--volume_factor` | IQ-only mass-balance overrides |

Figure generation is defensive: if it fails (e.g. `shap`/`matplotlib` missing) the trained
agent is still saved and `run.py` prints the standalone command to retry.

The two stages are also runnable on their own — `python iqlearn/bc_tuning.py --reservoir <name>`
then `python iqlearn/iq_tuning.py --reservoir <name>` (defaults to the latest BC run folder).

---

## 6. Re-generating / customizing results

```bash
python iqlearn/results.py --reservoir <name>        # defaults to the latest run
```

Knobs: `--run_id`, `--device`, `--n_mc`, `--grid_size`, `--contour_max_inflows`,
`--shap_split {train|val|test|all}`, `--shap_n_background`, `--shap_nsamples`,
`--shap_max_explain`, `--seed`. It reuses the exact training physics + seed from
`iq_best_config.json`.

---

## 7. Outputs (`results/<reservoir>/iqlearn/<run_id>/`)

| File | Stage | Contents |
|---|---|---|
| `bc_policy.pt`, `bc_best_config.json` | BC | winning policy weights + config (incl. `policy_family` + `dist_params`) |
| `iq_agent.pt`, `iq_best_config.json` | IQ | final agent + resolved mass-balance + seed |
| `run_args.json`, `iq_run_args.json` | both | provenance (candidate + winning family) |
| `figures/reward_contours.png` | results | learned `min(Q₁,Q₂)` reward landscape, observations overlaid (per-month if month encoding) |
| `figures/mc_fan_test.png`, `mc_fan_full.png` | results | Monte-Carlo rollout fans (median + 25–75% IQR) vs observed |
| `figures/shap_policy_*.png`, `shap_critic_*.png` | results | SHAP feature importance (overall + monthly) |

---

## 8. Method notes

- **Parametric policy.** The BC/IQ actor is a parametric distribution over the continuous
  normalised release; the twin critic scores that continuous release. The deterministic 
  action is the family mean.
- **Monte-Carlo soft value.** A continuous action precludes the exact bin-sum, so
  `V(s)=E_π[Q − α·log π]` is averaged over `n_action_samples` policy draws.
- **HardGating vs SoftGating.** Hard emits *exact zeros* (threshold / Bernoulli gate);
  Soft multiplies by the gate probability continuously (smooth toward zero). Both share
  the zero-inflated NLL and the gate+Beta KL.
- **Physics in the loop.** Trials are ranked on closed-loop operational fidelity, not
  one-step accuracy.
