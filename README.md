# IRL for Reservoir Operations

> **Status — under active development.** **IQ-Learn is the only end-to-end runnable pipeline today.**
> AIRL and Deep MaxEnt are under development: the interfaces shown for them below are the *planned*
> target API and are not yet functional.

A research codebase for learning reservoir release policies from historical operating records using
Inverse Reinforcement Learning (IRL), benchmarking several algorithms across nine U.S. reservoirs.

---

## Status

| Algorithm | Status | Entry point |
|---|---|---|
| **IQ-Learn** | **Ready** — train **and** figures in one command | `iqlearn/run.py` |
| **AIRL** | **Under development** | `airl/` *(planned)* |
| **Deep MaxEnt** | **Under development** | `deepmaxent/` *(planned)* |

This README documents the **ready** IQ-Learn pipeline in full and sketches the planned interface for
the under-development algorithms. For the complete IQ-Learn reference (file-by-file, all flags and
outputs) see **`iqlearn/README.md`**.

---

## Algorithms

| Algorithm | Type | Key idea | Status |
|---|---|---|---|
| **IQ-Learn** | Offline IRL | Categorical policy over discretized releases + a twin Q-network learned directly from expert data; no environment rollouts in the loss | **Ready** |
| **Deep MaxEnt IRL** | IRL | Learn a reward maximizing the entropy of the demonstrated policy | Under development |
| **AIRL** | IRL + RL | Adversarial reward (discriminator) + PPO policy optimization | Under development |

> The runnable **IQ-Learn** pipeline runs Bheavioral Cloning automatically as its first stage and warm-starts IQ from
> it — you do **not** run BC separately or pass a BC run id. The BC-first dependency in the original
> design applies only to the under-development AIRL path.

---

## Repository layout

```
irl-reservoir-operations/
│
├── iqlearn/                     # ── READY ── (see iqlearn/README.md)
│   ├── run.py                   # driver: BC ─► IQ ─► figures (one command)
│   ├── bc_tuning.py             # Stage 1: behavioral-cloning tuning (Optuna)
│   ├── iq_tuning.py             # Stage 2: IQ-Learn tuning (Optuna)
│   ├── results.py               # Stage 3: figures (auto-run; also standalone)
│   ├── agent.py                 # IQLearnAgent (actor + twin critic + target)
│   ├── loss.py                  # categorical IQ-Learn loss
│   ├── environment.py           # mass-balance closed-loop rollout
│   ├── expert_buffer.py         # in-memory expert transition buffer
│   ├── hyperparameter_metrics.py# trial scoring (rollout fidelity: r, nRMSE)
│   ├── networks/
│   │   ├── policy.py            # CategoricalPolicy (state ─► release-bin logits)
│   │   └── critic.py            # TwinCritic  Q(state, release)
│   └── utils/
│       ├── bc_binning.py        # release discretization (bin edges/means)
│       ├── distribution.py      # categorical-action math (probs, sample, soft-value)
│       └── runs.py              # run-folder / device / YAML write-back plumbing
│
├── airl/                        # ── UNDER DEVELOPMENT ── (planned: tune/train/generate_results + core.py)
├── deepmaxent/                  # ── UNDER DEVELOPMENT ── (planned: tune/train/generate_results + core.py)
│
├── configs/
│   ├── algorithms/              # one YAML per algorithm (search spaces, runtime, scoring)
│   │   ├── iqlearn.yaml         #   bc_tuning + iq_tuning blocks   (READY)
│   │   ├── airl.yaml            #   (under development)
│   │   └── deepmaxent.yaml      #   (under development)
│   └── reservoirs/              # per-reservoir settings (data path, columns, splits, bounds, physics)
│       ├── conchas.yaml
│       ├── cottage_grove.yaml
│       ├── dexter.cyamlsv
│       ├── englebright.yaml
│       ├── fern_ridge.yaml
│       ├── garrison.yaml
│       ├── libby.yaml
│       ├── stockton.yaml
│       └── walter_geroge.yaml   
│
├── data/                        # raw CSV files (one per reservoir)
│   ├── conchas.csv
│   ├── cottage_grove.csv
│   ├── dexter.csv
│   ├── englebright.csv
│   ├── fern_ridge.csv
│   ├── garrison.csv
│   ├── libby.csv
│   ├── stockton.csv
│   └── walter_geroge.csv               
│
├── utils/                       # shared helpers
│   ├── data.py                  # load_reservoir_data() → DataSplits (read, normalize, split)
│   └── metrics.py               # nrmse(), safe_pearsonr()
│
└── results/                     # all outputs land here (auto-created)
    └── <reservoir>/
        ├── iqlearn/<run_id>/        # READY — bare-integer run folders
        ├── airl/<run_id>_<policy>/  # planned
        └── deepmaxent/<run_id>/     # planned
```

> Run everything **from the repository root**.

---

## IQ-Learn — quick start (ready)

A single command loads the data once, tunes BC (Stage 1), warm-starts and tunes IQ-Learn (Stage 2),
and renders all figures (Stage 3):

```bash
python iqlearn/run.py \
    --reservoir garrison \
    --data_path data/garrison.csv \
    --state_variables storage net_inflow \
    --use_month_encoding true \
    --split_train 14 --split_val 1 --split_test 3 \
    --device cpu \
    --bc_n_trials 1000 --bc_n_jobs 8 \
    --iq_n_trials 300  --iq_n_jobs 8
```

Everything lands in `results/<reservoir>/iqlearn/<run_id>/` (the `run_id` is a **bare integer**,
auto-incremented). Figure generation is defensive — if it fails (e.g. `shap` missing) the trained
agent is still saved and the run prints the standalone retry command.

Re-render or customize figures for an existing run (defaults to the latest):

```bash
python iqlearn/results.py --reservoir garrison
# knobs: --run_id  --n_mc  --grid_size  --shap_split {train,val,test,all}  --shap_nsamples  ...
```

Notes specific to the ready pipeline:
- **No `--policy_type`** — IQ-Learn uses a single **categorical** policy over discretized release bins.
- **No `--bc_run_id`** — BC runs automatically as Stage 1 and is warm-started into IQ in the same folder.
- Full flag list, per-file descriptions, and output details: **`iqlearn/README.md`**.

---

## Reservoir config (IQ-Learn)

Each `configs/reservoirs/<name>.yaml` controls a reservoir run. The keys IQ-Learn needs:

```yaml
data_path: data/<reservoir>.csv

columns:
  date:  date
  state: [storage, net_inflow]    # state variables fed to the policy
  action: release
  use_month_encoding: true        # appends sin/cos(2π·month/12) → state_dim = 4
  storage: storage                # physics role: mass-balance state  (must be in `state`)
  inflow:  net_inflow             # physics role: exogenous inflow forcing (must be in `state`)

split:                            # years
  train: 14
  val:   1
  test:  3

reservoir:
  bounds:                         # auto-filled on first run from the TRAIN split (normalization source)
    storage:    {min: 88.13, max: 393.985}
    net_inflow: {min: -8.68, max: 104.48}
    release:    {min: 0.0,   max: 19.567}
  mass_balance:                   # physical clamp/spill + unit conversion (closed-loop simulation)
    seconds_per_day: 86400
    volume_factor:   1.0e6        # m³ → storage units (Mm³)
    max_storage: null             # null → bounds.storage.max
    min_storage: null
    max_release: null
    min_release: null
```

The `storage`/`inflow` roles and the `mass_balance` block are **required by IQ-Learn** (its rollout
propagates storage by mass balance). `use_month_encoding: true` gives the policy seasonal awareness
without treating month as a linear feature. Algorithm hyperparameters and Optuna search spaces live in
`configs/algorithms/iqlearn.yaml` (two blocks: `bc_tuning` and `iq_tuning`). CLI overrides on `run.py`
are written back to the reservoir YAML, so a later standalone `results.py` reproduces the same run.

---

## IQ-Learn outputs

Everything in `results/<reservoir>/iqlearn/<run_id>/`:

| File | Produced by | Contents |
|---|---|---|
| `bc_policy.pt` | BC (Stage 1) | warm-start policy weights |
| `iq_agent.pt` | IQ (Stage 2) | final agent (actor + twin critic + target) |
| `iq_best_config.json` | IQ | best hyperparameters + **resolved mass-balance** + seed |
| `iq_run_args.json` | IQ | run arguments (provenance) |
| `figures/` | Results (Stage 3) | all plots below |

**Figures** (`figures/`):

| File | Description |
|---|---|
| `reward_contours.png` | **Reward contour** — learned `min(Q₁, Q₂)` over a storage × release grid, all observations overlaid (one panel per month with month encoding) |
| `mc_fan_test.png` / `mc_fan_full.png` | Monte-Carlo rollout fans (storage + release): observed vs. median + 25–75% IQR, titled with *r* and nRMSE — **Test data** and **All data** |
| `shap_policy_overall.png` / `shap_critic_overall.png` | global mean\|SHAP\| per feature (policy = state; critic = state **+** release) |
| `shap_policy_monthly.png` / `shap_critic_monthly.png` | per-month SHAP heatmaps (seasonal sin/cos rows dropped, rest renormalized to 100%/month) |

---

## Under development: AIRL & Deep MaxEnt

These follow the planned three-step interface and are **not yet runnable** — the commands below are the
target API. Per-algorithm folders use `<run_id>_<policy_type>` run folders (Deep MaxEnt uses `<run_id>`
only, no policy type), and AIRL's planned flow warm-starts from a BC checkpoint via `--bc_run_id`.

```
tune.py  →  train.py  →  generate_results.py        # planned, per algorithm
```

```bash
# AIRL (planned)
python airl/tune.py  --reservoir conchas --policy_type hardgating --bc_run_id 1 \
    --device cpu --num_workers 4 --n_trials 100 --run_id 1
python airl/train.py --reservoir conchas --policy_type hardgating --run_id 1 --device cpu
python airl/generate_results.py --reservoir conchas --policy_type hardgating --run_id 1 \
    --device cpu --n_mc 500

# Deep MaxEnt (planned)
python deepmaxent/tune.py  --reservoir conchas --device cpu --n_trials 50 --run_id 1
python deepmaxent/train.py --reservoir conchas --run_id 1 --device cpu
python deepmaxent/generate_results.py --reservoir conchas --run_id 1 --device cpu
```

Planned per-algorithm outputs include `best_config.json`, `model.pt`, `train_log.json`,
`test_metrics.json`, time-series/scatter PNGs, reward contours, and SHAP figures (the reward network
for AIRL). These will be documented here as each algorithm lands.

---

## Reservoirs

| Name | File | Notes |
|---|---|---|
| Conchas | `conchas.yaml` | Flood control + irrigation |
| Garrison | `garrison.yaml` | Flood Control + navigation + hydroelectric + water supply + irrigation |
| Libby | `libby.yaml` | Flood Control + hydroelectric |
| Englebright | `englebright.yaml` | Hydroelectric + recreation |
| Cottage Grove | `cottage_grove.yaml` | Flood Control + irrigation |
| Fern Ridge | `fern_ridge.yaml` | Flood Control + irrigation |
| Dexter | `dexter.yaml` | Hydroelectric + recreation |
| Stockton | `stockton.yaml` | Flood Control + hydroelectric |
| Walter George | `walter_george.yaml` | Navigation + hydroelectric |

---

## Adding a new reservoir

1. Add a CSV to `data/` with at least a `date` column, a storage column, an inflow column, and a
   release column.
2. Copy an existing `configs/reservoirs/<name>.yaml` and update `data_path`, `columns` (including the
   `storage` / `inflow` physics roles), and `split`. For IQ-Learn, also set `reservoir.mass_balance`
   (or leave the bounds-derived defaults as `null`).
3. Run the IQ-Learn pipeline as shown above. Normalization `bounds` are computed from the training
   split and written back to the YAML on the first run.
```