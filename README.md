# Decoding Reservoir Operations with Inverse Reinforcement Learning

A collection of inverse reinforcement learning (IRL) methods that **recover the
operating policy — and the implied reward — of a real reservoir from its observed
storage/inflow/release record**. Each method trains an agent to imitate the human
operator, then interrogates it with closed-loop simulation, a learned reward
landscape, and SHAP attributions.

All methods share one data pipeline, one config scheme, and one results layout;
each lives in its own package with its own detailed README.

> 📄 **Associated paper.** This repository accompanies the paper 
> **"Decoding decision-making in managed hydrologic systems with inverse reinforcement learning"**, currently under review at *Water Resources Research*.

## Methods

| Method | Folder | Status | README |
|---|---|---|---|
| **IQ-Learn** — inverse soft-Q imitation, BC warm-start | [`iqlearn/`](iqlearn) | ✅ available | [iqlearn/readme.md](iqlearn/readme.md) |
| **DeepMaxEnt** — discretized-grid maximum-entropy IRL | [`deepmaxent/`](deepmaxent) | ✅ available | [deepmaxent/readme.md](deepmaxent/readme.md) |
| **AIRL** — adversarial IRL (BC warm-start + PPO/discriminator) | [`airl/`](airl) | ✅ available | [airl/readme.md](airl/readme.md) |

Start with the **[IQ-Learn](iqlearn/readme.md)**, **[Deep MaxEnt](deepmaxent/readme.md)**, or **[AIRL](airl/readme.md)** README for full pipelines and usage.

**Three complementary flavours.** *IQ-Learn* is actor–critic over a **continuous**
release (the policy family — Beta/LogNormal/gating — is auto-selected from the
data). *Deep MaxEnt* **discretizes** the state–action space, solves a small MDP
exactly, and fits a reward net by state-visitation-frequency matching. *AIRL* is
**adversarial** — a BC-warm-started PPO actor versus a discriminator whose reward
term `g(s,a)` is the recovered reward. IQ-Learn and AIRL share the BC policy
families; all three share the data, configs, and run-folder conventions, and each
has its own `run.py` (`tune → save best → results`).

## What's shared across methods

```
reservoir-irl/
├── iqlearn/                    # IQ-Learn pipeline      (see iqlearn/readme.md)
├── deepmaxent/                 # Deep MaxEnt pipeline   (see deepmaxent/readme.md)
├── airl/                       # AIRL pipeline          (see airl/readme.md)
│       each: run.py (tune → save best → results) + its own readme.md
├── utils/                      # SHARED across all methods
│   ├── data.py                 #   load_reservoir_data: read, month-encode, normalize, chronological split
│   └── metrics.py              #   RMSE, safe Pearson, …
├── configs/
│   ├── reservoirs/<name>.yaml  #   SHARED per-reservoir config (columns, physics roles, split, bounds)
│   └── algorithms/<method>.yaml#   one per method (iqlearn.yaml, deepmaxent.yaml, airl.yaml)
├── data/<name>.csv             # SHARED daily storage / inflow / release records (9 reservoirs)
├── results/<name>/<method>/<run_id>/   # per-method, per-run outputs
├── requirements.txt
└── README.md                   # this file
```

The reservoir configs (`configs/reservoirs/`) and data (`data/`) are method-agnostic, so a
new method reuses them as-is and only adds its own `configs/algorithms/<method>.yaml` and
package folder.  IQ-Learn and AIRL also share the `iqlearn/distributions` policy families;
Deep MaxEnt is self-contained (it discretizes the grid and learns a reward network).

## Policy families (IQ-Learn and AIRL)

For the two **continuous-action** methods (IQ-Learn, AIRL) the policy distribution is **not
chosen by the user** — it is selected from the data and the Paper-1 pairing is enforced
automatically during Behavioral Cloning:

- **release has zero-release days** → tune **HardGating** + **SoftGating** (zero-inflated)
- **release is continuous** → tune **Beta** + **LogNormal**

BC tunes both families in the matched pair and keeps the better policy; that winner warm-starts
the IQ-Learn / AIRL stage. See [iqlearn/readme.md](iqlearn/readme.md) for the per-family math.

**Deep MaxEnt does not use these families** — it discretizes storage/release/inflow into a grid
and learns a reward network over `[storage, release, sin/cos month, inflow]` (see
[deepmaxent/readme.md](deepmaxent/readme.md)).

## Installation

Python **≥ 3.10**:

```bash
pip install -r requirements.txt
```

## Quick start

Run from the repository root — `data/<name>.csv` and `configs/…` resolve relative to it. Each
method's `run.py` does the whole study (`tune → save best → results`); figures auto-generate, and
each method has a standalone `results.py` to re-render an existing run.

```bash
# IQ-Learn  (BC → IQ-Learn → results)
python iqlearn/run.py --reservoir conchas --device cuda \
  --bc_n_trials 1000 --bc_n_jobs 8 --iq_n_trials 2000 --iq_n_jobs 8
python iqlearn/results.py --reservoir conchas          # re-render latest run

# Deep MaxEnt  (tune → save best → results)
python deepmaxent/run.py --reservoir conchas --device cpu --n_trials 500 --num_workers 4
python deepmaxent/results.py --reservoir conchas      # re-render latest run

# AIRL  (BC → adversarial PPO → results)
python airl/run.py --reservoir conchas --device cpu \
  --bc_n_trials 1000 --bc_n_jobs 8 --airl_n_trials 100 --airl_n_jobs 4
python airl/results.py --reservoir conchas            # re-render latest run
```

See each method's README for the full flag list and per-method notes.

## Bundled reservoirs

Nine U.S. Army Corps / Reclamation reservoirs ship in `data/` (each a daily CSV with
`date, storage, net_inflow, release`) with matching configs in `configs/reservoirs/`:

| Name | Config | Operating purposes |
|---|---|---|
| Conchas | `conchas.yaml` | flood control + irrigation |
| Garrison | `garrison.yaml` | flood control + navigation + hydroelectric + water supply + irrigation |
| Libby | `libby.yaml` | flood control + hydroelectric |
| Englebright | `englebright.yaml` | hydroelectric + recreation (run-of-river) |
| Cottage Grove | `cottage_grove.yaml` | flood control + irrigation |
| Fern Ridge | `fern_ridge.yaml` | flood control + irrigation |
| Dexter | `dexter.yaml` | hydroelectric + recreation |
| Stockton | `stockton.yaml` | flood control + hydroelectric |
| Walter George | `walter_george.yaml` | navigation + hydroelectric |

## Run on your own data

The pipelines are **data-agnostic** — point any method at a new reservoir/dataset in three steps:

1. **Add the CSV** as `data/<name>.csv` with daily rows and at least: a `date` column, a
   `storage` column, an inflow column (e.g. `net_inflow`), and a `release` column.
2. **Add a reservoir config** `configs/reservoirs/<name>.yaml` — copy an existing one and edit
   `data_path`, `columns` (including the `storage`/`inflow` physics roles), and `split` (years).
   Use the annotated template below.
3. **Run any method** from the repo root (see [Quick start](#quick-start)). On the **first run**,
   normalization bounds are computed from the **training split** and written back into the YAML,
   so a later standalone `results.py` reproduces the same run.

### Reservoir config (`configs/reservoirs/<name>.yaml`)

```yaml
data_path: data/<name>.csv
columns:
  date: date
  state: [storage, net_inflow]    # state variables fed to the policy
  action: release
  use_month_encoding: true        # append sin/cos(2π·month/12) -> state_dim += 2
  storage: storage                # physics role: mass-balance state   (must be listed in `state`)
  inflow:  net_inflow             # physics role: exogenous inflow forcing (must be listed in `state`)
split:                            # in years, chronological
  train: 14
  val:   1
  test:  3
reservoir:
  bounds:                         # AUTO-FILLED on first run from the TRAIN split — do not hand-edit
    storage:    {min: ..., max: ...}
    net_inflow: {min: ..., max: ...}
    release:    {min: ..., max: ...}
  mass_balance:                   # closed-loop storage: S' = S + (inflow - release)*seconds_per_day/volume_factor
    seconds_per_day: 86400
    volume_factor:   1.0e6        # flow units -> storage units (e.g. m³/s·day -> Mm³)
    max_storage: null             # null -> bounds.storage.max
    min_storage: null
    max_release: null
    min_release: null
```

- The `storage`/`inflow` roles and the `mass_balance` block drive the **closed-loop simulation**
  every method uses (IQ-Learn & AIRL PPO/rollouts; Deep MaxEnt storage propagation), so set them
  for any new reservoir (or leave the `null` mass-balance bounds to fall back to the data bounds).
- `use_month_encoding: true` gives the policy seasonal awareness and enables the **monthly** SHAP /
  reward panels.
- Algorithm hyperparameters and Optuna search spaces live in `configs/algorithms/<method>.yaml`,
  kept separate from the reservoir config. CLI flags on `run.py` are written back to the reservoir
  YAML so runs are reproducible.