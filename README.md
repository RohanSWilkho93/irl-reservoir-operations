# IRL Reservoir Operations

THIS REPOSITORY IS STILL UNDER DEVELOPMENT

A research codebase for learning reservoir release policies from historical operational data using Inverse Reinforcement Learning (IRL). Four algorithms are implemented and benchmarked against each other across nine U.S. reservoirs.

---

## Algorithms

| Algorithm | Type | Key idea |
|---|---|---|
| **Behavioral Cloning (BC)** | Imitation learning baseline | Supervised regression: directly mimic historical release decisions |
| **Deep MaxEnt IRL** | IRL | Learn a reward function maximizing the entropy of the demonstrated policy |
| **AIRL** | IRL + RL | Adversarial training: discriminator learns a reward function; PPO optimizes the policy against it |
| **IQ-Learn** | Offline IRL | Batch-based; learns a Q-function directly from expert data without environment rollouts |

> **Dependency**: AIRL and IQ-Learn both initialize their actor from a pretrained BC checkpoint. BC must be run first for any reservoir before AIRL or IQ-Learn can be tuned or trained.

---

## Repository Layout

```
irl-reservoir-operations/
│
├── configs/
│   ├── algorithms/          # Hyperparameter search spaces (one YAML per algorithm)
│   │   ├── behavioral_cloning.yaml
│   │   ├── airl.yaml
│   │   ├── iqlearn.yaml
│   │   └── deepmaxent.yaml
│   └── reservoirs/          # Per-reservoir settings (data path, columns, splits, bounds)
│       ├── conchas.yaml
│       ├── garrison.yaml
│       └── ...              # one file per reservoir
│
├── data/                    # Raw CSV files (one per reservoir)
│   ├── conchas.csv
│   └── ...
│
├── networks/                # Shared PyTorch network definitions
│   ├── policy.py            # Factory: build_policy_network(type, config)
│   ├── airl.py              # AIRLDiscriminator (reward net + shaping net + critic)
│   ├── iqlearn.py           # IQCriticNetwork (twin Q-network)
│   └── reward_deepmax.py    # Reward network for Deep MaxEnt
│
├── utils/
│   ├── data.py              # load_reservoir_data() → DataSplits
│   ├── metrics.py           # nrmse(), safe_pearsonr()
│   └── runs.py              # _find_run_folder() — locates run folders by run_id
│
├── behavioral_cloning/
│   ├── tune.py              # Optuna search → best_config.json
│   ├── train.py             # Final training run → model.pt
│   └── generate_results.py  # Evaluation + plots
│
├── airl/
│   ├── tune.py
│   ├── train.py
│   ├── generate_results.py
│   └── core.py              # AIRLConfig, AIRLAgent, ReservoirEnvironment
│
├── iqlearn/
│   ├── tune.py
│   ├── train.py
│   ├── generate_results.py
│   └── core.py              # IQLearnConfig, IQLearnAgent, ExpertBuffer
│
├── deepmaxent/
│   ├── tune.py
│   ├── train.py
│   ├── generate_results.py
│   └── core.py
│
└── results/                 # All outputs land here (auto-created)
    └── <reservoir>/
        ├── behavioral_cloning/<run_id>_<policy_type>/
        ├── airl/<run_id>_<policy_type>/
        ├── iqlearn/<run_id>_<policy_type>/
        └── deepmaxent/<run_id>/
```

---

## Policy Network Types

All algorithms except Deep MaxEnt share the same four actor distributions.

| Type | Use when | Notes |
|---|---|---|
| `beta` | Releases always > 0 | Bounded [0,1] output after normalization |
| `lognormal` | Releases always > 0 | Heavy right tail; good for skewed distributions |
| `hardgating` | Zero releases present | Hard gate: samples zero with learned probability, otherwise Beta |
| `softgating` | Zero releases present | Soft gate: weighted mixture with learnable MSE + gate losses |

Check the data for zeros before choosing. For Conchas (flood control), `hardgating` or `softgating` is appropriate.

---

## Reservoir Config

Each `configs/reservoirs/<name>.yaml` controls everything about a reservoir run:

```yaml
data_path: data/conchas.csv

columns:
  date: date
  state: [storage, net_inflow]   # state variables fed into the policy
  action: release
  use_month_encoding: true        # appends sin/cos(month) to state → state_dim = 4

split:
  train: 14   # years
  val:   1
  test:  3

reservoir:
  bounds:                         # auto-filled on first run from training data
    storage:   {min: 88.13,  max: 393.985}
    net_inflow:{min: -8.68,  max: 104.48}
    release:   {min: 0.0,    max: 19.567}
```

`use_month_encoding: true` adds `sin(2π·month/12)` and `cos(2π·month/12)` to every state vector, giving the policy seasonal awareness without treating month as a linear feature. With two raw state variables and month encoding, `state_dim = 4`.

---

## Three-Step Pipeline

Every algorithm follows the same three steps. Run them in order.

```
tune.py  →  train.py  →  generate_results.py
```

---

## Step 1 — Hyperparameter Tuning (`tune.py`)

Runs an Optuna study over the search space defined in `configs/algorithms/<algo>.yaml`. Saves the best trial's hyperparameters to `results/<reservoir>/<algo>/<run_id>_<policy_type>/best_config.json`.

### Behavioral Cloning

```bash
python behavioral_cloning/tune.py \
    --reservoir conchas \
    --policy_type hardgating \
    --device cpu \
    --num_workers 1 \
    --n_trials 50 \
    --run_id 1
```

### AIRL

AIRL reads its BC checkpoint automatically from `results/<reservoir>/behavioral_cloning/`. You must specify which BC run to use with `--bc_run_id`.

```bash
python airl/tune.py \
    --reservoir conchas \
    --policy_type hardgating \
    --bc_run_id 1 \
    --device cpu \
    --num_workers 1 \
    --n_trials 100 \
    --run_id 1
```

### IQ-Learn

Same pattern as AIRL — requires a completed BC run.

```bash
python iqlearn/tune.py \
    --reservoir conchas \
    --policy_type hardgating \
    --bc_run_id 1 \
    --device cpu \
    --num_workers 1 \
    --n_trials 100 \
    --run_id 1
```

Optional: override month encoding (writes back to the reservoir YAML):

```bash
python iqlearn/tune.py ... --use_month_encoding true
```

### Deep MaxEnt

Deep MaxEnt does not use a policy type — the folder is named `<run_id>/` instead of `<run_id>_<policy_type>/`.

```bash
python deepmaxent/tune.py \
    --reservoir conchas \
    --device cpu \
    --num_workers 1 \
    --n_trials 50 \
    --run_id 1
```

**Key `tune.py` arguments**

| Argument | Description |
|---|---|
| `--reservoir` | Must match a file in `configs/reservoirs/` |
| `--policy_type` | `beta`, `lognormal`, `hardgating`, or `softgating` |
| `--bc_run_id` | (AIRL/IQ-Learn only) Run ID of the BC run to use as actor initialization |
| `--run_id` | Integer ID for this tuning run; auto-increments if omitted |
| `--n_trials` | Number of Optuna trials |
| `--num_workers` | Parallel trial workers (set to match available CPU cores) |
| `--device` | `cpu`, `cuda`, `cuda:0`, `mps`, or `auto` |

---

## Step 2 — Final Training (`train.py`)

Reads `best_config.json` from the run folder, trains a single full model with those hyperparameters, and saves the checkpoint.

### Behavioral Cloning

```bash
python behavioral_cloning/train.py \
    --reservoir conchas \
    --policy_type hardgating \
    --run_id 1 \
    --device cpu
```

### AIRL

```bash
python airl/train.py \
    --reservoir conchas \
    --policy_type hardgating \
    --run_id 1 \
    --device cpu
```

### IQ-Learn

```bash
python iqlearn/train.py \
    --reservoir conchas \
    --policy_type hardgating \
    --run_id 1 \
    --device cpu
```

### Deep MaxEnt

```bash
python deepmaxent/train.py \
    --reservoir conchas \
    --run_id 1 \
    --device cpu
```

**Key `train.py` arguments**

| Argument | Description |
|---|---|
| `--reservoir` | Reservoir name |
| `--policy_type` | Can be omitted — inferred from the run folder name |
| `--run_id` | Must match an existing folder created by `tune.py` |
| `--device` | Override the device stored in `best_config.json` |
| `--seed` | Override the seed (use to train ensemble members) |
| `--verbose` | Print per-epoch progress |

**Outputs written to `results/<reservoir>/<algo>/<run_id>_<policy_type>/`:**

| File | Contents |
|---|---|
| `model.pt` | Full checkpoint: network weights + config + metadata |
| `train_log.json` | Per-eval training history, best val score and epoch |
| `run_args.json` | Updated with this run's CLI arguments |

---

## Step 3 — Evaluation and Figures (`generate_results.py`)

Loads `model.pt`, runs Monte Carlo rollouts on the held-out test split, computes metrics, and saves all publication figures. Must be run after `train.py`.

### Behavioral Cloning

```bash
python behavioral_cloning/generate_results.py \
    --reservoir conchas \
    --policy_type hardgating \
    --run_id 1 \
    --device cpu
```

### AIRL

```bash
python airl/generate_results.py \
    --reservoir conchas \
    --policy_type hardgating \
    --run_id 1 \
    --device cpu \
    --n_mc 500
```

### IQ-Learn

```bash
python iqlearn/generate_results.py \
    --reservoir conchas \
    --policy_type hardgating \
    --run_id 1 \
    --device cpu \
    --n_mc 100
```

Skip SHAP for a quick diagnostic run:

```bash
python iqlearn/generate_results.py \
    --reservoir conchas --policy_type hardgating --run_id 1 \
    --device cpu --skip_shap
```

### Deep MaxEnt

```bash
python deepmaxent/generate_results.py \
    --reservoir conchas \
    --run_id 1 \
    --device cpu
```

**Key `generate_results.py` arguments**

| Argument | Description |
|---|---|
| `--n_mc` | Number of stochastic Monte Carlo rollouts (default 100 for IQ-Learn, 500 for AIRL) |
| `--shap_background` | Training samples used as SHAP background (default 100) |
| `--shap_test_size` | Test samples explained by SHAP (default 300) |
| `--skip_shap` | Skip all SHAP computation |

**Outputs (AIRL and IQ-Learn):**

| File | Description |
|---|---|
| `test_metrics.json` | Pearson r and nRMSE for release and storage across MC rollouts |
| `release_test.png` | Observed vs. MC median + IQR band (time series) |
| `storage_test.png` | Same for storage |
| `scatter_release.png` | Observed vs. simulated scatter + 1:1 line (release) |
| `scatter_storage.png` | Same for storage |
| `training_curves.png` | Loss and validation score history |
| `reward_contours.png` / `reward_contour.png` | Learned reward/Q-function contour grid |
| `shap_policy_total.png` | Mean \|SHAP\| per feature — policy network |
| `shap_policy_monthly.png` | Monthly SHAP heatmap — policy (sin/cos month excluded) |
| `shap_qnetwork_total.png` | Mean \|SHAP\| per feature — Q-network (IQ-Learn only) |
| `shap_qnetwork_monthly.png` | Monthly SHAP heatmap — Q-network (IQ-Learn only) |
| `shap_reward_total.png` | Mean \|SHAP\| per feature — reward network (AIRL only) |
| `shap_reward_monthly.png` | Monthly SHAP heatmap — reward network (AIRL only) |

---

## Complete Example: Conchas, IQ-Learn, `hardgating`

```bash
# 1. Tune BC (actor architecture)
python behavioral_cloning/tune.py \
    --reservoir conchas --policy_type hardgating \
    --device cpu --num_workers 4 --n_trials 50 --run_id 1

# 2. Train BC
python behavioral_cloning/train.py \
    --reservoir conchas --policy_type hardgating --run_id 1 --device cpu

# 3. Tune IQ-Learn
python iqlearn/tune.py \
    --reservoir conchas --policy_type hardgating \
    --bc_run_id 1 --device cpu --num_workers 4 --n_trials 100 --run_id 1

# 4. Train IQ-Learn
python iqlearn/train.py \
    --reservoir conchas --policy_type hardgating --run_id 1 --device cpu

# 5. Generate results
python iqlearn/generate_results.py \
    --reservoir conchas --policy_type hardgating --run_id 1 --device cpu
```

---

## Complete Example: Conchas, AIRL, `hardgating`

```bash
# BC steps 1–2 are shared with the example above.

# 3. Tune AIRL
python airl/tune.py \
    --reservoir conchas --policy_type hardgating \
    --bc_run_id 1 --device cpu --num_workers 4 --n_trials 100 --run_id 1

# 4. Train AIRL
python airl/train.py \
    --reservoir conchas --policy_type hardgating --run_id 1 --device cpu

# 5. Generate results
python airl/generate_results.py \
    --reservoir conchas --policy_type hardgating --run_id 1 --device cpu --n_mc 500
```

---

## Results Folder Structure

```
results/
└── conchas/
    ├── behavioral_cloning/
    │   └── 1_hardgating/
    │       ├── best_config.json
    │       ├── model.pt
    │       └── train_log.json
    ├── airl/
    │   └── 1_hardgating/
    │       ├── best_config.json
    │       ├── model.pt
    │       ├── train_log.json
    │       ├── test_metrics.json
    │       └── *.png
    ├── iqlearn/
    │   └── 1_hardgating/
    │       ├── best_config.json
    │       ├── model.pt
    │       ├── train_log.json
    │       ├── test_metrics.json
    │       └── *.png
    └── deepmaxent/
        └── 1/
            ├── best_config.json
            ├── model.pt
            └── *.png
```

Run folder names follow the convention `<run_id>_<policy_type>` (e.g., `1_hardgating`, `2_lognormal`). Deep MaxEnt uses `<run_id>` only (no policy type). The `run_id` auto-increments if `--run_id` is not passed to `tune.py`, so you can run multiple searches side by side without collisions.

---

## Reservoirs

| Name | File | Notes |
|---|---|---|
| Conchas | `conchas.yaml` | Flood control + irrigation; zero releases present |
| Garrison | `garrison.yaml` | |
| Libby | `libby.yaml` | |
| Englebright | `englebright.yaml` | |
| Cottage Grove | `cottage_grove.yaml` | |
| Fern Ridge | `fern_ridge.yaml` | |
| Dexter | `dexter.yaml` | |
| Stockton | `stockton.yaml` | |
| Walter George | `walter_george.yaml` | |

---

## Adding a New Reservoir

1. Add a CSV to `data/` with at minimum `date`, a storage column, an inflow column, and a release column.
2. Copy an existing YAML from `configs/reservoirs/` and update `data_path`, `columns`, and `split`.
3. Run the pipeline as shown above. Normalization bounds are computed automatically from the training split and written back to the YAML on first run.
