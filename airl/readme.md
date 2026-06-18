# AIRL for Reservoir Operations

Adversarial Inverse RL (Fu et al., 2018) that decodes a reservoir operating
policy **and** a recovered reward `g(s,a)` from the observed record. Same
two-stage shape as IQ-Learn, different Stage-2 engine (adversarial PPO instead of
inverse soft-Q):

```
BC tuning (save best policy)  ─►  discriminator warm-up + joint adversarial
        bc_policy.pt              PPO tuning (save best agent)  ─►  results
                                          airl_agent.pt              figures/*.png
```

---

## 1. Method

- **Actor** — the shared `iqlearn` parametric policy (Beta / LogNormal /
  HardGating / SoftGating), BC-warm-started, with a KL-to-BC anchor.
- **Value critic** — `V(s)` for PPO/GAE.
- **Reward net** `g(s,a)` (spectral-normalised) + **shaping net** `h(s)`.
- **Discriminator** `D = sigmoid(f − log pi)`, `f = g(s,a) + gamma·h(s′) − h(s)`.
  Trained by BCE (expert=1 / policy=0) + gradient penalty + label smoothing.
- **Policy** — PPO (clipped surrogate, GAE) on the discriminator-logit reward,
  with entropy bonus + KL-to-BC.
- **Objective (Optuna, val, maximize)**:
  `0.50·disc_balance + 0.125·release_corr + 0.125·(1−release_nRMSE) + 0.125·storage_corr + 0.125·(1−storage_nRMSE)`,
  `disc_balance = max(0, 1 − |expert_acc−0.5| − |policy_acc−0.5|)` (adversarial equilibrium).

The actor family is inherited from the BC checkpoint — the same Stage-2 code
trains every family. AIRL's Stage-1 BC is its own tuner (`airl/bc_tuning.py`)
but reuses IQ-Learn's BC internals, so the BC math is shared.

## 2. Layout

```
airl/
├── run.py          # driver: BC ─► AIRL ─► results
├── bc_tuning.py    # Stage 1: AIRL's BC tuner (data-driven family pair, keep winner)
├── airl_tuning.py  # Stage 2: discriminator warm-up + joint adversarial tuning (Optuna)
├── agent.py        # AIRLAgent: actor + critic + discriminator; PPO + disc updates
├── networks.py     # CriticNetwork, RewardNetwork g(s,a), ShapingNetwork h(s), AIRLDiscriminator
├── environment.py  # gym-style PPO env (repo state convention) + buffers + expert transitions
├── scoring.py      # composite score + rollout fidelity
└── results.py      # MC fans, reward contour g(s,a), SHAP (reward + policy); auto-run + standalone
```
Reuses `iqlearn/distributions` (+ the new `log_prob`/`entropy`), `iqlearn.networks.policy`,
`iqlearn.environment.ReservoirRollout` (rollouts/MC fans), `iqlearn.iq_tuning` (mass-balance
resolution), `iqlearn.results` (MC-fan + SHAP plotting), `utils/`, and `iqlearn.utils.runs`.

## 3. Installation

Python **≥ 3.10**, from the repository root:

```bash
pip install -r requirements.txt
```

`shap` is only needed for the SHAP figures — `results.py` skips them with a warning if it is
absent. Run everything **from the repository root** (paths resolve relative to it).

## 4. Running

```bash
python airl/run.py --reservoir englebright --device cpu \
    --bc_n_trials 1000 --bc_n_jobs 8 --airl_n_trials 100 --airl_n_jobs 4
# figures auto-generate; re-render an existing run with:
python airl/results.py --reservoir englebright            # defaults to latest run
```

| Flag | Meaning |
|---|---|
| `--reservoir` | matches `configs/reservoirs/<name>.yaml` |
| `--data_path`, `--state_variables`, `--use_month_encoding`, `--split_train/val/test` | feed/override the single data load |
| `--device` | `auto \| cpu \| cuda \| cuda:N` (both stages) |
| `--run_id` | reuse a folder id; omitted → BC auto-increments |
| `--bc_n_trials`, `--bc_n_jobs` | Stage-1 BC Optuna budget (trials are **per family**) |
| `--airl_n_trials`, `--airl_n_jobs` | Stage-2 adversarial Optuna budget |
| `--storage_variable`, `--inflow_variable` | which state columns are the physics roles |
| `--max/min_storage`, `--max/min_release`, `--seconds_per_day`, `--volume_factor` | mass-balance overrides (CLI > config > data) |

> **Compute note.** AIRL is the heaviest method: each Stage-2 trial does
> discriminator warm-up + adversarial PPO across **five** networks. Keep
> `airl_n_trials` modest on a laptop; a real sweep wants a cluster.

## 5. Outputs (`results/<reservoir>/airl/<run_id>/`)

| File | Contents |
|---|---|
| `bc_policy.pt`, `bc_best_config.json` | Stage-1 winner (policy + family) |
| `airl_agent.pt` | policy + critic + reward_net + shaping_net + configs |
| `airl_best_config.json`, `metrics.json`, `*run_args.json` | best params + resolved mass-balance; val/test fidelity; provenance |
| `figures/mc_fan_full.png`, `mc_fan_test.png` | Monte-Carlo rollout fans (median + IQR) — all data, test split |
| `figures/reward_contours.png` | recovered reward `g(s,a)` over storage×release, per month, expert overlaid |
| `figures/shap_reward_overall.png` (+ `_monthly`) | SHAP of the recovered reward |
| `figures/shap_policy_overall.png` (+ `_monthly`) | SHAP of the policy's expected release |

(`_monthly` SHAP only when `use_month_encoding` is true.)

## 6. Re-generating / customizing results

The pipeline produces the figures automatically (Stage 3). Run `results.py` directly to re-render
an existing run or change what is plotted — it reloads `airl_agent.pt` + `airl_best_config.json`
(physics + seed), so it needs nothing beyond the run folder:

```bash
python airl/results.py --reservoir englebright            # defaults to the latest run
```

| Flag | Default | Meaning |
|---|---|---|
| `--run_id` | latest | which run folder to visualize |
| `--device` | cpu | inference device |
| `--n_mc` | 200 | Monte-Carlo rollouts for the fans |
| `--grid_size`, `--contour_max_inflows` | 120 / 80 | reward-contour resolution / inflows averaged per panel |
| `--shap_split` | all | split SHAP explains (`train\|val\|test\|all`) |
| `--shap_n_background`, `--shap_nsamples`, `--shap_max_explain` | 100 / 100 / 300 | KernelSHAP budget |
| `--seed` | from `airl_best_config.json` | RNG seed |

See the top-level [README](../README.md) for the project overview.
