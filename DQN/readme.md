# DQN for FrozenLake (8x8)

> **Authorship / AI Assistance Disclaimer**
> - `base_DQN.py`, `onehot_wrapper.py`, and `reward_shaping_wrapper.py` are my own implementation.
> - Sweep/plot automation scripts and generated sweep plot artifacts were produced with LLM assistance.
> - `base_DQN.py` was also updated by LLM to stay consistent with the sweep-run workflow and experiment setup.
>

Run from project root:

```bash
python3 DQN/run_dqn_sweep.py [options]
```

---

## Project Files

| File | Role |
|------|------|
| `DQN/base_DQN.py` | Core DQN train/test loop for FrozenLake |
| `DQN/onehot_wrapper.py` | One-hot observation wrapper |
| `DQN/reward_shaping_wrapper.py` | Dense reward shaping wrapper |
| `DQN/run_dqn_sweep.py` | Hyperparameter sweeps + cross-repeat aggregation |
| `DQN/DQN_plots.py` | Regenerate training figures from CSV logs |
| `DQN/plot_onehot_vs_baseline.py` | One-hot vs discrete comparison plots |
| `DQN/plot_shaped_vs_baseline.py` | Reward-shaping vs baseline comparison plots |
| `DQN/plot_sweep_across_runs.py` | Helper plot for recent sweep summaries |
| `maps.json` | Custom FrozenLake map definitions |

---

## `run_dqn_sweep.py` CLI Arguments

| Flag | Default | Type | Description |
|------|---------|------|-------------|
| `--run-tag` | timestamp | `str` | Output folder prefix under `dqn_plots/sweep/` |
| `--one-hot` / `--no-one-hot` | from file default | bool flag | Enable/disable one-hot state encoding |
| `--reward-shaping` / `--no-reward-shaping` | from file default | bool flag | Enable/disable reward shaping wrapper |
| `--n-repeats` | `N_REPEATS` in file | `int` | Number of independent repeated sweeps |
| `--analyze-only` | `False` | flag | Skip training; aggregate existing `<run-tag>_rep*/sweep_summary.csv` |
| `--also-replot-config` | `None` | `str` | In addition to BEST config, also regenerate figures for a specific config |

### Current sweep defaults in code

- Map: 8x8 (`is_slippery=True`)
- Training timesteps: `200_000`
- Test episodes per config: `1000`
- Config set: `cfg1` to `cfg6` in `sweep_configs_all()`

---

## Main Experiment Commands

### 1) Baseline (discrete state, no reward shaping)

```bash
python3 DQN/run_dqn_sweep.py --no-one-hot --no-reward-shaping --run-tag baseline_discrete_200k --n-repeats 5
```

### 2) One-hot (no reward shaping)

```bash
python3 DQN/run_dqn_sweep.py --one-hot --no-reward-shaping --run-tag onehot_discrete_200k --n-repeats 5
```

### 3) Reward shaping (discrete state)

```bash
python3 DQN/run_dqn_sweep.py --no-one-hot --reward-shaping --run-tag shaped_discrete_200k --n-repeats 5
```

### 4) Analyze existing runs only (no retraining)

```bash
python3 DQN/run_dqn_sweep.py --analyze-only --run-tag baseline_discrete_200k --n-repeats 5
```

---

## Comparison Plot Commands

### One-hot vs baseline (average over all configs per repeat)

```bash
python3 DQN/plot_onehot_vs_baseline.py \
  --discrete-base-tag baseline_discrete_200k \
  --onehot-base-tag onehot_discrete_200k \
  --map-size 8 --discover-repeats
```

### One-hot vs baseline (fixed config, e.g. cfg5)

```bash
python3 DQN/plot_onehot_vs_baseline.py \
  --discrete-base-tag baseline_discrete_200k \
  --onehot-base-tag onehot_discrete_200k \
  --map-size 8 --discover-repeats \
  --fixed-config cfg5_more_explore
```

### Reward shaping vs baseline

```bash
python3 DQN/plot_shaped_vs_baseline.py \
  --baseline-base-tag baseline_discrete_200k \
  --shaped-base-tag shaped_discrete_200k \
  --map-size 8 --discover-repeats --plot all
```

### Reward shaping vs baseline (fixed config)

```bash
python3 DQN/plot_shaped_vs_baseline.py \
  --baseline-base-tag baseline_discrete_200k \
  --shaped-base-tag shaped_discrete_200k \
  --map-size 8 --discover-repeats \
  --fixed-config cfg5_more_explore
```

---

## Plot Script Options (Summary)

### `plot_onehot_vs_baseline.py`

| Flag | Description |
|------|-------------|
| `--discrete-base-tag`, `--onehot-base-tag` | Base tags for run folders |
| `--discover-repeats` | Auto-detect `<tag>_rep*` folders |
| `--n-repeats` | Manual repeat count if not auto-discovering |
| `--map-size` | Map size filter |
| `--fixed-config` | Compare only one config (e.g., `cfg5_more_explore`) |

### `plot_shaped_vs_baseline.py`

| Flag | Description |
|------|-------------|
| `--baseline-base-tag`, `--shaped-base-tag` | Base tags for run folders |
| `--discover-repeats` | Auto-detect `<tag>_rep*` folders |
| `--plot avg` | Average-over-configs view |
| `--plot best-worst` | Best/worst-per-run view |
| `--plot all` | Generate both avg and best-worst plots |
| `--fixed-config` | Compare only one config (e.g., `cfg5_more_explore`) |

---

## Run Checklist (DQN)

- [x] Baseline discrete sweep (5 repeats)
- [x] One-hot sweep (5 repeats)
- [x] Reward-shaping sweep (5 repeats)
- [x] Cross-repeat aggregate CSVs in `dqn_plots/sweep/compare_latest/`
- [x] One-hot vs baseline comparison plots
- [x] Reward-shaping vs baseline comparison plots
- [x] Fixed-config comparison plots (cfg5)

---

## Notes

- For reproducible references, sweep outputs (CSV + PNG) are kept under `dqn_plots/sweep/` and `compare_latest/`.