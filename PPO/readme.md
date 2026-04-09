# PPO Module

This folder contains the PPO implementation and experiment pipeline used for the FrozenLake comparison project.

## Overview

The PPO workflow is built around a lightweight subclass of Stable-Baselines3 PPO, along with wrappers, logging callbacks, plotting utilities, and an experiment runner.

The default configuration is the tuned PPO setup used for the 8x8 FrozenLake experiments:
- map size: 8x8
- learning rate: 1e-3
- timesteps: 200,000
- rollout size (`n_steps`): 2048
- network: MLP with two hidden layers of size 64
- evaluation: 1000 episodes
- seeds: 3 at first and then 5 for verification if promising

## File Structure

- `__init__.py`  
  Makes the `PPO/` folder importable as a package.

- `base_ppo.py`  
  Defines `ModPPO`, a lightweight SB3 PPO subclass used for the project experiments.

- `onehot_wrapper.py`  
  Converts FrozenLake integer states into one-hot encoded vectors for neural network input.

- `reward_shaping_wrapper.py`  
  Implements optional reward shaping:
  - step penalty
  - potential-based Manhattan distance shaping

- `callbacks.py`  
  Defines `TrainingLoggerCallback`, which logs:
  - episode rewards
  - episode lengths
  - approximate KL
  - entropy loss
  - training timesteps

- `metrics.py`  
  Contains evaluation utilities and summary metric computation.

- `naming.py`  
  Formats run names into readable labels for plots and tables.

- `plots.py`  
  Generates:
  - learning curves
  - entropy loss plots
  - stability signal (KL) plots
  - episode length plots
  - per-seed summary tables
  - aggregate summary tables

- `experiment.py`  
  Main PPO experiment pipeline:
  - loads maps
  - builds configs
  - creates environments
  - trains models
  - evaluates models
  - saves outputs

- `run_ppo.py`  
  Thin command-line entry point for launching PPO experiments.

## CLI Arguments

| Flag | Default | Type | Description |
|------|---------|------|-------------|
| `--timesteps` | `200000` | `int` | Total training timesteps |
| `--hidden_size` | `64` | `int` | Neurons per hidden layer in MLP |
| `--n_steps` | `2048` | `int` | Rollout buffer size |
| `--lr` | `1e-3` | `float` | Learning rate |
| `--deterministic` | `False` | flag | Use deterministic (non-slippery) environment |
| `--seeds` | `1 2 3` | `int+` | Seeds to run |
| `--n_eval` | `1000` | `int` | Number of evaluation episodes |
| `--ent_coef` | `0.0` | `float` | Entropy coefficient |
| `--gae_lambda` | `0.95` | `float` | GAE lambda |
| `--vf_coef` | `0.5` | `float` | Value loss coefficient |
| `--clip_range` | `0.2` | `float` | PPO clip range |
| `--step_penalty` | `0.0` | `float` | Step penalty for reward shaping |
| `--manhattan_scale` | `0.0` | `float` | PBRS Manhattan distance shaping scale |
| `--map_size` | `8` | `int` | Grid size (must exist in `maps.json`) |
| `--run_name` | `None` | `str` | Name for output folder and plot labels |
| `--output_dir` | `None` | `str` | Override default output directory |

## Output Structure

Results are saved under `PPO/results/8x8/<run_name>/`:

## Maps

This module loads maps from the shared project-level `maps.json` file located in the repository root.

All algorithms in the project use the same predefined maps for fair comparison.

## Running Experiments

Run from the project root using module execution:

```bash
python -m PPO.run_ppo
