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

## Maps

This module loads maps from the shared project-level `maps.json` file located in the repository root.

All algorithms in the project use the same predefined maps for fair comparison.

## Running Experiments

Run from the project root using module execution:

```bash
python -m PPO.run_ppo