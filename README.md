# RL Algorithms Comparison on FrozenLake

Comparison of planning and reinforcement learning algorithms on the FrozenLake-v1 environment (Gymnasium). All algorithms use the same pre-generated maps for fair comparison.

## Algorithms

### DQN — Deep Q-Network
SB3-based DQN with one-hot state encoding, reward shaping wrappers, and sweep tooling across grid sizes.

### PPO — Proximal Policy Optimization
SB3-based PPO with one-hot encoding, potential-based reward shaping, training callbacks, and a full experiment pipeline. Default tuned config: 8x8 grid, 200k timesteps, MLP 64x64.

### A2C — Advantage Actor-Critic
SB3-based A2C baseline.

## Structure

```
MCTS/           - MCTS implementation and experiments
DQN/            - DQN implementation and sweep scripts
PPO/            - PPO implementation and experiment pipeline
A2C/            - A2C implementation
maps.json       - Shared pre-generated maps (sizes 4–128)
```

## Usage

```bash
# MCTS
cd MCTS
python run_mcts.py --selection uct --rollout random --final robust_child --grid 8 --episodes 100
python run_batch.py   # run all configurations

# PPO
cd PPO
python3  run_ppo.py

# DQN
cd DQN
python3 run_dqn_sweep.py

# A2C
cd A2C
python3 a2c.py
```

## Environment

FrozenLake-v1 on 8x8 grid, with and without slipperiness.

## Requirements

```bash
pip3 install gymnasium stable-baselines3 torch matplotlib tqdm
```
