"""
run_ppo.py — Train and evaluate ModPPO on FrozenLake-v1
Produces:
  1. Training reward curve (saved as ppo_training_curve.png)
  2. Evaluation results printed to console
  3. Trained model saved as ppo_frozenlake.zip

Usage:
  python run_ppo.py                           # defaults: 50k timesteps, 4x4 map
  python run_ppo.py --timesteps 100000        # longer training
  python run_ppo.py --map_size 8              # 8x8 map
  python run_ppo.py --deterministic           # non-slippery mode
  python run_ppo.py --slip_prob 0.2           # custom slip probability (Dragan et al.)

Baseline config from the proposal (Dragan et al., 2022):
  - Stochastic mode, slip probability ≈ 0.2
  - Sparse reward (goal=1, otherwise 0)
  - Two hidden layers, sizes: [2, 4, 8, 16]
  - One-hot encoded state input
  - 50,000 timesteps
"""

import argparse
import os

import gymnasium as gym
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving plots
import matplotlib.pyplot as plt
import numpy as np
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from PPO.base_PPO import ModPPO

# Record episode rewards during training
class RewardLoggerCallback(BaseCallback):
    """
    Logs episode rewards and lengths during training
    Works with Monitor-wrapped environments
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.timesteps_at_episode = []

    def _on_step(self) -> bool:
        # Monitor wrapper stores completed episode info in "infos"
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
                self.episode_lengths.append(info["episode"]["l"])
                self.timesteps_at_episode.append(self.num_timesteps)
        return True

# Custom FrozenLake w/ adjustable slip probability
def make_frozenlake_env(map_size=4, is_slippery=True, slip_prob=1/3, seed=None):
    """
    Create a FrozenLake environment

    To match Dragan et al. (2022), set slip_prob=0.2
    Default Gymnasium FrozenLake uses slip_prob=1/3
    """
    def _init():
        desc = generate_random_map(size=map_size, seed=seed)
        env = gym.make("FrozenLake-v1", desc=desc, is_slippery=is_slippery)

        # Modify transition probabilities if custom slip_prob
        if is_slippery and slip_prob != 1/3:
            for state in env.unwrapped.P:
                for action in env.unwrapped.P[state]:
                    transitions = env.unwrapped.P[state][action]
                    # Original: 1/3 for intended, 1/3 each for two perpendicular
                    # We want: (1 - slip_prob) for intended, slip_prob/2 each for perpendicular
                    if len(transitions) == 3:
                        intended_prob = 1.0 - slip_prob
                        side_prob = slip_prob / 2.0
                        new_transitions = []
                        for i, (prob, next_state, reward, done) in enumerate(transitions):
                            if i == 0:
                                new_transitions.append((side_prob, next_state, reward, done))
                            elif i == 1:
                                new_transitions.append((intended_prob, next_state, reward, done))
                            elif i == 2:
                                new_transitions.append((side_prob, next_state, reward, done))
                        env.unwrapped.P[state][action] = new_transitions

        env = Monitor(env)
        return env
    return _init

# Plot the training curve
def plot_training_curve(timesteps, rewards, window=50, title="PPO Training on FrozenLake",
                        filename="ppo_training_curve.png"):
    """
    Plot episode rewards over training with a moving average.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    ax.plot(timesteps, rewards, alpha=0.3, color="steelblue", label="Episode Reward")

    # Moving average
    if len(rewards) >= window:
        moving_avg = np.convolve(rewards, np.ones(window) / window, mode="valid")
        # Align timesteps with moving average
        avg_timesteps = timesteps[window - 1:]
        ax.plot(avg_timesteps, moving_avg, color="darkblue", linewidth=2,
                label=f"Moving Avg ({window} episodes)")

    ax.set_xlabel("Timesteps", fontsize=12)
    ax.set_ylabel("Episode Reward", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Training curve saved to: {filename}")

# Evaluations 
def evaluate_agent(model, env_fn, n_eval_episodes=100, deterministic=True):
    """
    Evaluate a trained agent over n episodes
    Returns mean reward, std reward, and success rate
    """
    eval_env = env_fn()
    rewards = []
    successes = 0

    for _ in range(n_eval_episodes):
        obs, info = eval_env.reset()
        done = False
        episode_reward = 0.0

        while not done:
            action = model.select_action(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            episode_reward += reward
            done = terminated or truncated

        rewards.append(episode_reward)
        if episode_reward > 0:
            successes += 1

    eval_env.close()

    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)
    success_rate = successes / n_eval_episodes

    return mean_reward, std_reward, success_rate

def main():
    parser = argparse.ArgumentParser(description="Train PPO on FrozenLake-v1")
    parser.add_argument("--timesteps", type=int, default=50000,
                        help="Total training timesteps (default: 50000, matches Dragan et al.)")
    parser.add_argument("--map_size", type=int, default=4,
                        help="FrozenLake grid size (default: 4)")
    parser.add_argument("--hidden_size", type=int, default=16,
                        help="Neurons per hidden layer (default: 16; try 2, 4, 8, 16)")
    parser.add_argument("--deterministic", action="store_true",
                        help="Use non-slippery (deterministic) mode")
    parser.add_argument("--slip_prob", type=float, default=0.2,
                        help="Slip probability (default: 0.2, matching Dragan et al.)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--n_eval", type=int, default=500,
                        help="Number of evaluation episodes")
    parser.add_argument("--ent_coef", type=float, default=0.01,
                        help="Entropy coefficient (default: 0.01 from proposal)")
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Directory to save outputs")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    is_slippery = not args.deterministic

    print("=" * 60)
    print("PPO on FrozenLake-v1 — Configuration")
    print("=" * 60)
    print(f"  Total timesteps:   {args.timesteps}")
    print(f"  Map size:          {args.map_size}x{args.map_size}")
    print(f"  Hidden layer size: {args.hidden_size}")
    print(f"  Slippery:          {is_slippery}")
    print(f"  Slip probability:  {args.slip_prob}")
    print(f"  Entropy coef:      {args.ent_coef}")
    print(f"  Seed:              {args.seed}")
    print(f"  Eval episodes:     {args.n_eval}")
    print("=" * 60)

    # Create environment
    env_fn = make_frozenlake_env(
        map_size=args.map_size,
        is_slippery=is_slippery,
        slip_prob=args.slip_prob,
        seed=args.seed,
    )
    vec_env = DummyVecEnv([env_fn])

    # Initialize ModPPO with baseline config
    # Policy network: two hidden layers of `hidden_size` neurons each
    # This matches the classical baseline in Dragan et al. (2022)
    policy_kwargs = dict(
        net_arch=dict(pi=[args.hidden_size, args.hidden_size],
                      vf=[args.hidden_size, args.hidden_size]),
    )

    model = ModPPO(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        # Value function clipping is set to None (disabled) by default.
        # Unlike the policy clipping (clip_range), value clipping was NOT
        # part of the original PPO paper (Schulman et al., 2017). It was
        # an addition by OpenAI's implementation to prevent large value
        # function updates. In practice it doesn't consistently help and
        # can hurt, especially with sparse rewards like FrozenLake (0 or 1).
        # Set to a float (e.g. 0.2) to enable it as an experiment.
        clip_range_vf=None,
        ent_coef=args.ent_coef,
        vf_coef=0.5,
        max_grad_norm=0.5,
        seed=args.seed,
        verbose=1,
        policy_kwargs=policy_kwargs,
    )

    print(f"\nPolicy architecture:\n{model.policy}\n")

    # Train 
    callback = RewardLoggerCallback()
    model.learn(total_timesteps=args.timesteps, callback=callback, progress_bar=True)

    # Save model 
    model_path = os.path.join(args.output_dir, "ppo_frozenlake")
    model.save(model_path)
    print(f"\nModel saved to: {model_path}.zip")

    # Plot training curve 
    if len(callback.episode_rewards) > 0:
        timesteps = np.array(callback.timesteps_at_episode)
        rewards = np.array(callback.episode_rewards)

        slip_label = f"slip={args.slip_prob}" if is_slippery else "deterministic"
        title = (f"PPO Training on FrozenLake {args.map_size}x{args.map_size} "
                 f"({slip_label}, hidden={args.hidden_size})")
        plot_path = os.path.join(args.output_dir, "ppo_training_curve.png")
        plot_training_curve(timesteps, rewards, window=50, title=title, filename=plot_path)

        # Also save raw data for later analysis
        np_path = os.path.join(args.output_dir, "ppo_training_data.npz")
        np.savez(np_path, timesteps=timesteps, rewards=rewards,
                 lengths=np.array(callback.episode_lengths))
        print(f"Raw training data saved to: {np_path}")
    else:
        print("Warning: No episode completions recorded during training.")

    # Evaluate 
    print(f"\nEvaluating over {args.n_eval} episodes...")
    mean_r, std_r, success = evaluate_agent(
        model, env_fn, n_eval_episodes=args.n_eval, deterministic=True
    )
    print(f"\nEvaluation Results:")
    print(f"  Mean Reward:  {mean_r:.3f} ± {std_r:.3f}")
    print(f"  Success Rate: {success:.1%} ({int(success * args.n_eval)}/{args.n_eval})")

    # Summary for slides 
    print("\n" + "=" * 60)
    print("SUMMARY FOR SLIDES")
    print("=" * 60)
    print(f"Algorithm:      PPO (SB3 baseline)")
    print(f"Environment:    FrozenLake-v1 {args.map_size}x{args.map_size}")
    print(f"Stochastic:     {is_slippery} (slip prob = {args.slip_prob})")
    print(f"Network:        MLP [{args.hidden_size}, {args.hidden_size}]")
    print(f"Timesteps:      {args.timesteps}")
    print(f"Eval Success:   {success:.1%}")
    print(f"Eval Mean Reward: {mean_r:.3f} ± {std_r:.3f}")
    if len(callback.episode_rewards) > 0:
        last_100 = callback.episode_rewards[-100:]
        print(f"Last 100 ep avg: {np.mean(last_100):.3f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
