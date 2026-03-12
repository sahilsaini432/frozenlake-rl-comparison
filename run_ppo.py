"""
run_ppo.py - Train and evaluate ModPPO on FrozenLake-v1

Outputs (saved to output_dir):
  - ppo_training_curve.png  - moving avg reward over timesteps
  - ppo_entropy_loss.png    - entropy loss over training
  - ppo_summary.png         - metrics table for slides
  - ppo_frozenlake.zip      - saved model
  - ppo_training_data.npz   - raw data for later analysis

Usage:
  python run_ppo.py --hidden_size 16 --output_dir results/ppo_results/hs16
  python run_ppo.py --hidden_size 8 --output_dir results/ppo_results/hs8
"""

import argparse
import os

import gymnasium as gym
from gymnasium import spaces
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from PPO.base_PPO import ModPPO


# One-hot observation wrapper
# Dragan et al (2022): "The input to the classical solution is a
# one-hot vector encoding of the state"
class OneHotWrapper(gym.ObservationWrapper):
    """
    Converts FrozenLake's integer observation into a one-hot vector
    Raw integers imply ordering (7 > 3) that doesn't exist on the grid
    """

    def __init__(self, env):
        super().__init__(env)
        self.n_states = env.observation_space.n
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.n_states,), dtype=np.float32
        )

    def observation(self, obs):
        one_hot = np.zeros(self.n_states, dtype=np.float32)
        one_hot[obs] = 1.0
        return one_hot


# Callback to record episode rewards and training losses
class TrainingLoggerCallback(BaseCallback):

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.timesteps_at_episode = []
        self.entropy_losses = []
        self.training_timesteps = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
                self.episode_lengths.append(info["episode"]["l"])
                self.timesteps_at_episode.append(self.num_timesteps)
        return True

    def _on_rollout_end(self) -> None:
        try:
            logger = self.model.logger.name_to_value
            self.entropy_losses.append(logger.get("train/entropy_loss", float("nan")))
            self.training_timesteps.append(self.num_timesteps)
        except AttributeError:
            pass


# Environment factory
def make_frozenlake_env(map_size=4, is_slippery=True, slip_prob=1/3, seed=None):
    """
    Create FrozenLake with custom slip probability
    Dragan et al (2022) uses slip_prob=0.2, Gymnasium default is 1/3
    """
    def _init():
        desc = generate_random_map(size=map_size, seed=seed)
        env = gym.make("FrozenLake-v1", desc=desc, is_slippery=is_slippery)

        if is_slippery and slip_prob != 1/3:
            for state in env.unwrapped.P:
                for action in env.unwrapped.P[state]:
                    transitions = env.unwrapped.P[state][action]
                    if len(transitions) == 3:
                        intended_prob = 1.0 - slip_prob
                        side_prob = slip_prob / 2.0
                        new_transitions = []
                        for i, (prob, ns, r, d) in enumerate(transitions):
                            if i == 1:
                                new_transitions.append((intended_prob, ns, r, d))
                            else:
                                new_transitions.append((side_prob, ns, r, d))
                        env.unwrapped.P[state][action] = new_transitions

        env = OneHotWrapper(env)
        env = Monitor(env)
        return env
    return _init


# Evaluation
def evaluate_agent(model, env_fn, n_episodes=1000):
    """Run trained agent for n episodes, return list of rewards"""
    eval_env = env_fn()
    rewards = []

    for _ in range(n_episodes):
        obs, _ = eval_env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            action = model.select_action(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = eval_env.step(action)
            ep_reward += reward
            done = terminated or truncated
        rewards.append(ep_reward)

    eval_env.close()
    return rewards


# Metrics (from proposal + Dragan et al methodology)
def compute_metrics(timesteps, rewards, eval_rewards, window=50):
    """
    Sample Efficiency: timestep where 50-episode moving avg first crosses 0.5
    Convergence: Dragan et al method - bin into 1000-step intervals, smooth
      with window 10, first point where smoothed value stays within 0.2
    """
    metrics = {}
    metrics["eval_mean"] = np.mean(eval_rewards)
    metrics["eval_std"] = np.std(eval_rewards)
    metrics["success_rate"] = np.mean([1 if r > 0 else 0 for r in eval_rewards])

    # Sample efficiency
    if len(rewards) >= window:
        moving_avg = np.convolve(rewards, np.ones(window) / window, mode="valid")
        avg_ts = timesteps[window - 1:]
        crossed = np.where(moving_avg >= 0.5)[0]
        metrics["sample_eff_05"] = int(avg_ts[crossed[0]]) if len(crossed) > 0 else "Not reached"
        metrics["peak_moving_avg"] = float(np.max(moving_avg))
    else:
        metrics["sample_eff_05"] = "Not enough data"
        metrics["peak_moving_avg"] = "N/A"

    # Convergence (Dragan et al method)
    max_ts = int(timesteps[-1])
    bin_edges = np.arange(0, max_ts + 1000, 1000)
    binned_rewards, binned_ts = [], []
    for i in range(len(bin_edges) - 1):
        mask = (timesteps >= bin_edges[i]) & (timesteps < bin_edges[i + 1])
        if np.any(mask):
            binned_rewards.append(np.mean(rewards[mask]))
            binned_ts.append(bin_edges[i + 1])

    smooth_window = 10
    if len(binned_rewards) >= smooth_window:
        smoothed = np.convolve(binned_rewards, np.ones(smooth_window) / smooth_window, mode="valid")
        smoothed_ts = np.array(binned_ts)[smooth_window - 1:]
        converged_idx = None
        for i in range(len(smoothed)):
            if np.all(np.abs(smoothed[i:] - smoothed[i]) <= 0.2):
                converged_idx = i
                break
        metrics["convergence_timestep"] = int(smoothed_ts[converged_idx]) if converged_idx is not None else "Not converged"
    else:
        metrics["convergence_timestep"] = "Not enough data"

    return metrics


# Plotting
def plot_training_curve(timesteps, rewards, window, title, filename):
    fig, ax = plt.subplots(figsize=(10, 5))
    if len(rewards) >= window:
        moving_avg = np.convolve(rewards, np.ones(window) / window, mode="valid")
        ax.plot(timesteps[window - 1:], moving_avg, color="darkblue", linewidth=2,
                label=f"Success Rate (moving avg, {window} episodes)")
    ax.set_xlabel("Timesteps", fontsize=12)
    ax.set_ylabel("Episode Reward", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {filename}")


def plot_entropy_loss(training_ts, entropy_losses, hidden_size, filename):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(training_ts, entropy_losses, color="purple", linewidth=2)
    ax.set_xlabel("Timesteps", fontsize=12)
    ax.set_ylabel("Entropy Loss", fontsize=12)
    ax.set_title(f"Entropy Loss During Training (hidden={hidden_size})", fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.annotate("Lower (more negative) = less exploration",
                xy=(0.98, 0.98), xycoords="axes fraction",
                ha="right", va="top", fontsize=9, color="gray")
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {filename}")


def save_summary_table(args, metrics, entropy_loss, filename):
    is_slippery = not args.deterministic
    se = metrics["sample_eff_05"]
    conv = metrics["convergence_timestep"]
    peak = metrics.get("peak_moving_avg", "N/A")

    table_data = [
        ["Environment", f"FrozenLake-v1 {args.map_size}x{args.map_size}"],
        ["Stochastic", f"{is_slippery}"],
        ["Slip Probability", f"{args.slip_prob}"],
        ["Network", f"MLP [{args.hidden_size}, {args.hidden_size}]"],
        ["Total Timesteps", f"{args.timesteps:,}"],
        ["Eval Mean Reward", f"{metrics['eval_mean']:.3f} +/- {metrics['eval_std']:.3f}"],
        ["Eval Success Rate", f"{metrics['success_rate']:.1%}"],
        ["Peak Training Reward (moving avg)", f"{peak:.3f}" if isinstance(peak, float) else peak],
        ["Sample Efficiency (threshold=0.5)", f"{se:,} timesteps" if isinstance(se, int) else se],
        ["Convergence Timestep", f"{conv:,} timesteps" if isinstance(conv, int) else conv],
        ["Final Entropy Loss", f"{entropy_loss:.4f}"],
    ]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.axis("off")
    table = ax.table(cellText=table_data, colLabels=["Metric", "Value"],
                     cellLoc="left", colLoc="left", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    for j in range(2):
        table[0, j].set_facecolor("#4472C4")
        table[0, j].set_text_props(color="white", fontweight="bold")
    for i in range(1, len(table_data) + 1):
        color = "#D9E2F3" if i % 2 == 0 else "white"
        for j in range(2):
            table[i, j].set_facecolor(color)

    slip_label = f"slip={args.slip_prob}" if is_slippery else "deterministic"
    ax.set_title(f"PPO Baseline Results - hidden={args.hidden_size} ({slip_label})",
                 fontsize=13, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {filename}")


# Main
def main():
    parser = argparse.ArgumentParser(description="Train PPO on FrozenLake-v1")
    parser.add_argument("--timesteps", type=int, default=50000)
    parser.add_argument("--map_size", type=int, default=4)
    parser.add_argument("--hidden_size", type=int, default=64,
                        help="Neurons per hidden layer (SB3 default: 64)")
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--slip_prob", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ent_coef", type=float, default=0.0,
                    help="Entropy coefficient (SB3 default: 0.0)")
    parser.add_argument("--n_eval", type=int, default=1000)
    parser.add_argument("--output_dir", type=str, default="results")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    is_slippery = not args.deterministic
    slip_label = f"slip={args.slip_prob}" if is_slippery else "deterministic"

    print("=" * 60)
    print("PPO on FrozenLake-v1")
    print("=" * 60)
    print(f"  Timesteps:     {args.timesteps}")
    print(f"  Map:           {args.map_size}x{args.map_size}")
    print(f"  Hidden size:   {args.hidden_size}")
    print(f"  Slippery:      {is_slippery} ({slip_label})")
    print(f"  Seed:          {args.seed}")
    print(f"  Eval episodes: {args.n_eval}")
    print("=" * 60)

    # Environment
    env_fn = make_frozenlake_env(
        map_size=args.map_size, is_slippery=is_slippery,
        slip_prob=args.slip_prob, seed=args.seed,
    )

    # Model - all SB3 PPO defaults except net_arch
    model = ModPPO(
        policy="MlpPolicy",
        env=DummyVecEnv([env_fn]),
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=None,    # not in original PPO paper, disabled by default
        ent_coef=args.ent_coef,    # SB3 default = 0.0
        vf_coef=0.5,
        max_grad_norm=0.5,
        seed=args.seed,
        verbose=1,
        tensorboard_log=os.path.join(args.output_dir, "ppo_tensorboard"),
        policy_kwargs=dict(
            net_arch=dict(pi=[args.hidden_size, args.hidden_size],
                          vf=[args.hidden_size, args.hidden_size]),
        ),
    )

    print(f"\nArchitecture:\n{model.policy}\n")

    # Train
    callback = TrainingLoggerCallback()
    model.learn(total_timesteps=args.timesteps, callback=callback, progress_bar=True)
    model.save(os.path.join(args.output_dir, "ppo_frozenlake"))

    # Collect data
    timesteps = np.array(callback.timesteps_at_episode)
    rewards = np.array(callback.episode_rewards)

    # Evaluate
    print(f"\nEvaluating over {args.n_eval} episodes...")
    eval_rewards = evaluate_agent(model, env_fn, n_episodes=args.n_eval)
    metrics = compute_metrics(timesteps, rewards, eval_rewards)

    # Get final entropy loss
    entropy_loss = callback.entropy_losses[-1] if callback.entropy_losses else float("nan")

    # Generate plots
    plot_training_curve(
        timesteps, rewards, window=50,
        title=f"PPO Training on FrozenLake {args.map_size}x{args.map_size} ({slip_label}, hidden={args.hidden_size})",
        filename=os.path.join(args.output_dir, "ppo_training_curve.png"),
    )
    if callback.entropy_losses:
        plot_entropy_loss(
            callback.training_timesteps, callback.entropy_losses,
            args.hidden_size, filename=os.path.join(args.output_dir, "ppo_entropy_loss.png"),
        )
    save_summary_table(args, metrics, entropy_loss,
                       filename=os.path.join(args.output_dir, "ppo_summary.png"))

    # Save raw data
    np.savez(os.path.join(args.output_dir, "ppo_training_data.npz"),
             timesteps=timesteps, rewards=rewards,
             lengths=np.array(callback.episode_lengths))

    # Print results
    print(f"\nResults:")
    print(f"  Success Rate:      {metrics['success_rate']:.1%}")
    print(f"  Eval Reward:       {metrics['eval_mean']:.3f} +/- {metrics['eval_std']:.3f}")
    print(f"  Sample Efficiency: {metrics['sample_eff_05']}")
    print(f"  Convergence:       {metrics['convergence_timestep']}")
    print(f"  Entropy Loss:      {entropy_loss:.4f}")


if __name__ == "__main__":
    main()