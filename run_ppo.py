"""
run_ppo.py - Train and evaluate PPO baseline on FrozenLake-v1

Baseline definition:
  - FrozenLake-v1, standard 4x4 map, is_slippery=True (default 1/3 slip)
  - One-hot encoded state input
  - MlpPolicy with [64, 64] hidden layers
  - All SB3 PPO default hyperparameters
  - 50,000 training timesteps
  - Deterministic evaluation over 1000 episodes
  - 3 seeds for stochastic, 1 for deterministic

Usage:
  python run_ppo.py                                    # 3-seed stochastic baseline
  python run_ppo.py --seeds 1 2 3                      # explicit seeds
  python run_ppo.py --deterministic --seeds 1           # deterministic mode
  python run_ppo.py --hidden_size 16                   # vary architecture
"""

import argparse
import os

import gymnasium as gym
from gymnasium import spaces
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from PPO.base_PPO import ModPPO


# One-hot encoding is required for FrozenLake because raw integer
# states (0-15) imply false numeric ordering
class OneHotWrapper(gym.ObservationWrapper):

    def __init__(self, env):
        super().__init__(env)
        self.n_states = env.observation_space.n
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(self.n_states,), dtype=np.float32
        )

    def observation(self, obs):
        one_hot = np.zeros(self.n_states, dtype=np.float32)
        one_hot[int(obs)] = 1.0
        return one_hot


# Callback to record episode rewards and entropy loss during training
class TrainingLoggerCallback(BaseCallback):

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.timesteps_at_episode = []
        self.entropy_losses = []
        self.approx_kls = []
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
            self.approx_kls.append(logger.get("train/approx_kl", float("nan")))
            self.training_timesteps.append(self.num_timesteps)
        except AttributeError:
            pass


# Environment factory - standard FrozenLake 4x4 map
def make_frozenlake_env(is_slippery=True):
    def _init():
        env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=is_slippery)
        env = OneHotWrapper(env)
        env = Monitor(env)
        return env
    return _init


# Evaluation
def evaluate_agent(model, env_fn, n_episodes=1000):
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


# Metrics
def compute_metrics(eval_rewards):
    metrics = {}
    metrics["eval_mean"] = np.mean(eval_rewards)
    metrics["eval_std"] = np.std(eval_rewards)
    metrics["success_rate"] = np.mean([1 if r > 0 else 0 for r in eval_rewards])
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


def plot_entropy_loss(training_ts, entropy_losses, title, filename):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(training_ts, entropy_losses, color="purple", linewidth=2)
    ax.set_xlabel("Timesteps", fontsize=12)
    ax.set_ylabel("Entropy Loss", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.annotate("Lower (more negative) = less exploration",
                xy=(0.98, 0.98), xycoords="axes fraction",
                ha="right", va="top", fontsize=9, color="gray")
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()


def plot_approx_kl(training_ts, approx_kls, title, filename):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(training_ts, approx_kls, color="red", linewidth=2)
    ax.set_xlabel("Timesteps", fontsize=12)
    ax.set_ylabel("Approx KL Divergence", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.annotate("Small + stable = healthy, exploding = policy collapse",
                xy=(0.98, 0.98), xycoords="axes fraction",
                ha="right", va="top", fontsize=9, color="gray")
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()


def plot_episode_length(timesteps, lengths, window, title, filename):
    fig, ax = plt.subplots(figsize=(10, 5))
    if len(lengths) >= window:
        moving_avg = np.convolve(lengths, np.ones(window) / window, mode="valid")
        ax.plot(timesteps[window - 1:], moving_avg, color="green", linewidth=2,
                label=f"Avg Episode Length ({window} episodes)")
    ax.set_xlabel("Timesteps", fontsize=12)
    ax.set_ylabel("Episode Length", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()


def save_summary_table(config, metrics, entropy_loss, seed, filename):
    table_data = [
        ["Environment", "FrozenLake-v1 4x4"],
        ["Stochastic", f"{config['is_slippery']}"],
        ["Network", f"MLP [{config['hidden_size']}, {config['hidden_size']}]"],
        ["Total Timesteps", f"{config['timesteps']:,}"],
        ["Seed", f"{seed}"],
        ["Eval Mean Reward", f"{metrics['eval_mean']:.3f} +/- {metrics['eval_std']:.3f}"],
        ["Eval Success Rate", f"{metrics['success_rate']:.1%}"],
        ["Final Entropy Loss", f"{entropy_loss:.4f}"],
    ]

    fig, ax = plt.subplots(figsize=(9, 4.5))
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

    mode = "stochastic" if config["is_slippery"] else "deterministic"
    ax.set_title(f"PPO Baseline - {mode} (seed={seed})",
                 fontsize=13, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()


# Single training run
def run_single_seed(seed, config, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    is_slippery = config["is_slippery"]
    hidden_size = config["hidden_size"]
    timesteps = config["timesteps"]
    n_eval = config["n_eval"]

    print(f"\n  Seed {seed} | {'stochastic' if is_slippery else 'deterministic'} | hidden={hidden_size}")

    env_fn = make_frozenlake_env(is_slippery=is_slippery)

    model = ModPPO(
        policy="MlpPolicy",
        env=DummyVecEnv([env_fn]),
        learning_rate=config["lr"],
        n_steps=config["n_steps"],
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        clip_range_vf=None,
        ent_coef=config.get("ent_coef", 0.0),
        vf_coef=0.5,
        max_grad_norm=0.5,
        seed=seed,
        verbose=0,
        tensorboard_log=os.path.join(output_dir, "ppo_tensorboard"),
        policy_kwargs=dict(
            net_arch=dict(pi=[hidden_size, hidden_size],
                          vf=[hidden_size, hidden_size]),
        ),
    )

    callback = TrainingLoggerCallback()
    model.learn(total_timesteps=timesteps, callback=callback, progress_bar=True)
    model.save(os.path.join(output_dir, f"ppo_seed{seed}"))

    timesteps_arr = np.array(callback.timesteps_at_episode)
    rewards_arr = np.array(callback.episode_rewards)

    eval_rewards = evaluate_agent(model, env_fn, n_episodes=n_eval)
    metrics = compute_metrics(eval_rewards)
    entropy_loss = callback.entropy_losses[-1] if callback.entropy_losses else float("nan")

    mode = "stochastic" if is_slippery else "deterministic"
    plot_training_curve(
        timesteps_arr, rewards_arr, window=100,
        title=f"Baseline Learning Curve (seed={seed})",
        filename=os.path.join(output_dir, f"ppo_training_curve_seed{seed}.png"),
    )
    if callback.entropy_losses:
        plot_entropy_loss(
            callback.training_timesteps, callback.entropy_losses,
            title=f"Baseline Entropy Loss (seed={seed})",
            filename=os.path.join(output_dir, f"ppo_entropy_loss_seed{seed}.png"),
        )
    if callback.approx_kls:
        plot_approx_kl(
            callback.training_timesteps, callback.approx_kls,
            title=f"Baseline PPO Stability Signal (seed={seed})",
            filename=os.path.join(output_dir, f"ppo_approx_kl_seed{seed}.png"),
        )
    if len(callback.episode_lengths) >= 100:
        plot_episode_length(
            timesteps_arr, np.array(callback.episode_lengths), window=100,
            title=f"Baseline Episode Length Curve (seed={seed})",
            filename=os.path.join(output_dir, f"ppo_episode_length_seed{seed}.png"),
        )
    save_summary_table(config, metrics, entropy_loss, seed,
                       filename=os.path.join(output_dir, f"ppo_summary_seed{seed}.png"))

    np.savez(os.path.join(output_dir, f"ppo_data_seed{seed}.npz"),
             timesteps=timesteps_arr, rewards=rewards_arr,
             lengths=np.array(callback.episode_lengths))

    print(f"    Success: {metrics['success_rate']:.1%} | "
          f"Reward: {metrics['eval_mean']:.3f} +/- {metrics['eval_std']:.3f} | "
          f"Entropy: {entropy_loss:.4f}")

    return metrics, entropy_loss


# Main
def main():
    parser = argparse.ArgumentParser(description="PPO Baseline on FrozenLake-v1")
    parser.add_argument("--timesteps", type=int, default=100000)
    parser.add_argument("--hidden_size", type=int, default=64)
    parser.add_argument("--n_steps", type=int, default=2048)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--n_eval", type=int, default=1000)
    parser.add_argument("--ent_coef", type=float, default=0.0)
    parser.add_argument("--output_dir", type=str, default="results/ppo_baseline")
    args = parser.parse_args()

    is_slippery = not args.deterministic
    config = {
        "is_slippery": is_slippery,
        "hidden_size": args.hidden_size,
        "n_steps": args.n_steps,
        "lr": args.lr,
        "timesteps": args.timesteps,
        "n_eval": args.n_eval,
        "ent_coef": args.ent_coef,
    }

    mode = "stochastic" if is_slippery else "deterministic"
    print("=" * 60)
    print(f"PPO Baseline on FrozenLake-v1 ({mode})")
    print("=" * 60)
    print(f"  Map:           4x4 (standard)")
    print(f"  Slippery:      {is_slippery}")
    print(f"  Hidden size:   {args.hidden_size}")
    print(f"  n_steps:       {args.n_steps}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Timesteps:     {args.timesteps}")
    print(f"  Seeds:         {args.seeds}")
    print(f"  Eval episodes: {args.n_eval}")
    print(f"  ent_coef:      {args.ent_coef}")
    print("=" * 60)

    all_metrics = []
    for seed in args.seeds:
        seed_dir = os.path.join(args.output_dir, f"seed{seed}")
        metrics, entropy = run_single_seed(seed, config, seed_dir)
        all_metrics.append(metrics)

    if len(all_metrics) > 1:
        success_rates = [m["success_rate"] for m in all_metrics]
        eval_means = [m["eval_mean"] for m in all_metrics]

        print(f"\n{'=' * 60}")
        print(f"AGGREGATE ({len(args.seeds)} seeds)")
        print(f"{'=' * 60}")
        print(f"  Success Rate:      {np.mean(success_rates):.1%} +/- {np.std(success_rates):.1%}")
        print(f"  Eval Reward:       {np.mean(eval_means):.3f} +/- {np.std(eval_means):.3f}")
        print(f"{'=' * 60}")


if __name__ == "__main__":
    main()