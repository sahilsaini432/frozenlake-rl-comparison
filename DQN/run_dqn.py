"""
run_dqn.py - Train and evaluate SB3 DQN on FrozenLake-v1

Default experiment setup:
  - FrozenLake-v1, standard 4x4 map, is_slippery=True
  - One-hot encoded state input
  - MlpPolicy with [128, 128] hidden layers
  - Deterministic evaluation over 1000 episodes
  - Hard-coded 4-config sweep from the project progress report

Usage examples:
  python run_dqn.py --config all
  python run_dqn.py --config config2 --seeds 1 2 3
  python run_dqn.py --config config3 --timesteps 50000
  python run_dqn.py --config config2 --deterministic --seeds 1
"""

import argparse
import json
import os

import gymnasium as gym
from gymnasium import spaces
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv


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


class DQNTrainingLoggerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.timesteps_at_episode = []
        self.losses = []
        self.exploration_rates = []
        self.training_timesteps = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
                self.episode_lengths.append(info["episode"]["l"])
                self.timesteps_at_episode.append(self.num_timesteps)

        try:
            logger = self.model.logger.name_to_value
            loss = logger.get("train/loss", None)
            if loss is not None and not np.isnan(loss):
                self.losses.append(loss)
                self.training_timesteps.append(self.num_timesteps)
            exploration = logger.get("rollout/exploration_rate", None)
            if exploration is not None:
                self.exploration_rates.append((self.num_timesteps, exploration))
        except AttributeError:
            pass
        return True


def make_frozenlake_env(is_slippery=True):
    def _init():
        env = gym.make("FrozenLake-v1", map_name="4x4", is_slippery=is_slippery)
        env = OneHotWrapper(env)
        env = Monitor(env)
        return env
    return _init


def evaluate_agent(model, env_fn, n_episodes=1000):
    eval_env = env_fn()
    rewards = []
    for _ in range(n_episodes):
        obs, _ = eval_env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = eval_env.step(int(action))
            ep_reward += reward
            done = terminated or truncated
        rewards.append(ep_reward)
    eval_env.close()
    return rewards


def compute_metrics(eval_rewards):
    return {
        "eval_mean": float(np.mean(eval_rewards)),
        "eval_std": float(np.std(eval_rewards)),
        "success_rate": float(np.mean([1 if r > 0 else 0 for r in eval_rewards])),
    }


def plot_training_curve(timesteps, rewards, window, title, filename):
    fig, ax = plt.subplots(figsize=(10, 5))
    if len(rewards) >= window:
        moving_avg = np.convolve(rewards, np.ones(window) / window, mode="valid")
        ax.plot(
            timesteps[window - 1:],
            moving_avg,
            linewidth=2,
            label=f"Success Rate (moving avg, {window} episodes)",
        )
    elif len(rewards) > 0:
        ax.plot(timesteps, rewards, linewidth=1.5, label="Episode reward")
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Episode Reward")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()


def plot_loss(training_ts, losses, title, filename):
    if len(losses) == 0:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(training_ts, losses, linewidth=2)
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("TD Loss")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()


def plot_exploration_rate(exploration_points, title, filename):
    if len(exploration_points) == 0:
        return
    ts = [x[0] for x in exploration_points]
    eps = [x[1] for x in exploration_points]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(ts, eps, linewidth=2)
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Exploration Rate")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()


def plot_episode_length(timesteps, lengths, window, title, filename):
    if len(lengths) == 0:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    if len(lengths) >= window:
        moving_avg = np.convolve(lengths, np.ones(window) / window, mode="valid")
        ax.plot(
            timesteps[window - 1:],
            moving_avg,
            linewidth=2,
            label=f"Avg Episode Length ({window} episodes)",
        )
    else:
        ax.plot(timesteps, lengths, linewidth=1.5, label="Episode length")
    ax.set_xlabel("Timesteps")
    ax.set_ylabel("Episode Length")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()


def save_summary_table(config_name, config, metrics, final_loss, seed, filename):
    table_data = [
        ["Config", config_name],
        ["Environment", "FrozenLake-v1 4x4"],
        ["Slippery", f"{config['is_slippery']}"],
        ["Network", f"MLP {config['net_arch']}"],
        ["Total Timesteps", f"{config['timesteps']:,}"],
        ["Seed", f"{seed}"],
        ["Learning Rate", f"{config['learning_rate']}"],
        ["Gamma", f"{config['gamma']}"],
        ["Batch Size", f"{config['batch_size']}"],
        ["Buffer Size", f"{config['buffer_size']:,}"],
        ["Target Update", f"{config['target_update_interval']}"],
        ["Epsilon", f"{config['exploration_initial_eps']} -> {config['exploration_final_eps']}"],
        ["Eval Mean Reward", f"{metrics['eval_mean']:.3f} +/- {metrics['eval_std']:.3f}"],
        ["Eval Success Rate", f"{metrics['success_rate']:.1%}"],
        ["Final TD Loss", f"{final_loss:.4f}" if not np.isnan(final_loss) else "nan"],
    ]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.axis("off")
    table = ax.table(
        cellText=table_data,
        colLabels=["Metric", "Value"],
        cellLoc="left",
        colLoc="left",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.15, 1.25)
    for j in range(2):
        table[0, j].set_facecolor("#4472C4")
        table[0, j].set_text_props(color="white", fontweight="bold")
    for i in range(1, len(table_data) + 1):
        color = "#D9E2F3" if i % 2 == 0 else "white"
        for j in range(2):
            table[i, j].set_facecolor(color)
    ax.set_title(f"DQN Summary - {config_name} (seed={seed})", fontsize=13, fontweight="bold", pad=18)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()


DQN_CONFIGS = {
    "config1": {
        "label": "SB3-style default baseline",
        "learning_rate": 1e-4,
        "gamma": 0.99,
        "batch_size": 32,
        "buffer_size": 100_000,
        "target_update_interval": 10_000,
        "exploration_initial_eps": 1.0,
        "exploration_final_eps": 0.05,
        "exploration_fraction": 0.10,
        "learning_starts": 100,
        "train_freq": 4,
        "gradient_steps": 1,
        "timesteps": 50_000,
        "net_arch": [128, 128],
        "is_slippery": True,
    },
    "config2": {
        "label": "Reference-inspired baseline",
        "learning_rate": 1e-3,
        "gamma": 0.99,
        "batch_size": 64,
        "buffer_size": 10_000,
        "target_update_interval": 100,
        "exploration_initial_eps": 0.5,
        "exploration_final_eps": 0.01,
        "exploration_fraction": 0.10,
        "learning_starts": 100,
        "train_freq": 4,
        "gradient_steps": 1,
        "timesteps": 50_000,
        "net_arch": [128, 128],
        "is_slippery": True,
    },
    "config3": {
        "label": "Config2 with more early exploration",
        "learning_rate": 1e-3,
        "gamma": 0.99,
        "batch_size": 64,
        "buffer_size": 10_000,
        "target_update_interval": 100,
        "exploration_initial_eps": 0.6,
        "exploration_final_eps": 0.01,
        "exploration_fraction": 0.10,
        "learning_starts": 100,
        "train_freq": 4,
        "gradient_steps": 1,
        "timesteps": 50_000,
        "net_arch": [128, 128],
        "is_slippery": True,
    },
    "config4": {
        "label": "Config2 with lower learning rate",
        "learning_rate": 5e-4,
        "gamma": 0.99,
        "batch_size": 64,
        "buffer_size": 10_000,
        "target_update_interval": 100,
        "exploration_initial_eps": 0.5,
        "exploration_final_eps": 0.01,
        "exploration_fraction": 0.10,
        "learning_starts": 100,
        "train_freq": 4,
        "gradient_steps": 1,
        "timesteps": 50_000,
        "net_arch": [128, 128],
        "is_slippery": True,
    },
}


def run_single_seed(seed, config_name, config, output_dir, n_eval=1000):
    os.makedirs(output_dir, exist_ok=True)
    env_fn = make_frozenlake_env(is_slippery=config["is_slippery"])

    print(f"\n  {config_name} | Seed {seed} | {'stochastic' if config['is_slippery'] else 'deterministic'}")

    model = DQN(
        policy="MlpPolicy",
        env=DummyVecEnv([env_fn]),
        learning_rate=config["learning_rate"],
        buffer_size=config["buffer_size"],
        learning_starts=config["learning_starts"],
        batch_size=config["batch_size"],
        gamma=config["gamma"],
        train_freq=config["train_freq"],
        gradient_steps=config["gradient_steps"],
        target_update_interval=config["target_update_interval"],
        exploration_fraction=config["exploration_fraction"],
        exploration_initial_eps=config["exploration_initial_eps"],
        exploration_final_eps=config["exploration_final_eps"],
        seed=seed,
        verbose=0,
        tensorboard_log=os.path.join(output_dir, "dqn_tensorboard"),
        policy_kwargs=dict(net_arch=config["net_arch"]),
    )

    callback = DQNTrainingLoggerCallback()
    model.learn(total_timesteps=config["timesteps"], callback=callback, progress_bar=True)
    model.save(os.path.join(output_dir, f"dqn_{config_name}_seed{seed}"))

    timesteps_arr = np.array(callback.timesteps_at_episode)
    rewards_arr = np.array(callback.episode_rewards)
    lengths_arr = np.array(callback.episode_lengths)

    eval_rewards = evaluate_agent(model, env_fn, n_episodes=n_eval)
    metrics = compute_metrics(eval_rewards)
    final_loss = callback.losses[-1] if callback.losses else float("nan")

    plot_training_curve(
        timesteps_arr,
        rewards_arr,
        window=100,
        title=f"DQN Learning Curve - {config_name} (seed={seed})",
        filename=os.path.join(output_dir, f"dqn_training_curve_{config_name}_seed{seed}.png"),
    )
    plot_loss(
        callback.training_timesteps,
        callback.losses,
        title=f"DQN TD Loss - {config_name} (seed={seed})",
        filename=os.path.join(output_dir, f"dqn_loss_{config_name}_seed{seed}.png"),
    )
    plot_exploration_rate(
        callback.exploration_rates,
        title=f"DQN Exploration Schedule - {config_name} (seed={seed})",
        filename=os.path.join(output_dir, f"dqn_exploration_{config_name}_seed{seed}.png"),
    )
    plot_episode_length(
        timesteps_arr,
        lengths_arr,
        window=100,
        title=f"DQN Episode Length - {config_name} (seed={seed})",
        filename=os.path.join(output_dir, f"dqn_episode_length_{config_name}_seed{seed}.png"),
    )
    save_summary_table(
        config_name,
        config,
        metrics,
        final_loss,
        seed,
        filename=os.path.join(output_dir, f"dqn_summary_{config_name}_seed{seed}.png"),
    )

    np.savez(
        os.path.join(output_dir, f"dqn_data_{config_name}_seed{seed}.npz"),
        timesteps=timesteps_arr,
        rewards=rewards_arr,
        lengths=lengths_arr,
        eval_rewards=np.array(eval_rewards),
        losses=np.array(callback.losses),
    )

    metrics_path = os.path.join(output_dir, f"metrics_{config_name}_seed{seed}.json")
    with open(metrics_path, "w") as f:
        json.dump(
            {
                "config_name": config_name,
                "seed": seed,
                "config": config,
                "metrics": metrics,
                "final_loss": None if np.isnan(final_loss) else float(final_loss),
            },
            f,
            indent=2,
        )

    print(
        f"    Success: {metrics['success_rate']:.1%} | "
        f"Reward: {metrics['eval_mean']:.3f} +/- {metrics['eval_std']:.3f} | "
        f"Final loss: {final_loss:.4f}" if not np.isnan(final_loss)
        else f"    Success: {metrics['success_rate']:.1%} | Reward: {metrics['eval_mean']:.3f} +/- {metrics['eval_std']:.3f} | Final loss: nan"
    )

    return metrics


def aggregate_metrics(metrics_list, output_path, config_name):
    success_rates = [m["success_rate"] for m in metrics_list]
    eval_means = [m["eval_mean"] for m in metrics_list]
    eval_stds = [m["eval_std"] for m in metrics_list]

    summary = {
        "config_name": config_name,
        "num_seeds": len(metrics_list),
        "success_rate_mean": float(np.mean(success_rates)),
        "success_rate_std": float(np.std(success_rates)),
        "eval_mean_mean": float(np.mean(eval_means)),
        "eval_mean_std": float(np.std(eval_means)),
        "eval_std_mean": float(np.mean(eval_stds)),
    }
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def main():
    parser = argparse.ArgumentParser(description="SB3 DQN on FrozenLake-v1")
    parser.add_argument("--config", type=str, default="all", choices=["all", "config1", "config2", "config3", "config4"])
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--n_eval", type=int, default=1000)
    parser.add_argument("--output_dir", type=str, default="results/dqn_configs")
    args = parser.parse_args()

    config_names = ["config1", "config2", "config3", "config4"] if args.config == "all" else [args.config]

    print("=" * 70)
    print("DQN Config Sweep on FrozenLake-v1")
    print("=" * 70)
    print(f"Configs:        {config_names}")
    print(f"Seeds:          {args.seeds}")
    print(f"Eval episodes:  {args.n_eval}")
    print(f"Output dir:     {args.output_dir}")
    print("=" * 70)

    for config_name in config_names:
        config = dict(DQN_CONFIGS[config_name])
        if args.timesteps is not None:
            config["timesteps"] = args.timesteps
        if args.deterministic:
            config["is_slippery"] = False

        print(f"\n{'-' * 70}")
        print(f"Running {config_name}: {config['label']}")
        print(f"{'-' * 70}")
        print(json.dumps(config, indent=2))

        all_metrics = []
        for seed in args.seeds:
            seed_dir = os.path.join(args.output_dir, config_name, f"seed{seed}")
            metrics = run_single_seed(seed, config_name, config, seed_dir, n_eval=args.n_eval)
            all_metrics.append(metrics)

        agg_path = os.path.join(args.output_dir, config_name, "aggregate_metrics.json")
        os.makedirs(os.path.dirname(agg_path), exist_ok=True)
        summary = aggregate_metrics(all_metrics, agg_path, config_name)

        print(f"\n{'=' * 70}")
        print(f"AGGREGATE RESULTS - {config_name}")
        print(f"{'=' * 70}")
        print(f"Success Rate: {summary['success_rate_mean']:.1%} +/- {summary['success_rate_std']:.1%}")
        print(f"Eval Reward:  {summary['eval_mean_mean']:.3f} +/- {summary['eval_mean_std']:.3f}")
        print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
