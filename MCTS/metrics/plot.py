import os
from os import path
import sys
import matplotlib.pyplot as plt
import numpy as np


def plot_progress(total_rewards, args):
    n = len(total_rewards)
    # Set window size to 10% of total episode length, at least 1
    window = max(1, n // 10)
    episodes = np.arange(1, n + 1)
    rolling_avg = np.convolve(total_rewards, np.ones(window) / window, mode="valid")

    success_rate = sum(r > 0 for r in total_rewards) / n * 100
    avg_reward = np.mean(total_rewards)
    min_reward = np.min(total_rewards)
    max_reward = np.max(total_rewards)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(episodes, total_rewards, alpha=0.3, label="Episode reward")
    ax.plot(episodes[window - 1 :], rolling_avg, label=f"{window}-episode avg")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    config_label = (
        f"Selection: {args.selection} | Rollout: {args.rollout} | "
        f"Final: {args.final_action} | C={args.exploration_constant}"
    )
    ax.set_title(f"MCTS Performance\n{config_label}", fontsize=11)
    ax.legend(loc="upper left")

    stats = (
        f"Episodes: {n}\n"
        f"Success rate: {success_rate:.1f}%\n"
        f"Avg reward: {avg_reward:.4f}\n"
        f"Min / Max: {min_reward:.4f} / {max_reward:.4f}"
    )
    ax.text(
        0.98,
        0.05,
        stats,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()
    filename = path.join(
        "graphs",
        f"sel-{args.selection}_roll-{args.rollout}_final-{args.final_action}_"
        f"C-{args.exploration_constant}_episodes-{args.episodes}_grid-{args.grid}_slip-{args.slip}_progress.png",
    )
    plt.savefig(filename, dpi=500)
    plt.show()


def plot_time_stats(episode_times, steps_per_episode, avg_search_times, args):
    n = len(episode_times)
    window = max(1, n // 10)
    episodes = np.arange(1, n + 1)

    search_times_ms = [t * 1000 for t in avg_search_times]

    def rolling(data):
        return np.convolve(data, np.ones(window) / window, mode="valid")

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=False)
    config_label = (
        f"Selection: {args.selection} | Rollout: {args.rollout} | "
        f"Final: {args.final_action} | C={args.exploration_constant}"
    )
    fig.suptitle(f"MCTS Time Statistics\n{config_label}", fontsize=11)

    # Episode time
    axes[0].plot(episodes, episode_times, alpha=0.3, color="tab:blue", label="Episode time (s)")
    axes[0].plot(episodes[window - 1 :], rolling(episode_times), color="tab:blue", label=f"{window}-ep avg")
    axes[0].set_ylabel("Time (s)")
    axes[0].set_title("Time per Episode")
    axes[0].legend(loc="upper right")

    # Steps per episode
    axes[1].plot(episodes, steps_per_episode, alpha=0.3, color="tab:orange", label="Steps")
    axes[1].plot(
        episodes[window - 1 :], rolling(steps_per_episode), color="tab:orange", label=f"{window}-ep avg"
    )
    axes[1].set_ylabel("Steps")
    axes[1].set_title("Steps per Episode")
    axes[1].legend(loc="upper right")

    # Avg search time per step
    axes[2].plot(episodes, search_times_ms, alpha=0.3, color="tab:green", label="Avg search time (ms)")
    axes[2].plot(
        episodes[window - 1 :], rolling(search_times_ms), color="tab:green", label=f"{window}-ep avg"
    )
    axes[2].set_xlabel("Episode")
    axes[2].set_ylabel("Time (ms)")
    axes[2].set_title("Avg Search Time per Step")
    axes[2].legend(loc="upper right")

    avg_episode_time = np.mean(episode_times)
    avg_steps = np.mean(steps_per_episode)
    avg_search_time = np.mean(search_times_ms)
    stats = (
        f"Avg episode time: {avg_episode_time:.2f}s\n"
        f"Avg steps/episode: {avg_steps:.1f}\n"
        f"Avg search time: {avg_search_time:.1f}ms/step"
    )
    fig.text(
        0.98,
        0.02,
        stats,
        fontsize=9,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout(rect=[0, 0.06, 1, 1])
    filename = path.join(
        "graphs",
        f"sel-{args.selection}_roll-{args.rollout}_final-{args.final_action}_"
        f"C-{args.exploration_constant}_episodes-{args.episodes}_grid-{args.grid}_slip-{args.slip}_time_stats.png",
    )
    plt.savefig(filename, dpi=500)
    plt.show()
