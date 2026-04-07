from os import path
import matplotlib.pyplot as plt
import numpy as np


def plot_progress(total_rewards, outcomes, args):
    n = len(total_rewards)
    window = max(1, n // 10)
    episodes = np.arange(1, n + 1)

    # Cumulative success rate: running mean from episode 1..i
    binary = np.array([1 if r > 0 else 0 for r in total_rewards], dtype=float)
    cumulative_success = np.cumsum(binary) / episodes * 100
    rolling_success = np.convolve(binary, np.ones(window) / window, mode="valid") * 100

    n_success = outcomes.count("success")
    n_hole = outcomes.count("hole")
    n_timeout = outcomes.count("timeout")
    success_rate = n_success / n * 100

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(episodes, cumulative_success, color="tab:blue", linewidth=1.5, label="Cumulative success rate")
    ax.plot(
        episodes[window - 1 :],
        rolling_success,
        color="tab:orange",
        linewidth=2,
        label=f"{window}-episode rolling avg",
    )
    ax.axhline(
        success_rate, color="tab:green", linestyle="--", linewidth=1, label=f"Overall: {success_rate:.1f}%"
    )
    ax.set_ylim(0, 105)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Success rate (%)")
    config_label = (
        f"Selection: {args.selection} | Rollout: {args.rollout} | "
        f"Final: {args.final_action} | C={args.exploration_constant}"
    )
    ax.set_title(f"MCTS Success Rate\n{config_label}", fontsize=11)
    ax.legend(loc="upper left")

    stats = (
        f"Episodes: {n}\n"
        f"Success:  {n_success} ({success_rate:.1f}%)\n"
        f"Hole:     {n_hole} ({n_hole / n * 100:.1f}%)\n"
        f"Timeout:  {n_timeout} ({n_timeout / n * 100:.1f}%)"
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
    plt.close("all")


def plot_time_stats(episode_times, steps_per_episode, avg_search_times, args):
    n = len(episode_times)
    window = max(1, n // 10)
    episodes = np.arange(1, n + 1)
    search_times_ms = [t * 1000 for t in avg_search_times]

    def rolling(data):
        return np.convolve(data, np.ones(window) / window, mode="valid")

    config_label = (
        f"Selection: {args.selection} | Rollout: {args.rollout} | "
        f"Final: {args.final_action} | C={args.exploration_constant}"
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"MCTS Time & Step Statistics\n{config_label}", fontsize=11)

    # Episode time — line over episodes
    axes[0].plot(episodes, episode_times, alpha=0.3, color="tab:blue", label="Episode time (s)")
    axes[0].plot(episodes[window - 1 :], rolling(episode_times), color="tab:blue", label=f"{window}-ep avg")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Time (s)")
    axes[0].set_title("Time per Episode")
    axes[0].legend(loc="upper right")

    # Steps per episode — histogram (distribution matters more than sequence)
    axes[1].hist(steps_per_episode, bins=20, color="tab:orange", alpha=0.8, edgecolor="white")
    axes[1].axvline(
        np.mean(steps_per_episode),
        color="black",
        linestyle="--",
        linewidth=1,
        label=f"Mean: {np.mean(steps_per_episode):.1f}",
    )
    axes[1].set_xlabel("Steps")
    axes[1].set_ylabel("Episodes")
    axes[1].set_title("Steps per Episode (distribution)")
    axes[1].legend(fontsize=8)

    # Avg search time per step — line over episodes
    axes[2].plot(episodes, search_times_ms, alpha=0.3, color="tab:green", label="Search time (ms)")
    axes[2].plot(
        episodes[window - 1 :], rolling(search_times_ms), color="tab:green", label=f"{window}-ep avg"
    )
    axes[2].set_xlabel("Episode")
    axes[2].set_ylabel("Time (ms)")
    axes[2].set_title("Avg Search Time per Step")
    axes[2].legend(loc="upper right")

    stats = (
        f"Avg episode time: {np.mean(episode_times):.2f}s\n"
        f"Avg steps/episode: {np.mean(steps_per_episode):.1f}\n"
        f"Avg search time: {np.mean(search_times_ms):.1f}ms/step"
    )
    fig.text(
        0.99,
        0.01,
        stats,
        fontsize=8,
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()
    filename = path.join(
        "graphs",
        f"sel-{args.selection}_roll-{args.rollout}_final-{args.final_action}_"
        f"C-{args.exploration_constant}_episodes-{args.episodes}_grid-{args.grid}_slip-{args.slip}_time_stats.png",
    )
    plt.savefig(filename, dpi=500)
    plt.close("all")
