import os
from os import path

# Intel OpenMP SHM issues in some environments; set before NumPy/Matplotlib pull MKL.
for _key, _val in (
    ("KMP_USE_SHM", "0"),
    ("KMP_DISABLE_SHM", "1"),
    ("OMP_NUM_THREADS", "1"),
):
    os.environ.setdefault(_key, _val)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# Plotting (MCTS-style + DQN curves)
# =========================
def plot_dqn_progress(total_rewards, outcomes, args, output_dir):
    n = len(total_rewards)
    if n == 0:
        return
    window = min(50, n)
    episodes = np.arange(1, n + 1)

    binary = np.array([1.0 if r > 0 else 0.0 for r in total_rewards])
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
        success_rate,
        color="tab:green",
        linestyle="--",
        linewidth=1,
        label=f"Overall: {success_rate:.1f}%",
    )
    ax.set_ylim(0, 105)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Success rate (%)")
    config_label = getattr(args, "config_label", str(args))
    ax.set_title(f"DQN Success Rate\n{config_label}", fontsize=11)
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
    slug = getattr(args, "file_slug", "dqn")
    filename = path.join(output_dir, f"{slug}_progress.png")
    plt.savefig(filename, dpi=500)
    plt.close("all")


def _pad_float_seq(seq, n, fill=0.0):
    seq = [float(x) for x in seq]
    if len(seq) < n:
        seq = seq + [fill] * (n - len(seq))
    return seq[:n]


def plot_dqn_time_stats(episode_times, steps_per_episode, episode_rewards, args, output_dir):
    n = len(episode_rewards)
    if n == 0:
        return
    episode_times = _pad_float_seq(episode_times, n, 0.0)
    steps_per_episode = _pad_float_seq(steps_per_episode, n, 0.0)
    window = min(50, n)
    episodes = np.arange(1, n + 1)

    def rolling(data):
        return np.convolve(np.asarray(data, dtype=float), np.ones(window) / window, mode="valid")

    config_label = getattr(args, "config_label", str(args))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"DQN Time & Step Statistics\n{config_label}", fontsize=11)

    axes[0].plot(episodes, episode_times, alpha=0.3, color="tab:blue", label="Episode time (s)")
    axes[0].plot(episodes[window - 1 :], rolling(episode_times), color="tab:blue", label=f"{window}-ep avg")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Time (s)")
    axes[0].set_title("Time per Episode")
    axes[0].legend(loc="upper right")

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

    axes[2].plot(episodes, episode_rewards, alpha=0.3, color="tab:green", label="Episode reward")
    axes[2].plot(
        episodes[window - 1 :], rolling(episode_rewards), color="tab:green", label=f"{window}-ep avg"
    )
    axes[2].set_xlabel("Episode")
    axes[2].set_ylabel("Reward")
    axes[2].set_title("Reward per Episode")
    axes[2].legend(loc="upper right")

    stats = (
        f"Avg episode time: {np.mean(episode_times):.2f}s\n"
        f"Avg steps/episode: {np.mean(steps_per_episode):.1f}\n"
        f"Avg reward: {np.mean(episode_rewards):.3f}"
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
    slug = getattr(args, "file_slug", "dqn")
    filename = path.join(output_dir, f"{slug}_time_stats.png")
    plt.savefig(filename, dpi=500)
    plt.close("all")


def make_plot_args_offline(total_timesteps, *, map_size=None, is_slippery=False, file_slug=None):
    """Titles when re-plotting from CSV (no SB3 model object)."""
    from types import SimpleNamespace

    map_part = f"{map_size}x{map_size} (maps.json)" if map_size is not None else "from episode_log.csv"
    if file_slug is None and map_size is not None:
        file_slug = f"dqn_{map_size}x{map_size}"
    if file_slug is None:
        file_slug = "dqn"
    config_label = f"Map: {map_part} | slip={is_slippery} | train_steps={total_timesteps}"
    return SimpleNamespace(config_label=config_label, file_slug=file_slug)


def make_plot_args(model, total_timesteps, *, map_size=None, is_slippery=False, file_slug=None):
    """Titles / filenames for figures (optional map_size from maps.json)."""
    from types import SimpleNamespace

    lr = float(getattr(model, "learning_rate", 0.0))
    buf = int(getattr(model, "buffer_size", 0))
    gamma = float(getattr(model, "gamma", 0.0))
    batch = int(getattr(model, "batch_size", 0))
    map_part = f"{map_size}x{map_size} (maps.json)" if map_size is not None else "custom desc"
    if file_slug is None and map_size is not None:
        file_slug = f"dqn_{map_size}x{map_size}"
    if file_slug is None:
        file_slug = "dqn"
    config_label = (
        f"Map: {map_part} | slip={is_slippery} | "
        f"lr={lr:g} | buffer={buf} | batch={batch} | γ={gamma:g} | train_steps={total_timesteps}"
    )
    return SimpleNamespace(config_label=config_label, file_slug=file_slug)


def _recompute_rolling_metrics(episode_rewards, episode_success, timesteps, success_window=50):
    avg_reward_steps = []
    avg_rewards = []
    success_rate_episodes = []
    success_rates = []
    n = len(episode_rewards)
    for i in range(n):
        lo = max(0, i + 1 - success_window)
        avg_reward_steps.append(timesteps[i])
        avg_rewards.append(float(np.mean(episode_rewards[lo : i + 1])))
        success_rate_episodes.append(i + 1)
        success_rates.append(float(np.mean(episode_success[lo : i + 1])))
    return avg_reward_steps, avg_rewards, success_rate_episodes, success_rates


def logger_namespace_from_run_dir(run_dir, success_window=50):
    """Build a logger-like object from episode_log.csv (+ optional td_error_log.csv)."""
    from types import SimpleNamespace

    ep_path = path.join(run_dir, "episode_log.csv")
    if not path.isfile(ep_path):
        raise FileNotFoundError(f"Missing {ep_path}")
    df = pd.read_csv(ep_path)
    episode_rewards = df["reward"].astype(float).tolist()
    episode_end_steps = df["timesteps"].astype(int).tolist()
    if "outcome" in df.columns:
        outcomes = df["outcome"].astype(str).tolist()
    else:
        outcomes = ["success" if r > 0 else "hole" for r in episode_rewards]
    if "success" in df.columns:
        episode_success = df["success"].astype(int).tolist()
    else:
        episode_success = [1 if r > 0 else 0 for r in episode_rewards]
    episode_times = (
        df["wall_time_s"].astype(float).tolist() if "wall_time_s" in df.columns else []
    )
    steps_per_episode = df["steps"].astype(float).tolist() if "steps" in df.columns else []

    ars, arr, sre, srr = _recompute_rolling_metrics(
        episode_rewards, episode_success, episode_end_steps, success_window
    )

    td_path = path.join(run_dir, "td_error_log.csv")
    if path.isfile(td_path):
        tdd = pd.read_csv(td_path)
        td_steps = tdd["timesteps"].tolist()
        td_errors = tdd["td_error"].tolist()
    else:
        td_steps, td_errors = [], []

    total_timesteps = int(episode_end_steps[-1]) if episode_end_steps else 0

    return SimpleNamespace(
        episode_rewards=episode_rewards,
        episode_success=episode_success,
        episode_end_steps=episode_end_steps,
        outcomes=outcomes,
        episode_times=episode_times,
        steps_per_episode=steps_per_episode,
        avg_reward_steps=ars,
        avg_rewards=arr,
        success_rate_episodes=sre,
        success_rates=srr,
        td_steps=td_steps,
        td_errors=td_errors,
        _total_timesteps_from_csv=total_timesteps,
    )


def save_training_visualization(
    logger,
    model,
    total_timesteps,
    output_dir,
    *,
    plot_args=None,
    map_size=None,
    is_slippery=False,
    write_csv=True,
    save_figures=True,
):
    """
    Write episode_log.csv, td_error_log.csv, and all PNGs under output_dir.
    logger: DQNTrainingLogger from base_DQN.train_agent (duck-typed attributes).
    Set write_csv=False when re-plotting from existing CSVs in output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)

    if write_csv:
        n = len(logger.episode_rewards)
        rewards_df = pd.DataFrame({
            "episode": np.arange(1, n + 1),
            "timesteps": logger.episode_end_steps,
            "reward": logger.episode_rewards,
            "success": logger.episode_success,
            "outcome": logger.outcomes,
        })
        _steps = getattr(logger, "steps_per_episode", None) or []
        _wall = getattr(logger, "episode_times", None) or []
        if len(_steps) == n:
            rewards_df["steps"] = _steps
        if len(_wall) == n:
            rewards_df["wall_time_s"] = _wall
        rewards_df.to_csv(os.path.join(output_dir, "episode_log.csv"), index=False)

        td_df = pd.DataFrame({
            "timesteps": logger.td_steps,
            "td_error": logger.td_errors,
        })
        td_df.to_csv(os.path.join(output_dir, "td_error_log.csv"), index=False)

    if save_figures:
        plt.figure(figsize=(7, 4))
        plt.plot(logger.avg_reward_steps, logger.avg_rewards)
        plt.xlabel("Training Timesteps")
        plt.ylabel("Average Reward (last 50 episodes)")
        plt.title("DQN Sample Efficiency")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "sample_efficiency.png"), dpi=300)
        plt.close()

        ts = total_timesteps if total_timesteps is not None else (
            logger.episode_end_steps[-1] if logger.episode_end_steps else 0
        )
        ts = ts or getattr(logger, "_total_timesteps_from_csv", 0)

        args = plot_args
        if args is None:
            if model is not None:
                args = make_plot_args(model, ts, map_size=map_size, is_slippery=is_slippery)
            elif map_size is not None:
                args = make_plot_args_offline(ts, map_size=map_size, is_slippery=is_slippery)
            else:
                from types import SimpleNamespace

                args = SimpleNamespace(
                    config_label=f"DQN | train_steps≈{ts}",
                    file_slug="dqn",
                )
        plot_dqn_progress(logger.episode_rewards, logger.outcomes, args, output_dir)
        plot_dqn_time_stats(
            logger.episode_times,
            logger.steps_per_episode,
            logger.episode_rewards,
            args,
            output_dir,
        )

        if logger.success_rate_episodes:
            plt.figure(figsize=(7, 4))
            plt.plot(logger.success_rate_episodes, logger.success_rates)
            plt.xlabel("Episodes")
            plt.ylabel("Success Rate (last 50 episodes)")
            plt.title("DQN Success Rate vs Episodes (50-ep window)")
            plt.tight_layout()
            plt.savefig(
                os.path.join(output_dir, "success_rate_window50.png"),
                dpi=300,
            )
            plt.close()

        plt.figure(figsize=(7, 4))
        plt.plot(logger.td_steps, logger.td_errors)
        plt.xlabel("Training Timesteps")
        plt.ylabel("Mean TD Error")
        plt.title("DQN Temporal Difference Error")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "td_error.png"), dpi=300)
        plt.close()

    if write_csv and save_figures:
        print(f"Plots and logs saved to: {output_dir}")
    elif write_csv:
        print(f"Logs saved to: {output_dir}")
    elif save_figures:
        print(f"Figures saved to: {output_dir}")


def replot_from_directory(run_dir, *, map_size=None, is_slippery=False):
    """Regenerate PNGs from existing episode_log.csv (and td_error_log.csv if present)."""
    run_dir = path.abspath(run_dir)
    logger = logger_namespace_from_run_dir(run_dir)
    ts = getattr(logger, "_total_timesteps_from_csv", 0) or None
    save_training_visualization(
        logger,
        None,
        ts,
        run_dir,
        map_size=map_size,
        is_slippery=is_slippery,
        write_csv=False,
    )


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="DQN figures from training logs (no training here).")
    parser.add_argument(
        "run_dir",
        nargs="?",
        default=None,
        help="Directory containing episode_log.csv — only regenerate PNGs.",
    )
    parser.add_argument("--map-size", type=int, default=None, help="Figure title, e.g. 4 / 8 / 16.")
    parser.add_argument("--slip", action="store_true", help="Title shows is_slippery=True.")
    args = parser.parse_args()

    if args.run_dir is None:
        print("Usage:")
        print("  Train + save figures:  python3 DQN/base_DQN.py")
        print("  Replot from CSV only:  python3 DQN/DQN_plots.py dqn_plots/4x4 [--map-size 4] [--slip]")
        sys.exit(0)

    replot_from_directory(args.run_dir, map_size=args.map_size, is_slippery=args.slip)
