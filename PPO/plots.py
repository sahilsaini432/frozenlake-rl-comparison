import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


def plot_training_curve(timesteps, rewards, window, title, filename):
    fig, ax = plt.subplots(figsize=(10, 5))
    if len(rewards) >= window:
        moving_avg = np.convolve(rewards, np.ones(window) / window, mode="valid")
        ax.plot(
            timesteps[window - 1:],
            moving_avg,
            color = "darkblue",
            linewidth = 2,
            label = f"Success Rate (moving avg, {window} episodes)"
        )
    ax.set_xlabel("Timesteps", fontsize=12)
    ax.set_ylabel("Episode Reward", fontsize=12)
    ax.set_title(title, fontsize = 14)
    ax.legend(fontsize = 10)
    ax.grid(True, alpha = 0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()


def plot_entropy_loss(training_timesteps, entropy_losses, title, filename):
    fig, ax = plt.subplots(figsize = (10, 5))
    ax.plot(training_timesteps, entropy_losses, color="purple", linewidth=2)
    ax.set_xlabel("Timesteps", fontsize = 12)
    ax.set_ylabel("Entropy Loss", fontsize = 12)
    ax.set_title(title, fontsize =14)
    ax.grid(True, alpha = 0.3)
    ax.annotate("Lower (more negative) = less exploration", xy = (0.98, 0.98), xycoords = "axes fraction",
        ha = "right", va = "top", fontsize = 9, color = "gray")

    plt.tight_layout()
    plt.savefig(filename, dpi = 150, bbox_inches = "tight")
    plt.close()


def plot_approx_kl(training_timesteps, approx_kls, title, filename):
    fig, ax = plt.subplots(figsize = (10, 5))
    ax.plot(training_timesteps, approx_kls, color="red", linewidth=2)
    ax.set_xlabel("Timesteps", fontsize = 12)
    ax.set_ylabel("Approx KL Divergence", fontsize = 12)
    ax.set_title(title, fontsize = 14)
    ax.grid(True, alpha = 0.3)
    ax.annotate("Small + stable = healthy, exploding = policy collapse", xy = (0.98, 0.98),
        xycoords = "axes fraction", ha = "right", va = "top", fontsize = 9, color = "gray")
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()


def plot_episode_length(timesteps, lengths, window, title, filename):
    fig, ax = plt.subplots(figsize=(10, 5))
    if len(lengths) >= window:
        moving_avg = np.convolve(lengths, np.ones(window) / window, mode = "valid")
        ax.plot(
            timesteps[window - 1:],
            moving_avg,
            color = "green",
            linewidth = 2,
            label = f"Avg Episode Length ({window} episodes)")
    ax.set_xlabel("Timesteps", fontsize=12)
    ax.set_ylabel("Episode Length", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()

def save_summary_table(config, metrics, seed, pretty_name, train_reward_l100, episode_length_l100, final_kl, filename):
    table_data = [
        ["Environment", f"FrozenLake-v1 {config['map_size']}x{config['map_size']}"],
        ["Stochastic", f"{config['is_slippery']}"],
        ["Network", f"MLP [{config['hidden_size']}, {config['hidden_size']}]"],
        ["Total Timesteps", f"{config['timesteps']:,}"],
        ["Seed", f"{seed}"],
        ["Eval Success Rate", f"{metrics['success_rate']:.1%}"],
        ["Eval Reward Std", f"{metrics['eval_std']:.3f}"],
        ["Train Reward (last 100)", f"{train_reward_l100:.3f}"],
        ["Episode Length (last 100)", f"{episode_length_l100:.1f}"],
        ["Final KL", f"{final_kl:.5f}"]
    ]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.axis("off")

    table = ax.table(
        cellText = table_data,
        colLabels = ["Metric", "Value"],
        cellLoc = "left",
        colLoc = "left",
        loc = "center"
    )
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
    ax.set_title(f"{pretty_name} Summary (seed={seed})", fontsize=13, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()

def save_aggregate_summary_table(config, all_metrics, seeds, pretty_name, filename):
    success_rates = [metric["success_rate"] for metric in all_metrics]
    per_seed_rows = [
        [f"Seed {seed} Success Rate", f"{metric['success_rate']:.1%}"]
        for seed, metric in zip(seeds, all_metrics)
    ]

    table_data = [
        ["Environment", f"FrozenLake-v1 {config['map_size']}x{config['map_size']}"],
        ["Stochastic", f"{config['is_slippery']}"],
        ["Network", f"MLP [{config['hidden_size']}, {config['hidden_size']}]"],
        ["Learning Rate", f"{config['lr']}"],
        ["Entropy Coef", f"{config['ent_coef']}"],
        ["Clip Range", f"{config['clip_range']}"],
        ["Step Penalty", f"{config.get('step_penalty', 0.0)}"],
        ["Manhattan Scale", f"{config.get('manhattan_scale', 0.0)}"],
        ["Total Timesteps", f"{config['timesteps']:,}"],
        ["Seeds", f"{seeds}"],
        ["Agg Success Rate", f"{np.mean(success_rates):.1%} +/- {np.std(success_rates):.1%}"],
    ] + per_seed_rows

    fig, ax = plt.subplots(figsize=(9, 0.5 * (len(table_data) + 2) + 1.5))
    ax.axis("off")
    table = ax.table(
        cellText=table_data,
        colLabels=["Metric", "Value"],
        cellLoc="left",
        colLoc="left",
        loc="center"
    )
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

    ax.set_title(f"{pretty_name} Aggregate Summary", fontsize = 13, fontweight = "bold", pad=20)
    plt.tight_layout()
    plt.savefig(filename, dpi = 150, bbox_inches = "tight")
    plt.close()