"""
AI Generated Script to create comprehensive comparison plots from MCTS experiment results.

Reads results.jsonl and produces comparison plots across all recorded configurations.

Usage (from the MCTS directory):
    python -m metrics.plot_comparison
    python -m metrics.plot_comparison --metric avg_reward
    python -m metrics.plot_comparison --out my_folder
"""

import argparse
import json
import os
from os import path
from itertools import product

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

RESULTS_FILE = path.join(path.dirname(path.abspath(__file__)), "..", "results.jsonl")
DEFAULT_OUT = path.join(path.dirname(path.abspath(__file__)), "..", "graphs", "comparison")


def load_results(results_file=RESULTS_FILE):
    if not path.exists(results_file):
        raise FileNotFoundError(f"No results file found at {results_file}. Run run_mcts.py first.")
    rows = []
    with open(results_file) as f:
        for line in f:
            line = line.strip()
            if line:
                row = json.loads(line)
                if "avg_search_time_ms" in row:
                    row["avg_search_time_s"] = round(row.pop("avg_search_time_ms") / 1000, 6)
                rows.append(row)
    return rows


def _bar_chart(ax, categories, values, errors=None, title="", ylabel="", xlabel="", color="steelblue"):
    x = np.arange(len(categories))
    bars = ax.bar(x, values, yerr=errors, capsize=4, color=color, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=30, ha="right", fontsize=8)
    ax.set_title(title, fontsize=10)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{val:.1f}",
            ha="center",
            va="bottom",
            fontsize=7,
        )
    mean_val = np.nanmean(values)
    ax.axhline(mean_val, color="black", linestyle=":", linewidth=1.2, label=f"mean={mean_val:.1f}")
    ax.legend(fontsize=7, loc="upper right", framealpha=0.6)
    return bars


def _grouped_bar(ax, group_labels, series_labels, matrix, title="", ylabel="", xlabel=""):
    """matrix shape: (n_groups, n_series)"""
    n_groups, n_series = matrix.shape
    x = np.arange(n_groups)
    width = 0.8 / n_series
    colors = plt.cm.tab10(np.linspace(0, 0.9, n_series))
    for i, (label, color) in enumerate(zip(series_labels, colors)):
        offset = (i - n_series / 2 + 0.5) * width
        bars = ax.bar(x + offset, matrix[:, i], width, label=label, color=color, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(group_labels, rotation=20, ha="right", fontsize=8)
    ax.set_title(title, fontsize=10)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    group_means = np.where(np.isnan(matrix), 0, matrix).mean(axis=1)
    ax.plot(
        x,
        group_means,
        color="black",
        linestyle=":",
        linewidth=1.5,
        marker="o",
        markersize=4,
        label="group mean",
    )
    ax.legend(fontsize=7, loc="upper right", framealpha=0.6)


# ── Section 1: Selection × Rollout ──────────────────────────────────────────


def plot_selection_vs_rollout(rows, metric, out_dir):
    selections = sorted({r["selection"] for r in rows})
    rollouts = sorted({r["rollout"] for r in rows})

    matrix = np.full((len(selections), len(rollouts)), np.nan)
    for r in rows:
        i = selections.index(r["selection"])
        j = rollouts.index(r["rollout"])
        matrix[i, j] = r[metric]

    fig, ax = plt.subplots(figsize=(max(8, len(rollouts) * 1.5), 5))
    _grouped_bar(
        ax,
        selections,
        rollouts,
        matrix,
        title=f"Selection × Rollout — {metric}",
        ylabel=metric,
        xlabel="Selection strategy",
    )
    plt.tight_layout()
    _save(fig, out_dir, "1_selection_vs_rollout.png")


# ── Section 2: Final Action × Backprop ──────────────────────────────────────


def plot_final_action_vs_backprop(rows, metric, out_dir):
    final_actions = sorted({r["final_action"] for r in rows})
    backprops = sorted({r["backprop"] for r in rows})

    matrix = np.full((len(final_actions), len(backprops)), np.nan)
    for r in rows:
        i = final_actions.index(r["final_action"])
        j = backprops.index(r["backprop"])
        matrix[i, j] = r[metric]

    fig, ax = plt.subplots(figsize=(max(6, len(backprops) * 2), 5))
    _grouped_bar(
        ax,
        final_actions,
        backprops,
        matrix,
        title=f"Final Action × Backprop — {metric}",
        ylabel=metric,
        xlabel="Final action strategy",
    )
    plt.tight_layout()
    _save(fig, out_dir, "2_final_action_vs_backprop.png")


# ── Section 2b: Final Action × Selection ─────────────────────────────────────


def plot_final_action_vs_selection(rows, metric, out_dir):
    final_actions = sorted({r["final_action"] for r in rows})
    selections = sorted({r["selection"] for r in rows})

    matrix = np.full((len(final_actions), len(selections)), np.nan)
    counts = np.zeros((len(final_actions), len(selections)), dtype=int)
    for r in rows:
        i = final_actions.index(r["final_action"])
        j = selections.index(r["selection"])
        if np.isnan(matrix[i, j]):
            matrix[i, j] = r[metric]
        else:
            matrix[i, j] = (matrix[i, j] * counts[i, j] + r[metric]) / (counts[i, j] + 1)
        counts[i, j] += 1

    fig, ax = plt.subplots(figsize=(max(8, len(selections) * 1.5), 5))
    _grouped_bar(
        ax,
        final_actions,
        selections,
        matrix,
        title=f"Final Action × Selection — {metric}",
        ylabel=metric,
        xlabel="Final action strategy",
    )
    plt.tight_layout()
    _save(fig, out_dir, "2b_final_action_vs_selection.png")


# ── Section 3 & 4: Grid × Slip heatmap ──────────────────────────────────────


def plot_grid_slip_heatmap(rows, metric, out_dir):
    grids = sorted({r["grid"] for r in rows})
    slips = sorted({r["slip"] for r in rows})

    matrix = np.full((len(grids), len(slips)), np.nan)
    for r in rows:
        i = grids.index(r["grid"])
        j = slips.index(r["slip"])
        # average across configs if multiple entries share same grid/slip
        if np.isnan(matrix[i, j]):
            matrix[i, j] = r[metric]
        else:
            matrix[i, j] = (matrix[i, j] + r[metric]) / 2

    fig, ax = plt.subplots(figsize=(max(4, len(slips) * 2), max(4, len(grids) * 0.8)))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=np.nanmin(matrix), vmax=np.nanmax(matrix))
    ax.set_xticks(range(len(slips)))
    ax.set_xticklabels([f"slip={s}" for s in slips])
    ax.set_yticks(range(len(grids)))
    ax.set_yticklabels([f"{g}×{g}" for g in grids])
    ax.set_title(f"Grid Size × Slip — {metric}", fontsize=10)
    for i, j in product(range(len(grids)), range(len(slips))):
        if not np.isnan(matrix[i, j]):
            ax.text(j, i, f"{matrix[i, j]:.1f}", ha="center", va="center", fontsize=9, color="black")
    plt.colorbar(im, ax=ax, label=metric)
    plt.tight_layout()
    _save(fig, out_dir, "3_grid_slip_heatmap.png")


# ── Section 4a: Expansion × Selection ───────────────────────────────────────


def plot_expansion_vs_selection(rows, metric, out_dir):
    expansions = sorted({r["expansion"] for r in rows})
    selections = sorted({r["selection"] for r in rows})

    matrix = np.full((len(expansions), len(selections)), np.nan)
    counts = np.zeros((len(expansions), len(selections)), dtype=int)
    for r in rows:
        i = expansions.index(r["expansion"])
        j = selections.index(r["selection"])
        if np.isnan(matrix[i, j]):
            matrix[i, j] = r[metric]
        else:
            matrix[i, j] = (matrix[i, j] * counts[i, j] + r[metric]) / (counts[i, j] + 1)
        counts[i, j] += 1

    fig, ax = plt.subplots(figsize=(max(8, len(selections) * 1.5), 5))
    _grouped_bar(
        ax,
        expansions,
        selections,
        matrix,
        title=f"Expansion × Selection — {metric}",
        ylabel=metric,
        xlabel="Expansion strategy",
    )
    plt.tight_layout()
    _save(fig, out_dir, "4a_expansion_vs_selection.png")


# ── Section 4b: Rollout x Expansion ─────────────────────────────────────────


def plot_expansion_vs_rollout(rows, metric, out_dir):
    rollouts = sorted({r["rollout"] for r in rows})
    expansions = sorted({r["expansion"] for r in rows})

    matrix = np.full((len(rollouts), len(expansions)), np.nan)
    counts = np.zeros((len(rollouts), len(expansions)), dtype=int)
    for r in rows:
        i = rollouts.index(r["rollout"])
        j = expansions.index(r["expansion"])
        if np.isnan(matrix[i, j]):
            matrix[i, j] = r[metric]
        else:
            matrix[i, j] = (matrix[i, j] * counts[i, j] + r[metric]) / (counts[i, j] + 1)
        counts[i, j] += 1

    fig, ax = plt.subplots(figsize=(max(8, len(expansions) * 1.5), 5))
    _grouped_bar(
        ax,
        rollouts,
        expansions,
        matrix,
        title=f"Rollout x Expansion — {metric}",
        ylabel=metric,
        xlabel="Rollout strategy",
    )
    plt.tight_layout()
    _save(fig, out_dir, "4b_expansion_vs_rollout.png")


# ── Combined 2×2: Sections 1, 2b, 4a, 4b ────────────────────────────────────


def _build_matrix_averaged(rows, row_keys, col_keys, row_field, col_field, metric):
    """Build a (n_rows × n_cols) matrix, averaging duplicate (row, col) entries."""
    matrix = np.full((len(row_keys), len(col_keys)), np.nan)
    counts = np.zeros((len(row_keys), len(col_keys)), dtype=int)
    for r in rows:
        i = row_keys.index(r[row_field])
        j = col_keys.index(r[col_field])
        if np.isnan(matrix[i, j]):
            matrix[i, j] = r[metric]
        else:
            matrix[i, j] = (matrix[i, j] * counts[i, j] + r[metric]) / (counts[i, j] + 1)
        counts[i, j] += 1
    return matrix


def plot_combined_overview(rows, metric, out_dir):
    selections = sorted({r["selection"] for r in rows})
    rollouts = sorted({r["rollout"] for r in rows})
    final_actions = sorted({r["final_action"] for r in rows})
    expansions = sorted({r["expansion"] for r in rows})

    panels = [
        # (row_groups, series, row_field, col_field, title, xlabel)
        (
            selections,
            rollouts,
            "selection",
            "rollout",
            f"Selection × Rollout — {metric}",
            "Selection strategy",
        ),
        (
            final_actions,
            selections,
            "final_action",
            "selection",
            f"Final Action × Selection — {metric}",
            "Final action strategy",
        ),
        (
            expansions,
            selections,
            "expansion",
            "selection",
            f"Expansion × Selection — {metric}",
            "Expansion strategy",
        ),
        (rollouts, expansions, "rollout", "expansion", f"Rollout × Expansion — {metric}", "Rollout strategy"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    for ax, (row_keys, col_keys, row_field, col_field, title, xlabel) in zip(axes.flat, panels):
        matrix = _build_matrix_averaged(rows, row_keys, col_keys, row_field, col_field, metric)
        _grouped_bar(ax, row_keys, col_keys, matrix, title=title, ylabel=metric, xlabel=xlabel)

    fig.suptitle(f"{metric}", fontsize=12, fontweight="bold")
    plt.tight_layout()
    _save(fig, out_dir, f"{metric}_overview.png")


# ── Section 5: Exploration constant C ───────────────────────────────────────


def plot_exploration_constant(rows, metric, out_dir):
    c_values = sorted({r["C"] for r in rows})
    if len(c_values) < 2:
        print("Not enough distinct C values to plot exploration constant chart — skipping.")
        return

    # Average metric across all other configs for each C value
    c_metric = {}
    c_counts = {}
    for r in rows:
        c = r["C"]
        c_metric[c] = c_metric.get(c, 0) + r[metric]
        c_counts[c] = c_counts.get(c, 0) + 1
    c_avg = [c_metric[c] / c_counts[c] for c in c_values]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(c_values, c_avg, marker="o", color="steelblue", linewidth=2)
    ax.set_xlabel("Exploration constant C")
    ax.set_ylabel(f"Avg {metric} (across all configs)")
    ax.set_title(f"Effect of Exploration Constant C on {metric}", fontsize=10)
    ax.xaxis.set_major_locator(mticker.MultipleLocator(0.5))
    plt.tight_layout()
    _save(fig, out_dir, "5_exploration_constant.png")


# ── Section 6: Top-N configs ─────────────────────────────────────────────────


def plot_top_configs(rows, metric, out_dir, top_n=10):
    sorted_rows = sorted(rows, key=lambda r: r[metric], reverse=True)[:top_n]
    labels = [
        f"{r['selection']}\n{r['rollout']}\n{r['final_action']}\nC={r['C']} g={r['grid']}"
        for r in sorted_rows
    ]
    values = [r[metric] for r in sorted_rows]
    runs = [r.get("runs", 1) for r in sorted_rows]

    fig, ax = plt.subplots(figsize=(max(10, top_n * 1.2), 5))
    colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(values)))
    bars = ax.bar(range(len(values)), values, color=colors, alpha=0.9)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=7, rotation=0, ha="center")
    ax.set_ylabel(metric)
    ax.set_title(f"Top {top_n} Configurations by {metric}", fontsize=10)
    for bar, val, n in zip(bars, values, runs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{val:.1f}\n(n={n})",
            ha="center",
            va="bottom",
            fontsize=7,
        )
    plt.tight_layout()
    _save(fig, out_dir, "6_top_configs.png")


# ── Section 7: All configs summary bar ──────────────────────────────────────


def plot_all_configs_summary(rows, metric, out_dir):
    sorted_rows = sorted(rows, key=lambda r: r[metric], reverse=True)
    labels = [
        f"{r['selection']}|{r['rollout']}|{r['final_action']}|C={r['C']}|g={r['grid']}|slip={r['slip']}"
        for r in sorted_rows
    ]
    values = [r[metric] for r in sorted_rows]

    fig, ax = plt.subplots(figsize=(max(12, len(labels) * 0.9), 5))
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(values)))
    ax.barh(range(len(labels)), values, color=colors, alpha=0.85)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel(metric)
    ax.set_title(f"All Configurations Ranked by {metric}", fontsize=10)
    plt.tight_layout()
    _save(fig, out_dir, "7_all_configs_summary.png")


# ── Section 8: Strategy dominance leaderboard ────────────────────────────────

ALL_METRICS = ["success_rate", "avg_reward", "avg_episode_time", "avg_steps", "avg_search_time_s"]
LOWER_IS_BETTER = {"avg_episode_time", "avg_steps", "avg_search_time_s"}
CATEGORY_COLORS = {
    "selection": "#4C72B0",
    "rollout": "#55A868",
    "expansion": "#C44E52",
    "final_action": "#8172B2",
}


def plot_strategy_dominance(rows, out_dir):
    rows = [r for r in rows if not r["slip"]]
    categories = ["selection", "rollout", "expansion", "final_action"]
    metrics = [m for m in ALL_METRICS if m != "avg_reward"]

    # Macro-average: for each (cat, strategy, metric), average per combination of
    # the OTHER two categories first, then average those group means.
    # This ensures a strategy that only works with one partner combination
    # is penalised — every partner combination counts equally.
    other = {
        "selection": ["rollout", "expansion"],
        "rollout": ["selection", "expansion"],
        "expansion": ["selection", "rollout"],
        "final_action": ["selection", "rollout"],
    }

    def macro_avg(cat, strategy, metric):
        group_sums, group_counts = {}, {}
        for r in rows:
            if r[cat] != strategy:
                continue
            combo = tuple(r[c] for c in other[cat])
            group_sums[combo] = group_sums.get(combo, 0) + r[metric]
            group_counts[combo] = group_counts.get(combo, 0) + 1
        if not group_sums:
            return 0.0
        per_group = [group_sums[c] / group_counts[c] for c in group_sums]
        return sum(per_group) / len(per_group)

    all_strategies = {cat: sorted({r[cat] for r in rows}) for cat in categories}
    avgs = {
        (cat, s): {m: macro_avg(cat, s, m) for m in metrics}
        for cat in categories
        for s in all_strategies[cat]
    }

    # Count dominance: +1 for each metric where this strategy has the best avg
    wins = {k: 0 for k in avgs}
    dominated_by = {}  # (cat, metric) -> winning key
    for cat in categories:
        cat_keys = [k for k in avgs if k[0] == cat]
        for metric in metrics:
            reverse = metric not in LOWER_IS_BETTER
            best = max(cat_keys, key=lambda k: avgs[k][metric] * (1 if reverse else -1))
            wins[best] += 1
            dominated_by[(cat, metric)] = best

    # Plot: one subplot per category, bars = strategies, height = wins
    fig, axes = plt.subplots(1, len(categories), figsize=(len(categories) * 4, 5), sharey=True)

    for ax, cat in zip(axes, categories):
        cat_keys = sorted([k for k in avgs if k[0] == cat], key=lambda k: wins[k], reverse=True)
        names = [k[1] for k in cat_keys]
        win_counts = [wins[k] for k in cat_keys]
        x = np.arange(len(names))
        color = CATEGORY_COLORS[cat]

        ax.bar(x, win_counts, color=color, alpha=0.8, width=0.5)

        # Annotate each bar with which metrics it won
        for xi, key in enumerate(cat_keys):
            won = [m for (c, m), winner in dominated_by.items() if c == cat and winner == key]
            if won:
                label = "\n".join(w.replace("avg_", "").replace("_", " ") for w in won)
                ax.text(xi, win_counts[xi] + 0.05, label, ha="center", va="bottom", fontsize=6, color="black")

        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
        ax.set_title(cat, fontsize=10, color=color, fontweight="bold")
        ax.set_ylim(0, len(metrics) + 1.5)
        ax.yaxis.set_major_locator(mticker.MultipleLocator(1))
        ax.grid(axis="y", linestyle=":", alpha=0.4)

    axes[0].set_ylabel("Metrics dominated", fontsize=9)
    fig.suptitle("Wins per metric", fontsize=9)
    plt.tight_layout()
    _save(fig, out_dir, "8_dominance.png")


# ── Shared normalisation helper ──────────────────────────────────────────────

PLOT_METRICS = ["success_rate", "avg_reward", "avg_steps", "avg_episode_time", "avg_search_time_s"]
PLOT_METRIC_LABELS = [
    "Success Rate",
    "Avg Reward",
    "Avg Steps",
    "Avg Episode\nTime (s)",
    "Avg Search\nTime (s)",
]


def _normalise(rows, metrics=PLOT_METRICS):
    """Return (raw, norm, composite) arrays, all shape (N, M).

    norm is min-max scaled to [0,1] where 1 = best for every metric.
    composite is the row-wise mean of norm.
    """
    raw = np.array([[r[m] for m in metrics] for r in rows], dtype=float)
    norm = np.zeros_like(raw)
    for j, m in enumerate(metrics):
        col = raw[:, j]
        lo, hi = np.nanmin(col), np.nanmax(col)
        norm[:, j] = 1.0 if hi == lo else (col - lo) / (hi - lo)
        if m in LOWER_IS_BETTER:
            norm[:, j] = 1.0 - norm[:, j]
    composite = norm.mean(axis=1)
    return raw, norm, composite


# ── Section 10: Parallel Coordinates ────────────────────────────────────────


def plot_parallel_coordinates(rows, out_dir):
    """One line per configuration across all normalised metric axes.

    Lines are coloured by selection strategy so the reader can immediately
    see which strategy family dominates each metric.
    """
    metrics = PLOT_METRICS
    metric_labels = PLOT_METRIC_LABELS
    n_axes = len(metrics)

    raw, norm, composite = _normalise(rows, metrics)

    # Colour by selection strategy
    selections = sorted({r["selection"] for r in rows})
    cmap = plt.cm.tab10
    sel_color = {s: cmap(i / max(len(selections) - 1, 1)) for i, s in enumerate(selections)}

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.set_xlim(-0.05, n_axes - 0.95)
    ax.set_ylim(-0.08, 1.08)

    # Draw faint background grid lines for each axis
    for j in range(n_axes):
        ax.axvline(j, color="grey", linewidth=0.8, alpha=0.4, zorder=1)

    # Sort so the best-composite configs are drawn on top
    order = np.argsort(composite)
    for idx in order:
        r = rows[idx]
        color = sel_color[r["selection"]]
        # alpha: dim weak configs, highlight strong ones
        alpha = 0.15 + 0.65 * composite[idx]
        lw = 0.6 + 1.2 * composite[idx]
        y_vals = norm[idx]
        ax.plot(range(n_axes), y_vals, color=color, alpha=alpha, linewidth=lw, zorder=2)

    # Axis labels with raw-value tick annotations
    for j, (m, label) in enumerate(zip(metrics, metric_labels)):
        col_raw = raw[:, j]
        lo, hi = np.nanmin(col_raw), np.nanmax(col_raw)
        ax.text(j, 1.05, label, ha="center", va="bottom", fontsize=8, fontweight="bold")
        # top tick = best raw value, bottom = worst
        best_raw = lo if m in LOWER_IS_BETTER else hi
        worst_raw = hi if m in LOWER_IS_BETTER else lo
        fmt = ".1f" if m in ("success_rate", "avg_reward", "avg_steps") else ".3f"
        ax.text(j, 1.01, f"{best_raw:{fmt}}", ha="center", va="bottom", fontsize=6.5, color="green")
        ax.text(j, -0.06, f"{worst_raw:{fmt}}", ha="center", va="top", fontsize=6.5, color="red")

    # Legend for selection strategies
    handles = [plt.Line2D([0], [0], color=sel_color[s], linewidth=2, label=s) for s in selections]
    ax.legend(
        handles=handles, title="Selection", fontsize=7, title_fontsize=8, loc="lower right", framealpha=0.8
    )

    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["worst", "", "mid", "", "best"], fontsize=7)
    ax.set_xticks([])
    ax.set_title(
        "Parallel Coordinates — All Configurations × All Metrics\n"
        "(normalised: 1 = best, coloured by selection strategy)",
        fontsize=10,
    )
    plt.tight_layout()
    _save(fig, out_dir, "10_parallel_coordinates.png")


# ── Section 11: Top-K heatmap ────────────────────────────────────────────────


def plot_top_k_heatmap(rows, out_dir, top_k=15):
    """Clean heatmap restricted to the top-K configs by composite score.

    Readable at report size and still shows every metric for each config.
    """
    metrics = PLOT_METRICS
    metric_labels = [
        "Success\nRate",
        "Avg\nReward",
        "Avg\nSteps",
        "Avg Episode\nTime (s)",
        "Avg Search\nTime (s)",
    ]

    raw, norm, composite = _normalise(rows, metrics)

    # Keep only top_k
    order = np.argsort(composite)[::-1][:top_k]
    raw_k = raw[order]
    norm_k = norm[order]
    composite_k = composite[order]

    def short_label(r):
        return (
            f"{r['selection']} · {r['rollout']} · {r['final_action']}\n"
            f"{r['expansion']} · C={r['C']} · g={r['grid']} · slip={r['slip']}"
        )

    labels_k = [short_label(rows[i]) for i in order]
    n_configs, n_metrics = norm_k.shape

    fig, (ax_heat, ax_score) = plt.subplots(
        1,
        2,
        figsize=(n_metrics * 2.0 + 3.5, max(5, n_configs * 0.6)),
        gridspec_kw={"width_ratios": [n_metrics, 1.2]},
    )

    # Heatmap
    im = ax_heat.imshow(norm_k, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    ax_heat.set_xticks(range(n_metrics))
    ax_heat.set_xticklabels(metric_labels, fontsize=9)
    ax_heat.set_yticks(range(n_configs))
    ax_heat.set_yticklabels(labels_k, fontsize=8)
    ax_heat.set_title(
        f"Top {top_k} Configurations — All Metrics\n(green = best, red = worst within each metric)",
        fontsize=10,
    )
    for i in range(n_configs):
        for j in range(n_metrics):
            val = raw_k[i, j]
            text_color = "black" if 0.2 < norm_k[i, j] < 0.85 else "white"
            fmt = ".1f" if metrics[j] in ("success_rate", "avg_reward", "avg_steps") else ".3f"
            ax_heat.text(j, i, f"{val:{fmt}}", ha="center", va="center", fontsize=8, color=text_color)
    plt.colorbar(im, ax=ax_heat, label="Normalised score (1 = best)", shrink=0.7)

    # Composite score sidebar
    bar_colors = plt.cm.RdYlGn(composite_k)
    ax_score.barh(range(n_configs), composite_k, color=bar_colors, alpha=0.9)
    ax_score.set_yticks(range(n_configs))
    ax_score.set_yticklabels([], fontsize=8)
    ax_score.invert_yaxis()
    ax_score.set_xlim(0, 1.15)
    ax_score.set_xlabel("Composite\nscore", fontsize=8)
    ax_score.set_title("Overall", fontsize=9)
    ax_score.xaxis.set_major_locator(mticker.MultipleLocator(0.5))
    for i, v in enumerate(composite_k):
        ax_score.text(v + 0.02, i, f"{v:.2f}", va="center", fontsize=8)

    plt.tight_layout()
    _save(fig, out_dir, "11_top_k_heatmap.png")


def _save(fig, out_dir, filename):
    os.makedirs(out_dir, exist_ok=True)
    fpath = path.join(out_dir, filename)
    fig.savefig(fpath, dpi=500, bbox_inches="tight")
    print(f"  Saved: {fpath}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot comparison charts from results.jsonl")
    parser.add_argument(
        "--metric",
        default="success_rate",
        choices=["success_rate", "avg_reward", "avg_episode_time", "avg_steps", "avg_search_time_s"],
        help="Primary metric to compare (default: success_rate)",
    )
    parser.add_argument("--out", default=DEFAULT_OUT, help="Output directory for plots")
    parser.add_argument("--top", type=int, default=10, help="Number of top configs to show (default: 10)")
    args = parser.parse_args()

    print(f"Loading results from {RESULTS_FILE} ...")
    rows = load_results()
    print(f"  {len(rows)} unique configuration(s) found.")

    print(f"\nGenerating plots (metric: {args.metric}) → {args.out}")
    # plot_selection_vs_rollout(rows, args.metric, args.out)
    # plot_final_action_vs_backprop(rows, args.metric, args.out)
    # plot_final_action_vs_selection(rows, args.metric, args.out)
    # plot_grid_slip_heatmap(rows, args.metric, args.out)
    # plot_expansion_vs_selection(rows, args.metric, args.out)
    # plot_expansion_vs_rollout(rows, args.metric, args.out)
    # plot_exploration_constant(rows, args.metric, args.out)
    # plot_top_configs(rows, args.metric, args.out, top_n=args.top)
    # plot_all_configs_summary(rows, args.metric, args.out)
    # plot_strategy_dominance(rows, args.out)

    choices = ["success_rate", "avg_reward", "avg_episode_time", "avg_steps", "avg_search_time_s"]
    for choice in choices:
        plot_combined_overview(rows, choice, args.out)
    # plot_parallel_coordinates(rows, args.out)
    # plot_top_k_heatmap(rows, args.out, top_k=args.top)
    print("\nDone.")


if __name__ == "__main__":
    main()
