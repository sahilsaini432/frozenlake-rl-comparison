import csv
import json
import os
import statistics
from datetime import datetime
from pathlib import Path

# Headless plotting after sweep
if "MPLBACKEND" not in os.environ:
    os.environ["MPLBACKEND"] = "Agg"
os.environ.setdefault("KMP_USE_SHM", "0")
os.environ.setdefault("KMP_DISABLE_SHM", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import matplotlib.pyplot as plt

from base_DQN import (
    load_map,
    max_episode_steps_for_map_size,
    test_agent,
    train_agent,
)

try:
    from DQN.DQN_plots import replot_from_directory, save_training_visualization
except ImportError:
    from DQN_plots import replot_from_directory, save_training_visualization

# Default when neither `--one-hot` nor `--no-one-hot` is passed on the CLI.
# cfg5 one-hot (3 repeats):  python DQN/run_dqn_sweep.py --one-hot --run-tag my_onehot
USE_ONE_HOT = False
USE_REWARD_SHAPING = False
# How many full train+test runs to execute in one invocation (separate folder + CSV each).
N_REPEATS = 3


def sweep_configs_all():
    """Sweep configs. Uncomment entries to include them in a run."""
    return [
        {
            "name": "cfg1_default_sb3",
            "dqn_kwargs": {
                "learning_rate": 1e-4,
                "gamma": 0.99,
                "batch_size": 32,
                "buffer_size": 100000,
                "target_update_interval": 10000,
                "exploration_initial_eps": 1.0,
                "exploration_final_eps": 0.05,
            },
        },
        {
            "name": "cfg2_reference",
            "dqn_kwargs": {
                "learning_rate": 1e-3,
                "gamma": 0.99,
                "batch_size": 64,
                "buffer_size": 10000,
                "target_update_interval": 100,
                "exploration_initial_eps": 0.5,
                "exploration_final_eps": 0.01,
            },
        },
        {
            "name": "cfg3_best_so_far",
            "dqn_kwargs": {
                "learning_rate": 1e-3,
                "gamma": 0.99,
                "batch_size": 64,
                "buffer_size": 10000,
                "target_update_interval": 100,
                "exploration_initial_eps": 0.6,
                "exploration_final_eps": 0.01,
            },
        },
        {
            "name": "cfg4_lr_5e4",
            "dqn_kwargs": {
                "learning_rate": 5e-4,
                "gamma": 0.99,
                "batch_size": 64,
                "buffer_size": 10000,
                "target_update_interval": 100,
                "exploration_initial_eps": 0.5,
                "exploration_final_eps": 0.01,
            },
        },
        {
            "name": "cfg5_more_explore",
            "dqn_kwargs": {
                "learning_rate": 1e-3,
                "gamma": 0.99,
                "batch_size": 64,
                "buffer_size": 10000,
                "target_update_interval": 100,
                "exploration_initial_eps": 1.0,
                "exploration_final_eps": 0.01,
            },
        },
        {
            "name": "cfg6_target_slow",
            "dqn_kwargs": {
                "learning_rate": 1e-3,
                "gamma": 0.99,
                "batch_size": 64,
                "buffer_size": 10000,
                "target_update_interval": 1000,
                "exploration_initial_eps": 0.6,
                "exploration_final_eps": 0.01,
            },
        },
    ]


def _aggregate_across_repeats(
    summary_paths: list[Path],
    *,
    map_size: int,
) -> tuple[list[dict], dict | None]:
    """Return (table rows per config with mean/std, best row dict)."""
    by_cfg: dict[str, list[float]] = {}
    for sp in summary_paths:
        if not sp.is_file():
            continue
        with open(sp, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if int(row["map_size"]) != int(map_size):
                    continue
                by_cfg.setdefault(row["config_name"], []).append(float(row["success_rate"]))

    table = []
    for cfg, rates in sorted(by_cfg.items()):
        m = statistics.mean(rates)
        s = statistics.stdev(rates) if len(rates) > 1 else 0.0
        table.append(
            {
                "config_name": cfg,
                "n_runs": len(rates),
                "success_rate_mean": m,
                "success_rate_std": s,
                "success_rates": rates,
            }
        )

    if not table:
        return [], None

    # Best: maximize mean test success; tie-break lower std (more stable).
    best = sorted(table, key=lambda r: (-r["success_rate_mean"], r["success_rate_std"]))[0]
    return table, best


def _write_aggregate_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["config_name", "n_runs", "success_rate_mean", "success_rate_std"],
        )
        w.writeheader()
        for r in sorted(rows, key=lambda x: -x["success_rate_mean"]):
            w.writerow(
                {
                    "config_name": r["config_name"],
                    "n_runs": r["n_runs"],
                    "success_rate_mean": f"{r['success_rate_mean']:.6f}",
                    "success_rate_std": f"{r['success_rate_std']:.6f}",
                }
            )


def _plot_config_comparison(
    rows: list[dict],
    *,
    out_path: Path,
    base_tag: str,
    map_size: int,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(rows, key=lambda r: -r["success_rate_mean"])
    names = [r["config_name"] for r in ordered]
    means = [r["success_rate_mean"] for r in ordered]
    stds = [r["success_rate_std"] for r in ordered]
    fig, ax = plt.subplots(figsize=(11, 4.5))
    x = range(len(names))
    ax.bar(x, means, yerr=stds, capsize=5, color="steelblue", alpha=0.85, ecolor="black")
    ax.set_xticks(list(x))
    ax.set_xticklabels(names, rotation=25, ha="right")
    ax.set_ylabel("Test success rate")
    ax.set_ylim(0, 1.05)
    ax.set_title(
        f"Mean ± std over {ordered[0]['n_runs']} repeats | map {map_size}×{map_size} | tag {base_tag}"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def _replot_config_runs(
    summary_paths: list[Path],
    config_name: str,
    *,
    map_size: int,
    is_slippery: bool,
) -> list[Path]:
    """Regenerate PNGs from CSV logs for one named config in every repeat."""
    replotted = []
    for sp in summary_paths:
        if not sp.is_file():
            continue
        with open(sp, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row["config_name"] != config_name:
                    continue
                if int(row["map_size"]) != int(map_size):
                    continue
                run_dir = row["output_dir"]
                replot_from_directory(
                    run_dir,
                    map_size=map_size,
                    is_slippery=is_slippery,
                )
                replotted.append(Path(run_dir))
                break
    return replotted


def _replot_best_config_runs(
    summary_paths: list[Path],
    best_cfg: str,
    *,
    map_size: int,
    is_slippery: bool,
) -> list[Path]:
    """Regenerate PNGs from CSV logs for the winning config in every repeat."""
    return _replot_config_runs(
        summary_paths,
        best_cfg,
        map_size=map_size,
        is_slippery=is_slippery,
    )


def finalize_sweep_comparison(
    summary_paths: list[Path],
    sweep_base: Path,
    base_tag: str,
    *,
    map_size: int,
    is_slippery: bool,
    also_replot_config: str | None = None,
) -> None:
    if not summary_paths:
        return
    table, best = _aggregate_across_repeats(summary_paths, map_size=map_size)
    if not table or best is None:
        print("No rows to aggregate for comparison.")
        return

    compare_dir = sweep_base / "compare_latest"
    agg_csv = compare_dir / f"sweep_aggregate_{base_tag}.csv"
    _write_aggregate_csv(agg_csv, table)
    cmp_png = compare_dir / f"sweep_cfg_mean_std_{base_tag}.png"
    _plot_config_comparison(table, out_path=cmp_png, base_tag=base_tag, map_size=map_size)

    best_txt = compare_dir / f"best_config_{base_tag}.txt"
    lines = [
        f"base_tag={base_tag}",
        f"map_size={map_size}",
        f"is_slippery={is_slippery}",
        "",
        "Selection rule: max mean(success_rate), tie-break min std(success_rate).",
        "",
        f"BEST_CONFIG={best['config_name']}",
        f"  mean={best['success_rate_mean']:.4f}  std={best['success_rate_std']:.4f}  n={best['n_runs']}",
        "",
        f"Aggregate CSV: {agg_csv}",
        f"Comparison figure: {cmp_png}",
        "",
        "Per-repeat success rates:",
    ]
    for r in table:
        if r["config_name"] == best["config_name"]:
            lines.append(f"  {r['config_name']}: {r['success_rates']}")
    best_txt.write_text("\n".join(lines), encoding="utf-8")

    replotted = _replot_best_config_runs(
        summary_paths,
        best["config_name"],
        map_size=map_size,
        is_slippery=is_slippery,
    )
    replotted_extra: list[Path] = []
    if also_replot_config and also_replot_config != best["config_name"]:
        replotted_extra = _replot_config_runs(
            summary_paths,
            also_replot_config,
            map_size=map_size,
            is_slippery=is_slippery,
        )

    print("\n=== Cross-repeat comparison (all configs) ===")
    for r in sorted(table, key=lambda x: (-x["success_rate_mean"], x["success_rate_std"])):
        print(
            f"  {r['config_name']}: mean={r['success_rate_mean']:.4f} "
            f"std={r['success_rate_std']:.4f} (n={r['n_runs']})"
        )
    print(f"\nSelected BEST: {best['config_name']}")
    print(f"  Wrote: {agg_csv}")
    print(f"  Wrote: {cmp_png}")
    print(f"  Wrote: {best_txt}")
    print(f"  Replot (PNGs) for best config in {len(replotted)} run folder(s):")
    for p in replotted:
        print(f"    {p}")
    if replotted_extra:
        print(f"  Extra replot for '{also_replot_config}' in {len(replotted_extra)} run folder(s):")
        for p in replotted_extra:
            print(f"    {p}")


def run_sweep(
    run_tag: str | None = None,
    one_hot: bool | None = None,
    reward_shaping: bool | None = None,
    n_repeats: int | None = None,
    skip_training: bool = False,
    also_replot_config: str | None = None,
):
    project_root = Path(__file__).resolve().parent.parent
    sweep_base = project_root / "dqn_plots" / "sweep"
    sweep_base.mkdir(parents=True, exist_ok=True)

    if one_hot is None:
        one_hot = USE_ONE_HOT
    if reward_shaping is None:
        reward_shaping = USE_REWARD_SHAPING
    if n_repeats is None:
        n_repeats = N_REPEATS
    if n_repeats < 1:
        raise ValueError("n_repeats must be >= 1")

    base_tag = run_tag if run_tag is not None else datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_paths: list[Path] = []

    map_sizes = [8]
    is_slippery = True
    total_timesteps = 200_000
    test_episodes = 1000
    sweep_configs = sweep_configs_all()

    if not skip_training:
        for rep in range(n_repeats):
            run_tag_cur = base_tag if n_repeats == 1 else f"{base_tag}_rep{rep + 1}"
            out_root = sweep_base / run_tag_cur
            out_root.mkdir(parents=True, exist_ok=True)

            if n_repeats > 1:
                print(f"\n########## Repeat {rep + 1}/{n_repeats} -> {run_tag_cur} ##########")

            summary_rows = []

            for map_size in map_sizes:
                map_desc = load_map(map_size)
                horizon = max_episode_steps_for_map_size(map_size)

                for cfg in sweep_configs:
                    cfg_name = cfg["name"]
                    kwargs = cfg.get("dqn_kwargs", {})
                    rel = Path(f"{map_size}x{map_size}")
                    if one_hot:
                        rel = rel / "onehot"
                    if reward_shaping:
                        rel = rel / "shaped"
                    out_dir = out_root / rel / cfg_name
                    out_dir.mkdir(parents=True, exist_ok=True)

                    print(
                        f"\n=== map={map_size}x{map_size}, config={cfg_name}, "
                        f"one_hot={one_hot}, reward_shaping={reward_shaping} ==="
                    )
                    model, logger = train_agent(
                        total_timesteps=total_timesteps,
                        map_desc=map_desc,
                        max_episode_steps=horizon,
                        is_slippery=is_slippery,
                        reward_shaping=reward_shaping,
                        one_hot=one_hot,
                        **kwargs,
                    )
                    save_training_visualization(
                        logger,
                        model,
                        total_timesteps,
                        str(out_dir),
                        map_size=map_size,
                        is_slippery=is_slippery,
                        write_csv=True,
                        save_figures=False,
                    )
                    avg_reward, success_rate = test_agent(
                        model,
                        num_episodes=test_episodes,
                        map_desc=map_desc,
                        max_episode_steps=horizon,
                        is_slippery=is_slippery,
                        reward_shaping=reward_shaping,
                        one_hot=one_hot,
                    )

                    summary_rows.append(
                        {
                            "run_tag": run_tag_cur,
                            "repeat": rep + 1,
                            "map_size": map_size,
                            "config_name": cfg_name,
                            "one_hot": one_hot,
                            "reward_shaping": reward_shaping,
                            "is_slippery": is_slippery,
                            "total_timesteps": total_timesteps,
                            "test_episodes": test_episodes,
                            "avg_reward": avg_reward,
                            "success_rate": success_rate,
                            "dqn_kwargs_json": json.dumps(kwargs, sort_keys=True),
                            "output_dir": str(out_dir),
                        }
                    )

            summary_path = out_root / "sweep_summary.csv"
            with open(summary_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "run_tag",
                        "repeat",
                        "map_size",
                        "config_name",
                        "one_hot",
                        "reward_shaping",
                        "is_slippery",
                        "total_timesteps",
                        "test_episodes",
                        "avg_reward",
                        "success_rate",
                        "dqn_kwargs_json",
                        "output_dir",
                    ],
                )
                writer.writeheader()
                writer.writerows(summary_rows)

            summary_paths.append(summary_path)

            print(f"\nRun tag: {run_tag_cur}")
            print(f"Sweep summary saved to: {summary_path}")
            print("\n=== Summary (sorted by success_rate desc) ===")
            for row in sorted(summary_rows, key=lambda r: r["success_rate"], reverse=True):
                print(
                    f"map={row['map_size']}x{row['map_size']} | cfg={row['config_name']} | "
                    f"one_hot={row['one_hot']} | shaped={row['reward_shaping']} | "
                    f"success_rate={row['success_rate']:.4f} | avg_reward={row['avg_reward']:.4f}"
                )

        if n_repeats > 1:
            print(f"\n=== All {n_repeats} repeats done (CSV during training; no sweep PNGs yet) ===")
            for p in summary_paths:
                print(f"  {p}")
    else:
        if run_tag is None:
            raise ValueError("--analyze-only requires --run-tag <base_tag> (folders <base_tag>_rep* on disk).")
        for rep in range(n_repeats):
            run_tag_cur = base_tag if n_repeats == 1 else f"{base_tag}_rep{rep + 1}"
            sp = sweep_base / run_tag_cur / "sweep_summary.csv"
            if sp.is_file():
                summary_paths.append(sp)
        if len(summary_paths) != n_repeats:
            raise RuntimeError(
                f"--analyze-only: expected {n_repeats} sweep_summary.csv files under {sweep_base}, "
                f"found {len(summary_paths)}."
            )

    if summary_paths and n_repeats >= 1 and map_sizes:
        finalize_sweep_comparison(
            summary_paths,
            sweep_base,
            base_tag,
            map_size=map_sizes[0],
            is_slippery=is_slippery,
            also_replot_config=also_replot_config,
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="DQN hyperparameter sweeps: CSV during training; then compare repeats and replot best."
    )
    parser.add_argument("--run-tag", type=str, default=None, help="Output subdir under dqn_plots/sweep/.")
    parser.add_argument(
        "--one-hot",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Use one-hot state observations (same as base_DQN USE_ONE_HOT). "
            "If omitted, uses USE_ONE_HOT in this file."
        ),
    )
    parser.add_argument(
        "--reward-shaping",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Enable FrozenLake dense reward shaping wrapper. "
            "If omitted, uses USE_REWARD_SHAPING in this file."
        ),
    )
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=None,
        help=f"Number of independent runs (each gets its own folder + sweep_summary.csv). Default: {N_REPEATS}.",
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Skip training; only read existing <run-tag>_rep*/sweep_summary.csv and compare + replot best.",
    )
    parser.add_argument(
        "--also-replot-config",
        type=str,
        default=None,
        metavar="CONFIG_NAME",
        help=(
            "After comparison, also regenerate PNGs from CSV for this config in every repeat "
            "(e.g. cfg5_more_explore). Does not change which config is selected as BEST."
        ),
    )
    args = parser.parse_args()
    run_sweep(
        run_tag=args.run_tag,
        one_hot=args.one_hot,
        reward_shaping=args.reward_shaping,
        n_repeats=args.n_repeats,
        skip_training=args.analyze_only,
        also_replot_config=args.also_replot_config,
    )
