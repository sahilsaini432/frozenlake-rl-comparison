import csv
import os
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt


def _find_recent_runs(sweep_base: Path, n: int = 3):
    runs = []
    for p in sweep_base.iterdir():
        if not p.is_dir():
            continue
        summ = p / "sweep_summary.csv"
        if not summ.exists():
            continue
        runs.append(p)
    runs.sort(key=lambda d: (d / "sweep_summary.csv").stat().st_mtime, reverse=True)
    return runs[:n]


def _read_summary(run_dir: Path):
    path = run_dir / "sweep_summary.csv"
    rows = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def plot_across_runs(sweep_base: Path, n_runs: int = 3, map_size: int | None = None):
    run_dirs = _find_recent_runs(sweep_base, n=n_runs)
    if len(run_dirs) < n_runs:
        raise RuntimeError(
            f"Found only {len(run_dirs)} run dirs with sweep_summary.csv under {sweep_base}. "
            f"Expected {n_runs}."
        )

    all_rows = []
    for rd in run_dirs:
        rows = _read_summary(rd)
        for r in rows:
            r2 = dict(r)
            r2["run_dir"] = str(rd)
            all_rows.append(r2)

    # Filter by map_size if requested
    if map_size is not None:
        all_rows = [r for r in all_rows if int(r["map_size"]) == int(map_size)]

    # Group by (map_size, config_name)
    grouped = {}
    for r in all_rows:
        key = (int(r["map_size"]), r["config_name"])
        grouped.setdefault(key, []).append(float(r["success_rate"]))

    # Determine output folder
    out_dir = sweep_base / "compare_latest"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Plot separately per map_size
    by_map = {}
    for (m, cfg), vals in grouped.items():
        by_map.setdefault(m, []).append((cfg, vals))

    for m, cfg_vals in by_map.items():
        cfg_vals.sort(key=lambda x: sum(x[1]) / len(x[1]), reverse=True)

        cfg_names = [cfg for cfg, _ in cfg_vals]
        means = []
        stdevs = []
        for _, vals in cfg_vals:
            mean = sum(vals) / len(vals)
            var = sum((v - mean) ** 2 for v in vals) / max(1, len(vals) - 1)
            stdev = var ** 0.5
            means.append(mean)
            stdevs.append(stdev)

        # Performance plot (mean success rate)
        plt.figure(figsize=(12, 4))
        plt.bar(range(len(cfg_names)), means)
        plt.xticks(range(len(cfg_names)), cfg_names, rotation=30, ha="right")
        plt.ylabel("Mean success rate")
        plt.ylim(0, 1.05)
        plt.title(f"Performance (mean) across last {n_runs} runs | map={m}x{m}")
        plt.tight_layout()
        plt.savefig(out_dir / f"performance_mean_map{m}_last{n_runs}_{ts}.png", dpi=300)
        plt.close()

        # Stability plot (std of success rate)
        plt.figure(figsize=(12, 4))
        plt.bar(range(len(cfg_names)), stdevs)
        plt.xticks(range(len(cfg_names)), cfg_names, rotation=30, ha="right")
        plt.ylabel("Std dev of success rate")
        plt.ylim(0, max(stdevs + [0.01]) * 1.1)
        plt.title(f"Stability (std) across last {n_runs} runs | map={m}x{m}")
        plt.tight_layout()
        plt.savefig(out_dir / f"stability_std_map{m}_last{n_runs}_{ts}.png", dpi=300)
        plt.close()

    # Print run dirs for traceability
    print("Selected recent run dirs:")
    for rd in run_dirs:
        print(f"  - {rd}")
    print(f"Plots saved under: {out_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compare performance + stability across multiple sweep runs.")
    parser.add_argument("--n-runs", type=int, default=3, help="Number of recent runs to compare.")
    parser.add_argument("--map-size", type=int, default=None, help="Optional map size filter (e.g., 8).")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    sweep_base = project_root / "dqn_plots" / "sweep"
    plot_across_runs(sweep_base, n_runs=args.n_runs, map_size=args.map_size)

