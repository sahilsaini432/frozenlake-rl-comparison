"""
One-hot vs discrete-state baseline: mean test success rate averaged over all
hyperparameter configs in each sweep, then mean ± std across repeats.
Only reads sweep_summary.csv (no training).

  python DQN/plot_onehot_vs_baseline.py \\
    --discrete-base-tag baseline_discrete_200k --onehot-base-tag onehot_discrete_200k \\
    --map-size 8 --discover-repeats

Explicit folder lists:

  python DQN/plot_onehot_vs_baseline.py \\
    --baseline-tags B_rep1 B_rep2 --onehot-tags O_rep1 O_rep2 --map-size 8

  Fixed hyperparameter config (separate PNG, does not overwrite avg plot):

  python DQN/plot_onehot_vs_baseline.py \\
    --discrete-base-tag baseline_discrete_200k --onehot-base-tag onehot_discrete_200k \\
    --map-size 8 --discover-repeats --fixed-config cfg5_more_explore
"""

from __future__ import annotations

import os

os.environ.setdefault("KMP_USE_SHM", "0")
os.environ.setdefault("KMP_DISABLE_SHM", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
if "MPLBACKEND" not in os.environ:
    os.environ["MPLBACKEND"] = "Agg"

import argparse
import csv
import re
import statistics
from pathlib import Path

import matplotlib.pyplot as plt


def _resolve_run_dir(sweep_base: Path, tag_or_path: str) -> Path:
    p = Path(tag_or_path)
    if p.is_dir():
        return p.resolve()
    return (sweep_base / tag_or_path).resolve()


def _parse_one_hot(row: dict) -> bool | None:
    raw = row.get("one_hot")
    if raw is None or raw == "":
        return None
    s = str(raw).strip().lower()
    if s in ("true", "1", "yes"):
        return True
    if s in ("false", "0", "no"):
        return False
    return None


def _infer_one_hot_from_row(row: dict) -> bool:
    oh = _parse_one_hot(row)
    if oh is not None:
        return oh
    out_dir = row.get("output_dir", "").replace("\\", "/").lower()
    return "onehot" in out_dir


def _row_matches_one_hot_filter(row: dict, one_hot_want: bool | None) -> bool:
    if one_hot_want is None:
        return True
    return _infer_one_hot_from_row(row) == one_hot_want


def _mean_rate_over_all_configs_from_summary(
    summary_path: Path,
    *,
    map_size: int,
    one_hot_want: bool,
) -> tuple[float, int] | None:
    if not summary_path.is_file():
        return None
    rates: list[float] = []
    with open(summary_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if int(row["map_size"]) != int(map_size):
                continue
            if not _row_matches_one_hot_filter(row, one_hot_want):
                continue
            rates.append(float(row["success_rate"]))
    if not rates:
        return None
    return (statistics.mean(rates), len(rates))


def _rate_for_config_from_summary(
    summary_path: Path,
    *,
    config_name: str,
    map_size: int,
    one_hot_want: bool,
) -> float | None:
    if not summary_path.is_file():
        return None
    with open(summary_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("config_name") != config_name:
                continue
            if int(row["map_size"]) != int(map_size):
                continue
            if not _row_matches_one_hot_filter(row, one_hot_want):
                continue
            return float(row["success_rate"])
    return None


def _collect_fixed_config_rates(
    run_dirs: list[Path],
    *,
    config_name: str,
    map_size: int,
    one_hot_want: bool,
) -> tuple[list[float], list[Path]]:
    rates: list[float] = []
    used: list[Path] = []
    for rd in run_dirs:
        r = _rate_for_config_from_summary(
            rd / "sweep_summary.csv",
            config_name=config_name,
            map_size=map_size,
            one_hot_want=one_hot_want,
        )
        if r is None:
            continue
        rates.append(r)
        used.append(rd)
    return rates, used


def _safe_config_filename(config_name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", config_name)


def plot_onehot_vs_discrete_fixed_config(
    sweep_base: Path,
    *,
    onehot_run_dirs: list[Path],
    baseline_run_dirs: list[Path],
    map_size: int,
    config_name: str,
    out_dir: Path | None = None,
) -> Path:
    oh_rates, oh_used = _collect_fixed_config_rates(
        onehot_run_dirs, config_name=config_name, map_size=map_size, one_hot_want=True
    )
    bl_rates, bl_used = _collect_fixed_config_rates(
        baseline_run_dirs, config_name=config_name, map_size=map_size, one_hot_want=False
    )

    if not oh_rates:
        raise RuntimeError(
            f"No one-hot rows for config {config_name!r}; check sweep_summary.csv and --map-size."
        )
    if not bl_rates:
        raise RuntimeError(f"No discrete baseline rows for config {config_name!r}.")

    oh_mean = statistics.mean(oh_rates)
    oh_err = statistics.stdev(oh_rates) if len(oh_rates) > 1 else 0.0
    bl_mean = statistics.mean(bl_rates)
    bl_err = statistics.stdev(bl_rates) if len(bl_rates) > 1 else 0.0

    if out_dir is None:
        out_dir = sweep_base / "compare_latest"
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_fn = _safe_config_filename(config_name)
    out_path = out_dir / f"onehot_vs_discrete_fixed_{cfg_fn}_map{map_size}x{map_size}.png"

    labels = [
        f"Discrete state\n(n={len(bl_rates)} run{'s' if len(bl_rates) != 1 else ''})",
        f"One-hot state\n(n={len(oh_rates)} run{'s' if len(oh_rates) != 1 else ''})",
    ]
    means = [bl_mean, oh_mean]
    errs = [bl_err, oh_err]

    if len(bl_rates) == len(oh_rates):
        runs_title = f"{len(bl_rates)} runs"
    else:
        runs_title = f"{len(bl_rates)} discrete / {len(oh_rates)} one-hot runs"

    x = range(len(labels))
    fig, ax = plt.subplots(figsize=(6, 4.5))
    colors = ["tab:blue", "tab:orange"]
    ax.bar(x, means, yerr=errs, capsize=6, color=colors, alpha=0.85, ecolor="black")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Test success rate")
    ax.set_ylim(0, 1.05)
    ax.set_title(
        f"Mean ± std over {runs_title} | map {map_size}×{map_size}\n"
        f"(fixed config: {config_name})"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

    log_path = out_path.with_suffix(".txt")
    lines = [
        f"Fixed hyperparameter config: {config_name}",
        f"map_size={map_size}",
        "",
        "Discrete:",
        *[f"  - {p.name} | success_rate={r:.4f}" for p, r in zip(bl_used, bl_rates)],
        f"  mean={bl_mean:.4f} std={bl_err:.4f}",
        "",
        "One-hot:",
        *[f"  - {p.name} | success_rate={r:.4f}" for p, r in zip(oh_used, oh_rates)],
        f"  mean={oh_mean:.4f} std={oh_err:.4f}",
        "",
        f"figure: {out_path}",
    ]
    log_path.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return out_path


def _collect_avg_over_configs_per_repeat(
    run_dirs: list[Path],
    *,
    map_size: int,
    one_hot_want: bool,
) -> tuple[list[float], list[Path], list[int]]:
    rates: list[float] = []
    used: list[Path] = []
    n_cfgs: list[int] = []
    for rd in run_dirs:
        pair = _mean_rate_over_all_configs_from_summary(
            rd / "sweep_summary.csv",
            map_size=map_size,
            one_hot_want=one_hot_want,
        )
        if pair is None:
            continue
        rates.append(pair[0])
        used.append(rd)
        n_cfgs.append(pair[1])
    return rates, used, n_cfgs


def plot_onehot_vs_discrete(
    sweep_base: Path,
    *,
    onehot_run_dirs: list[Path],
    baseline_run_dirs: list[Path],
    map_size: int,
    out_dir: Path | None = None,
) -> Path:
    oh_rates, oh_used, oh_n = _collect_avg_over_configs_per_repeat(
        onehot_run_dirs, map_size=map_size, one_hot_want=True
    )
    bl_rates, bl_used, bl_n = _collect_avg_over_configs_per_repeat(
        baseline_run_dirs, map_size=map_size, one_hot_want=False
    )

    if not oh_rates:
        raise RuntimeError("No one-hot rates; check run dirs and sweep_summary.csv.")
    if not bl_rates:
        raise RuntimeError("No discrete baseline rates; check run dirs.")

    oh_mean = statistics.mean(oh_rates)
    oh_err = statistics.stdev(oh_rates) if len(oh_rates) > 1 else 0.0
    bl_mean = statistics.mean(bl_rates)
    bl_err = statistics.stdev(bl_rates) if len(bl_rates) > 1 else 0.0

    if out_dir is None:
        out_dir = sweep_base / "compare_latest"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"onehot_vs_discrete_avg_over_configs_map{map_size}x{map_size}.png"

    labels = [
        f"Discrete state\n(n={len(bl_rates)} run{'s' if len(bl_rates) != 1 else ''})",
        f"One-hot state\n(n={len(oh_rates)} run{'s' if len(oh_rates) != 1 else ''})",
    ]
    means = [bl_mean, oh_mean]
    errs = [bl_err, oh_err]

    if len(bl_rates) == len(oh_rates):
        runs_title = f"{len(bl_rates)} runs"
    else:
        runs_title = f"{len(bl_rates)} discrete / {len(oh_rates)} one-hot runs"

    x = range(len(labels))
    fig, ax = plt.subplots(figsize=(6, 4.5))
    colors = ["tab:blue", "tab:orange"]
    ax.bar(x, means, yerr=errs, capsize=6, color=colors, alpha=0.85, ecolor="black")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Test success rate")
    ax.set_ylim(0, 1.05)
    ax.set_title(
        f"Mean ± std over {runs_title} | map {map_size}×{map_size}\n"
        f"(each run: avg over hyperparameter configs)"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

    log_path = out_path.with_suffix(".txt")
    lines = [
        "Method: per repeat, value = mean(success_rate) over all configs in that sweep; "
        "then bars = mean ± std of those values across repeats.",
        f"map_size={map_size}",
        "",
        "Discrete:",
        *[
            f"  - {p.name} | mean_over_{k}_configs={r:.4f}"
            for p, k, r in zip(bl_used, bl_n, bl_rates)
        ],
        f"  across repeats: mean={bl_mean:.4f} std={bl_err:.4f}",
        "",
        "One-hot:",
        *[
            f"  - {p.name} | mean_over_{k}_configs={r:.4f}"
            for p, k, r in zip(oh_used, oh_n, oh_rates)
        ],
        f"  across repeats: mean={oh_mean:.4f} std={oh_err:.4f}",
        "",
        f"figure: {out_path}",
    ]
    log_path.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return out_path


def _expand_repeat_tags(base: str, n: int) -> list[str]:
    if n == 1:
        return [base]
    return [f"{base}_rep{i}" for i in range(1, n + 1)]


def _discover_run_dirs(sweep_base: Path, base_tag: str) -> list[Path]:
    found: list[tuple[int, Path]] = []
    prefix = f"{base_tag}_rep"
    for p in sweep_base.iterdir():
        if not p.is_dir():
            continue
        if not p.name.startswith(prefix):
            continue
        m = re.match(rf"^{re.escape(base_tag)}_rep(\d+)$", p.name)
        if not m:
            continue
        if not (p / "sweep_summary.csv").is_file():
            continue
        found.append((int(m.group(1)), p.resolve()))
    found.sort(key=lambda x: x[0])
    if found:
        return [p for _, p in found]
    single = (sweep_base / base_tag).resolve()
    if single.is_dir() and (single / "sweep_summary.csv").is_file():
        return [single]
    return []


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Bar plot: discrete vs one-hot from sweep_summary.csv — "
            "per repeat, average success rate over all configs; bars = mean ± std across repeats."
        )
    )
    parser.add_argument(
        "--sweep-base",
        type=Path,
        default=None,
        help="Parent of run folders (default: <project>/dqn_plots/sweep).",
    )
    parser.add_argument(
        "--onehot-tags",
        nargs="*",
        default=None,
        metavar="RUN_TAG",
        help="Run folder names (each with sweep_summary.csv).",
    )
    parser.add_argument(
        "--baseline-tags",
        nargs="*",
        default=None,
        metavar="RUN_TAG",
        help="Discrete baseline run folder names.",
    )
    parser.add_argument(
        "--discrete-base-tag",
        type=str,
        default=None,
        help="With --discover-repeats or --n-repeats: discrete runs <tag>_rep1..",
    )
    parser.add_argument(
        "--onehot-base-tag",
        type=str,
        default=None,
        help="With --discover-repeats or --n-repeats: one-hot runs <tag>_rep1..",
    )
    parser.add_argument(
        "--n-repeats",
        type=int,
        default=3,
        help="With *-base-tag (no --discover-repeats): expect <base>_rep1..rep<n>. Default 3.",
    )
    parser.add_argument(
        "--discover-repeats",
        action="store_true",
        help="Auto-find all <tag>_rep* dirs that contain sweep_summary.csv.",
    )
    parser.add_argument("--map-size", type=int, default=8)
    parser.add_argument(
        "--fixed-config",
        type=str,
        default=None,
        metavar="CONFIG_NAME",
        help=(
            "If set, write onehot_vs_discrete_fixed_<config>_map{N}x{N}.png/.txt only "
            "(does not overwrite the avg-over-configs figure)."
        ),
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    plot_path = Path(__file__).resolve().parent
    sweep_base = args.sweep_base or (plot_path / "dqn_plots" / "sweep")

    if args.discover_repeats:
        if not args.discrete_base_tag or not args.onehot_base_tag:
            parser.error("--discover-repeats requires both --discrete-base-tag and --onehot-base-tag.")
        baseline_dirs = _discover_run_dirs(sweep_base, args.discrete_base_tag)
        onehot_dirs = _discover_run_dirs(sweep_base, args.onehot_base_tag)
        if not baseline_dirs:
            parser.error(f"No runs with sweep_summary.csv for discrete tag {args.discrete_base_tag!r}.")
        if not onehot_dirs:
            parser.error(f"No runs with sweep_summary.csv for one-hot tag {args.onehot_base_tag!r}.")
    else:
        oh = list(args.onehot_tags) if args.onehot_tags else []
        bl = list(args.baseline_tags) if args.baseline_tags else []
        if args.onehot_base_tag:
            oh = _expand_repeat_tags(args.onehot_base_tag, args.n_repeats)
        if args.discrete_base_tag:
            bl = _expand_repeat_tags(args.discrete_base_tag, args.n_repeats)
        if not oh or not bl:
            parser.error(
                "Provide (--onehot-tags and --baseline-tags) or "
                "(--onehot-base-tag and --discrete-base-tag), optionally with --discover-repeats."
            )
        onehot_dirs = [_resolve_run_dir(sweep_base, t) for t in oh]
        baseline_dirs = [_resolve_run_dir(sweep_base, t) for t in bl]

    if args.fixed_config:
        plot_onehot_vs_discrete_fixed_config(
            sweep_base,
            onehot_run_dirs=onehot_dirs,
            baseline_run_dirs=baseline_dirs,
            map_size=args.map_size,
            config_name=args.fixed_config,
        )
    else:
        plot_onehot_vs_discrete(
            sweep_base,
            onehot_run_dirs=onehot_dirs,
            baseline_run_dirs=baseline_dirs,
            map_size=args.map_size,
        )


if __name__ == "__main__":
    main()
