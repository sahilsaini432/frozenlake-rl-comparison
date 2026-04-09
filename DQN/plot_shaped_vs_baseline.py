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


def _parse_bool_col(row: dict, key: str) -> bool | None:
    raw = row.get(key)
    if raw is None or raw == "":
        return None
    s = str(raw).strip().lower()
    if s in ("true", "1", "yes"):
        return True
    if s in ("false", "0", "no"):
        return False
    return None


def _infer_one_hot_from_row(row: dict) -> bool:
    oh = _parse_bool_col(row, "one_hot")
    if oh is not None:
        return oh
    out_dir = row.get("output_dir", "").replace("\\", "/").lower()
    return "onehot" in out_dir


def _infer_reward_shaping_from_row(row: dict) -> bool:
    sh = _parse_bool_col(row, "reward_shaping")
    if sh is not None:
        return sh
    out_dir = row.get("output_dir", "").replace("\\", "/").lower()
    return "/shaped/" in out_dir


def _row_matches(
    row: dict,
    *,
    map_size: int,
    one_hot_want: bool,
    reward_shaping_want: bool,
) -> bool:
    if int(row["map_size"]) != int(map_size):
        return False
    if _infer_one_hot_from_row(row) != one_hot_want:
        return False
    if _infer_reward_shaping_from_row(row) != reward_shaping_want:
        return False
    return True


def _mean_rate_over_all_configs_from_summary(
    summary_path: Path,
    *,
    map_size: int,
    reward_shaping_want: bool,
) -> tuple[float, int] | None:
    if not summary_path.is_file():
        return None
    rates: list[float] = []
    with open(summary_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if not _row_matches(
                row,
                map_size=map_size,
                one_hot_want=False,
                reward_shaping_want=reward_shaping_want,
            ):
                continue
            rates.append(float(row["success_rate"]))
    if not rates:
        return None
    return (statistics.mean(rates), len(rates))


def _collect_avg_over_configs_per_repeat(
    run_dirs: list[Path],
    *,
    map_size: int,
    reward_shaping_want: bool,
) -> tuple[list[float], list[Path], list[int]]:
    rates: list[float] = []
    used: list[Path] = []
    n_cfgs: list[int] = []
    for rd in run_dirs:
        pair = _mean_rate_over_all_configs_from_summary(
            rd / "sweep_summary.csv",
            map_size=map_size,
            reward_shaping_want=reward_shaping_want,
        )
        if pair is None:
            continue
        rates.append(pair[0])
        used.append(rd)
        n_cfgs.append(pair[1])
    return rates, used, n_cfgs


def _min_max_from_summary(
    summary_path: Path,
    *,
    map_size: int,
    reward_shaping_want: bool,
) -> tuple[float, str, float, str, int] | None:
    """Return (min_rate, min_cfg, max_rate, max_cfg, n_configs) for matching rows."""
    if not summary_path.is_file():
        return None
    pairs: list[tuple[float, str]] = []
    with open(summary_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if not _row_matches(
                row,
                map_size=map_size,
                one_hot_want=False,
                reward_shaping_want=reward_shaping_want,
            ):
                continue
            pairs.append((float(row["success_rate"]), row["config_name"]))
    if not pairs:
        return None
    min_r, min_c = min(pairs, key=lambda x: x[0])
    max_r, max_c = max(pairs, key=lambda x: x[0])
    return (min_r, min_c, max_r, max_c, len(pairs))


def _collect_min_max_per_repeat(
    run_dirs: list[Path],
    *,
    map_size: int,
    reward_shaping_want: bool,
) -> tuple[list[tuple[float, str]], list[tuple[float, str]], list[Path]]:
    """Per repeat: (max_rate, max_cfg), (min_rate, min_cfg)."""
    max_list: list[tuple[float, str]] = []
    min_list: list[tuple[float, str]] = []
    used: list[Path] = []
    for rd in run_dirs:
        mm = _min_max_from_summary(
            rd / "sweep_summary.csv",
            map_size=map_size,
            reward_shaping_want=reward_shaping_want,
        )
        if mm is None:
            continue
        min_r, min_c, max_r, max_c, _n = mm
        max_list.append((max_r, max_c))
        min_list.append((min_r, min_c))
        used.append(rd)
    return max_list, min_list, used


def plot_shaped_vs_baseline_best_worst(
    sweep_base: Path,
    *,
    shaped_run_dirs: list[Path],
    baseline_run_dirs: list[Path],
    map_size: int,
    out_dir: Path | None = None,
) -> Path:
    bl_max, bl_min, bl_used = _collect_min_max_per_repeat(
        baseline_run_dirs, map_size=map_size, reward_shaping_want=False
    )
    sh_max, sh_min, sh_used = _collect_min_max_per_repeat(
        shaped_run_dirs, map_size=map_size, reward_shaping_want=True
    )

    if not bl_max or not sh_max:
        raise RuntimeError("Need baseline and shaped runs with sweep_summary.csv rows.")

    bl_best_rates = [r for r, _ in bl_max]
    sh_best_rates = [r for r, _ in sh_max]
    bl_worst_rates = [r for r, _ in bl_min]
    sh_worst_rates = [r for r, _ in sh_min]

    def _m_s(vals: list[float]) -> tuple[float, float]:
        m = statistics.mean(vals)
        s = statistics.stdev(vals) if len(vals) > 1 else 0.0
        return m, s

    b_bm, b_bs = _m_s(bl_best_rates)
    s_bm, s_bs = _m_s(sh_best_rates)
    b_wm, b_ws = _m_s(bl_worst_rates)
    s_wm, s_ws = _m_s(sh_worst_rates)

    if out_dir is None:
        out_dir = sweep_base / "compare_latest"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"shaped_vs_baseline_best_worst_per_run_map{map_size}x{map_size}.png"

    nbl, nsh = len(bl_best_rates), len(sh_best_rates)
    runs_title = f"{nbl} runs" if nbl == nsh else f"{nbl} baseline / {nsh} shaped runs"
    bar_labels = [
        f"No reward shaping\n(n={nbl} run{'s' if nbl != 1 else ''})",
        f"Reward shaping\n(n={nsh} run{'s' if nsh != 1 else ''})",
    ]
    colors = ["tab:blue", "tab:green"]
    x = range(2)

    fig, axes = plt.subplots(2, 1, figsize=(6, 7.2), sharex=False)
    for ax, title, bm1, bs1, bm2, bs2 in (
        (
            axes[0],
            f"Best config per run (max success in sweep)\nMean ± std over {runs_title} | map {map_size}×{map_size}",
            b_bm,
            b_bs,
            s_bm,
            s_bs,
        ),
        (
            axes[1],
            f"Worst config per run (min success in sweep)\nMean ± std over {runs_title} | map {map_size}×{map_size}",
            b_wm,
            b_ws,
            s_wm,
            s_ws,
        ),
    ):
        ax.bar(x, [bm1, bm2], yerr=[bs1, bs2], capsize=6, color=colors, alpha=0.85, ecolor="black")
        ax.set_xticks(list(x))
        ax.set_xticklabels(bar_labels)
        ax.set_ylabel("Test success rate")
        ax.set_ylim(0, 1.05)
        ax.set_title(title, fontsize=10)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)

    log_path = out_path.with_suffix(".txt")
    lines = [
        "Discrete only. Per repeat: best = max(success_rate), worst = min(success_rate) over configs in that sweep.",
        "Bars: mean ± std of those per-run values across repeats (best-config name can differ each run).",
        f"map_size={map_size}",
        "",
        "=== BEST (max) per run ===",
        "Baseline:",
        *[
            f"  - {p.name} | {cfg} | success_rate={r:.4f}"
            for p, (r, cfg) in zip(bl_used, bl_max)
        ],
        f"  mean={b_bm:.4f} std={b_bs:.4f}",
        "Shaped:",
        *[
            f"  - {p.name} | {cfg} | success_rate={r:.4f}"
            for p, (r, cfg) in zip(sh_used, sh_max)
        ],
        f"  mean={s_bm:.4f} std={s_bs:.4f}",
        "",
        "=== WORST (min) per run ===",
        "Baseline:",
        *[
            f"  - {p.name} | {cfg} | success_rate={r:.4f}"
            for p, (r, cfg) in zip(bl_used, bl_min)
        ],
        f"  mean={b_wm:.4f} std={b_ws:.4f}",
        "Shaped:",
        *[
            f"  - {p.name} | {cfg} | success_rate={r:.4f}"
            for p, (r, cfg) in zip(sh_used, sh_min)
        ],
        f"  mean={s_wm:.4f} std={s_ws:.4f}",
        "",
        f"figure: {out_path}",
    ]
    log_path.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return out_path


def plot_shaped_vs_baseline(
    sweep_base: Path,
    *,
    shaped_run_dirs: list[Path],
    baseline_run_dirs: list[Path],
    map_size: int,
    out_dir: Path | None = None,
) -> Path:
    sh_rates, sh_used, sh_n = _collect_avg_over_configs_per_repeat(
        shaped_run_dirs, map_size=map_size, reward_shaping_want=True
    )
    bl_rates, bl_used, bl_n = _collect_avg_over_configs_per_repeat(
        baseline_run_dirs, map_size=map_size, reward_shaping_want=False
    )

    if not sh_rates:
        raise RuntimeError("No reward-shaping runs matched; check dirs and sweep_summary.csv.")
    if not bl_rates:
        raise RuntimeError("No baseline (no shaping) runs matched; check dirs.")

    sh_mean = statistics.mean(sh_rates)
    sh_err = statistics.stdev(sh_rates) if len(sh_rates) > 1 else 0.0
    bl_mean = statistics.mean(bl_rates)
    bl_err = statistics.stdev(bl_rates) if len(bl_rates) > 1 else 0.0

    if out_dir is None:
        out_dir = sweep_base / "compare_latest"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"shaped_vs_baseline_avg_over_configs_map{map_size}x{map_size}.png"

    labels = [
        f"No reward shaping\n(n={len(bl_rates)} run{'s' if len(bl_rates) != 1 else ''})",
        f"Reward shaping\n(n={len(sh_rates)} run{'s' if len(sh_rates) != 1 else ''})",
    ]
    means = [bl_mean, sh_mean]
    errs = [bl_err, sh_err]

    if len(bl_rates) == len(sh_rates):
        runs_title = f"{len(bl_rates)} runs"
    else:
        runs_title = f"{len(bl_rates)} baseline / {len(sh_rates)} shaped runs"

    x = range(len(labels))
    fig, ax = plt.subplots(figsize=(6, 4.5))
    colors = ["tab:blue", "tab:green"]
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
        "Discrete observations only (one_hot=False). "
        "Per repeat: mean(success_rate) over all configs; bars = mean ± std across repeats.",
        f"map_size={map_size}",
        "",
        "No reward shaping:",
        *[
            f"  - {p.name} | mean_over_{k}_configs={r:.4f}"
            for p, k, r in zip(bl_used, bl_n, bl_rates)
        ],
        f"  across repeats: mean={bl_mean:.4f} std={bl_err:.4f}",
        "",
        "Reward shaping:",
        *[
            f"  - {p.name} | mean_over_{k}_configs={r:.4f}"
            for p, k, r in zip(sh_used, sh_n, sh_rates)
        ],
        f"  across repeats: mean={sh_mean:.4f} std={sh_err:.4f}",
        "",
        f"figure: {out_path}",
    ]
    log_path.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return out_path


def _safe_config_filename(config_name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", config_name)


def _rate_for_config_shaped(
    summary_path: Path,
    *,
    config_name: str,
    map_size: int,
    reward_shaping_want: bool,
) -> float | None:
    if not summary_path.is_file():
        return None
    with open(summary_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("config_name") != config_name:
                continue
            if not _row_matches(
                row,
                map_size=map_size,
                one_hot_want=False,
                reward_shaping_want=reward_shaping_want,
            ):
                continue
            return float(row["success_rate"])
    return None


def _collect_fixed_config_rates_shaped(
    run_dirs: list[Path],
    *,
    config_name: str,
    map_size: int,
    reward_shaping_want: bool,
) -> tuple[list[float], list[Path]]:
    rates: list[float] = []
    used: list[Path] = []
    for rd in run_dirs:
        r = _rate_for_config_shaped(
            rd / "sweep_summary.csv",
            config_name=config_name,
            map_size=map_size,
            reward_shaping_want=reward_shaping_want,
        )
        if r is None:
            continue
        rates.append(r)
        used.append(rd)
    return rates, used


def plot_shaped_vs_baseline_fixed_config(
    sweep_base: Path,
    *,
    shaped_run_dirs: list[Path],
    baseline_run_dirs: list[Path],
    map_size: int,
    config_name: str,
    out_dir: Path | None = None,
) -> Path:
    sh_rates, sh_used = _collect_fixed_config_rates_shaped(
        shaped_run_dirs, config_name=config_name, map_size=map_size, reward_shaping_want=True
    )
    bl_rates, bl_used = _collect_fixed_config_rates_shaped(
        baseline_run_dirs, config_name=config_name, map_size=map_size, reward_shaping_want=False
    )

    if not sh_rates:
        raise RuntimeError(
            f"No reward-shaping rows for config {config_name!r}; check sweep_summary.csv."
        )
    if not bl_rates:
        raise RuntimeError(f"No baseline rows for config {config_name!r}.")

    sh_mean = statistics.mean(sh_rates)
    sh_err = statistics.stdev(sh_rates) if len(sh_rates) > 1 else 0.0
    bl_mean = statistics.mean(bl_rates)
    bl_err = statistics.stdev(bl_rates) if len(bl_rates) > 1 else 0.0

    if out_dir is None:
        out_dir = sweep_base / "compare_latest"
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_fn = _safe_config_filename(config_name)
    out_path = out_dir / f"shaped_vs_baseline_fixed_{cfg_fn}_map{map_size}x{map_size}.png"

    labels = [
        f"No reward shaping\n(n={len(bl_rates)} run{'s' if len(bl_rates) != 1 else ''})",
        f"Reward shaping\n(n={len(sh_rates)} run{'s' if len(sh_rates) != 1 else ''})",
    ]
    means = [bl_mean, sh_mean]
    errs = [bl_err, sh_err]

    if len(bl_rates) == len(sh_rates):
        runs_title = f"{len(bl_rates)} runs"
    else:
        runs_title = f"{len(bl_rates)} baseline / {len(sh_rates)} shaped runs"

    x = range(len(labels))
    fig, ax = plt.subplots(figsize=(6, 4.5))
    colors = ["tab:blue", "tab:green"]
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
        f"Fixed hyperparameter config: {config_name} (discrete obs, no one-hot).",
        f"map_size={map_size}",
        "",
        "No reward shaping:",
        *[f"  - {p.name} | success_rate={r:.4f}" for p, r in zip(bl_used, bl_rates)],
        f"  mean={bl_mean:.4f} std={bl_err:.4f}",
        "",
        "Reward shaping:",
        *[f"  - {p.name} | success_rate={r:.4f}" for p, r in zip(sh_used, sh_rates)],
        f"  mean={sh_mean:.4f} std={sh_err:.4f}",
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
            "Bar plot: discrete baseline vs reward shaping from sweep_summary.csv — "
            "same aggregation as plot_onehot_vs_baseline (avg over configs per repeat)."
        )
    )
    parser.add_argument(
        "--sweep-base",
        type=Path,
        default=None,
        help="Parent of run folders (default: <project>/dqn_plots/sweep).",
    )
    parser.add_argument(
        "--shaped-tags",
        nargs="*",
        default=None,
        metavar="RUN_TAG",
        help="Reward-shaping run folders (each with sweep_summary.csv).",
    )
    parser.add_argument(
        "--baseline-tags",
        nargs="*",
        default=None,
        metavar="RUN_TAG",
        help="Baseline (no shaping) run folders.",
    )
    parser.add_argument(
        "--baseline-base-tag",
        type=str,
        default=None,
        help="Shorthand: baseline <tag>_rep1.. (with --n-repeats or --discover-repeats).",
    )
    parser.add_argument(
        "--shaped-base-tag",
        type=str,
        default=None,
        help="Shorthand: shaped runs <tag>_rep1..",
    )
    parser.add_argument("--n-repeats", type=int, default=3)
    parser.add_argument(
        "--discover-repeats",
        action="store_true",
        help="Auto-find all <tag>_rep* dirs that contain sweep_summary.csv.",
    )
    parser.add_argument("--map-size", type=int, default=8)
    parser.add_argument(
        "--plot",
        choices=("avg", "best-worst", "all"),
        default="avg",
        help="Which figure(s) to write when --fixed-config is not set (default: avg).",
    )
    parser.add_argument(
        "--fixed-config",
        type=str,
        default=None,
        metavar="CONFIG_NAME",
        help=(
            "If set, write shaped_vs_baseline_fixed_<config>_map{N}x{N}.png/.txt only "
            "(ignores --plot; does not overwrite avg / best-worst figures)."
        ),
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    plot_path = Path(__file__).resolve().parent
    sweep_base = args.sweep_base or (plot_path / "dqn_plots" / "sweep")

    if args.discover_repeats:
        if not args.baseline_base_tag or not args.shaped_base_tag:
            parser.error("--discover-repeats requires --baseline-base-tag and --shaped-base-tag.")
        baseline_dirs = _discover_run_dirs(sweep_base, args.baseline_base_tag)
        shaped_dirs = _discover_run_dirs(sweep_base, args.shaped_base_tag)
        if not baseline_dirs:
            parser.error(f"No sweep_summary.csv for baseline tag {args.baseline_base_tag!r}.")
        if not shaped_dirs:
            parser.error(f"No sweep_summary.csv for shaped tag {args.shaped_base_tag!r}.")
    else:
        sh = list(args.shaped_tags) if args.shaped_tags else []
        bl = list(args.baseline_tags) if args.baseline_tags else []
        if args.shaped_base_tag:
            sh = _expand_repeat_tags(args.shaped_base_tag, args.n_repeats)
        if args.baseline_base_tag:
            bl = _expand_repeat_tags(args.baseline_base_tag, args.n_repeats)
        if not sh or not bl:
            parser.error(
                "Provide (--shaped-tags and --baseline-tags) or both *-base-tag, "
                "optionally with --discover-repeats."
            )
        shaped_dirs = [_resolve_run_dir(sweep_base, t) for t in sh]
        baseline_dirs = [_resolve_run_dir(sweep_base, t) for t in bl]

    if args.fixed_config:
        plot_shaped_vs_baseline_fixed_config(
            sweep_base,
            shaped_run_dirs=shaped_dirs,
            baseline_run_dirs=baseline_dirs,
            map_size=args.map_size,
            config_name=args.fixed_config,
        )
    else:
        if args.plot in ("avg", "all"):
            plot_shaped_vs_baseline(
                sweep_base,
                shaped_run_dirs=shaped_dirs,
                baseline_run_dirs=baseline_dirs,
                map_size=args.map_size,
            )
        if args.plot in ("best-worst", "all"):
            plot_shaped_vs_baseline_best_worst(
                sweep_base,
                shaped_run_dirs=shaped_dirs,
                baseline_run_dirs=baseline_dirs,
                map_size=args.map_size,
            )


if __name__ == "__main__":
    main()
