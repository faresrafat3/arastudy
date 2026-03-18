from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def normalize_tokenizer(tokenizer: str) -> str:
    return tokenizer.replace("morph_8k", "morph_bpe_8k").replace("morph_16k", "morph_bpe_16k")


def load_runs(exp: str) -> list[dict]:
    run_root = ROOT / f"results/{exp}"
    if not run_root.exists():
        return []

    rows: list[dict] = []
    for summary_path in sorted(run_root.glob("*/summary.json")):
        run_dir = summary_path.parent
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        metrics_path = run_dir / "metrics.csv"
        if not metrics_path.exists():
            continue

        metrics = []
        with metrics_path.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if str(row.get("step", "")).isdigit():
                    metrics.append(row)
        if not metrics:
            continue

        eval_rows = [row for row in metrics if str(row.get("val_loss", "")).strip()]
        if not eval_rows:
            continue

        best = min(eval_rows, key=lambda r: float(r["val_loss"]))
        tps_values = [
            float(row["tokens_sec"])
            for row in metrics
            if str(row.get("tokens_sec", "")).strip()
        ]

        run_id = str(summary.get("run_id", run_dir.name))
        tokenizer = normalize_tokenizer(str(summary.get("tokenizer", "unknown")))

        rows.append(
            {
                "run_id": run_id,
                "tokenizer": tokenizer,
                "seed": int(summary.get("seed", 42)),
                "best_val_loss": float(summary.get("best_val_loss", best["val_loss"])),
                "best_bpc": float(summary.get("best_bpc", best.get("bpc") or 0.0)),
                "best_step": int(summary.get("best_step", best["step"])),
                "final_step": int(summary.get("final_step", metrics[-1]["step"])),
                "training_time_h": float(summary.get("training_time_h", 0.0)),
                "peak_vram_gb": float(summary.get("peak_vram_gb", 0.0)),
                "tokens_sec": sum(tps_values) / max(len(tps_values), 1),
                "total_params": int(summary.get("total_params", 0)),
                "hardware": str(summary.get("hardware", "unknown")),
                "summary_json": str(summary_path.relative_to(ROOT)),
                "metrics_csv": str(metrics_path.relative_to(ROOT)),
            }
        )

    return rows


def save_per_run_table(exp: str, rows: list[dict]) -> Path:
    out = ROOT / f"experiments/{exp}/comparison_table.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "run_id",
        "tokenizer",
        "seed",
        "best_val_loss",
        "best_bpc",
        "best_step",
        "final_step",
        "training_time_h",
        "peak_vram_gb",
        "tokens_sec",
        "total_params",
        "hardware",
        "summary_json",
        "metrics_csv",
    ]
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    return out


def grouped_stats(rows: list[dict], metric: str) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[row["tokenizer"]].append(float(row[f"best_{metric}"]))

    stats: dict[str, dict[str, float]] = {}
    for tok, values in grouped.items():
        mean = statistics.mean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0.0
        stats[tok] = {"mean": mean, "std": std, "n": float(len(values))}
    return stats


def save_mean_std_table(exp: str, metric: str, stats: dict[str, dict[str, float]]) -> Path:
    out = ROOT / f"experiments/{exp}/mean_std_table.csv"
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["tokenizer", "metric", "mean", "std", "n"])
        w.writeheader()
        for tok, values in sorted(stats.items(), key=lambda kv: kv[1]["mean"]):
            w.writerow(
                {
                    "tokenizer": tok,
                    "metric": metric,
                    "mean": values["mean"],
                    "std": values["std"],
                    "n": int(values["n"]),
                }
            )
    return out


def bootstrap_pvalue(a: list[float], b: list[float], n_boot: int = 5000) -> tuple[float, float, float]:
    if not a or not b:
        return float("nan"), float("nan"), float("nan")

    random.seed(42)
    diffs = []
    for _ in range(n_boot):
        sa = [random.choice(a) for _ in range(len(a))]
        sb = [random.choice(b) for _ in range(len(b))]
        diffs.append(statistics.mean(sa) - statistics.mean(sb))

    diffs.sort()
    lo = diffs[int(0.025 * len(diffs))]
    hi = diffs[int(0.975 * len(diffs))]
    p_left = sum(1 for d in diffs if d <= 0) / len(diffs)
    p_right = sum(1 for d in diffs if d >= 0) / len(diffs)
    p_two = min(1.0, 2 * min(p_left, p_right))
    return lo, hi, p_two


def save_significance_table(exp: str, rows: list[dict], metric: str) -> Path:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[row["tokenizer"]].append(float(row[f"best_{metric}"]))

    toks = sorted(grouped.keys())
    out = ROOT / f"experiments/{exp}/significance_test.csv"
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["tokenizer_a", "tokenizer_b", "ci_low", "ci_high", "p_value"])
        w.writeheader()
        for i in range(len(toks)):
            for j in range(i + 1, len(toks)):
                a, b = toks[i], toks[j]
                lo, hi, p = bootstrap_pvalue(grouped[a], grouped[b])
                w.writerow(
                    {
                        "tokenizer_a": a,
                        "tokenizer_b": b,
                        "ci_low": lo,
                        "ci_high": hi,
                        "p_value": p,
                    }
                )
    return out


def save_latex_table(exp: str, metric: str, stats: dict[str, dict[str, float]]) -> Path:
    out = ROOT / f"experiments/{exp}/paper_table_{metric}.tex"
    lines = [
        "\\begin{tabular}{lcc}",
        "\\hline",
        r"Tokenizer & Mean & Std \\",
        "\\hline",
    ]
    for tok, values in sorted(stats.items(), key=lambda kv: kv[1]["mean"]):
        lines.append(rf"{tok} & {values['mean']:.6f} & {values['std']:.6f} \\")
    lines += ["\\hline", "\\end{tabular}"]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def plot_metric(exp: str, rows: list[dict], metric: str) -> Path:
    figures_dir = ROOT / f"experiments/{exp}/figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    out = figures_dir / f"{metric}_comparison.png"

    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[row["tokenizer"]].append(float(row[f"best_{metric}"]))

    labels = sorted(grouped.keys(), key=lambda k: statistics.mean(grouped[k]))
    means = [statistics.mean(grouped[k]) for k in labels]
    stds = [statistics.stdev(grouped[k]) if len(grouped[k]) > 1 else 0.0 for k in labels]

    plt.figure(figsize=(9, 4.8))
    bars = plt.bar(labels, means, yerr=stds, capsize=4)
    plt.ylabel(metric.upper())
    plt.title(f"{exp.upper()} {metric.upper()} mean ± std")
    plt.xticks(rotation=20)
    for bar, val in zip(bars, means):
        plt.text(bar.get_x() + bar.get_width() / 2, val, f"{val:.3f}", ha="center", va="bottom")
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare runs for one experiment")
    parser.add_argument("--exp", required=True, help="Experiment folder under results/, e.g. exp01")
    parser.add_argument("--metric", default="bpc", choices=["bpc", "val_loss"])
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()

    rows = load_runs(args.exp)
    if not rows:
        raise RuntimeError(f"No completed runs found under results/{args.exp}/")

    metric_key = f"best_{args.metric}"
    sorted_rows = sorted(rows, key=lambda row: float(row[metric_key]))

    per_run = save_per_run_table(args.exp, sorted_rows)
    stats = grouped_stats(sorted_rows, args.metric)
    mean_std = save_mean_std_table(args.exp, args.metric, stats)
    sig = save_significance_table(args.exp, sorted_rows, args.metric)
    tex = save_latex_table(args.exp, args.metric, stats)

    print(f"Saved per-run table: {per_run}")
    print(f"Saved mean±std table: {mean_std}")
    print(f"Saved bootstrap significance: {sig}")
    print(f"Saved paper-ready LaTeX table: {tex}")

    if args.plot:
        fig = plot_metric(args.exp, sorted_rows, args.metric)
        print(f"Saved plot: {fig}")


if __name__ == "__main__":
    main()
