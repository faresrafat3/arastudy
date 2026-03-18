from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_registry(exp: str) -> dict:
    registry_path = ROOT / "experiments/registry.yaml"
    data = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    for item in data.get("experiments", []):
        if item.get("id") == exp:
            return item
    return {"id": exp, "name": exp}


def rows_from_manifest(exp: str) -> list[dict]:
    manifest_path = ROOT / f"runs/{exp}/run_manifest.yaml"
    if not manifest_path.exists():
        return []
    data = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    return list(data.get("runs", []))


def rows_from_logs(exp: str) -> list[dict]:
    log_root = ROOT / f"results/logs/{exp}"
    if not log_root.exists():
        return []

    rows: list[dict] = []
    for summary_path in sorted(log_root.glob("*/*_summary.json")):
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        tokenizer = str(summary.get("tokenizer_id", ""))
        run_id = str(summary.get("run_id", ""))
        metrics_path = summary_path.with_name(f"{tokenizer}_metrics.csv")
        if not metrics_path.exists():
            continue

        metrics = []
        with metrics_path.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if str(row.get("step", "")).isdigit():
                    metrics.append(row)
        if not metrics:
            continue

        eval_rows = [m for m in metrics if str(m.get("val_loss", "")).strip()]
        if not eval_rows:
            continue

        best = min(eval_rows, key=lambda r: float(r["val_loss"]))
        tps_vals = [
            float(m["tokens_per_sec"])
            for m in metrics
            if str(m.get("tokens_per_sec", "")).strip()
        ]

        seed = 42
        if "_s" in run_id:
            try:
                seed = int(run_id.rsplit("_s", 1)[1])
            except ValueError:
                seed = 42

        rows.append(
            {
                "id": f"{tokenizer}_s{seed}",
                "tokenizer": tokenizer,
                "seed": seed,
                "best_val_loss": float(best["val_loss"]),
                "best_bpc": float(best["bpc"]),
                "best_step": int(best["step"]),
                "final_step": int(metrics[-1]["step"]),
                "training_time_h": float(summary.get("elapsed_sec", 0.0)) / 3600.0,
                "peak_vram_gb": float(summary.get("peak_vram_gb", 0.0)),
                "tokens_per_sec": sum(tps_vals) / max(len(tps_vals), 1),
                "total_params": 0,
                "hardware": "RTX_4050_or_Colab",
                "metrics_csv": str(metrics_path.relative_to(ROOT)),
            }
        )

    return rows


def load_runs(exp: str) -> list[dict]:
    rows = rows_from_manifest(exp)
    if rows:
        return rows
    return rows_from_logs(exp)


def save_comparison_csv(exp: str, rows: list[dict]) -> Path:
    out_path = ROOT / f"runs/{exp}/comparison_table.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
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
        "tokens_per_sec",
        "total_params",
        "hardware",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "run_id": row["id"],
                    "tokenizer": row["tokenizer"],
                    "seed": row["seed"],
                    "best_val_loss": row["best_val_loss"],
                    "best_bpc": row["best_bpc"],
                    "best_step": row["best_step"],
                    "final_step": row.get("final_step", ""),
                    "training_time_h": row["training_time_h"],
                    "peak_vram_gb": row["peak_vram_gb"],
                    "tokens_per_sec": row["tokens_per_sec"],
                    "total_params": row.get("total_params", 0),
                    "hardware": row.get("hardware", ""),
                }
            )
    return out_path


def markdown_table(rows: list[dict]) -> str:
    lines = [
        "| Run | Tokenizer | Seed | Val Loss | BPC | Best Step | Tok/s | VRAM (GB) |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {id} | {tok} | {seed} | {vl:.6f} | {bpc:.6f} | {bs} | {tps:.2f} | {vram:.4f} |".format(
                id=row["id"],
                tok=row["tokenizer"],
                seed=row["seed"],
                vl=float(row["best_val_loss"]),
                bpc=float(row["best_bpc"]),
                bs=row["best_step"],
                tps=float(row["tokens_per_sec"]),
                vram=float(row["peak_vram_gb"]),
            )
        )
    return "\n".join(lines)


def grouped_stats(rows: list[dict], metric: str) -> dict[str, dict[str, float]]:
    grp: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grp[row["tokenizer"]].append(float(row[f"best_{metric}"]))

    out: dict[str, dict[str, float]] = {}
    for tok, values in grp.items():
        mean = statistics.mean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0.0
        out[tok] = {"mean": mean, "std": std, "n": float(len(values))}
    return out


def bootstrap_pvalue(
    a: list[float], b: list[float], n_boot: int = 5000
) -> tuple[float, float, float]:
    if not a or not b:
        return float("nan"), float("nan"), float("nan")

    random.seed(42)
    diffs = []
    for _ in range(n_boot):
        sa = [random.choice(a) for _ in range(len(a))]
        sb = [random.choice(b) for _ in range(len(b))]
        diffs.append(statistics.mean(sa) - statistics.mean(sb))

    diffs_sorted = sorted(diffs)
    lo = diffs_sorted[int(0.025 * len(diffs_sorted))]
    hi = diffs_sorted[int(0.975 * len(diffs_sorted))]
    p_left = sum(1 for d in diffs if d <= 0) / len(diffs)
    p_right = sum(1 for d in diffs if d >= 0) / len(diffs)
    p_two = 2 * min(p_left, p_right)
    return lo, hi, min(p_two, 1.0)


def significance_report(rows: list[dict], metric: str) -> str:
    grp: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grp[row["tokenizer"]].append(float(row[f"best_{metric}"]))

    toks = sorted(grp.keys())
    lines = [
        "## Bootstrap significance (difference of means)",
        "",
        "| A | B | 95% CI (A-B) | p-value |",
        "|---|---|---|---:|",
    ]
    for i in range(len(toks)):
        for j in range(i + 1, len(toks)):
            a, b = toks[i], toks[j]
            lo, hi, p = bootstrap_pvalue(grp[a], grp[b])
            lines.append(f"| {a} | {b} | [{lo:.6f}, {hi:.6f}] | {p:.4f} |")
    return "\n".join(lines)


def summary_markdown(stats: dict[str, dict[str, float]], metric: str) -> str:
    lines = [
        f"## Tokenizer summary ({metric})",
        "",
        "| Tokenizer | Mean ± Std | n |",
        "|---|---:|---:|",
    ]
    for tok, s in sorted(stats.items(), key=lambda kv: kv[1]["mean"]):
        lines.append(f"| {tok} | {s['mean']:.6f} ± {s['std']:.6f} | {int(s['n'])} |")
    return "\n".join(lines)


def plot_metric(exp: str, rows: list[dict], metric: str) -> Path:
    figures_dir = ROOT / f"runs/{exp}/figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    out = figures_dir / f"{metric}_comparison.png"

    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[row["tokenizer"]].append(float(row[f"best_{metric}"]))

    labels = sorted(grouped.keys(), key=lambda k: statistics.mean(grouped[k]))
    means = [statistics.mean(grouped[k]) for k in labels]
    stds = [
        statistics.stdev(grouped[k]) if len(grouped[k]) > 1 else 0.0 for k in labels
    ]

    plt.figure(figsize=(9, 4.8))
    bars = plt.bar(labels, means, yerr=stds, capsize=4)
    plt.ylabel(metric.upper())
    plt.title(f"{exp.upper()} {metric.upper()} mean ± std")
    plt.xticks(rotation=20)
    for bar, val in zip(bars, means):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            val,
            f"{val:.3f}",
            ha="center",
            va="bottom",
        )
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare runs inside one experiment")
    parser.add_argument("--exp", required=True, help="Experiment id, e.g. exp01")
    parser.add_argument(
        "--metric",
        default="bpc",
        choices=["bpc", "val_loss"],
        help="Primary ranking metric",
    )
    parser.add_argument("--plot", action="store_true", help="Generate plots")
    args = parser.parse_args()

    registry_item = load_registry(args.exp)
    rows = load_runs(args.exp)
    if not rows:
        raise RuntimeError(
            f"No runs found for {args.exp}. Check runs manifest or results/logs/{args.exp}."
        )

    metric_key = f"best_{args.metric}"
    rows_sorted = sorted(rows, key=lambda r: float(r[metric_key]))

    comparison_csv = save_comparison_csv(args.exp, rows_sorted)
    table = markdown_table(rows_sorted)
    stats = grouped_stats(rows_sorted, args.metric)
    stats_md = summary_markdown(stats, args.metric)
    sig_md = significance_report(rows_sorted, args.metric)

    analysis_dir = ROOT / f"runs/{args.exp}/analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    out_md = analysis_dir / f"comparison_{args.metric}.md"
    out_md.write_text(
        f"# {registry_item.get('name', args.exp)}\n\n"
        f"Ranking metric: `{args.metric}`\n\n"
        + table
        + "\n\n"
        + stats_md
        + "\n\n"
        + sig_md
        + "\n",
        encoding="utf-8",
    )

    print(table)
    print("\n" + stats_md)
    print("\n" + sig_md)
    print(f"\nSaved CSV: {comparison_csv}")
    print(f"Saved Markdown: {out_md}")

    if args.plot:
        metric_plot = plot_metric(args.exp, rows_sorted, args.metric)
        print(f"Saved plot: {metric_plot}")


if __name__ == "__main__":
    main()
