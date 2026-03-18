from __future__ import annotations

import argparse
import csv
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
    raise ValueError(f"Experiment {exp} not found in {registry_path}")


def load_runs(exp: str) -> list[dict]:
    manifest_path = ROOT / f"runs/{exp}/run_manifest.yaml"
    data = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    return data.get("runs", [])


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
                    "total_params": row["total_params"],
                    "hardware": row.get("hardware", ""),
                }
            )
    return out_path


def markdown_table(rows: list[dict]) -> str:
    lines = [
        "| Run | Tokenizer | Val Loss | BPC | Best Step | Tok/s | VRAM (GB) | Params |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {id} | {tok} | {vl:.6f} | {bpc:.6f} | {bs} | {tps:.2f} | {vram:.4f} | {params} |".format(
                id=row["id"],
                tok=row["tokenizer"],
                vl=float(row["best_val_loss"]),
                bpc=float(row["best_bpc"]),
                bs=row["best_step"],
                tps=float(row["tokens_per_sec"]),
                vram=float(row["peak_vram_gb"]),
                params=row["total_params"],
            )
        )
    return "\n".join(lines)


def plot_metric(exp: str, rows: list[dict], metric: str) -> Path:
    figures_dir = ROOT / f"runs/{exp}/figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    out = figures_dir / f"{metric}_comparison.png"

    labels = [r["tokenizer"] for r in rows]
    values = [float(r[f"best_{metric}"]) for r in rows]

    plt.figure(figsize=(9, 4.8))
    bars = plt.bar(labels, values)
    plt.ylabel(metric.upper())
    plt.title(f"{exp.upper()} {metric.upper()} comparison")
    plt.xticks(rotation=20)
    for bar, val in zip(bars, values):
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


def plot_training_curves(exp: str, rows: list[dict]) -> Path:
    figures_dir = ROOT / f"runs/{exp}/figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    out = figures_dir / "training_curves.png"

    plt.figure(figsize=(9, 5))
    for row in rows:
        metrics_path = ROOT / row["metrics_csv"]
        if not metrics_path.exists():
            continue
        steps: list[int] = []
        val_losses: list[float] = []
        with metrics_path.open(encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if not str(r.get("step", "")).isdigit():
                    continue
                v = str(r.get("val_loss", "")).strip()
                if not v:
                    continue
                steps.append(int(r["step"]))
                val_losses.append(float(v))
        if steps and val_losses:
            plt.plot(steps, val_losses, label=row["tokenizer"])

    plt.xlabel("Step")
    plt.ylabel("Validation loss")
    plt.title(f"{exp.upper()} validation curves")
    plt.legend(fontsize=8)
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
    metric_key = f"best_{args.metric}"
    rows = sorted(rows, key=lambda r: float(r[metric_key]))

    comparison_csv = save_comparison_csv(args.exp, rows)
    table = markdown_table(rows)

    analysis_dir = ROOT / f"runs/{args.exp}/analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    out_md = analysis_dir / f"comparison_{args.metric}.md"
    out_md.write_text(
        f"# {registry_item.get('name', args.exp)}\n\n"
        f"Ranking metric: `{args.metric}`\n\n" + table + "\n",
        encoding="utf-8",
    )

    print(table)
    print(f"\nSaved CSV: {comparison_csv}")
    print(f"Saved Markdown: {out_md}")

    if args.plot:
        metric_plot = plot_metric(args.exp, rows, args.metric)
        curves_plot = plot_training_curves(args.exp, rows)
        print(f"Saved plot: {metric_plot}")
        print(f"Saved plot: {curves_plot}")


if __name__ == "__main__":
    main()
