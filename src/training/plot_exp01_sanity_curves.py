import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from omegaconf import OmegaConf


def read_metrics(path: Path) -> tuple[list[int], list[float], list[tuple[int, float]]]:
    steps: list[int] = []
    train_losses: list[float] = []
    val_points: list[tuple[int, float]] = []

    with open(path, encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            step = int(row["step"])
            train_loss = float(row["train_loss"])
            steps.append(step)
            train_losses.append(train_loss)

            val_raw = row.get("val_loss", "")
            if val_raw:
                val_points.append((step, float(val_raw)))

    return steps, train_losses, val_points


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot exp01 sanity training curves")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiments/exp01_sanity_train.yaml",
    )
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    log_dir = Path(cfg.paths.log_dir)
    out_dir = log_dir / "curves"
    out_dir.mkdir(parents=True, exist_ok=True)

    combined_fig = plt.figure(figsize=(10, 6))
    combined_ax = combined_fig.add_subplot(1, 1, 1)

    for tok in cfg.tokenizers:
        tok_id = tok.id
        label = tok.label
        metrics_path = log_dir / f"{tok_id}_metrics.csv"
        if not metrics_path.exists():
            continue

        steps, train_losses, val_points = read_metrics(metrics_path)
        if not steps:
            continue

        fig = plt.figure(figsize=(8, 5))
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(steps, train_losses, label="train_loss", linewidth=1.5)
        if val_points:
            vx = [x for x, _ in val_points]
            vy = [y for _, y in val_points]
            ax.plot(vx, vy, label="val_loss", marker="o", linewidth=1.2)
        ax.set_title(f"Sanity Curve - {label}")
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"{tok_id}_curve.png", dpi=150)
        plt.close(fig)

        combined_ax.plot(steps, train_losses, label=label, linewidth=1.3)

    combined_ax.set_title("Sanity Train Loss Curves (All Tokenizers)")
    combined_ax.set_xlabel("Step")
    combined_ax.set_ylabel("Train Loss")
    combined_ax.grid(alpha=0.3)
    combined_ax.legend(fontsize=8)
    combined_fig.tight_layout()
    combined_fig.savefig(out_dir / "all_train_curves.png", dpi=150)
    plt.close(combined_fig)

    print(f"[exp01-sanity] curves_dir={out_dir}")


if __name__ == "__main__":
    main()
