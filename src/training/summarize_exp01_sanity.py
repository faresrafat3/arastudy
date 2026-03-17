import argparse
from pathlib import Path

from omegaconf import OmegaConf


def parse_summary(path: Path) -> dict[str, str]:
    data: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            data[k.strip()] = v.strip()
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize exp01 sanity runs")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiments/exp01_sanity_train.yaml",
    )
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    log_dir = Path(cfg.paths.log_dir)
    rows = []

    for tok_cfg in cfg.tokenizers:
        tok_id = tok_cfg.id
        label = tok_cfg.label
        summary_path = log_dir / f"{tok_id}_summary.txt"
        if not summary_path.exists():
            continue
        info = parse_summary(summary_path)
        rows.append(
            {
                "tokenizer": label,
                "id": tok_id,
                "best_val_loss": float(info.get("best_val_loss", "inf")),
                "vocab_size": int(float(info.get("vocab_size", "0"))),
                "total_params": int(float(info.get("total_params", "0"))),
                "tokens_seen": int(float(info.get("tokens_seen", "0"))),
                "elapsed_sec": float(info.get("elapsed_sec", "0")),
            }
        )

    rows = sorted(rows, key=lambda x: x["best_val_loss"])
    output = log_dir / "sanity_leaderboard.md"
    with open(output, "w", encoding="utf-8") as handle:
        handle.write(
            "| Rank | Tokenizer | Best Val Loss | Vocab | Params | "
            "Tokens Seen | Time (s) |\n"
        )
        handle.write("|---:|---|---:|---:|---:|---:|---:|\n")
        for idx, row in enumerate(rows, start=1):
            handle.write(
                f"| {idx} | {row['tokenizer']} | {row['best_val_loss']:.6f} | "
                f"{row['vocab_size']} | {row['total_params']} | {row['tokens_seen']} | "
                f"{row['elapsed_sec']:.2f} |\n"
            )

    print(f"[exp01-sanity] leaderboard={output}")


if __name__ == "__main__":
    main()
