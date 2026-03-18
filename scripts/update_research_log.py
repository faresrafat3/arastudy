from __future__ import annotations

import argparse
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def append_entry(run: dict) -> None:
    log_path = ROOT / "RESEARCH_LOG.md"
    tag = f"{run['experiment']}::{run['id']}"
    current = log_path.read_text(encoding="utf-8") if log_path.exists() else ""
    if tag in current:
        print(f"Entry already exists for {tag}")
        return

    entry = (
        "\n---\n\n"
        f"### Auto Log — {run['experiment']} ({run['id']})\n\n"
        f"Tag: `{tag}`\n\n"
        f"- Tokenizer: `{run['tokenizer']}`\n"
        f"- Seed: `{run['seed']}`\n"
        f"- Best Val Loss: `{run['best_val_loss']}`\n"
        f"- Best BPC: `{run['best_bpc']}`\n"
        f"- Best Step: `{run['best_step']}`\n"
        f"- Training Time (h): `{run['training_time_h']}`\n"
        f"- Peak VRAM (GB): `{run['peak_vram_gb']}`\n"
        f"- Tokens/sec: `{run['tokens_per_sec']}`\n"
        f"- Metrics: `{run['metrics_csv']}`\n"
    )
    log_path.write_text(current + entry, encoding="utf-8")
    print(f"Appended {tag} to RESEARCH_LOG.md")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Append one run from manifest to RESEARCH_LOG"
    )
    parser.add_argument("--exp", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    manifest_path = ROOT / f"runs/{args.exp}/run_manifest.yaml"
    data = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    for run in data.get("runs", []):
        if run.get("id") == args.run_id:
            run = dict(run)
            run["experiment"] = args.exp
            append_entry(run)
            return
    raise RuntimeError(f"Run id {args.run_id} not found in {manifest_path}")


if __name__ == "__main__":
    main()
