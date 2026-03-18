from __future__ import annotations

import argparse
import csv
import json
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
        f"- Final Step: `{run.get('final_step', '')}`\n"
        f"- Training Time (h): `{run['training_time_h']}`\n"
        f"- Peak VRAM (GB): `{run['peak_vram_gb']}`\n"
        f"- Tokens/sec: `{run['tokens_per_sec']}`\n"
        f"- Metrics: `{run['metrics_csv']}`\n"
    )
    log_path.write_text(current + entry, encoding="utf-8")
    print(f"Appended {tag} to RESEARCH_LOG.md")


def upsert_manifest_run(run: dict) -> None:
    exp = run["experiment"]
    manifest_path = ROOT / f"runs/{exp}/run_manifest.yaml"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    if manifest_path.exists():
        data = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
    else:
        data = {}

    runs = data.get("runs", [])
    idx = next((i for i, r in enumerate(runs) if r.get("id") == run["id"]), None)
    if idx is None:
        runs.append(run)
    else:
        runs[idx] = run

    data["runs"] = runs
    manifest_path.write_text(
        yaml.safe_dump(data, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    print(f"Updated manifest: {manifest_path}")


def resolve_run_from_manifest(exp: str, run_id: str) -> dict:
    manifest_path = ROOT / f"runs/{exp}/run_manifest.yaml"
    data = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    for run in data.get("runs", []):
        if run.get("id") == run_id:
            run = dict(run)
            run["experiment"] = exp
            return run
    raise RuntimeError(f"Run id {run_id} not found in {manifest_path}")


def parse_summary_mode(exp: str, tokenizer: str, seed: int) -> dict:
    run_id = f"{exp}_{tokenizer}_s{seed}"
    log_dir = ROOT / f"results/logs/{exp}/{run_id}"

    summary_path = log_dir / f"{tokenizer}_summary.json"
    metrics_path = log_dir / f"{tokenizer}_metrics.csv"
    generation_path = log_dir / f"{tokenizer}_generation.md"

    if not summary_path.exists() or not metrics_path.exists():
        raise FileNotFoundError(
            f"Expected files missing for {run_id}: {summary_path} / {metrics_path}"
        )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    rows = []
    with metrics_path.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if str(row.get("step", "")).isdigit():
                rows.append(row)

    if not rows:
        raise RuntimeError(f"No numeric rows found in {metrics_path}")

    eval_rows = [r for r in rows if str(r.get("val_loss", "")).strip()]
    best_row = min(eval_rows, key=lambda r: float(r["val_loss"]))
    final_row = rows[-1]

    tps = [
        float(r["tokens_per_sec"])
        for r in rows
        if str(r.get("tokens_per_sec", "")).strip()
    ]
    avg_tps = sum(tps) / max(len(tps), 1)

    vocab_map = {
        "bpe_32k": 32000,
        "bpe_16k": 16000,
        "morph_bpe_16k": 16000,
        "bpe_8k": 8000,
        "morph_bpe_8k": 8000,
        "char": 188,
    }
    shared = 20_716_032
    total_params = shared + vocab_map.get(tokenizer, 0) * 512

    return {
        "id": f"{tokenizer}_s{seed}",
        "experiment": exp,
        "tokenizer": tokenizer,
        "seed": seed,
        "status": "completed",
        "best_val_loss": float(best_row["val_loss"]),
        "best_bpc": float(best_row["bpc"])
        if str(best_row.get("bpc", "")).strip()
        else float(summary.get("best_bpc", 0.0)),
        "best_step": int(best_row["step"]),
        "final_step": int(final_row["step"]),
        "training_time_h": float(summary.get("elapsed_sec", 0.0)) / 3600.0,
        "peak_vram_gb": float(summary.get("peak_vram_gb", 0.0)),
        "tokens_per_sec": float(avg_tps),
        "total_params": int(total_params),
        "hardware": "RTX_4050_or_Colab",
        "checkpoint": f"results/checkpoints/{exp}/{run_id}/{tokenizer}/best.pt",
        "metrics_csv": str(metrics_path.relative_to(ROOT)),
        "generation_md": str(generation_path.relative_to(ROOT))
        if generation_path.exists()
        else "",
        "hf_model": "",
        "notes": "",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Append run results to RESEARCH_LOG and manifest"
    )
    parser.add_argument("--exp", required=True)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--tokenizer", default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    if args.run_id:
        run = resolve_run_from_manifest(args.exp, args.run_id)
        append_entry(run)
        return

    if args.tokenizer is None or args.seed is None:
        raise RuntimeError("Provide either --run-id OR both --tokenizer and --seed")

    run = parse_summary_mode(args.exp, args.tokenizer, int(args.seed))
    upsert_manifest_run(run)
    append_entry(run)


if __name__ == "__main__":
    main()
