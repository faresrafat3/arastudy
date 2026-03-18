from __future__ import annotations

import argparse
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_runs(exp: str) -> list[dict]:
    p = ROOT / f"runs/{exp}/run_manifest.yaml"
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    rows = data.get("runs", [])
    return sorted(rows, key=lambda r: float(r["best_bpc"]))


def markdown_table(rows: list[dict]) -> str:
    lines = [
        "| Tokenizer | Params | Val Loss | BPC | Best Step | Tok/s | Peak VRAM (GB) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            "| {tok} | {params} | {vl:.3f} | {bpc:.3f} | {step} | {tps:.0f} | {vram:.3f} |".format(
                tok=r["tokenizer"],
                params=r["total_params"],
                vl=float(r["best_val_loss"]),
                bpc=float(r["best_bpc"]),
                step=r["best_step"],
                tps=float(r["tokens_per_sec"]),
                vram=float(r["peak_vram_gb"]),
            )
        )
    return "\n".join(lines)


def latex_table(rows: list[dict]) -> str:
    header = (
        "\\begin{table}[t]\n"
        "\\centering\n"
        "\\small\n"
        "\\begin{tabular}{lrrrrrr}\n"
        "\\toprule\n"
        "Tokenizer & Params & Val Loss & BPC & Best Step & Tok/s & VRAM (GB) \\\\\n"
        "\\midrule\n"
    )
    body = ""
    for r in rows:
        body += (
            f"{r['tokenizer']} & {r['total_params']} & {float(r['best_val_loss']):.3f} & "
            f"{float(r['best_bpc']):.3f} & {r['best_step']} & {float(r['tokens_per_sec']):.0f} & "
            f"{float(r['peak_vram_gb']):.3f} \\\\n"
        )
    footer = (
        "\\bottomrule\n"
        "\\end{tabular}\n"
        "\\caption{AraStudy Exp results sorted by corrected BPC (lower is better).}\n"
        "\\label{tab:arastudy-exp-results}\n"
        "\\end{table}\n"
    )
    return header + body + footer


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate paper tables for one experiment"
    )
    parser.add_argument("--exp", required=True, help="Experiment id, e.g. exp01")
    args = parser.parse_args()

    rows = load_runs(args.exp)
    analysis_dir = ROOT / f"runs/{args.exp}/analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    md_path = analysis_dir / "paper_table.md"
    tex_path = analysis_dir / "paper_table.tex"

    md_path.write_text(markdown_table(rows) + "\n", encoding="utf-8")
    tex_path.write_text(latex_table(rows), encoding="utf-8")

    print(f"Saved Markdown table: {md_path}")
    print(f"Saved LaTeX table: {tex_path}")


if __name__ == "__main__":
    main()
