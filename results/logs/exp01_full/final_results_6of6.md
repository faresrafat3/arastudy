# Exp01 Phase 2C — Final Results (6/6)

Updated: 2026-03-17

## Full Runs Table

| Tokenizer | Run ID | Best Val Loss | Best BPC (corrected) | Best Step | Final Step | Stop Reason | Training Time (h) | Peak VRAM (GB) | Tokens/sec | Total Params |
|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|
| BPE-32K | `night1_bpe_32k` | 4.403984 | 1.703897 | 12500 | 14499 | early_stopping | 3.36 | 2.826 | 39297.0 | 37,100,032 |
| BPE-16K | `night1_bpe_16k` | 4.072959 | 1.729788 | 16500 | 18204 | early_stopping | 0.10* | 3.549 | 46864.5 | 28,908,032 |
| Morph-BPE-16K | `night2_morph_bpe_16k` | 3.476209 | 1.759526 | 10500 | 12499 | early_stopping | 2.26 | 1.956 | 50298.0 | 28,908,032 |
| BPE-8K | `night2_bpe_8k` | 3.841254 | 1.824984 | 10500 | 12499 | early_stopping | 2.15 | 2.611 | 52841.5 | 24,812,032 |
| Morph-BPE-8K | `day1_local_morph8k` | 3.322887 | 1.829851 | 13000 | 14999 | early_stopping | 2.62 | 2.640 | 51584.4 | 24,812,032 |
| Char-Level | `night2_char` | 1.215111 | 2.114531 | 5500 | 7499 | early_stopping | 0.97 | 0.745 | 70864.8 | 20,812,288 |

\* `night1_bpe_16k` reported time is from the final resumed segment in summary JSON.

## Fair Pair Comparison (Key Question)

### BPE-8K vs Morph-BPE-8K (same parameter budget)

| Metric | BPE-8K | Morph-BPE-8K | Better |
|---|---:|---:|---|
| Total Params | 24,812,032 | 24,812,032 | Tie |
| Best Val Loss | 3.841254 | 3.322887 | Morph-BPE-8K |
| Best BPC (corrected) | 1.824984 | 1.829851 | **BPE-8K** |
| Best Step | 10500 | 13000 | BPE-8K (earlier) |
| Avg Tokens/sec | 52841.5 | 51584.4 | BPE-8K |

Interpretation:
- Morph-BPE-8K wins per-token loss.
- BPE-8K wins corrected BPC at the same parameter budget.

## Artifact Index

- Aggregate CSV: `results/logs/exp01_full/final_runs_aggregate_6of6.csv`
- Per-run metrics/generation/summary are in each run directory under `results/logs/exp01_full/`.
