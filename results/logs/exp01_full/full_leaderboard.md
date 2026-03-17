# Exp01 Phase 2C Full Runs Leaderboard

Updated: 2026-03-17

## Completed Runs (Corrected BPC)

| Rank (by corrected BPC) | Tokenizer | Run ID | Best Val Loss | Best BPC (corrected) | Best Step | Final Step | Stop Reason | Training Time (h) | Peak VRAM (GB) | Avg Tokens/sec |
|---|---|---|---:|---:|---:|---:|---|---:|---:|---:|
| 1 | BPE-32K | `night1_bpe_32k` | 4.403984 | 1.703897 | 12500 | 14499 | Early stopping | 3.36 | 2.826 | 39266.8 |
| 2 | BPE-16K | `night1_bpe_16k` | 4.072959 | 1.729788 | 16500 | 18204 | Early stopping | 0.10* | 3.549 | 46470.2 |
| 3 | Morph-BPE-8K | `day1_local_morph8k` | 3.322887 | 1.829851 | 13000 | 14999 | Early stopping | 2.62 | 2.640 | 52181.0 |

\* `night1_bpe_16k` time in this table reflects the latest resumed segment in summary JSON, not total wall-clock across all resumes.

## Pending Full Runs

| Tokenizer | Planned Run ID | Status |
|---|---|---|
| BPE-8K | `day1_colab_bpe8k` | Not found in `results/logs/exp01_full` |
| Morph-BPE-16K | `day2_colab_morph16k` | Not found in `results/logs/exp01_full` |
| Char-Level | `day3_colab_char` | Not found in `results/logs/exp01_full` |

## Artifact Paths

- Aggregate CSV: `results/logs/exp01_full/final_runs_aggregate.csv`
- Morph-BPE-8K metrics: `results/logs/exp01_full/day1_local_morph8k/morph_bpe_8k_metrics.csv`
- BPE-16K metrics: `results/logs/exp01_full/night1_bpe_16k/bpe_16k_metrics.csv`
- BPE-32K metrics: `results/logs/exp01_full/night1_bpe_32k/bpe_32k_metrics.csv`
- Morph-BPE-8K samples: `results/logs/exp01_full/day1_local_morph8k/morph_bpe_8k_generation.md`
- BPE-16K samples: `results/logs/exp01_full/night1_bpe_16k/bpe_16k_generation.md`
- BPE-32K samples: `results/logs/exp01_full/night1_bpe_32k/bpe_32k_generation.md`
