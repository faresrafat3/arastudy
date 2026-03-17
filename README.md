# AraStudy

Ultra-small Arabic language modeling with a strict research-first pipeline.

Current milestone: **Exp01 complete (6/6 runs)** — controlled tokenizer study for Arabic tiny LMs on RTX 4050.

## TL;DR

- We trained the same tiny decoder-only Transformer with **6 tokenizer strategies**.
- We report both token-level and character-level metrics.
- Main finding: **raw val loss alone is misleading across tokenizers**.
- For fair cross-tokenizer comparison, use **corrected BPC**:

$$
BPC = \log_2(PPL_{token}) \times tokens\_per\_char
$$

## Final Exp01 Results (6/6)

Sorted by corrected BPC (lower is better):

| Tokenizer | Params | Best Val Loss | Best BPC | Best Step | Tokens/s | Peak VRAM |
|---|---:|---:|---:|---:|---:|---:|
| BPE-32K | 37.1M | 4.404 | 1.704 | 12500 | 39297 | 2.83 GB |
| BPE-16K | 28.9M | 4.073 | 1.730 | 16500 | 46865 | 3.55 GB |
| Morph-BPE-16K | 28.9M | 3.476 | 1.760 | 10500 | 50298 | 1.96 GB |
| BPE-8K | 24.8M | 3.841 | 1.825 | 10500 | 52842 | 2.61 GB |
| Morph-BPE-8K | 24.8M | 3.323 | 1.830 | 13000 | 51584 | 2.64 GB |
| Char-Level | 20.8M | 1.215 | 2.115 | 5500 | 70865 | 0.75 GB |

## Fair Pair Comparisons (same parameter budget)

### 8K Pair (24.8M vs 24.8M)

- **Val loss:** Morph-BPE-8K better (~13.5%).
- **Corrected BPC:** near tie; BPE-8K slightly better (~0.27%).
- **Interpretation:** Morph improves per-token prediction, but higher tokens/char offsets part of the gain in BPC.

### 16K Pair (28.9M vs 28.9M)

- **Val loss:** Morph-BPE-16K better (~14.7%).
- **Corrected BPC:** BPE-16K better (~1.7%).
- **Interpretation:** segmentation helps token-level modeling but does not guarantee better char-level compression.

## Generation Samples

All generation dumps are available in:

- `results/logs/exp01_full/night1_bpe_32k/bpe_32k_generation.md`
- `results/logs/exp01_full/night1_bpe_16k/bpe_16k_generation.md`
- `results/logs/exp01_full/night2_morph_bpe_16k/morph_bpe_16k_generation.md`
- `results/logs/exp01_full/night2_bpe_8k/bpe_8k_generation.md`
- `results/logs/exp01_full/day1_local_morph8k/morph_bpe_8k_generation.md`
- `results/logs/exp01_full/night2_char/char_generation.md`

## Key Artifacts

- Final aggregate CSV: `results/logs/exp01_full/final_runs_aggregate_6of6.csv`
- Final report (markdown): `results/logs/exp01_full/final_results_6of6.md`
- Leaderboard snapshot: `results/logs/exp01_full/full_leaderboard.md`
- Tokenizer stats (phase2b): `results/logs/phase2b/tokenizer_stats.csv`

## Project Structure

```text
configs/        # experiment/training yaml configs
scripts/        # run/supervision helpers
src/data/       # data cleaning + tokenization pipelines
src/models/     # transformer model implementation
src/training/   # sanity/full training loops + summaries
results/logs/   # metrics, generation outputs, aggregated reports
```

## Quick Start

### 1) Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Run tokenizer phase (Exp01)

```bash
bash scripts/run_exp01_tokenization.sh
```

### 3) Run full training queue (local)

```bash
bash scripts/run_overnight_local_queue.sh
```

### 4) Monitor/repair long runs

```bash
bash scripts/supervise_overnight_queue.sh
bash scripts/monitor_training_health.sh
```

## Reproducibility Notes

- Same architecture/training loop across tokenizer variants.
- Metrics flushed during training for crash-safe resumption.
- Full-run trainer uses on-disk token memmap caching for large corpora.
- Compare tokenizers with corrected BPC, not raw token-level loss alone.

## Current Status

- [x] Exp01 pipeline implementation
- [x] 6/6 full runs completed
- [x] Final metrics aggregation and reporting
- [ ] Formal multi-rater human evaluation (rubric)
- [ ] Preprint (arXiv) and HF release package

## Citation (placeholder)

If you use this repository, please cite the upcoming AraStudy preprint.
