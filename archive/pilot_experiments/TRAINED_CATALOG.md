# Trained Runs Catalog

- Generated: 2026-03-18T06:09:58
- Repository: arastudy
- Source file: `experiments/trained_runs_catalog.csv`

## Scope
- Includes every `*_metrics.csv` found under `results/logs/`.
- A run is marked `completed` if a paired summary file exists (`*_summary.json` or `*_summary.txt`).

## Summary
- Completed runs: 12
- Interrupted/partial runs: 0

## Completed Runs

| Experiment | Run ID | Tokenizer | Best Val Loss | Best BPC | Best Step | Final Step | Elapsed (h) | Peak VRAM (GB) | HF |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| exp01_full | day1_local_morph8k | morph_bpe_8k | 3.322887 | 0.155985 | 13000 | 14999 | 2.617 | 2.640 | [link](https://huggingface.co/faresrafat/AraStudy-Morph8K-25M) |
| exp01_full | night1_bpe_16k | bpe_16k | 4.072959 | 1.733176 | 16500 | 18204 | 0.098 | 3.549 | [link](https://huggingface.co/faresrafat/AraStudy-BPE16K-29M) |
| exp01_full | night1_bpe_32k | bpe_32k | 4.403984 | 1.710173 | 12500 | 14499 | 3.361 | 2.826 | [link](https://huggingface.co/faresrafat/AraStudy-BPE32K-37M) |
| exp01_full | night2_bpe_8k | bpe_8k | 3.841254 | 1.827454 | 10500 | 12499 | 2.149 | 2.611 |  |
| exp01_full | night2_char | char | 1.215111 | 2.116795 | 5500 | 7499 | 0.974 | 0.745 |  |
| exp01_full | night2_morph_bpe_16k | morph_bpe_16k | 3.476209 | 1.758961 | 10500 | 12499 | 2.260 | 1.956 |  |
| exp01_sanity | exp01_sanity_single | bpe_16k | 8.006859 |  | 250 | 300 |  |  | [link](https://huggingface.co/faresrafat/AraStudy-BPE16K-29M) |
| exp01_sanity | exp01_sanity_single | bpe_32k | 8.331104 |  | 300 | 300 |  |  | [link](https://huggingface.co/faresrafat/AraStudy-BPE32K-37M) |
| exp01_sanity | exp01_sanity_single | bpe_8k | 7.591602 |  | 300 | 300 |  |  |  |
| exp01_sanity | exp01_sanity_single | char | 2.519621 |  | 300 | 300 |  |  |  |
| exp01_sanity | exp01_sanity_single | morph_bpe_16k | 7.040647 |  | 250 | 300 |  |  |  |
| exp01_sanity | exp01_sanity_single | morph_bpe_8k | 6.700375 |  | 300 | 300 |  |  | [link](https://huggingface.co/faresrafat/AraStudy-Morph8K-25M) |

## Interrupted / Partial Runs

- None

## Artifact Roots
- Checkpoints: `results/checkpoints/`
- Tokenizers: `results/tokenizers/`
- Logs: `results/logs/`
- HF Dataset: https://huggingface.co/datasets/faresrafat/arastudy-arabic-wikipedia-cleaned
