# Exp02 Parallel Execution Checklist

## Data Upload
- [ ] Data zipped and uploaded to Google Drive
- [ ] Data uploaded as Kaggle Dataset
- [ ] Verified: data accessible from Colab
- [ ] Verified: data accessible from Kaggle

## Local Runs (RTX 4050)
- [x] bpe_16k_s42 (running!)
- [ ] bpe_32k_s42
- [ ] morph_bpe_8k_s42

## Colab Runs (T4)
- [ ] bpe_16k_s123
- [ ] bpe_32k_s123
- [ ] morph_bpe_8k_s123

## Kaggle Runs (T4/P100)
- [ ] bpe_16k_s2024
- [ ] bpe_32k_s2024
- [ ] morph_bpe_8k_s2024

## Results Collection
- [ ] All 9 summaries downloaded
- [ ] compare_runs.py executed
- [ ] mean±std calculated
- [ ] bootstrap significance tested
- [ ] RESEARCH_LOG updated

## Notes
- Data must be uploaded to Drive/Kaggle before starting cloud runs.
- If a Colab session disconnects, rerun with the same run id and `--resume` if needed.
- Every run should produce `summary.json`, `metrics.csv`, and `generation.md`.
