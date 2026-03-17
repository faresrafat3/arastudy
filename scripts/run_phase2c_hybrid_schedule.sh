#!/usr/bin/env bash
set -euo pipefail

cat <<'EOF'
[phase2c] Recommended hybrid schedule

Local (RTX 4050):
  - day1/run1: morph_bpe_8k
  - day2/run3: bpe_16k
  - day3/run5: bpe_32k

Colab (T4):
  - day1/run2: bpe_8k
  - day2/run4: morph_bpe_16k
  - day3/run6: char

Command template:
  .venv/bin/python -m src.training.train_exp01_full --config configs/experiments/exp01_full_train.yaml --tokenizer-id <id> --run-id <run-name>

Suggested run IDs:
  - day1_local_morph8k
  - day1_colab_bpe8k
  - day2_local_bpe16k
  - day2_colab_morph16k
  - day3_local_bpe32k
  - day3_colab_char

If interrupted, resume:
  .venv/bin/python -m src.training.train_exp01_full --config configs/experiments/exp01_full_train.yaml --tokenizer-id <id> --run-id <run-name> --resume
EOF
