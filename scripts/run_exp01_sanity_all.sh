#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/experiments/exp01_sanity_train.yaml}"
if [[ -x ".venv/bin/python" ]]; then
  PYTHON_BIN=".venv/bin/python"
else
  PYTHON_BIN="python3"
fi

TOKENIZERS=(
  "bpe_32k"
  "bpe_16k"
  "bpe_8k"
  "morph_bpe_16k"
  "morph_bpe_8k"
  "char"
)

for tok in "${TOKENIZERS[@]}"; do
  echo "[exp01-sanity] running tokenizer=${tok}"
  "$PYTHON_BIN" -m src.training.train_exp01_sanity --config "$CONFIG_PATH" --tokenizer-id "$tok"
done

echo "[exp01-sanity] all runs completed"
