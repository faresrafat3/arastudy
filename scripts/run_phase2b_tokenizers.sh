#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/experiments/exp01_phase2b_data.yaml}"
if [[ -x ".venv/bin/python" ]]; then
  PYTHON_BIN=".venv/bin/python"
else
  PYTHON_BIN="python3"
fi

echo "[phase2b] step 1/2: train tokenizers"
"$PYTHON_BIN" -m src.data.tokenization.train_exp01_tokenizers --config "$CONFIG_PATH" --skip-existing

echo "[phase2b] step 2/2: analyze tokenizers"
"$PYTHON_BIN" -m src.data.tokenization.analyze_exp01_tokenizers --config "$CONFIG_PATH"

echo "[phase2b] tokenizer pipeline completed"
