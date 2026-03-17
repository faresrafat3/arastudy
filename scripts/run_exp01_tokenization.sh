#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/experiments/exp01_tokenization.yaml}"
if [[ -x ".venv/bin/python" ]]; then
	PYTHON_BIN=".venv/bin/python"
else
	PYTHON_BIN="python3"
fi

echo "[exp01] Step 1/3: preparing cleaned corpus"
"$PYTHON_BIN" -m src.data.cleaning.prepare_exp01_corpus --config "$CONFIG_PATH"

echo "[exp01] Step 2/3: training tokenizers"
"$PYTHON_BIN" -m src.data.tokenization.train_exp01_tokenizers --config "$CONFIG_PATH"

echo "[exp01] Step 3/3: analyzing tokenizers"
"$PYTHON_BIN" -m src.data.tokenization.analyze_exp01_tokenizers --config "$CONFIG_PATH"

echo "[exp01] completed successfully"
