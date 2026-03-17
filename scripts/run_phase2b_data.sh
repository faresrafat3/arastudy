#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/experiments/exp01_phase2b_data.yaml}"
if [[ -x ".venv/bin/python" ]]; then
  PYTHON_BIN=".venv/bin/python"
else
  PYTHON_BIN="python3"
fi

echo "[phase2b] step 1/2: build raw wiki corpus"
"$PYTHON_BIN" -m src.data.collection.build_phase2b_wiki_corpus --config "$CONFIG_PATH"

echo "[phase2b] step 2/2: clean + dedup + split"
"$PYTHON_BIN" -m src.data.cleaning.prepare_phase2b_corpus --config "$CONFIG_PATH"

echo "[phase2b] data pipeline completed"
