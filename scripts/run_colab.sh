#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/experiments/exp01_full_train.yaml}"
TOKENIZER_ID="${2:-bpe_8k}"
RUN_ID="${3:-colab_run}"
RESUME_FLAG="${4:-}"

if [[ -x ".venv/bin/python" ]]; then
  PYTHON_BIN=".venv/bin/python"
else
  PYTHON_BIN="python3"
fi

CMD=(
  "$PYTHON_BIN" -m src.training.train_exp01_full
  --config "$CONFIG_PATH"
  --tokenizer-id "$TOKENIZER_ID"
  --run-id "$RUN_ID"
)

if [[ "$RESUME_FLAG" == "--resume" ]]; then
  CMD+=(--resume)
fi

echo "[colab] running tokenizer=$TOKENIZER_ID run_id=$RUN_ID resume=$RESUME_FLAG"
"${CMD[@]}"
