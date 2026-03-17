#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/experiments/exp01_full_train.yaml}"
RUN_PREFIX="${2:-overnight_local}"
MAX_RETRIES="${MAX_RETRIES:-3}"
QUEUE_TOKENIZERS="${QUEUE_TOKENIZERS:-}"

if [[ -x ".venv/bin/python" ]]; then
  PYTHON_BIN=".venv/bin/python"
else
  PYTHON_BIN="python3"
fi

if [[ -n "$QUEUE_TOKENIZERS" ]]; then
  IFS=',' read -r -a TOKENIZERS <<< "$QUEUE_TOKENIZERS"
else
  TOKENIZERS=(
    "bpe_16k"
    "bpe_32k"
  )
fi

MASTER_LOG_DIR="results/logs/exp01_full"
MASTER_LOG_FILE="$MASTER_LOG_DIR/${RUN_PREFIX}_queue.log"
mkdir -p "$MASTER_LOG_DIR"

timestamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

run_one() {
  local tokenizer_id="$1"
  local run_id="$2"
  local out_dir="results/checkpoints/exp01_full/${run_id}/${tokenizer_id}"
  local latest_ckpt="${out_dir}/latest.pt"
  local summary_file="results/logs/exp01_full/${run_id}/${tokenizer_id}_summary.json"

  if [[ -f "$summary_file" ]]; then
    echo "[$(timestamp)] [queue] skip tokenizer=${tokenizer_id} run_id=${run_id} (summary exists)" | tee -a "$MASTER_LOG_FILE"
    return 0
  fi

  local attempt=1
  while (( attempt <= MAX_RETRIES )); do
    local resume_flag=""
    if [[ -f "$latest_ckpt" ]]; then
      resume_flag="--resume"
    fi

    echo "[$(timestamp)] [queue] start tokenizer=${tokenizer_id} run_id=${run_id} attempt=${attempt} resume=${resume_flag}" | tee -a "$MASTER_LOG_FILE"

    local -a env_prefix=("env")
    env_prefix+=("PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True")
    if [[ "$tokenizer_id" == "bpe_32k" ]]; then
      env_prefix+=("TRAINING_BATCH_SIZE=8")
      env_prefix+=("TRAINING_GRAD_ACCUM_STEPS=8")
    elif [[ "$tokenizer_id" == "char" ]]; then
      env_prefix+=("TRAINING_BATCH_SIZE=4")
      env_prefix+=("TRAINING_GRAD_ACCUM_STEPS=16")
    elif [[ "$tokenizer_id" == "morph_bpe_16k" ]]; then
      env_prefix+=("TRAINING_BATCH_SIZE=8")
      env_prefix+=("TRAINING_GRAD_ACCUM_STEPS=8")
    fi

    if [[ -n "$resume_flag" ]]; then
      "${env_prefix[@]}" "$PYTHON_BIN" -m src.training.train_exp01_full \
        --config "$CONFIG_PATH" \
        --tokenizer-id "$tokenizer_id" \
        --run-id "$run_id" \
        --resume 2>&1 | tee -a "$MASTER_LOG_FILE"
    else
      "${env_prefix[@]}" "$PYTHON_BIN" -m src.training.train_exp01_full \
        --config "$CONFIG_PATH" \
        --tokenizer-id "$tokenizer_id" \
        --run-id "$run_id" 2>&1 | tee -a "$MASTER_LOG_FILE"
    fi

    local exit_code=${PIPESTATUS[0]}
    if (( exit_code == 0 )); then
      echo "[$(timestamp)] [queue] done tokenizer=${tokenizer_id} run_id=${run_id}" | tee -a "$MASTER_LOG_FILE"
      return 0
    fi

    echo "[$(timestamp)] [queue] fail tokenizer=${tokenizer_id} run_id=${run_id} attempt=${attempt} exit_code=${exit_code}" | tee -a "$MASTER_LOG_FILE"
    ((attempt++))
    sleep 5
  done

  echo "[$(timestamp)] [queue] gave up tokenizer=${tokenizer_id} run_id=${run_id} after ${MAX_RETRIES} attempts" | tee -a "$MASTER_LOG_FILE"
  return 1
}

echo "[$(timestamp)] [queue] started run_prefix=${RUN_PREFIX}" | tee -a "$MASTER_LOG_FILE"
echo "[$(timestamp)] [queue] tokenizers=${TOKENIZERS[*]}" | tee -a "$MASTER_LOG_FILE"

for tokenizer_id in "${TOKENIZERS[@]}"; do
  run_id="${RUN_PREFIX}_${tokenizer_id}"
  run_one "$tokenizer_id" "$run_id"
done

echo "[$(timestamp)] [queue] all scheduled runs completed" | tee -a "$MASTER_LOG_FILE"
