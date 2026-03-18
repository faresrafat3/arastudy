#!/usr/bin/env bash
set -euo pipefail

# Exp02: Multi-seed sweep
# Top 3 tokenizers × 3 seeds = 9 runs

TOKENIZERS=("bpe_16k" "bpe_32k" "morph_bpe_8k")
SEEDS=(42 123 2024)
CONFIG="configs/experiments/exp02_multiseed.yaml"
PYTHON_BIN="${PYTHON_BIN:-/home/fares/bug-hunter/.venv/bin/python}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python"
fi

run_one() {
  local tokenizer="$1"
  local seed="$2"
  local run_id="exp02_${tokenizer}_s${seed}"

  echo "========================================="
  echo "Running: ${tokenizer} | seed=${seed}"
  echo "========================================="

  "$PYTHON_BIN" -m src.training.train_exp01_full \
    --config "$CONFIG" \
    --tokenizer-id "$tokenizer" \
    --seed "$seed" \
    --run-id "$run_id"

  "$PYTHON_BIN" scripts/update_research_log.py \
    --exp exp02 \
    --tokenizer "$tokenizer" \
    --seed "$seed"

  "$PYTHON_BIN" scripts/compare_runs.py --exp exp02 --metric bpc --plot || true

  if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    git add RESEARCH_LOG.md runs/exp02 scripts/compare_runs.py scripts/update_research_log.py || true
    if ! git diff --cached --quiet; then
      git commit -m "exp02: add ${tokenizer} seed ${seed} results"
      git push
    fi
  fi

  echo "✅ Done: ${tokenizer} seed=${seed}"
}

# Priority order: bpe_16k first, then bpe_32k, then morph_bpe_8k
for tokenizer in "${TOKENIZERS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    run_one "$tokenizer" "$seed"
  done
done

echo "🎉 Exp02 Complete!"
"$PYTHON_BIN" scripts/compare_runs.py --exp exp02 --metric bpc --plot
