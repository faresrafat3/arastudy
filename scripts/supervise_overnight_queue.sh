#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/experiments/exp01_full_train.yaml}"
RUN_PREFIX="${2:-night1}"
CHECK_EVERY_SEC="${CHECK_EVERY_SEC:-60}"
MAX_RESTARTS="${MAX_RESTARTS:-20}"

QUEUE_CMD="./scripts/run_overnight_local_queue.sh ${CONFIG_PATH} ${RUN_PREFIX}"
QUEUE_PATTERN="run_overnight_local_queue.sh ${CONFIG_PATH} ${RUN_PREFIX}"

LOG_DIR="results/logs/exp01_full"
QUEUE_LOG="${LOG_DIR}/${RUN_PREFIX}_queue.log"
SUP_LOG="${LOG_DIR}/${RUN_PREFIX}_supervisor.log"
mkdir -p "${LOG_DIR}"

timestamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

is_completed() {
  [[ -f "${QUEUE_LOG}" ]] && grep -q "all scheduled runs completed" "${QUEUE_LOG}"
}

is_running() {
  pgrep -f "${QUEUE_PATTERN}" >/dev/null 2>&1
}

restart_count=0
echo "[$(timestamp)] [supervisor] started prefix=${RUN_PREFIX}" | tee -a "${SUP_LOG}"

while true; do
  if is_completed; then
    echo "[$(timestamp)] [supervisor] queue completed successfully" | tee -a "${SUP_LOG}"
    break
  fi

  if is_running; then
    echo "[$(timestamp)] [supervisor] queue running" >> "${SUP_LOG}"
    sleep "${CHECK_EVERY_SEC}"
    continue
  fi

  if (( restart_count >= MAX_RESTARTS )); then
    echo "[$(timestamp)] [supervisor] max restarts reached (${MAX_RESTARTS}), stopping" | tee -a "${SUP_LOG}"
    exit 1
  fi

  restart_count=$((restart_count + 1))
  echo "[$(timestamp)] [supervisor] queue not running, restart #${restart_count}" | tee -a "${SUP_LOG}"
  nohup bash -lc "${QUEUE_CMD}" >> "${SUP_LOG}" 2>&1 &
  sleep 5
done
