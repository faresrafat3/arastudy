#!/usr/bin/env bash
set -euo pipefail

RUN_PREFIX="${1:-night1}"
CHECK_EVERY_SEC="${CHECK_EVERY_SEC:-120}"
STALL_MINUTES="${STALL_MINUTES:-20}"

LOG_DIR="results/logs/exp01_full"
QUEUE_LOG="${LOG_DIR}/${RUN_PREFIX}_queue.log"
HEALTH_LOG="${LOG_DIR}/${RUN_PREFIX}_health.log"
STATE_DIR="${LOG_DIR}/.${RUN_PREFIX}_health_state"
mkdir -p "${LOG_DIR}" "${STATE_DIR}"

LAST_STEP_FILE="${STATE_DIR}/last_step.txt"
LAST_CHANGE_TS_FILE="${STATE_DIR}/last_change_ts.txt"

timestamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

epoch_now() {
  date +%s
}

get_current_run() {
  local proc_line
  proc_line="$(ps -eo args | grep 'train_exp01_full' | grep -v grep | head -n 1 || true)"
  if [[ -n "${proc_line}" ]]; then
    local tokenizer run_id
    tokenizer="$(echo "${proc_line}" | sed -n 's/.*--tokenizer-id \([^ ]*\).*/\1/p')"
    run_id="$(echo "${proc_line}" | sed -n 's/.*--run-id \([^ ]*\).*/\1/p')"
    if [[ -n "${tokenizer}" && -n "${run_id}" ]]; then
      echo "${tokenizer} ${run_id}"
      return 0
    fi
  fi

  if [[ ! -f "${QUEUE_LOG}" ]]; then
    return 1
  fi

  local line
  line="$(grep 'start tokenizer=' "${QUEUE_LOG}" | tail -n 1 || true)"
  if [[ -z "${line}" ]]; then
    return 1
  fi

  local tokenizer run_id
  tokenizer="$(echo "${line}" | sed -n 's/.*tokenizer=\([^ ]*\).*/\1/p')"
  run_id="$(echo "${line}" | sed -n 's/.*run_id=\([^ ]*\).*/\1/p')"

  if [[ -z "${tokenizer}" || -z "${run_id}" ]]; then
    return 1
  fi

  echo "${tokenizer} ${run_id}"
}

last_metrics_step() {
  local tokenizer="$1"
  local run_id="$2"
  local metrics="${LOG_DIR}/${run_id}/${tokenizer}_metrics.csv"

  if [[ ! -f "${metrics}" ]]; then
    echo ""
    return 0
  fi

  tail -n 1 "${metrics}" | cut -d',' -f1
}

kill_active_run_process() {
  local tokenizer="$1"
  local run_id="$2"
  local pids

  pids="$(pgrep -f "train_exp01_full.*--tokenizer-id ${tokenizer}.*--run-id ${run_id}" || true)"
  if [[ -z "${pids}" ]]; then
    pids="$(pgrep -f "train_exp01_full.*--run-id ${run_id}.*--tokenizer-id ${tokenizer}" || true)"
  fi

  if [[ -n "${pids}" ]]; then
    echo "[$(timestamp)] [health] stalled run detected, killing pids=${pids}" | tee -a "${HEALTH_LOG}"
    kill ${pids} || true
  fi
}

echo "[$(timestamp)] [health] monitor started prefix=${RUN_PREFIX} check=${CHECK_EVERY_SEC}s stall=${STALL_MINUTES}m" | tee -a "${HEALTH_LOG}"

while true; do
  if [[ -f "${QUEUE_LOG}" ]] && grep -q "all scheduled runs completed" "${QUEUE_LOG}"; then
    echo "[$(timestamp)] [health] queue completed, monitor exiting" | tee -a "${HEALTH_LOG}"
    exit 0
  fi

  run_info="$(get_current_run || true)"
  if [[ -z "${run_info}" ]]; then
    echo "[$(timestamp)] [health] waiting for queue run info" >> "${HEALTH_LOG}"
    sleep "${CHECK_EVERY_SEC}"
    continue
  fi

  tokenizer="${run_info%% *}"
  run_id="${run_info##* }"
  step="$(last_metrics_step "${tokenizer}" "${run_id}")"

  if [[ -z "${step}" ]]; then
    echo "[$(timestamp)] [health] waiting metrics tokenizer=${tokenizer} run_id=${run_id}" >> "${HEALTH_LOG}"
    sleep "${CHECK_EVERY_SEC}"
    continue
  fi

  now="$(epoch_now)"
  if [[ ! -f "${LAST_STEP_FILE}" ]]; then
    echo "${step}" > "${LAST_STEP_FILE}"
    echo "${now}" > "${LAST_CHANGE_TS_FILE}"
    echo "[$(timestamp)] [health] baseline step=${step} tokenizer=${tokenizer} run_id=${run_id}" | tee -a "${HEALTH_LOG}"
    sleep "${CHECK_EVERY_SEC}"
    continue
  fi

  prev_step="$(cat "${LAST_STEP_FILE}")"
  last_change_ts="$(cat "${LAST_CHANGE_TS_FILE}")"

  if [[ "${step}" != "${prev_step}" ]]; then
    echo "${step}" > "${LAST_STEP_FILE}"
    echo "${now}" > "${LAST_CHANGE_TS_FILE}"
    echo "[$(timestamp)] [health] progress step=${step} tokenizer=${tokenizer} run_id=${run_id}" >> "${HEALTH_LOG}"
  else
    stalled_for=$(( now - last_change_ts ))
    threshold=$(( STALL_MINUTES * 60 ))
    echo "[$(timestamp)] [health] no progress step=${step} stalled_for=${stalled_for}s tokenizer=${tokenizer} run_id=${run_id}" >> "${HEALTH_LOG}"

    if (( stalled_for >= threshold )); then
      kill_active_run_process "${tokenizer}" "${run_id}"
      echo "${now}" > "${LAST_CHANGE_TS_FILE}"
    fi
  fi

  sleep "${CHECK_EVERY_SEC}"
done
