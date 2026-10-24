#!/usr/bin/env bash
set -uo pipefail

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
SESSION_ROOT="outputs/sanity_final/server_${TIMESTAMP}"
LOG_DIR="${SESSION_ROOT}/_activity"

mkdir -p "${LOG_DIR}"

ENV_LOG="${LOG_DIR}/environment.txt"
GPU_CSV="${LOG_DIR}/gpu_usage.csv"
TRAIN_LOG="${LOG_DIR}/sanity_training.log"
PIP_FREEZE="${LOG_DIR}/pip_freeze.txt"

{
  echo "FINAL SANITY - RTX A5000 SERVER"
  echo "Timestamp: $(date --iso-8601=seconds)"
  echo "Host: $(hostname)"
  echo "--- Python ---"
  python --version
  python -c 'import sys; print(sys.executable)'
  echo "--- Git commit ---"
  git rev-parse HEAD
  echo "--- Git status ---"
  git status --short
  echo "--- NVIDIA-SMI ---"
  nvidia-smi
} > "${ENV_LOG}" 2>&1

python -m pip freeze > "${PIP_FREEZE}" 2>&1

echo "timestamp,index,name,gpu_util_percent,memory_util_percent,memory_used_mib,memory_total_mib,temperature_c,power_draw_w,power_limit_w" > "${GPU_CSV}"

(
  while true; do
    nvidia-smi \
      --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw,power.limit \
      --format=csv,noheader,nounits \
      | sed -n '2p' >> "${GPU_CSV}"
    sleep 10
  done
) &

GPU_MONITOR_PID=$!

cleanup() {
  kill "${GPU_MONITOR_PID}" 2>/dev/null || true
  wait "${GPU_MONITOR_PID}" 2>/dev/null || true
  nvidia-smi > "${LOG_DIR}/nvidia_smi_final.txt" 2>&1 || true
}
trap cleanup EXIT

set +e
python scripts/sanity_final.py \
  --methods fetril efc \
  --dataset aid \
  --protocol T5 \
  --backbone r34 \
  --attention none \
  --epochs 1 \
  --max-tasks 2 \
  --device 1 \
  --num-workers 8 \
  --batch fetril=32 \
  --batch efc=12 \
  --output-root "${SESSION_ROOT}" \
  2>&1 | tee "${TRAIN_LOG}"

EXIT_CODE=${PIPESTATUS[0]}
set -e

if [[ "${EXIT_CODE}" -ne 0 ]]; then
  echo "SERVER SANITY FAILED (exit=${EXIT_CODE})"
  exit "${EXIT_CODE}"
fi

echo
echo "============================================================"
echo "SERVER FINAL SANITY PASSED"
echo "OUTPUT: ${SESSION_ROOT}"
echo "============================================================"
