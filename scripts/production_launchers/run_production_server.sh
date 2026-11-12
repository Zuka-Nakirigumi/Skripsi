#!/usr/bin/env bash
set -uo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage:"
  echo "  bash scripts/run_production_server.sh SECTION"
  echo
  echo "Valid sections: 1-12"
  exit 2
fi

SECTION="$1"

if ! [[ "${SECTION}" =~ ^([1-9]|1[0-2])$ ]]; then
  echo "Invalid section: ${SECTION}"
  echo "Valid sections: 1-12"
  exit 2
fi

SECTION_PADDED="$(printf '%02d' "${SECTION}")"
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

SESSION_ROOT="outputs/activity_logs/server_section${SECTION_PADDED}_${TIMESTAMP}"

mkdir -p "${SESSION_ROOT}"

ENV_LOG="${SESSION_ROOT}/environment.txt"
PIP_FREEZE="${SESSION_ROOT}/pip_freeze.txt"
GPU_CSV="${SESSION_ROOT}/gpu_usage.csv"
SECTION_LOG="${SESSION_ROOT}/section.log"
SECTION_REPORT="${SESSION_ROOT}/section_report.json"


# ============================================================
# Environment snapshot
# ============================================================

{
  echo "PRODUCTION SECTION - RTX A5000 SERVER"
  echo "Timestamp: $(date --iso-8601=seconds)"
  echo "Host: $(hostname)"
  echo "Section: ${SECTION}"

  echo
  echo "--- Python ---"
  python --version
  python -c 'import sys; print(sys.executable)'

  echo
  echo "--- Git commit ---"
  git rev-parse HEAD

  echo
  echo "--- Git status ---"
  git status --short

  echo
  echo "--- NVIDIA-SMI ---"
  nvidia-smi

} > "${ENV_LOG}" 2>&1


python -m pip freeze \
  > "${PIP_FREEZE}" \
  2>&1


# ============================================================
# GPU monitor - GPU 1 only
# ============================================================

echo "timestamp,index,name,gpu_util_percent,memory_util_percent,memory_used_mib,memory_total_mib,temperature_c,power_draw_w,power_limit_w" \
  > "${GPU_CSV}"

(
  while true; do

    nvidia-smi \
      -i 1 \
      --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw,power.limit \
      --format=csv,noheader,nounits \
      >> "${GPU_CSV}"

    sleep 10

  done
) &

GPU_MONITOR_PID=$!


cleanup() {

  kill "${GPU_MONITOR_PID}" \
    2>/dev/null \
    || true

  wait "${GPU_MONITOR_PID}" \
    2>/dev/null \
    || true

  nvidia-smi \
    > "${SESSION_ROOT}/nvidia_smi_final.txt" \
    2>&1 \
    || true
}

trap cleanup EXIT


echo
echo "============================================================"
echo "STARTING SERVER PRODUCTION SECTION ${SECTION}"
echo "ACTIVITY LOG: ${SESSION_ROOT}"
echo "============================================================"
echo


set +e

python scripts/production_section.py \
  --machine server \
  --section "${SECTION}" \
  --device 1 \
  --num-workers 8 \
  --seed 1993 \
  --output-root outputs/experiments \
  --report-path "${SECTION_REPORT}" \
  2>&1 \
  | tee "${SECTION_LOG}"

EXIT_CODE=${PIPESTATUS[0]}

set -e


if [[ "${EXIT_CODE}" -ne 0 ]]; then

  echo
  echo "SERVER PRODUCTION SECTION ${SECTION} FAILED"
  echo "EXIT CODE: ${EXIT_CODE}"
  echo "LOG: ${SECTION_LOG}"

  exit "${EXIT_CODE}"
fi


echo
echo "============================================================"
echo "SERVER PRODUCTION SECTION ${SECTION} PASSED"
echo "OUTPUT: ${SESSION_ROOT}"
echo "============================================================"
