#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

GPU_ID="${GPU_ID:-0}"
MODE="${MODE:-jax-split-ipc}"
POLICY_CONFIG="${POLICY_CONFIG:-pi05_libero}"
POLICY_DIR="${POLICY_DIR:-/data1/miliang/models/pi05_libero}"
LOG_ROOT="${LOG_ROOT:-/data1/miliang/VL-A-Disaggregation/logs/JAX}"
RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
RUN_LOG_DIR="${RUN_LOG_DIR:-${LOG_ROOT}/jax-${RUN_TS}}"
MPS_PIPE_DIR="${MPS_PIPE_DIR:-${RUN_LOG_DIR}/mps-pipe}"
MPS_LOG_DIR="${MPS_LOG_DIR:-${RUN_LOG_DIR}}"
AE_SM_PERCENT="${AE_SM_PERCENT:-20}"
VLM_SM_PERCENT="${VLM_SM_PERCENT:-0}"
NUM_REQUESTS="${NUM_REQUESTS:-128}"
REQUEST_RATE_HZ="${REQUEST_RATE_HZ:-16}"
MAX_INFLIGHT="${MAX_INFLIGHT:-64}"
SEED="${SEED:-0}"
NUM_STEPS="${NUM_STEPS:-5}"
TIMEOUT_S="${TIMEOUT_S:-60}"
MAX_AE_BATCH_SIZE="${MAX_AE_BATCH_SIZE:-999}"
MAX_VLM_BATCH_SIZE="${MAX_VLM_BATCH_SIZE:-8}"
MAX_VLM_WAIT_MS="${MAX_VLM_WAIT_MS:-1.0}"
WARMUP_REQUESTS="${WARMUP_REQUESTS:-4}"
WARMUP_UNTIL_STEADY="${WARMUP_UNTIL_STEADY:-1}"
WARMUP_STEADY_WINDOW="${WARMUP_STEADY_WINDOW:-4}"
WARMUP_STEADY_MAX_REQUESTS="${WARMUP_STEADY_MAX_REQUESTS:-48}"
JAX_COMPILE="${JAX_COMPILE:-1}"
JAX_COMPILE_WARMUP="${JAX_COMPILE_WARMUP:-1}"
# Default warmup ceiling follows VA-split prefix capacity: max_vlm_batch_size * 3.
JAX_COMPILE_WARMUP_MAX_BATCH_SIZE="${JAX_COMPILE_WARMUP_MAX_BATCH_SIZE:-$((MAX_VLM_BATCH_SIZE * 3))}"
JSON_OUTPUT="${JSON_OUTPUT:-${RUN_LOG_DIR}/profile.json}"
PYTHON_BIN="${PYTHON_BIN:-/data1/miliang/RLinf/openpi_libero/bin/python}"

MPS_STARTED=0

if ! GPU_UUID="$(nvidia-smi -i "${GPU_ID}" --query-gpu=uuid --format=csv,noheader | tr -d '[:space:]')"; then
  echo "Failed to resolve UUID for GPU_ID=${GPU_ID}" >&2
  exit 1
fi
if [[ -z "${GPU_UUID}" || "${GPU_UUID}" == *"No devices were found"* ]]; then
  echo "GPU_ID=${GPU_ID} is not visible to nvidia-smi / this script." >&2
  exit 1
fi
GPU_NAME="$(nvidia-smi -i "${GPU_ID}" --query-gpu=name --format=csv,noheader | sed 's/^ *//')"

cleanup_mps_pipe_dir() {
  local pipe_dir="$1"
  rm -f \
    "${pipe_dir}/control" \
    "${pipe_dir}/control_privileged" \
    "${pipe_dir}/control_lock" \
    "${pipe_dir}/log" \
    "${pipe_dir}/nvidia-cuda-mps-control.pid" \
    2>/dev/null || true
}

cleanup() {
  if [[ "${MPS_STARTED}" -eq 1 ]]; then
    export CUDA_MPS_PIPE_DIRECTORY="${MPS_PIPE_DIR}"
    echo quit | nvidia-cuda-mps-control >/dev/null 2>&1 || true
    cleanup_mps_pipe_dir "${MPS_PIPE_DIR}"
  fi
}
trap cleanup EXIT

export CUDA_VISIBLE_DEVICES="${GPU_UUID}"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}/packages/openpi-client/src${PYTHONPATH:+:${PYTHONPATH}}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

mkdir -p "${RUN_LOG_DIR}"

if [[ "${MODE}" == "jax-split-ipc" ]]; then
  mkdir -p "${MPS_PIPE_DIR}" "${MPS_LOG_DIR}"
  export CUDA_MPS_PIPE_DIRECTORY="${MPS_PIPE_DIR}"
  export CUDA_MPS_LOG_DIRECTORY="${MPS_LOG_DIR}"
  cleanup_mps_pipe_dir "${MPS_PIPE_DIR}"
  nvidia-cuda-mps-control -d || true
  mps_ready=0
  for _ in $(seq 1 50); do
    if [[ -S "${MPS_PIPE_DIR}/control" || -e "${MPS_PIPE_DIR}/control_lock" || -p "${MPS_PIPE_DIR}/log" ]]; then
      mps_ready=1
      break
    fi
    sleep 0.2
  done
  if [[ "${mps_ready}" -ne 1 ]]; then
    echo "Failed to start MPS control daemon (missing control/control_lock/log in ${MPS_PIPE_DIR})" >&2
    echo "--- control.log ---" >&2
    cat "${MPS_LOG_DIR}/control.log" >&2 || true
    echo "--- pipe dir ---" >&2
    ls -la "${MPS_PIPE_DIR}" >&2 || true
    exit 1
  fi
  MPS_STARTED=1
fi

cmd=(
  "${PYTHON_BIN}" "${SCRIPT_DIR}/profile_va_split.py"
  --policy.config "${POLICY_CONFIG}"
  --policy.dir "${POLICY_DIR}"
  --mode "${MODE}"
  --num-requests "${NUM_REQUESTS}"
  --request-rate-hz "${REQUEST_RATE_HZ}"
  --max-inflight "${MAX_INFLIGHT}"
  --seed "${SEED}"
  --num-steps "${NUM_STEPS}"
  --timeout-s "${TIMEOUT_S}"
  --max-ae-batch-size "${MAX_AE_BATCH_SIZE}"
  --max-vlm-batch-size "${MAX_VLM_BATCH_SIZE}"
  --max-vlm-wait-ms "${MAX_VLM_WAIT_MS}"
  --warmup-requests "${WARMUP_REQUESTS}"
  --warmup-steady-window "${WARMUP_STEADY_WINDOW}"
  --warmup-steady-max-requests "${WARMUP_STEADY_MAX_REQUESTS}"
  --ae-sm-percent "${AE_SM_PERCENT}"
  --vlm-sm-percent "${VLM_SM_PERCENT}"
  --jax-compile-warmup-max-batch-size "${JAX_COMPILE_WARMUP_MAX_BATCH_SIZE}"
  --gpu-device-index "${GPU_ID}"
  --json-output "${JSON_OUTPUT}"
)

if [[ "${JAX_COMPILE}" == "0" || "${JAX_COMPILE}" == "false" ]]; then
  cmd+=(--no-jax-compile)
fi

if [[ "${JAX_COMPILE_WARMUP}" == "0" || "${JAX_COMPILE_WARMUP}" == "false" ]]; then
  cmd+=(--no-jax-compile-warmup)
fi

if [[ "${WARMUP_UNTIL_STEADY}" == "0" || "${WARMUP_UNTIL_STEADY}" == "false" ]]; then
  cmd+=(--no-warmup-until-steady)
fi

echo "Running JAX V-A profile: mode=${MODE} gpu=${GPU_ID} (${GPU_NAME}) policy_dir=${POLICY_DIR}"
echo "  python: ${PYTHON_BIN}"
echo "  logs:   ${RUN_LOG_DIR}"
echo "  json:   ${JSON_OUTPUT}"
echo "  cuda:   visible=${CUDA_VISIBLE_DEVICES}"
echo "  compile: enabled=${JAX_COMPILE} warmup=${JAX_COMPILE_WARMUP} warmup_max_batch=${JAX_COMPILE_WARMUP_MAX_BATCH_SIZE}"
echo "  e2e_warmup: min=${WARMUP_REQUESTS} until_steady=${WARMUP_UNTIL_STEADY} window=${WARMUP_STEADY_WINDOW} max=${WARMUP_STEADY_MAX_REQUESTS}"
echo "  mps:    pipe=${MPS_PIPE_DIR} ae_sm=${AE_SM_PERCENT} vlm_sm=${VLM_SM_PERCENT}"
"${cmd[@]}" "$@"
