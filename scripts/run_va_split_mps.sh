#!/usr/bin/env bash
# Start OpenPI VA-split policy server under NVIDIA MPS.
# All logs for one run go under: /data1/tianze/V-A schedule/logs/<timestamp>/*.log
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

GPU_ID="${GPU_ID:-1}"
PORT="${PORT:-8000}"
POLICY_CONFIG="${POLICY_CONFIG:-pi05_libero}"
POLICY_DIR="${POLICY_DIR:-/data2/gaobowen/model/RLinf-Pi05-LIBERO-SFT}"
RUN_MODE="${RUN_MODE:-server}"
RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="${LOG_ROOT:-/data1/tianze/V-A schedule/logs}"
RUN_LOG_DIR="${RUN_LOG_DIR:-${LOG_ROOT}/${RUN_TS}}"
# MPS pipe sockets stay under the run dir but are not *.log files.
MPS_PIPE_DIR="${MPS_PIPE_DIR:-${RUN_LOG_DIR}/mps-pipe}"
# NVIDIA MPS writes control.log / server.log into this directory.
MPS_LOG_DIR="${MPS_LOG_DIR:-${RUN_LOG_DIR}}"
SERVE_LOG="${SERVE_LOG:-${RUN_LOG_DIR}/serve_policy.log}"
AE_SM_PERCENT="${AE_SM_PERCENT:-20}"
VLM_SM_PERCENT="${VLM_SM_PERCENT:-0}"
VA_SPLIT_MAX_AE_BATCH_SIZE="${VA_SPLIT_MAX_AE_BATCH_SIZE:-8}"
MAX_AE_BATCH_SIZE="${MAX_AE_BATCH_SIZE:-${VA_SPLIT_MAX_AE_BATCH_SIZE}}"
if [[ "${RUN_MODE}" == "smoke" ]]; then
  NUM_REQUESTS="${NUM_REQUESTS:-1}"
  REQUEST_RATE_HZ="${REQUEST_RATE_HZ:-1}"
  MAX_INFLIGHT="${MAX_INFLIGHT:-1}"
  NUM_STEPS="${NUM_STEPS:-1}"
  TIMEOUT_S="${TIMEOUT_S:-60}"
else
  NUM_REQUESTS="${NUM_REQUESTS:-128}"
  REQUEST_RATE_HZ="${REQUEST_RATE_HZ:-16}"
  MAX_INFLIGHT="${MAX_INFLIGHT:-64}"
  NUM_STEPS="${NUM_STEPS:-10}"
  TIMEOUT_S="${TIMEOUT_S:-60}"
fi
SEED="${SEED:-0}"
PROFILE_LOG="${PROFILE_LOG:-${RUN_LOG_DIR}/profile.log}"
JSON_OUTPUT="${JSON_OUTPUT:-${RUN_LOG_DIR}/profile.json}"
PYTHON_BIN="${PYTHON_BIN:-/data1/tianze/RLinf-tianze/openpi05_libero_env/bin/python}"

case "${RUN_MODE}" in
  server|profile|smoke) ;;
  *)
    echo "Unsupported RUN_MODE=${RUN_MODE}; expected server|profile|smoke" >&2
    exit 1
    ;;
esac

mkdir -p "${MPS_PIPE_DIR}" "${MPS_LOG_DIR}"

# Resolve physical GPU -> UUID. Under MPS, prefer UUID so clients keep seeing
# the selected card instead of a remapped index 0 (which looks like "GPU 0").
if ! GPU_UUID="$(nvidia-smi -i "${GPU_ID}" --query-gpu=uuid --format=csv,noheader | tr -d '[:space:]')"; then
  echo "Failed to resolve UUID for GPU_ID=${GPU_ID}" >&2
  exit 1
fi
if [[ -z "${GPU_UUID}" || "${GPU_UUID}" == *"No devices were found"* ]]; then
  echo "GPU_ID=${GPU_ID} is not visible to nvidia-smi / this script." >&2
  exit 1
fi
GPU_NAME="$(nvidia-smi -i "${GPU_ID}" --query-gpu=name --format=csv,noheader | sed 's/^ *//')"

# Bind MPS and all CUDA clients to the same physical GPU via UUID.
export CUDA_VISIBLE_DEVICES="${GPU_UUID}"
export CUDA_MPS_PIPE_DIRECTORY="${MPS_PIPE_DIR}"
export CUDA_MPS_LOG_DIRECTORY="${MPS_LOG_DIR}"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}/packages/openpi-client/src${PYTHONPATH:+:${PYTHONPATH}}"

cleanup() {
  # Ensure quit talks to this run's MPS daemon.
  export CUDA_MPS_PIPE_DIRECTORY="${MPS_PIPE_DIR}"
  echo quit | nvidia-cuda-mps-control >/dev/null 2>&1 || true
}
trap cleanup EXIT

nvidia-cuda-mps-control -d

{
  echo "RUN_TS=${RUN_TS}"
  echo "RUN_MODE=${RUN_MODE}"
  echo "GPU_ID=${GPU_ID}"
  echo "GPU_UUID=${GPU_UUID}"
  echo "GPU_NAME=${GPU_NAME}"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  echo "CUDA_MPS_PIPE_DIRECTORY=${CUDA_MPS_PIPE_DIRECTORY}"
  echo "CUDA_MPS_LOG_DIRECTORY=${CUDA_MPS_LOG_DIRECTORY}"
} | tee "${RUN_LOG_DIR}/gpu_binding.log"

echo "RUN_LOG_DIR=${RUN_LOG_DIR}"
echo "  run_mode=${RUN_MODE}"
echo "  physical_gpu=${GPU_ID} (${GPU_NAME})"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "  MPS:   ${MPS_LOG_DIR}/control.log ${MPS_LOG_DIR}/server.log"
if [[ "${RUN_MODE}" == "server" ]]; then
  echo "  serve: ${SERVE_LOG}"
  echo "  server mode only listens for websocket clients; use RUN_MODE=profile or RUN_MODE=smoke to send synthetic requests."

  "${PYTHON_BIN}" scripts/serve_policy.py \
    --port "${PORT}" \
    --va-split \
    --va-split-max-ae-batch-size "${VA_SPLIT_MAX_AE_BATCH_SIZE}" \
    --ae-sm-percent "${AE_SM_PERCENT}" \
    --vlm-sm-percent "${VLM_SM_PERCENT}" \
    policy:checkpoint \
    --policy.config "${POLICY_CONFIG}" \
    --policy.dir "${POLICY_DIR}" \
    "$@" \
    2>&1 | tee "${SERVE_LOG}"
else
  echo "  profile: ${PROFILE_LOG}"
  echo "  json:    ${JSON_OUTPUT}"

  "${PYTHON_BIN}" scripts/profile_va_split.py \
    --policy.config "${POLICY_CONFIG}" \
    --policy.dir "${POLICY_DIR}" \
    --mode split-mps \
    --num-requests "${NUM_REQUESTS}" \
    --request-rate-hz "${REQUEST_RATE_HZ}" \
    --max-inflight "${MAX_INFLIGHT}" \
    --seed "${SEED}" \
    --num-steps "${NUM_STEPS}" \
    --timeout-s "${TIMEOUT_S}" \
    --max-ae-batch-size "${MAX_AE_BATCH_SIZE}" \
    --ae-sm-percent "${AE_SM_PERCENT}" \
    --vlm-sm-percent "${VLM_SM_PERCENT}" \
    --json-output "${JSON_OUTPUT}" \
    "$@" \
    2>&1 | tee "${PROFILE_LOG}"
fi
