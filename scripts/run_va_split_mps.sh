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
# ours mode : PROFILE_MODE=split-mps
# baseline mode : PROFILE_MODE=monolithic
PROFILE_MODE="${PROFILE_MODE:-split-mps}"
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
VA_SPLIT_MAX_AE_BATCH_SIZE="${VA_SPLIT_MAX_AE_BATCH_SIZE:-999}"
MAX_AE_BATCH_SIZE="${MAX_AE_BATCH_SIZE:-${VA_SPLIT_MAX_AE_BATCH_SIZE}}"
VA_SPLIT_MAX_VLM_BATCH_SIZE="${VA_SPLIT_MAX_VLM_BATCH_SIZE:-8}"
MAX_VLM_BATCH_SIZE="${MAX_VLM_BATCH_SIZE:-${VA_SPLIT_MAX_VLM_BATCH_SIZE}}"
VA_SPLIT_MAX_VLM_WAIT_MS="${VA_SPLIT_MAX_VLM_WAIT_MS:-2.0}"
MAX_VLM_WAIT_MS="${MAX_VLM_WAIT_MS:-${VA_SPLIT_MAX_VLM_WAIT_MS}}"
BATCH_SIZE="${BATCH_SIZE:-${MAX_VLM_BATCH_SIZE}}"
ENABLE_POLICY_BATCH="${ENABLE_POLICY_BATCH:-true}"
if [[ "${RUN_MODE}" == "smoke" ]]; then
  NUM_REQUESTS="${NUM_REQUESTS:-1}"
  REQUEST_RATE_HZ="${REQUEST_RATE_HZ:-1}"
  MAX_INFLIGHT="${MAX_INFLIGHT:-1}"
  NUM_STEPS="${NUM_STEPS:-1}"
  TIMEOUT_S="${TIMEOUT_S:-60}"
  WARMUP_REQUESTS="${WARMUP_REQUESTS:-2}"
else
  NUM_REQUESTS="${NUM_REQUESTS:-128}"
  REQUEST_RATE_HZ="${REQUEST_RATE_HZ:-8}"
  MAX_INFLIGHT="${MAX_INFLIGHT:-999}"
  NUM_STEPS="${NUM_STEPS:-5}"
  TIMEOUT_S="${TIMEOUT_S:-600}"
  WARMUP_REQUESTS="${WARMUP_REQUESTS:-2}"
fi
SEED="${SEED:-0}"
PROFILE_LOG="${PROFILE_LOG:-${RUN_LOG_DIR}/profile.log}"
JSON_OUTPUT="${JSON_OUTPUT:-${RUN_LOG_DIR}/profile.json}"
PYTHON_BIN="${PYTHON_BIN:-/data1/tianze/RLinf-tianze/openpi05_libero_env/bin/python}"
PYTORCH_COMPILE_MODE="${PYTORCH_COMPILE_MODE:-}"

append_pytorch_compile_mode_arg() {
  local -n cmd_ref="$1"
  local compile_mode_lc="${PYTORCH_COMPILE_MODE,,}"
  case "${compile_mode_lc}" in
    ""|none|null|off|false)
      return 0
      ;;
  esac
  cmd_ref+=(--pytorch-compile-mode "${PYTORCH_COMPILE_MODE}")
}

case "${RUN_MODE}" in
  server|profile|smoke) ;;
  *)
    echo "Unsupported RUN_MODE=${RUN_MODE}; expected server|profile|smoke" >&2
    exit 1
    ;;
esac

case "${PROFILE_MODE}" in
  monolithic|split-no-mps|split-mps) ;;
  *)
    echo "Unsupported PROFILE_MODE=${PROFILE_MODE}; expected monolithic|split-no-mps|split-mps" >&2
    exit 1
    ;;
esac

MPS_STARTED=0
NEEDS_MPS=0
if [[ "${RUN_MODE}" == "server" || "${PROFILE_MODE}" == "split-mps" ]]; then
  NEEDS_MPS=1
fi

mkdir -p "${RUN_LOG_DIR}"
if [[ "${NEEDS_MPS}" -eq 1 ]]; then
  mkdir -p "${MPS_PIPE_DIR}" "${MPS_LOG_DIR}"
fi

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
if [[ "${NEEDS_MPS}" -eq 1 ]]; then
  export CUDA_MPS_PIPE_DIRECTORY="${MPS_PIPE_DIR}"
  export CUDA_MPS_LOG_DIRECTORY="${MPS_LOG_DIR}"
else
  unset CUDA_MPS_PIPE_DIRECTORY
  unset CUDA_MPS_LOG_DIRECTORY
fi
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}/packages/openpi-client/src${PYTHONPATH:+:${PYTHONPATH}}"

cleanup() {
  if [[ "${MPS_STARTED}" -eq 1 ]]; then
    # Ensure quit talks to this run's MPS daemon.
    export CUDA_MPS_PIPE_DIRECTORY="${MPS_PIPE_DIR}"
    echo quit | nvidia-cuda-mps-control >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

if [[ "${NEEDS_MPS}" -eq 1 ]]; then
  nvidia-cuda-mps-control -d
  MPS_STARTED=1
fi

{
  echo "RUN_TS=${RUN_TS}"
  echo "RUN_MODE=${RUN_MODE}"
  echo "PROFILE_MODE=${PROFILE_MODE}"
  echo "GPU_ID=${GPU_ID}"
  echo "GPU_UUID=${GPU_UUID}"
  echo "GPU_NAME=${GPU_NAME}"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
  echo "CUDA_MPS_PIPE_DIRECTORY=${CUDA_MPS_PIPE_DIRECTORY:-}"
  echo "CUDA_MPS_LOG_DIRECTORY=${CUDA_MPS_LOG_DIRECTORY:-}"
  echo "POLICY_CONFIG=${POLICY_CONFIG}"
  echo "POLICY_DIR=${POLICY_DIR}"
  echo "NUM_REQUESTS=${NUM_REQUESTS}"
  echo "REQUEST_RATE_HZ=${REQUEST_RATE_HZ}"
  echo "MAX_INFLIGHT=${MAX_INFLIGHT}"
  echo "NUM_STEPS=${NUM_STEPS}"
  echo "TIMEOUT_S=${TIMEOUT_S}"
  echo "WARMUP_REQUESTS=${WARMUP_REQUESTS}"
  echo "PYTORCH_COMPILE_MODE=${PYTORCH_COMPILE_MODE:-disabled}"
  echo "AE_SM_PERCENT=${AE_SM_PERCENT}"
  echo "VLM_SM_PERCENT=${VLM_SM_PERCENT}"
  echo "VA_SPLIT_MAX_AE_BATCH_SIZE=${VA_SPLIT_MAX_AE_BATCH_SIZE}"
  echo "MAX_AE_BATCH_SIZE=${MAX_AE_BATCH_SIZE}"
  echo "VA_SPLIT_MAX_VLM_BATCH_SIZE=${VA_SPLIT_MAX_VLM_BATCH_SIZE}"
  echo "MAX_VLM_BATCH_SIZE=${MAX_VLM_BATCH_SIZE}"
  echo "VA_SPLIT_MAX_VLM_WAIT_MS=${VA_SPLIT_MAX_VLM_WAIT_MS}"
  echo "MAX_VLM_WAIT_MS=${MAX_VLM_WAIT_MS}"
  echo "BATCH_SIZE=${BATCH_SIZE}"
  echo "ENABLE_POLICY_BATCH=${ENABLE_POLICY_BATCH}"
} | tee "${RUN_LOG_DIR}/gpu_binding.log"

echo "RUN_LOG_DIR=${RUN_LOG_DIR}"
echo "  run_mode=${RUN_MODE}"
echo "  profile_mode=${PROFILE_MODE}"
echo "  physical_gpu=${GPU_ID} (${GPU_NAME})"
echo "  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "  workload: requests=${NUM_REQUESTS} rate=${REQUEST_RATE_HZ}/s max_inflight=${MAX_INFLIGHT} baseline_max_batch=${BATCH_SIZE} steps=${NUM_STEPS} warmup=${WARMUP_REQUESTS}"
echo "  split: max_ae_batch=${MAX_AE_BATCH_SIZE} max_vlm_batch=${MAX_VLM_BATCH_SIZE} max_vlm_wait_ms=${MAX_VLM_WAIT_MS} ae_sm=${AE_SM_PERCENT} vlm_sm=${VLM_SM_PERCENT}"
if [[ "${NEEDS_MPS}" -eq 1 ]]; then
  echo "  MPS:   ${MPS_LOG_DIR}/control.log ${MPS_LOG_DIR}/server.log"
else
  echo "  MPS:   disabled"
fi
if [[ "${RUN_MODE}" == "server" ]]; then
  echo "  serve: ${SERVE_LOG}"
  echo "  server mode only listens for websocket clients; use RUN_MODE=profile or RUN_MODE=smoke to send synthetic requests."

  server_cmd=(
    "${PYTHON_BIN}" scripts/serve_policy.py
    --port "${PORT}" \
    --va-split \
    --va-split-max-ae-batch-size "${VA_SPLIT_MAX_AE_BATCH_SIZE}" \
    --va-split-max-vlm-batch-size "${VA_SPLIT_MAX_VLM_BATCH_SIZE}" \
    --va-split-max-vlm-wait-ms "${VA_SPLIT_MAX_VLM_WAIT_MS}" \
    --ae-sm-percent "${AE_SM_PERCENT}" \
    --vlm-sm-percent "${VLM_SM_PERCENT}"
  )
  append_pytorch_compile_mode_arg server_cmd
  server_cmd+=(
    policy:checkpoint \
    --policy.config "${POLICY_CONFIG}" \
    --policy.dir "${POLICY_DIR}"
  )
  if [[ "${ENABLE_POLICY_BATCH}" == "true" ]]; then
    server_cmd+=(--enable-policy-batch)
  else
    server_cmd+=(--no-enable-policy-batch)
  fi
  server_cmd+=("$@")
  "${server_cmd[@]}" 2>&1 | tee "${SERVE_LOG}"
else
  echo "  profile: ${PROFILE_LOG}"
  echo "  json:    ${JSON_OUTPUT}"

  profile_cmd=(
    "${PYTHON_BIN}" scripts/profile_va_split.py
    --policy.config "${POLICY_CONFIG}" \
    --policy.dir "${POLICY_DIR}" \
    --mode "${PROFILE_MODE}" \
    --num-requests "${NUM_REQUESTS}" \
    --request-rate-hz "${REQUEST_RATE_HZ}" \
    --max-inflight "${MAX_INFLIGHT}" \
    --batch-size "${BATCH_SIZE}" \
    --seed "${SEED}" \
    --num-steps "${NUM_STEPS}" \
    --timeout-s "${TIMEOUT_S}" \
    --warmup-requests "${WARMUP_REQUESTS}" \
    --max-ae-batch-size "${MAX_AE_BATCH_SIZE}" \
    --max-vlm-batch-size "${MAX_VLM_BATCH_SIZE}" \
    --max-vlm-wait-ms "${MAX_VLM_WAIT_MS}" \
    --ae-sm-percent "${AE_SM_PERCENT}" \
    --vlm-sm-percent "${VLM_SM_PERCENT}" \
    --gpu-device-index "${GPU_ID}" \
    --json-output "${JSON_OUTPUT}"
  )
  append_pytorch_compile_mode_arg profile_cmd
  if [[ "${PROFILE_MODE}" != "split-mps" ]]; then
    profile_cmd+=(--no-require-mps-env)
  fi
  profile_cmd+=("$@")
  "${profile_cmd[@]}" 2>&1 | tee "${PROFILE_LOG}"
fi
