#!/usr/bin/env bash
# Start OpenPI VA-split policy server under NVIDIA MPS.
# All logs for one run go under: /data1/tianze/V-A schedule/logs/<timestamp>/*.log
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

GPU_ID="${GPU_ID:-0}"
PORT="${PORT:-8000}"
POLICY_CONFIG="${POLICY_CONFIG:-pi05_libero}"
POLICY_DIR="${POLICY_DIR:?POLICY_DIR must point to a PyTorch checkpoint directory containing model.safetensors}"
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
PYTHON_BIN="${PYTHON_BIN:-python}"

mkdir -p "${MPS_PIPE_DIR}" "${MPS_LOG_DIR}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export CUDA_MPS_PIPE_DIRECTORY="${MPS_PIPE_DIR}"
export CUDA_MPS_LOG_DIRECTORY="${MPS_LOG_DIR}"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}/packages/openpi-client/src${PYTHONPATH:+:${PYTHONPATH}}"

cleanup() {
  echo quit | nvidia-cuda-mps-control >/dev/null 2>&1 || true
}
trap cleanup EXIT

nvidia-cuda-mps-control -d

echo "RUN_LOG_DIR=${RUN_LOG_DIR}"
echo "  MPS:   ${MPS_LOG_DIR}/control.log ${MPS_LOG_DIR}/server.log"
echo "  serve: ${SERVE_LOG}"

"${PYTHON_BIN}" scripts/serve_policy.py \
  --port "${PORT}" \
  --policy.config "${POLICY_CONFIG}" \
  --policy.dir "${POLICY_DIR}" \
  --va-split \
  --va-split-max-ae-batch-size "${VA_SPLIT_MAX_AE_BATCH_SIZE}" \
  --ae-sm-percent "${AE_SM_PERCENT}" \
  --vlm-sm-percent "${VLM_SM_PERCENT}" \
  2>&1 | tee "${SERVE_LOG}"
