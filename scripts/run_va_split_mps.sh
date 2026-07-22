#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${GPU_ID:-0}"
PORT="${PORT:-8000}"
POLICY_CONFIG="${POLICY_CONFIG:-pi05_libero}"
POLICY_DIR="${POLICY_DIR:?POLICY_DIR must point to a PyTorch checkpoint directory containing model.safetensors}"
MPS_PIPE_DIR="${MPS_PIPE_DIR:-/tmp/openpi-mps-${USER}-gpu${GPU_ID}}"
MPS_LOG_DIR="${MPS_LOG_DIR:-/tmp/openpi-mps-log-${USER}-gpu${GPU_ID}}"
AE_SM_PERCENT="${AE_SM_PERCENT:-20}"
VLM_SM_PERCENT="${VLM_SM_PERCENT:-0}"
VA_SPLIT_MAX_AE_BATCH_SIZE="${VA_SPLIT_MAX_AE_BATCH_SIZE:-8}"
PYTHON_BIN="${PYTHON_BIN:-python}"

mkdir -p "${MPS_PIPE_DIR}" "${MPS_LOG_DIR}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export CUDA_MPS_PIPE_DIRECTORY="${MPS_PIPE_DIR}"
export CUDA_MPS_LOG_DIRECTORY="${MPS_LOG_DIR}"

cleanup() {
  echo quit | nvidia-cuda-mps-control >/dev/null 2>&1 || true
}
trap cleanup EXIT

nvidia-cuda-mps-control -d

"${PYTHON_BIN}" scripts/serve_policy.py \
  --port "${PORT}" \
  --policy.config "${POLICY_CONFIG}" \
  --policy.dir "${POLICY_DIR}" \
  --va-split \
  --va-split-max-ae-batch-size "${VA_SPLIT_MAX_AE_BATCH_SIZE}" \
  --ae-sm-percent "${AE_SM_PERCENT}" \
  --vlm-sm-percent "${VLM_SM_PERCENT}"
