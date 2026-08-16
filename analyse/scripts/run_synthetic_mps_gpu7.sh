#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/miliang/VL-A-Disaggregation}"
PYTHON="${PYTHON:-${REPO_ROOT}/.venv/bin/python}"
GPU="${GPU:-7}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/analyse/summary/synthetic_mps}"
MPS_ROOT="${MPS_ROOT:-/tmp/openpi-synthetic-mps-${USER:-user}-${GPU}}"

export JAXTYPING_DISABLE="${JAXTYPING_DISABLE:-1}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export CUDA_MPS_PIPE_DIRECTORY="${CUDA_MPS_PIPE_DIRECTORY:-${MPS_ROOT}/pipe}"
export CUDA_MPS_LOG_DIRECTORY="${CUDA_MPS_LOG_DIRECTORY:-${MPS_ROOT}/log}"

cd "${REPO_ROOT}"
mkdir -p "${OUTPUT_DIR}" analyse/logs "${CUDA_MPS_PIPE_DIRECTORY}" "${CUDA_MPS_LOG_DIRECTORY}"

cleanup_mps() {
  echo quit | nvidia-cuda-mps-control >/dev/null 2>&1 || true
}
trap cleanup_mps EXIT

CUDA_VISIBLE_DEVICES="${GPU}" nvidia-cuda-mps-control -d

"${PYTHON}" analyse/src/synthetic_mps_overlap.py \
  --gpu "${GPU}" \
  --output-dir "${OUTPUT_DIR}" \
  --duration-s 8 \
  2>&1 | tee "analyse/logs/synthetic_mps_gpu${GPU}.log"
