#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/miliang/VL-A-Disaggregation}"
PYTHON="${PYTHON:-${REPO_ROOT}/.venv/bin/python}"
GPU="${GPU:-7}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/home/miliang/model/openpi-assets/checkpoints/pi05_libero}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/analyse/summary/jax_va_overlap}"
SNAPSHOT_DIR="${SNAPSHOT_DIR:-${REPO_ROOT}/analyse/summary/memory_snapshots}"
MPS_ROOT="${MPS_ROOT:-/tmp/openpi-jax-va-overlap-mps-${USER:-user}-${GPU}}"

export JAXTYPING_DISABLE="${JAXTYPING_DISABLE:-1}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export CUDA_MPS_PIPE_DIRECTORY="${CUDA_MPS_PIPE_DIRECTORY:-${MPS_ROOT}/pipe}"
export CUDA_MPS_LOG_DIRECTORY="${CUDA_MPS_LOG_DIRECTORY:-${MPS_ROOT}/log}"

cd "${REPO_ROOT}"
mkdir -p "${OUTPUT_DIR}" "${SNAPSHOT_DIR}" analyse/logs "${CUDA_MPS_PIPE_DIRECTORY}" "${CUDA_MPS_LOG_DIRECTORY}"

cleanup_mps() {
  echo quit | nvidia-cuda-mps-control >/dev/null 2>&1 || true
}
trap cleanup_mps EXIT

CUDA_VISIBLE_DEVICES="${GPU}" nvidia-cuda-mps-control -d

"${PYTHON}" analyse/src/jax_va_overlap_bench.py \
    --gpu "${GPU}" \
    --checkpoint-dir "${CHECKPOINT_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --snapshot-dir "${SNAPSHOT_DIR}" \
    --batch-sizes 1,2,4,8,16,32,64 \
    --num-steps 5 \
    --warmup 3 \
    --repeats 10 \
    --concurrent-duration-s 12 \
    --experiment all \
    2>&1 | tee analyse/logs/run_all_gpu${GPU}.log

"${PYTHON}" analyse/src/summarize_overlap.py \
  --output-dir "${OUTPUT_DIR}" \
  2>&1 | tee analyse/logs/summarize_gpu${GPU}.log
