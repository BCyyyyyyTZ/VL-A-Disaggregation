#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/miliang/VL-A-Disaggregation}"
PYTHON="${PYTHON:-${REPO_ROOT}/.venv/bin/python}"
GPU="${GPU:-7}"
BATCH_SIZE="${BATCH_SIZE:-8}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/home/miliang/model/openpi-assets/checkpoints/pi05_libero}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/analyse/summary/nsys_b${BATCH_SIZE}}"
SNAPSHOT_DIR="${SNAPSHOT_DIR:-${REPO_ROOT}/analyse/summary/memory_snapshots}"
NSYS_OUT="${NSYS_OUT:-${REPO_ROOT}/analyse/summary/nsys_va_overlap_b${BATCH_SIZE}}"
MPS_ROOT="${MPS_ROOT:-/tmp/openpi-jax-va-overlap-nsys-mps-${USER:-user}-${GPU}-${BATCH_SIZE}}"

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

nsys profile \
  --trace=cuda,nvtx,osrt \
  --cuda-memory-usage=true \
  --force-overwrite=true \
  -o "${NSYS_OUT}" \
  "${PYTHON}" analyse/src/jax_va_overlap_bench.py \
    --gpu "${GPU}" \
    --checkpoint-dir "${CHECKPOINT_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --snapshot-dir "${SNAPSHOT_DIR}" \
    --batch-sizes "${BATCH_SIZE}" \
    --num-steps 5 \
    --warmup 3 \
    --repeats 6 \
    --concurrent-duration-s 10 \
    --experiment concurrent \
  2>&1 | tee "analyse/logs/nsys_b${BATCH_SIZE}_gpu${GPU}.log"
