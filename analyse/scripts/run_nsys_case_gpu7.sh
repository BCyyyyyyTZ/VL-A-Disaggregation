#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/home/miliang/VL-A-Disaggregation}"
PYTHON="${PYTHON:-${REPO_ROOT}/.venv/bin/python}"
NSYS="${NSYS:-nsys}"
# Physical GPU for MPS; clients use remapped device 0.
GPU="${GPU:-7}"
CLIENT_GPU="${CLIENT_GPU:-0}"
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

echo "Using nsys: ${NSYS}"
"${NSYS}" --version

echo quit | nvidia-cuda-mps-control >/dev/null 2>&1 || true
sleep 1
CUDA_VISIBLE_DEVICES="${GPU}" nvidia-cuda-mps-control -d

# Capture CUDA kernels + NVTX. Prefer the MEASURE_WINDOW range so compile is skipped.
NSYS_CAPTURE_MODE="${NSYS_CAPTURE_MODE:-nvtx}"
CAPTURE_ARGS=(
  --trace=cuda,nvtx
  --sample=none
  --cpuctxsw=none
  --export=none
  --force-overwrite=true
)
if [[ "${NSYS_CAPTURE_MODE}" == "nvtx" ]]; then
  CAPTURE_ARGS+=(
    --capture-range=nvtx
    --nvtx-capture=MEASURE_WINDOW
    --capture-range-end=stop
  )
fi

"${NSYS}" profile \
  "${CAPTURE_ARGS[@]}" \
  -o "${NSYS_OUT}" \
  "${PYTHON}" analyse/src/jax_va_overlap_bench.py \
    --gpu "${CLIENT_GPU}" \
    --checkpoint-dir "${CHECKPOINT_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --snapshot-dir "${SNAPSHOT_DIR}" \
    --batch-sizes "${BATCH_SIZE}" \
    --ae-batch-size "${BATCH_SIZE}" \
    --num-steps 5 \
    --warmup 3 \
    --repeats 6 \
    --concurrent-duration-s 10 \
    --experiment concurrent \
  2>&1 | tee "analyse/logs/nsys_b${BATCH_SIZE}_gpu${GPU}.log"

# Best-effort post-process. Prefer .nsys-rep when import succeeded.
if [[ -f "${NSYS_OUT}.nsys-rep" ]]; then
  "${NSYS}" stats --force-export=true "${NSYS_OUT}.nsys-rep" \
    > "${NSYS_OUT}_stats.txt" 2>&1 || true
  "${NSYS}" export \
    --type sqlite \
    --force-overwrite=true \
    -o "${NSYS_OUT}.sqlite" \
    "${NSYS_OUT}.nsys-rep" \
    > "${NSYS_OUT}_export.log" 2>&1 || true
elif [[ -f "${NSYS_OUT}.qdstrm" ]]; then
  "${NSYS}" stats --force-export=true "${NSYS_OUT}.qdstrm" \
    > "${NSYS_OUT}_stats.txt" 2>&1 || true
  "${NSYS}" export \
    --type sqlite \
    --force-overwrite=true \
    -o "${NSYS_OUT}.sqlite" \
    "${NSYS_OUT}.qdstrm" \
    > "${NSYS_OUT}_export.log" 2>&1 || true
fi

# Prefer reporting the final report/sqlite if import succeeded.
ls -lah "${NSYS_OUT}".* "${NSYS_OUT}"_* 2>/dev/null || true
