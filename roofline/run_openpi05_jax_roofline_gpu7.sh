#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/miliang/VL-A-Disaggregation"
PYTHON="${REPO_ROOT}/.venv/bin/python"
CHECKPOINT_DIR="/home/miliang/model/openpi-assets/checkpoints/pi05_libero"
OUTPUT_DIR="${REPO_ROOT}/roofline/logs/openpi05_jax_roofline"
GPU="7"

# Lowering passes ArgInfo into typed dataclasses; disable jaxtyping for this bench.
export JAXTYPING_DISABLE="${JAXTYPING_DISABLE:-1}"

cd "${REPO_ROOT}"

"${PYTHON}" roofline/bench_hardware_roofline.py \
  --gpu "${GPU}" \
  --output roofline/logs/hardware_roofline.json

"${PYTHON}" roofline/bench_openpi05_jax_roofline.py \
  --gpu "${GPU}" \
  --checkpoint-dir "${CHECKPOINT_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  --batch-sizes 1,4,8,16,32,64,128 \
  --denoise-steps 5,10 \
  --compile-warmup 5 \
  --measure-repeats 20 \
  --cost-source static

"${PYTHON}" roofline/plot_roofline.py \
  --hardware roofline/logs/hardware_roofline.json \
  --openpi "${OUTPUT_DIR}/summary.json" \
  --output roofline/logs/openpi05_jax_roofline.png
