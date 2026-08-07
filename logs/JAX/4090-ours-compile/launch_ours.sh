#!/usr/bin/env bash
set -euo pipefail
REPO=/mnt/tianze/VL-A-Disaggregation
cd "$REPO"
mkdir -p "$REPO/logs/JAX/4090-ours-compile"
ts=$(date +%H%M%S)
if [[ -f "$REPO/logs/JAX/4090-ours-compile/run.log" ]]; then
  mv -f "$REPO/logs/JAX/4090-ours-compile/run.log" "$REPO/logs/JAX/4090-ours-compile/run.prev_${ts}.log"
fi

# Use script defaults for batch sizes (MAX_VLM_BATCH_SIZE=8, MAX_AE_BATCH_SIZE=999).
env -u CUDA_VISIBLE_DEVICES \
  MODE=jax-split-ipc \
  GPU_ID=0 \
  NUM_REQUESTS=200 \
  REQUEST_RATE_HZ_LIST=8,16,32 \
  JAX_COMPILE=1 \
  JAX_COMPILE_WARMUP=1 \
  WARMUP_UNTIL_STEADY=1 \
  TIMEOUT_S=600 \
  PYTHONUNBUFFERED=1 \
  RUN_LOG_DIR="$REPO/logs/JAX/4090-ours-compile" \
  JSON_OUTPUT="$REPO/logs/JAX/4090-ours-compile/profile.json" \
  bash "$REPO/scripts/run_profile_va_split_jax.sh" \
  >"$REPO/logs/JAX/4090-ours-compile/run.log" 2>&1 &

echo "ours_pid=$!"
echo "launched"
