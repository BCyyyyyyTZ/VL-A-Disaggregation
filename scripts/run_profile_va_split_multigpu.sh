#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

BACKEND="${BACKEND:-jax}"
PROFILE_TARGET="${PROFILE_TARGET:-split}"
POLICY_CONFIG="${POLICY_CONFIG:-pi05_libero}"
POLICY_DIR="${POLICY_DIR:-/mnt/tianze/models/pi05_libero}"
LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/logs/tests/multigpu}"
RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
RUN_LOG_DIR="${RUN_LOG_DIR:-${LOG_ROOT}/${BACKEND}-${PROFILE_TARGET}-${RUN_TS}}"
NUM_REQUESTS="${NUM_REQUESTS:-128}"
REQUEST_RATE_HZ="${REQUEST_RATE_HZ:-16}"
REQUEST_RATE_HZ_VALUES="${REQUEST_RATE_HZ_VALUES:-${REQUEST_RATE_HZ_LIST:-}}"
MAX_INFLIGHT="${MAX_INFLIGHT:-64}"
SEED="${SEED:-0}"
NUM_STEPS="${NUM_STEPS:-5}"
TIMEOUT_S="${TIMEOUT_S:-60}"
MAX_AE_BATCH_SIZE="${MAX_AE_BATCH_SIZE:-999}"
MAX_VLM_BATCH_SIZE="${MAX_VLM_BATCH_SIZE:-8}"
MAX_VLM_WAIT_MS="${MAX_VLM_WAIT_MS:-1.0}"
VLM_DEVICES="${VLM_DEVICES:-0,1}"
AE_DEVICE="${AE_DEVICE:-2}"
BASELINE_DEVICES="${BASELINE_DEVICES:-0,1,2}"
WARMUP_REQUESTS="${WARMUP_REQUESTS:-4}"
WARMUP_UNTIL_STEADY="${WARMUP_UNTIL_STEADY:-1}"
WARMUP_STEADY_WINDOW="${WARMUP_STEADY_WINDOW:-4}"
WARMUP_STEADY_MAX_REQUESTS="${WARMUP_STEADY_MAX_REQUESTS:-48}"
WARMUP_CONCURRENT_INFLIGHT="${WARMUP_CONCURRENT_INFLIGHT:-4}"
JAX_COMPILE="${JAX_COMPILE:-1}"
JAX_COMPILE_WARMUP="${JAX_COMPILE_WARMUP:-1}"
JAX_COMPILE_WARMUP_MAX_BATCH_SIZE="${JAX_COMPILE_WARMUP_MAX_BATCH_SIZE:-$((MAX_VLM_BATCH_SIZE * 3))}"
JSON_OUTPUT="${JSON_OUTPUT:-${RUN_LOG_DIR}/profile.json}"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"

if [[ "${BACKEND}" != "jax" ]]; then
  echo "PyTorch multi-GPU split profile is not implemented; use BACKEND=jax." >&2
  exit 2
fi

case "${PROFILE_TARGET}" in
  split)
    MODE="jax-multigpu-split-ipc"
    ;;
  baseline)
    MODE="jax-multigpu-baseline"
    ;;
  *)
    echo "Unsupported PROFILE_TARGET=${PROFILE_TARGET}; expected split or baseline." >&2
    exit 2
    ;;
esac

mkdir -p "${RUN_LOG_DIR}"

export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}/packages/openpi-client/src${PYTHONPATH:+:${PYTHONPATH}}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false

cmd=(
  "${PYTHON_BIN}" "${SCRIPT_DIR}/profile_va_split.py"
  --policy.config "${POLICY_CONFIG}"
  --policy.dir "${POLICY_DIR}"
  --mode "${MODE}"
  --num-requests "${NUM_REQUESTS}"
  --request-rate-hz "${REQUEST_RATE_HZ}"
  --max-inflight "${MAX_INFLIGHT}"
  --seed "${SEED}"
  --num-steps "${NUM_STEPS}"
  --timeout-s "${TIMEOUT_S}"
  --max-ae-batch-size "${MAX_AE_BATCH_SIZE}"
  --max-vlm-batch-size "${MAX_VLM_BATCH_SIZE}"
  --max-vlm-wait-ms "${MAX_VLM_WAIT_MS}"
  --warmup-requests "${WARMUP_REQUESTS}"
  --warmup-steady-window "${WARMUP_STEADY_WINDOW}"
  --warmup-steady-max-requests "${WARMUP_STEADY_MAX_REQUESTS}"
  --warmup-concurrent-inflight "${WARMUP_CONCURRENT_INFLIGHT}"
  --jax-compile-warmup-max-batch-size "${JAX_COMPILE_WARMUP_MAX_BATCH_SIZE}"
  --no-require-mps-env
  --json-output "${JSON_OUTPUT}"
)

if [[ "${MODE}" == "jax-multigpu-split-ipc" ]]; then
  cmd+=(--vlm-devices "${VLM_DEVICES}" --ae-device "${AE_DEVICE}")
else
  cmd+=(--baseline-devices "${BASELINE_DEVICES}" --batch-size "${MAX_VLM_BATCH_SIZE}")
fi

if [[ -n "${REQUEST_RATE_HZ_VALUES}" ]]; then
  cmd+=(--request-rate-hz-values "${REQUEST_RATE_HZ_VALUES}")
fi

if [[ "${JAX_COMPILE}" == "0" || "${JAX_COMPILE}" == "false" ]]; then
  cmd+=(--no-jax-compile)
fi

if [[ "${JAX_COMPILE_WARMUP}" == "0" || "${JAX_COMPILE_WARMUP}" == "false" ]]; then
  cmd+=(--no-jax-compile-warmup)
fi

if [[ "${WARMUP_UNTIL_STEADY}" == "0" || "${WARMUP_UNTIL_STEADY}" == "false" ]]; then
  cmd+=(--no-warmup-until-steady)
fi

echo "Running multi-GPU V-A profile: target=${PROFILE_TARGET} backend=${BACKEND} mode=${MODE}"
echo "  python: ${PYTHON_BIN}"
echo "  logs:   ${RUN_LOG_DIR}"
echo "  json:   ${JSON_OUTPUT}"
if [[ "${MODE}" == "jax-multigpu-split-ipc" ]]; then
  echo "  split:  vlm_devices=${VLM_DEVICES} ae_device=${AE_DEVICE}"
else
  echo "  baseline replicas: devices=${BASELINE_DEVICES}"
fi
if [[ -n "${REQUEST_RATE_HZ_VALUES}" ]]; then
  echo "  rates:  ${REQUEST_RATE_HZ_VALUES} (single warmup before sweep)"
fi
echo "  batch:  max_vlm=${MAX_VLM_BATCH_SIZE} max_ae=${MAX_AE_BATCH_SIZE} max_vlm_wait_ms=${MAX_VLM_WAIT_MS}"
echo "  compile: enabled=${JAX_COMPILE} warmup=${JAX_COMPILE_WARMUP} warmup_max_batch=${JAX_COMPILE_WARMUP_MAX_BATCH_SIZE}"
echo "  e2e_warmup: min=${WARMUP_REQUESTS} until_steady=${WARMUP_UNTIL_STEADY} window=${WARMUP_STEADY_WINDOW} max=${WARMUP_STEADY_MAX_REQUESTS}"
"${cmd[@]}" "$@"
