#!/usr/bin/env bash
# Launch OpenPI V-A split / monolithic profile workload.
# For --mode split-mps, starts NVIDIA MPS before profiling and stops it on exit.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

GPU_ID="${GPU_ID:-3}"
MODE="${MODE:-split-mps}"
# MODE="${MODE:-monolithic}"
POLICY_CONFIG="${POLICY_CONFIG:-pi05_libero}"
POLICY_DIR="${POLICY_DIR:-/mnt/tianze/models/pi05_libero_pytorch}"
LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/logs}"
RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
RUN_LOG_DIR="${RUN_LOG_DIR:-${LOG_ROOT}/${RUN_TS}}"
MPS_PIPE_DIR="${MPS_PIPE_DIR:-${RUN_LOG_DIR}/mps-pipe}"
MPS_LOG_DIR="${MPS_LOG_DIR:-${RUN_LOG_DIR}}"
AE_SM_PERCENT="${AE_SM_PERCENT:-20}"
VLM_SM_PERCENT="${VLM_SM_PERCENT:-0}"
MAX_AE_BATCH_SIZE="${MAX_AE_BATCH_SIZE:-8}"
NUM_REQUESTS="${NUM_REQUESTS:-128}"
REQUEST_RATE_HZ="${REQUEST_RATE_HZ:-16}"
MAX_INFLIGHT="${MAX_INFLIGHT:-64}"
SEED="${SEED:-0}"
NUM_STEPS="${NUM_STEPS:-10}"
TIMEOUT_S="${TIMEOUT_S:-60}"
JSON_OUTPUT="${JSON_OUTPUT:-${RUN_LOG_DIR}/profile.json}"
CHECK_CONSISTENCY="${CHECK_CONSISTENCY:-0}"
PYTHON_BIN="${PYTHON_BIN:-${REPO_ROOT}/.venv/bin/python}"

# Prefer the local editable source over any site-packages openpi install.
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}/packages/openpi-client/src${PYTHONPATH:+:${PYTHONPATH}}"

MPS_STARTED=0

cleanup() {
  if [[ "${MPS_STARTED}" -eq 1 ]]; then
    echo quit | nvidia-cuda-mps-control >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

export CUDA_VISIBLE_DEVICES="${GPU_ID}"

case "${MODE}" in
  monolithic|split-no-mps|split-mps) ;;
  *)
    echo "Unsupported MODE=${MODE}; expected monolithic|split-no-mps|split-mps" >&2
    exit 1
    ;;
esac

mkdir -p "${RUN_LOG_DIR}"

if [[ "${MODE}" == "split-mps" ]]; then
  mkdir -p "${MPS_PIPE_DIR}" "${MPS_LOG_DIR}"
  export CUDA_MPS_PIPE_DIRECTORY="${MPS_PIPE_DIR}"
  export CUDA_MPS_LOG_DIRECTORY="${MPS_LOG_DIR}"
  nvidia-cuda-mps-control -d
  MPS_STARTED=1
fi

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
  --ae-sm-percent "${AE_SM_PERCENT}"
  --vlm-sm-percent "${VLM_SM_PERCENT}"
)

if [[ "${MODE}" != "split-mps" ]]; then
  cmd+=(--no-require-mps-env)
fi

if [[ -n "${JSON_OUTPUT}" ]]; then
  cmd+=(--json-output "${JSON_OUTPUT}")
fi

if [[ "${CHECK_CONSISTENCY}" == "1" || "${CHECK_CONSISTENCY}" == "true" ]]; then
  cmd+=(--check-consistency)
fi

# Extra tyro flags, e.g. --pytorch-device cuda:0 --fixed-noise False
cmd+=("$@")

echo "Running V-A profile: mode=${MODE} gpu=${GPU_ID} policy_dir=${POLICY_DIR}"
echo "  python:  ${PYTHON_BIN}"
echo "  logs:    ${RUN_LOG_DIR}"
echo "  json:    ${JSON_OUTPUT}"
"${cmd[@]}"
