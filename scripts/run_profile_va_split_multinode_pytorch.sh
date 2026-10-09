#!/usr/bin/env bash
# PyTorch multi-GPU / multi-node VLM–AE profile.
# Same machine (default): N VLM GPUs + 1 AE GPU, host shared-memory prefix.
#   GPU_LIST is not used; set VLM_DEVICES and AE_DEVICE.
#   MODE=ours VLM_DEVICES=0,1 AE_DEVICE=2 bash scripts/run_profile_va_split_multinode_pytorch.sh
# Cross machine:
#   # coordinator host
#   ROLE=coordinator TRANSPORT=tcp RESULT_ADDR=0.0.0.0:9200 VLM_ADDRS=10.0.0.2:9110,10.0.0.3:9111 \
#     bash scripts/run_profile_va_split_multinode_pytorch.sh
#   # AE host
#   ROLE=ae TRANSPORT=tcp RESULT_ADDR=<coordinator>:9200 AE_ADDR=0.0.0.0:9100 AE_DEVICE=0 \
#     bash scripts/run_profile_va_split_multinode_pytorch.sh
#   # VLM hosts (AE_ADDR is the reachable AE host:port)
#   ROLE=vlm TRANSPORT=tcp AE_ADDR=<ae-host>:9100 WORKER_INDEX=0 VLM_DEVICES=0 BIND=0.0.0.0:9110 \
#     bash scripts/run_profile_va_split_multinode_pytorch.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"

ROLE="${ROLE:-all}"
TRANSPORT="${TRANSPORT:-host_shm}"
MODE="${MODE:-ours}"
LAUNCH="${LAUNCH:-process}"
VLM_DEVICES="${VLM_DEVICES:-0,1}"
AE_DEVICE="${AE_DEVICE:-2}"
BASELINE_DEVICES="${BASELINE_DEVICES:-}"
POLICY_CONFIG="${POLICY_CONFIG:-pi05_libero}"
POLICY_DIR="${POLICY_DIR:-/mnt/tianze/models/pi05_libero_pytorch}"
NUM_REQUESTS="${NUM_REQUESTS:-64}"
RATE="${RATE:-16}"
RATES="${RATES:-}"
SEED="${SEED:-0}"
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-8}"
AE_MAX_BATCH_SIZE="${AE_MAX_BATCH_SIZE:-16}"
MAX_PREFIX_SLOTS="${MAX_PREFIX_SLOTS:-256}"
MAX_WAIT_MS="${MAX_WAIT_MS:-1.0}"
NUM_STEPS="${NUM_STEPS:-5}"
TIMEOUT_S="${TIMEOUT_S:-600}"
STARTUP_TIMEOUT_S="${STARTUP_TIMEOUT_S:-600}"
SLO_MS="${SLO_MS:-200}"
STATE_DIM="${STATE_DIM:-32}"
ACTION_DIM="${ACTION_DIM:-32}"
ACTION_HORIZON="${ACTION_HORIZON:-50}"
TOKEN_LEN="${TOKEN_LEN:-200}"
OVERLAP_D2H="${OVERLAP_D2H:-1}"
PACKED_NOISE="${PACKED_NOISE:-1}"
WARMUP="${WARMUP:-1}"
WARMUP_MAX_BATCH="${WARMUP_MAX_BATCH:-4}"
WORKER_INDEX="${WORKER_INDEX:-0}"
AE_ADDR="${AE_ADDR:-}"
VLM_ADDRS="${VLM_ADDRS:-}"
RESULT_ADDR="${RESULT_ADDR:-}"
BIND="${BIND:-0.0.0.0}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
LOG_ROOT="${LOG_ROOT:-${REPO_ROOT}/logs/multinode-pytorch}"
RUN_LOG_DIR="${RUN_LOG_DIR:-${LOG_ROOT}/${RUN_TS}}"
OUTPUT="${OUTPUT:-${RUN_LOG_DIR}/profile.json}"

mkdir -p "${RUN_LOG_DIR}"

ARGS=(
  --backend pytorch
  --mode "${MODE}"
  --role "${ROLE}"
  --transport "${TRANSPORT}"
  --launch "${LAUNCH}"
  --vlm-devices "${VLM_DEVICES}"
  --ae-device "${AE_DEVICE}"
  --policy-config "${POLICY_CONFIG}"
  --policy-dir "${POLICY_DIR}"
  --num-requests "${NUM_REQUESTS}"
  --rate "${RATE}"
  --seed "${SEED}"
  --max-batch-size "${MAX_BATCH_SIZE}"
  --ae-max-batch-size "${AE_MAX_BATCH_SIZE}"
  --max-prefix-slots "${MAX_PREFIX_SLOTS}"
  --max-wait-ms "${MAX_WAIT_MS}"
  --num-steps "${NUM_STEPS}"
  --timeout-s "${TIMEOUT_S}"
  --startup-timeout-s "${STARTUP_TIMEOUT_S}"
  --slo-ms "${SLO_MS}"
  --state-dim "${STATE_DIM}"
  --action-dim "${ACTION_DIM}"
  --action-horizon "${ACTION_HORIZON}"
  --token-len "${TOKEN_LEN}"
  --warmup-max-batch "${WARMUP_MAX_BATCH}"
  --worker-index "${WORKER_INDEX}"
  --bind-host "${BIND}"
  --output "${OUTPUT}"
)
if [[ -n "${RATES}" ]]; then
  ARGS+=(--rates "${RATES}")
fi
if [[ -n "${BASELINE_DEVICES}" ]]; then
  ARGS+=(--baseline-devices "${BASELINE_DEVICES}")
fi
if [[ -n "${AE_ADDR}" ]]; then
  ARGS+=(--ae-addr "${AE_ADDR}")
fi
if [[ -n "${VLM_ADDRS}" ]]; then
  ARGS+=(--vlm-addrs "${VLM_ADDRS}")
fi
if [[ -n "${RESULT_ADDR}" ]]; then
  ARGS+=(--result-addr "${RESULT_ADDR}")
fi
if [[ "${OVERLAP_D2H}" == "0" || "${OVERLAP_D2H}" == "false" ]]; then
  ARGS+=(--no-overlap-d2h)
fi
if [[ "${PACKED_NOISE}" == "0" || "${PACKED_NOISE}" == "false" ]]; then
  ARGS+=(--no-packed-noise)
fi
if [[ "${WARMUP}" == "0" || "${WARMUP}" == "false" ]]; then
  ARGS+=(--no-warmup)
fi

{
  echo "RUN_LOG_DIR=${RUN_LOG_DIR}"
  echo "ROLE=${ROLE} MODE=${MODE} TRANSPORT=${TRANSPORT} VLM_DEVICES=${VLM_DEVICES} AE_DEVICE=${AE_DEVICE}"
  PYTHONDONTWRITEBYTECODE=1 "${PYTHON_BIN}" "${SCRIPT_DIR}/profile_va_split_multinode.py" "${ARGS[@]}"
} 2>&1 | tee "${RUN_LOG_DIR}/run_stdout.log"
