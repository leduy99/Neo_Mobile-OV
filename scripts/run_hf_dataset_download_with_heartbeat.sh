#!/usr/bin/env bash

set -euo pipefail

: "${DATASET_REPO_ID:?DATASET_REPO_ID must be set}"
: "${DATASET_OUTPUT_DIR:?DATASET_OUTPUT_DIR must be set}"

CONDA_ENV="${CONDA_ENV:-/proj/cvl/users/x_fahkh2/envs/neo_mobileov}"
PYTHON_BIN="${PYTHON_BIN:-${CONDA_ENV}/bin/python}"
DOWNLOAD_WORKERS="${DOWNLOAD_WORKERS:-8}"
DOWNLOAD_RETRIES="${DOWNLOAD_RETRIES:-12}"
DATASET_REVISION="${DATASET_REVISION:-main}"
HEARTBEAT_INTERVAL="${HEARTBEAT_INTERVAL:-5}"
HEARTBEAT_TENSOR_MB="${HEARTBEAT_TENSOR_MB:-4}"
HEARTBEAT_WORK_SECONDS="${HEARTBEAT_WORK_SECONDS:-2}"
HEARTBEAT_STOP_FILE="${TMPDIR%/}/mobileov_dataset_download_${SLURM_JOB_ID:-local}.flag"

mkdir -p logs "${DATASET_OUTPUT_DIR}" "${HF_HOME}" "${TMPDIR}"

export PATH="${CONDA_ENV}/bin:${PATH}"
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-3600}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-60}"
# Eight file-level processes are already concurrent. Keep each Xet transfer
# bounded so aggregate requests stay fast without overwhelming the shared link.
export HF_XET_NUM_CONCURRENT_RANGE_GETS="${HF_XET_NUM_CONCURRENT_RANGE_GETS:-4}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Missing Python executable: ${PYTHON_BIN}" >&2
  exit 1
fi

echo "DATASET_REPO_ID=${DATASET_REPO_ID}"
echo "DATASET_OUTPUT_DIR=${DATASET_OUTPUT_DIR}"
echo "DATASET_REVISION=${DATASET_REVISION}"
echo "DOWNLOAD_WORKERS=${DOWNLOAD_WORKERS} DOWNLOAD_RETRIES=${DOWNLOAD_RETRIES}"
df -h "${DATASET_OUTPUT_DIR}" || true
nvidia-smi || true

rm -f "${HEARTBEAT_STOP_FILE}"
srun --overlap \
  --nodes=1 \
  --ntasks=1 \
  --gpus-per-task=1 \
  --gpu-bind=single:1 \
  bash -lc '
set -euo pipefail
export PATH="'"${CONDA_ENV}"'/bin:$PATH"
export PYTHONPATH="'"${PWD}"':${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1
"'"${PYTHON_BIN}"'" "'"${PWD}"'/tools/utils/gpu_heartbeat.py" \
  --devices all \
  --interval "'"${HEARTBEAT_INTERVAL}"'" \
  --tensor-mb "'"${HEARTBEAT_TENSOR_MB}"'" \
  --work-seconds "'"${HEARTBEAT_WORK_SECONDS}"'" \
  --stop-file "'"${HEARTBEAT_STOP_FILE}"'" \
  --label "'"${SLURM_JOB_NAME:-dataset-download}"'-'"${SLURM_JOB_ID:-local}"'"
' &
HEARTBEAT_PID=$!

cleanup() {
  touch "${HEARTBEAT_STOP_FILE}" 2>/dev/null || true
  for _ in $(seq 1 30); do
    if ! kill -0 "${HEARTBEAT_PID}" >/dev/null 2>&1; then
      break
    fi
    sleep 1
  done
  if kill -0 "${HEARTBEAT_PID}" >/dev/null 2>&1; then
    kill "${HEARTBEAT_PID}" >/dev/null 2>&1 || true
  fi
  wait "${HEARTBEAT_PID}" >/dev/null 2>&1 || true
  rm -f "${HEARTBEAT_STOP_FILE}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

sleep 3
if ! kill -0 "${HEARTBEAT_PID}" >/dev/null 2>&1; then
  wait "${HEARTBEAT_PID}" >/dev/null 2>&1 || true
  echo "GPU heartbeat failed to start; aborting the download job." >&2
  exit 1
fi

DOWNLOAD_ARGS=(
  --repo-id "${DATASET_REPO_ID}"
  --output-dir "${DATASET_OUTPUT_DIR}"
  --revision "${DATASET_REVISION}"
  --workers "${DOWNLOAD_WORKERS}"
  --retries "${DOWNLOAD_RETRIES}"
)
if [[ "${REFRESH_DATASET_REVISION:-0}" == "1" ]]; then
  DOWNLOAD_ARGS+=(--refresh-revision)
fi
if [[ "${SKIP_DISK_CHECK:-0}" == "1" ]]; then
  DOWNLOAD_ARGS+=(--skip-disk-check)
fi

"${PYTHON_BIN}" tools/data_prepare/download_hf_dataset_parallel.py "${DOWNLOAD_ARGS[@]}"
cleanup

tail -n 20 "${DATASET_OUTPUT_DIR}/download_summary.json"
