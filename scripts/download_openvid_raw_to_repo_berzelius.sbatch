#!/usr/bin/env bash
#SBATCH --job-name=mobile-ov-dl
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --output=logs/slurmdl-%j.out
#SBATCH --error=logs/slurmdl-%j.out
#SBATCH --account=berzelius-2026-196
#SBATCH --gres=gpu:1
##SBATCH --gres=gpu:A100-SXM4-80GB:1

set -euo pipefail

module load buildenv-nvhpc/24.5-cuda12.4
#module load buildenv-gcccuda/12.9.1-gcc11

export PYTHONPATH="./:${PYTHONPATH:-}"
export HF_HOME=/proj/cvl/users/x_fahkh2/caches
export TORCH_HOME=/proj/cvl/users/x_fahkh2/caches
export PIP_CACHE_DIR=/proj/cvl/users/x_fahkh2/caches
export TMPDIR=/proj/cvl/users/x_fahkh2/caches
export TRITON_CACHE_DIR=/proj/cvl/users/x_fahkh2/caches

CONDA_ENV="${CONDA_ENV:-/proj/cvl/users/x_fahkh2/envs/neo_mobileov}"
PYTHON_BIN="${PYTHON_BIN:-${CONDA_ENV}/bin/python}"
DOWNLOAD_WORKERS="${DOWNLOAD_WORKERS:-8}"
HEARTBEAT_INTERVAL="${HEARTBEAT_INTERVAL:-5}"
HEARTBEAT_SIZE="${HEARTBEAT_SIZE:-0}"
HEARTBEAT_TENSOR_MB="${HEARTBEAT_TENSOR_MB:-4}"
HEARTBEAT_WORK_SECONDS="${HEARTBEAT_WORK_SECONDS:-2}"
HEARTBEAT_STOP_FILE="${TMPDIR%/}/mobileov_download_heartbeat_stop_${SLURM_JOB_ID:-local}.flag"

mkdir -p logs "${HF_HOME}" "${TORCH_HOME}" "${PIP_CACHE_DIR}" "${TMPDIR}" "${TRITON_CACHE_DIR}"

export PATH="${CONDA_ENV}/bin:${PATH}"
export PYTHONPATH="${PWD}:${PYTHONPATH}"
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Missing python: ${PYTHON_BIN}" >&2
  exit 1
fi

echo "PWD=${PWD}"
echo "CONDA_ENV=${CONDA_ENV}"
echo "DOWNLOAD_WORKERS=${DOWNLOAD_WORKERS}"
echo "PARTS=${PARTS:-<infer-from-manifest>}"
echo "MAX_PARTS=${MAX_PARTS:--1} START_OFFSET=${START_OFFSET:-0}"
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
cd "'"${PWD}"'"
export PYTHONNOUSERSITE=1
export PYTHONPATH="./:${PYTHONPATH:-}"
"'"${PYTHON_BIN}"'" tools/utils/gpu_heartbeat.py \
  --devices all \
  --interval "'"${HEARTBEAT_INTERVAL}"'" \
  --size "'"${HEARTBEAT_SIZE}"'" \
  --tensor-mb "'"${HEARTBEAT_TENSOR_MB}"'" \
  --work-seconds "'"${HEARTBEAT_WORK_SECONDS}"'" \
  --stop-file "'"${HEARTBEAT_STOP_FILE}"'" \
  --label "openvid-download-'"${SLURM_JOB_ID:-local}"'-node-${SLURMD_NODENAME:-unknown}"
' &
HEARTBEAT_PID=$!

cleanup() {
  if kill -0 "${HEARTBEAT_PID}" >/dev/null 2>&1; then
    touch "${HEARTBEAT_STOP_FILE}" 2>/dev/null || true
    for _ in $(seq 1 30); do
      if ! kill -0 "${HEARTBEAT_PID}" >/dev/null 2>&1; then
        break
      fi
      sleep 1
    done
  fi
  if kill -0 "${HEARTBEAT_PID}" >/dev/null 2>&1; then
    kill "${HEARTBEAT_PID}" >/dev/null 2>&1 || true
    wait "${HEARTBEAT_PID}" >/dev/null 2>&1 || true
  fi
  rm -f "${HEARTBEAT_STOP_FILE}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

sleep 3
if ! kill -0 "${HEARTBEAT_PID}" >/dev/null 2>&1; then
  wait "${HEARTBEAT_PID}" >/dev/null 2>&1 || true
  echo "GPU heartbeat failed to start; aborting to avoid idle-GPU job cancellation." >&2
  exit 1
fi

DOWNLOAD_WORKERS="${DOWNLOAD_WORKERS}" bash scripts/download_openvid_raw_to_repo.sh

cleanup

if [[ -f download_data/data/openvid/openvid_raw_download_summary.json ]]; then
  tail -n 80 download_data/data/openvid/openvid_raw_download_summary.json || true
fi
echo "Completed OpenVid raw download/extract job."
