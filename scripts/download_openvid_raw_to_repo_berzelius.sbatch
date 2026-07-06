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
DOWNLOAD_WORKERS="${DOWNLOAD_WORKERS:-4}"
HEARTBEAT_INTERVAL="${HEARTBEAT_INTERVAL:-15}"
HEARTBEAT_SIZE="${HEARTBEAT_SIZE:-1024}"

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

"${PYTHON_BIN}" tools/utils/gpu_heartbeat.py \
  --devices all \
  --interval "${HEARTBEAT_INTERVAL}" \
  --size "${HEARTBEAT_SIZE}" \
  --label "openvid-download-${SLURM_JOB_ID:-local}" &
HEARTBEAT_PID=$!

cleanup() {
  if kill -0 "${HEARTBEAT_PID}" >/dev/null 2>&1; then
    kill "${HEARTBEAT_PID}" >/dev/null 2>&1 || true
    wait "${HEARTBEAT_PID}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT INT TERM

DOWNLOAD_WORKERS="${DOWNLOAD_WORKERS}" bash scripts/download_openvid_raw_to_repo.sh

cleanup

if [[ -f download_data/data/openvid/openvid_raw_download_summary.json ]]; then
  tail -n 80 download_data/data/openvid/openvid_raw_download_summary.json || true
fi
echo "Completed OpenVid raw download/extract job."
