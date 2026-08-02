#!/usr/bin/env bash
#SBATCH --job-name=mov-dream-dl
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --output=logs/mov-dream-dl-%j.out
#SBATCH --error=logs/mov-dream-dl-%j.out
#SBATCH --account=berzelius-2026-196
#SBATCH --gres=gpu:1
##SBATCH --gres=gpu:A100-SXM4-80GB:1

set -euo pipefail

module load buildenv-nvhpc/24.5-cuda12.4

export PYTHONPATH="./:${PYTHONPATH:-}"
export HF_HOME=/proj/cvl/users/x_fahkh2/caches
export TORCH_HOME=/proj/cvl/users/x_fahkh2/caches
export PIP_CACHE_DIR=/proj/cvl/users/x_fahkh2/caches
export TMPDIR=/proj/cvl/users/x_fahkh2/caches
export TRITON_CACHE_DIR=/proj/cvl/users/x_fahkh2/caches

CONDA_ENV="${CONDA_ENV:-/proj/cvl/users/x_fahkh2/envs/neo_mobileov}"
PYTHON_BIN="${PYTHON_BIN:-${CONDA_ENV}/bin/python}"
STOP_FILE="${TMPDIR%/}/dreamlite_download_${SLURM_JOB_ID:-local}.stop"

mkdir -p logs checkpoints
export PATH="${CONDA_ENV}/bin:${PATH}"
export PYTHONNOUSERSITE=1

rm -f "${STOP_FILE}"
"${PYTHON_BIN}" tools/utils/gpu_heartbeat.py \
  --all-devices \
  --interval 5 \
  --tensor-mb 4 \
  --work-seconds 1 \
  --stop-file "${STOP_FILE}" \
  --label "dreamlite-download-${SLURM_JOB_ID:-local}" &
HEARTBEAT_PID=$!

cleanup() {
  touch "${STOP_FILE}" 2>/dev/null || true
  wait "${HEARTBEAT_PID}" >/dev/null 2>&1 || true
  rm -f "${STOP_FILE}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

bash scripts/install_dreamlite_dependencies.sh
"${PYTHON_BIN}" tools/prepare_checkpoints.py \
  --config configs/mobile_ov_dreamlite.yaml \
  --skip-neodragon

echo "DreamLite checkpoint is ready under ./checkpoints/dreamlite-mobile"
