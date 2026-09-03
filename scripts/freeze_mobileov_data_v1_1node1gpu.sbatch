#!/usr/bin/env bash
#SBATCH --job-name=mov-data-freeze
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=logs/mov-data-freeze-%j.out
#SBATCH --error=logs/mov-data-freeze-%j.out
#SBATCH --account=berzelius-2026-196
#SBATCH --gres=gpu:1

set -euo pipefail

module load buildenv-nvhpc/24.5-cuda12.4

CONDA_ENV="${CONDA_ENV:-/proj/cvl/users/x_fahkh2/envs/neo_mobileov}"
PYTHON_BIN="${CONDA_ENV}/bin/python"
CONFIG="${CONFIG:-configs/mobileov_data_v1.yaml}"
RELEASE_ROOT="${RELEASE_ROOT:-data/mobileov_data_v1/releases/v1}"
STOP_FILE="${TMPDIR:-/proj/cvl/users/x_fahkh2/caches}/mov_data_freeze_${SLURM_JOB_ID:-local}.stop"

test -x "${PYTHON_BIN}"
test -s "${CONFIG}"
mkdir -p logs "$(dirname "${RELEASE_ROOT}")"

export PATH="${CONDA_ENV}/bin:${PATH}"
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1
export TMPDIR="${TMPDIR:-/proj/cvl/users/x_fahkh2/caches}"

rm -f "${STOP_FILE}"
"${PYTHON_BIN}" tools/utils/gpu_heartbeat.py \
  --devices 0 --interval 5 --tensor-mb 4 --work-seconds 1 \
  --stop-file "${STOP_FILE}" --label "mobileov-data-freeze-${SLURM_JOB_ID:-local}" &
HEARTBEAT_PID=$!
cleanup() {
  touch "${STOP_FILE}" 2>/dev/null || true
  wait "${HEARTBEAT_PID}" >/dev/null 2>&1 || true
  rm -f "${STOP_FILE}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

BUILD_ARGS=(--config "${CONFIG}" --output-dir "${RELEASE_ROOT}")
if [[ "${FORCE:-0}" == "1" ]]; then
  BUILD_ARGS+=(--force)
fi
"${PYTHON_BIN}" tools/build_mobileov_data_release.py "${BUILD_ARGS[@]}"
"${PYTHON_BIN}" tools/validate_mobileov_data_release.py \
  --release-root "${RELEASE_ROOT}" \
  --sample-checks-per-pool "${SAMPLE_CHECKS_PER_POOL:-64}"

cat "${RELEASE_ROOT}/stats/preflight.json"
echo "MobileOV-Data-v1 is frozen and training-ready: ${RELEASE_ROOT}"
