#!/usr/bin/env bash
# Apply the same 2x QuickSRNet post-process to videos that already exist.
# Example:
# INPUT_DIR=output/dreamlite_v7_exp1_64k_vbench5x/videos \
#   sbatch scripts/apply_quicksrnet_existing_videos_1gpu.sbatch
#
# Optional benchmark-only run (no new MP4 files):
# INPUT_VIDEO=output/.../one_video.mp4 MODE=benchmark \
#   sbatch scripts/apply_quicksrnet_existing_videos_1gpu.sbatch
#SBATCH --job-name=mov-quicksr
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --output=logs/mov-quicksr-%j.out
#SBATCH --error=logs/mov-quicksr-%j.out
#SBATCH --account=berzelius-2026-196
#SBATCH --gres=gpu:1
##SBATCH --gres=gpu:A100-SXM4-80GB:1

set -euo pipefail

module load buildenv-nvhpc/24.5-cuda12.4

ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
CONDA_ENV="${CONDA_ENV:-/proj/cvl/users/x_fahkh2/envs/neo_mobileov}"
PYTHON_BIN="${CONDA_ENV}/bin/python"
MODE="${MODE:-apply}"
VARIANT="${VARIANT:-medium}"
SCALE="${SCALE:-2}"
BATCH_FRAMES="${BATCH_FRAMES:-4}"

export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-/proj/cvl/users/x_fahkh2/caches}"
export TORCH_HOME="${TORCH_HOME:-/proj/cvl/users/x_fahkh2/caches}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/proj/cvl/users/x_fahkh2/caches}"
export TMPDIR="${TMPDIR:-/proj/cvl/users/x_fahkh2/caches}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/proj/cvl/users/x_fahkh2/caches}"

test -x "${PYTHON_BIN}"
test -f "${ROOT}/tools/apply_quicksrnet_video.py"
cd "${ROOT}"
mkdir -p logs
nvidia-smi || true

COMMON_ARGS=(
  --variant "${VARIANT}"
  --scale "${SCALE}"
  --checkpoint-dir "${ROOT}/checkpoints/quicksrnet"
  --device cuda
  --dtype "${DTYPE:-fp16}"
  --batch-frames "${BATCH_FRAMES}"
)

if [[ "${MODE}" == "benchmark" ]]; then
  : "${INPUT_VIDEO:?Set INPUT_VIDEO=/path/to/an-existing.mp4 for MODE=benchmark}"
  OUT="${OUT:-output/quicksrnet_benchmark/${SLURM_JOB_ID:-local}}"
  "${PYTHON_BIN}" tools/apply_quicksrnet_video.py \
    --input-video "${INPUT_VIDEO}" \
    --output-dir "${OUT}" \
    --benchmark-runs "${RUNS:-30}" \
    --warmup-runs "${WARMUP:-10}" \
    "${COMMON_ARGS[@]}"
elif [[ "${MODE}" == "apply" ]]; then
  : "${INPUT_DIR:?Set INPUT_DIR=output/.../videos; this script never regenerates videos}"
  OUT="${OUT:-${INPUT_DIR%/}_quicksrnet_${VARIANT}_${SCALE}x}"
  "${PYTHON_BIN}" tools/apply_quicksrnet_video.py \
    --input-dir "${INPUT_DIR}" \
    --output-dir "${OUT}" \
    --glob "${VIDEO_GLOB:-**/*.mp4}" \
    --limit "${LIMIT:-0}" \
    "${COMMON_ARGS[@]}"
else
  echo "MODE must be apply or benchmark; got ${MODE}" >&2
  exit 2
fi
