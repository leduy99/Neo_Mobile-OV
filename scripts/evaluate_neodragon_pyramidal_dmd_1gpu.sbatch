#!/usr/bin/env bash
#SBATCH --job-name=neo-dmd-eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=04:00:00
#SBATCH --output=logs/neo-dmd-eval-%j.out
#SBATCH --error=logs/neo-dmd-eval-%j.out
#SBATCH --account=berzelius-2026-196
#SBATCH --gres=gpu:1

set -euo pipefail
module load buildenv-nvhpc/24.5-cuda12.4

CONDA_ENV="${CONDA_ENV:-/proj/cvl/users/x_fahkh2/envs/neo_mobileov}"
CHECKPOINT="${CHECKPOINT:?Set CHECKPOINT=output/.../neodragon_pyramidal_dmd_student_latest.pt}"
MANIFEST="${MANIFEST:-data/neodragon_pyramidal_dmd_320p/manifest.csv}"
OUT="${OUT:-output/neodragon_pyramidal_dmd_evaluation/${SLURM_JOB_ID}}"

test -f "${CHECKPOINT}"
test -f "${MANIFEST}"
mkdir -p logs "${OUT}"
export PATH="${CONDA_ENV}/bin:${PATH}"
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1
export HF_HOME="${HF_HOME:-/proj/cvl/users/x_fahkh2/caches}"
export TORCH_HOME="${TORCH_HOME:-/proj/cvl/users/x_fahkh2/caches}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/proj/cvl/users/x_fahkh2/caches}"

"${CONDA_ENV}/bin/python" tools/evaluate_neodragon_pyramidal_dmd.py \
  --manifest "${MANIFEST}" --checkpoint "${CHECKPOINT}" --output-dir "${OUT}" \
  --index "${INDEX:-0}" --dtype "${DTYPE:-bf16}"
