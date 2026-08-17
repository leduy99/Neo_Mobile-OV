#!/usr/bin/env bash
# Validate that a monolithic-trained Mobile-OV bridge controls a corrected
# all-native-unit DMD-v2 checkpoint before using the pair for deployment.
#
# Submit after bridge training and DMD-v2 training complete:
# DMD_CHECKPOINT=output/.../neodragon_pyramidal_dmd_student_step010000.pt \
# BRIDGE_CHECKPOINT=output/.../neodragon_text_bridge_best.pt \
# sbatch scripts/audit_neodragon_monolithic_bridge_dmd_v2_1node1gpu.sbatch
#SBATCH --job-name=neo-mono-dmd-audit
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=02:00:00
#SBATCH --output=logs/neo-mono-dmd-audit-%j.out
#SBATCH --error=logs/neo-mono-dmd-audit-%j.out
#SBATCH --account=berzelius-2026-196
#SBATCH --gres=gpu:1

set -euo pipefail

module load buildenv-nvhpc/24.5-cuda12.4

CONDA_ENV="${CONDA_ENV:-/proj/cvl/users/x_fahkh2/envs/neo_mobileov}"
DMD_CHECKPOINT="${DMD_CHECKPOINT:?Set DMD_CHECKPOINT=output/.../neodragon_pyramidal_dmd_student_stepXXXXXX.pt}"
BRIDGE_CHECKPOINT="${BRIDGE_CHECKPOINT:?Set BRIDGE_CHECKPOINT=output/.../neodragon_text_bridge_{best,stepXXXXXX}.pt}"
RUN_NAME="${RUN_NAME:-monolithic_bridge_dmd_${SLURM_JOB_ID}}"
OUT="${OUT:-output/neo_monolithic_bridge_dmd_audit/${RUN_NAME}}"

test -x "${CONDA_ENV}/bin/python"
test -f "${DMD_CHECKPOINT}"
test -f "${BRIDGE_CHECKPOINT}"
mkdir -p logs "${OUT}"

export PATH="${CONDA_ENV}/bin:${PATH}"
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-/proj/cvl/users/x_fahkh2/caches}"
export TORCH_HOME="${TORCH_HOME:-/proj/cvl/users/x_fahkh2/caches}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/proj/cvl/users/x_fahkh2/caches}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

"${CONDA_ENV}/bin/python" -u tools/audit_neodragon_monolithic_bridge_dmd.py \
  --dmd-checkpoint "${DMD_CHECKPOINT}" \
  --bridge-checkpoint "${BRIDGE_CHECKPOINT}" \
  --output-dir "${OUT}" \
  --seed "${SEED:-20260817}" \
  --dtype "${DTYPE:-bf16}" \
  ${EXTRA_ARGS:-}

find "${OUT}" -maxdepth 1 -type f -printf '%f %k KB\n' | sort
