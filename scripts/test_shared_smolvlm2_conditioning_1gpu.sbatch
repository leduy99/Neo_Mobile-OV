#!/usr/bin/env bash
# Submit from the repository root:
#   sbatch scripts/test_shared_smolvlm2_conditioning_1gpu.sbatch
# This validates Exp1 video-condition parity and measures one versus two
# SmolVLM2 forwards. It does not validate visual quality of a legacy image head.
#SBATCH --job-name=mov-shared-smol
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:30:00
#SBATCH --output=logs/mov-shared-smol-%j.out
#SBATCH --error=logs/mov-shared-smol-%j.out
#SBATCH --account=berzelius-2026-196
#SBATCH --gres=gpu:1

set -euo pipefail

module load buildenv-nvhpc/24.5-cuda12.4

ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
CONDA_ENV="${CONDA_ENV:-/proj/cvl/users/x_fahkh2/envs/neo_mobileov}"
PYTHON_BIN="${CONDA_ENV}/bin/python"
IMAGE_CKPT="${IMAGE_CKPT:-checkpoints/hf_mobile_ov/dreamlite_compact_v7_mobileo/17242707/dreamlite_image_bridge_step120000.pt}"
VIDEO_CKPT="${VIDEO_CKPT:-checkpoints/hf_mobile_ov/neo_exp1_bridge_functional/17108893/neodragon_text_bridge_latest.pt}"
OUT="${OUT:-output/shared_smolvlm2/${SLURM_JOB_ID:-local}/conditioning_test.json}"

test -x "${PYTHON_BIN}"
test -f "${ROOT}/${IMAGE_CKPT}"
test -f "${ROOT}/${VIDEO_CKPT}"
cd "${ROOT}"
mkdir -p logs "$(dirname "${OUT}")"

export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-/proj/cvl/users/x_fahkh2/caches}"
export TORCH_HOME="${TORCH_HOME:-/proj/cvl/users/x_fahkh2/caches}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/proj/cvl/users/x_fahkh2/caches}"
export TMPDIR="${TMPDIR:-/proj/cvl/users/x_fahkh2/caches}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/proj/cvl/users/x_fahkh2/caches}"

"${PYTHON_BIN}" -u tools/test_shared_smolvlm2_conditioning.py \
  --image-bridge-ckpt "${IMAGE_CKPT}" \
  --video-bridge-ckpt "${VIDEO_CKPT}" \
  --warmup "${WARMUP:-2}" \
  --runs "${RUNS:-10}" \
  --output "${OUT}"
