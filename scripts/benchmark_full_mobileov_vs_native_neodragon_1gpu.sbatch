#!/usr/bin/env bash
# Fair warm-latency comparison. Both paths use NeoDragon Hybrid 1-1-1 at
# 512x320x49; Mobile-OV uses DreamLite V7 and the Exp1 64K video bridge.
# Submit from the repository root:
#   sbatch scripts/benchmark_full_mobileov_vs_native_neodragon_1gpu.sbatch
#SBATCH --job-name=mov-neo-e2e
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=04:00:00
#SBATCH --output=logs/mov-neo-e2e-%j.out
#SBATCH --error=logs/mov-neo-e2e-%j.out
#SBATCH --account=berzelius-2026-196
#SBATCH --gres=gpu:1
##SBATCH --gres=gpu:A100-SXM4-80GB:1

set -euo pipefail

module load buildenv-nvhpc/24.5-cuda12.4

ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
CONDA_ENV="${CONDA_ENV:-/proj/cvl/users/x_fahkh2/envs/neo_mobileov}"
PYTHON_BIN="${CONDA_ENV}/bin/python"
IMAGE_CKPT="${IMAGE_CKPT:-checkpoints/hf_mobile_ov/dreamlite_compact_v7_mobileo/17242707/dreamlite_image_bridge_step120000.pt}"
VIDEO_CKPT="${VIDEO_CKPT:-checkpoints/hf_mobile_ov/neo_exp1_bridge_functional/17108893/neodragon_text_bridge_latest.pt}"
OUT="${OUT:-output/full_pipeline_latency/${SLURM_JOB_ID:-local}}"

test -x "${PYTHON_BIN}"
test -f "${ROOT}/${IMAGE_CKPT}"
test -f "${ROOT}/${VIDEO_CKPT}"
cd "${ROOT}"
mkdir -p logs "${OUT}"

export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-/proj/cvl/users/x_fahkh2/caches}"
export TORCH_HOME="${TORCH_HOME:-/proj/cvl/users/x_fahkh2/caches}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/proj/cvl/users/x_fahkh2/caches}"
export TMPDIR="${TMPDIR:-/proj/cvl/users/x_fahkh2/caches}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/proj/cvl/users/x_fahkh2/caches}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

nvidia-smi || true
"${PYTHON_BIN}" tools/benchmark_full_mobileov_vs_native_neodragon.py \
  --image-config "${IMAGE_CONFIG:-configs/mobile_ov_dreamlite_compact_v7.yaml}" \
  --video-config "${VIDEO_CONFIG:-configs/mobile_ov_neodragon.yaml}" \
  --image-bridge-ckpt "${IMAGE_CKPT}" \
  --video-bridge-ckpt "${VIDEO_CKPT}" \
  --output-dir "${OUT}" \
  --warmup "${WARMUP:-1}" \
  --runs "${RUNS:-2}" \
  --with-quicksr
