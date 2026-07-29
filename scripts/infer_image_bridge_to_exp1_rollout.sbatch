#!/usr/bin/env bash
#SBATCH --job-name=mov-img2neo
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=220G
#SBATCH --time=04:00:00
#SBATCH --chdir=/share_4/users/duy/project/unified_video/New-Mobile-OV
#SBATCH --output=output/image_bridge_to_exp1_rollout/logs/%x_%j.out
#SBATCH --error=output/image_bridge_to_exp1_rollout/logs/%x_%j.out

set -euo pipefail

ROOT="/share_4/users/duy/project/unified_video/New-Mobile-OV"
NEO_ROOT="/share_4/users/duy/project/unified_video/neodragon"
CONDA_ENV="${CONDA_ENV:-/share_4/users/duy/.conda/envs/neo_mobileov}"
VIDEO_CKPT="${VIDEO_CKPT:-checkpoints/hf_mobile_ov/neo_exp1_rollout_64k_to100k/17174692/neodragon_rollout_bridge_latest.pt}"
IMAGE_CKPT="${IMAGE_CKPT:-checkpoints/ssd1b_image_bridge_distill/ssd1b_image_bridge_step100000.pt}"
OUT="${OUT:-output/image_bridge_to_exp1_rollout}"

mkdir -p "${OUT}/logs"
source /share_0/conda/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"

export PYTHONPATH="${ROOT}:${NEO_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export FSDP_USE_ORIG_PARAMS=true
export HF_HOME="${NEO_ROOT}/models/hf_home"
export TORCH_HOME="${NEO_ROOT}/models/torch_home"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

cd "${ROOT}"
python tools/infer_image_bridge_to_neodragon.py \
  --video-bridge-ckpt "${VIDEO_CKPT}" \
  --image-bridge-ckpt "${IMAGE_CKPT}" \
  --output-dir "${OUT}" \
  ${EXTRA_ARGS:-}
