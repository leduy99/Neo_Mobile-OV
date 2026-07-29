#!/usr/bin/env bash
#SBATCH --job-name=mov-ssd-infer
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=180G
#SBATCH --time=02:00:00
#SBATCH --chdir=/share_4/users/duy/project/unified_video/New-Mobile-OV
#SBATCH --output=output/ssd1b_image_bridge_step100k_test/logs/%x_%j.out
#SBATCH --error=output/ssd1b_image_bridge_step100k_test/logs/%x_%j.out

set -euo pipefail

ROOT="/share_4/users/duy/project/unified_video/New-Mobile-OV"
CONDA_ENV="${CONDA_ENV:-/share_4/users/duy/.conda/envs/neo_mobileov}"
CHECKPOINT="${CHECKPOINT:-checkpoints/ssd1b_image_bridge_distill/ssd1b_image_bridge_step100000.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-output/ssd1b_image_bridge_step100k_test}"

mkdir -p "${ROOT}/${OUTPUT_DIR}/logs"
source /share_0/conda/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"

export PYTHONPATH="${ROOT}:${ROOT}/checkpoints/neodragon_repo:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${ROOT}/checkpoints/neodragon"
export TORCH_HOME="${ROOT}/checkpoints/torch_home"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

cd "${ROOT}"
date
nvidia-smi

python tools/infer_ssd1b_image_bridge.py \
  --config configs/mobile_ov_ssd1b_image_bridge.yaml \
  --checkpoint "${CHECKPOINT}" \
  --output-dir "${OUTPUT_DIR}" \
  --seed "${SEED:-2026}" \
  --dtype bf16

find "${OUTPUT_DIR}" -maxdepth 2 -type f -print -exec ls -lh {} \;
date
