#!/usr/bin/env bash
#SBATCH --job-name=mov-ssd-eval
#SBATCH --partition=debug
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=220G
#SBATCH --time=04:00:00
#SBATCH --chdir=/share_4/users/duy/project/unified_video/New-Mobile-OV
#SBATCH --output=output/ssd1b_image_bridge_comprehensive_eval/logs/%x_%j.out
#SBATCH --error=output/ssd1b_image_bridge_comprehensive_eval/logs/%x_%j.out

set -euo pipefail

ROOT="/share_4/users/duy/project/unified_video/New-Mobile-OV"
CONDA_ENV="${CONDA_ENV:-/share_4/users/duy/.conda/envs/neo_mobileov}"
CHECKPOINT="${CHECKPOINT:-checkpoints/ssd1b_image_bridge_distill/ssd1b_image_bridge_step100000.pt}"
OUTPUT_DIR="${OUTPUT_DIR:-output/ssd1b_image_bridge_comprehensive_eval}"

mkdir -p "${ROOT}/${OUTPUT_DIR}/logs"
source /share_0/conda/etc/profile.d/conda.sh
conda activate "${CONDA_ENV}"

export PYTHONPATH="${ROOT}:${ROOT}/checkpoints/neodragon_repo:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export HF_HOME="${ROOT}/checkpoints/hf_home"
export TORCH_HOME="${ROOT}/checkpoints/torch_home"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

cd "${ROOT}"
date
nvidia-smi

python tools/evaluate_ssd1b_image_bridge.py \
  --config configs/mobile_ov_ssd1b_image_bridge.yaml \
  --checkpoint "${CHECKPOINT}" \
  --prompt-file checkpoints/neodragon_repo/prompts/vbench_prompts.txt \
  --output-dir "${OUTPUT_DIR}" \
  --condition-prompts "${CONDITION_PROMPTS:-384}" \
  --functional-prompts "${FUNCTIONAL_PROMPTS:-96}" \
  --image-prompts "${IMAGE_PROMPTS:-30}" \
  --condition-batch-size "${CONDITION_BATCH_SIZE:-16}" \
  --unet-batch-size "${UNET_BATCH_SIZE:-4}" \
  --clip-batch-size "${CLIP_BATCH_SIZE:-8}" \
  --seed "${SEED:-2026}" \
  --dtype bf16 \
  ${EXTRA_ARGS:-}

tail -n 20 "${OUTPUT_DIR}/metrics.json"
date
