#!/usr/bin/env bash
# Official-prompt VBench: V11-balanced image bridge + fixed Exp1-64K video bridge.
# The checkpoint is resolved from the completed local training output, avoiding
# an accidental fallback to a similarly named Hugging Face artifact.
#SBATCH --job-name=mov-vbench-v11b
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=72:00:00
#SBATCH --output=logs/mov-vbench-v11b-%j.out
#SBATCH --error=logs/mov-vbench-v11b-%j.out
#SBATCH --account=berzelius-2026-196
#SBATCH --gres=gpu:1
##SBATCH --gres=gpu:A100-SXM4-80GB:1

set -euo pipefail

export RUN_NAME="${RUN_NAME:-dreamlite_v11_balanced_exp1_64k_vbench1x}"
export SAMPLES_PER_PROMPT="${SAMPLES_PER_PROMPT:-1}"
export IMAGE_CONFIG="${IMAGE_CONFIG:-configs/mobile_ov_dreamlite_compact_v9.yaml}"
IMAGE_RUN_ROOT="${IMAGE_RUN_ROOT:-output/dreamlite_compact_v11_v10_rates_from_scratch}"
export IMAGE_CKPT="${IMAGE_CKPT:-$(ls -t "${IMAGE_RUN_ROOT}"/*/dreamlite_image_bridge_latest.pt 2>/dev/null | head -n 1)}"
export IMAGE_EXPECTED_STEP="${IMAGE_EXPECTED_STEP:-160000}"

test -n "${IMAGE_CKPT}"
test -f "${IMAGE_CKPT}"
IMAGE_JOB_ID="$(basename "$(dirname "${IMAGE_CKPT}")")"
export IMAGE_HF_FILE="${IMAGE_HF_FILE:-dreamlite_compact_v11_v10_rates_from_scratch/${IMAGE_JOB_ID}/dreamlite_image_bridge_latest.pt}"

# Keep the video branch and raw VBench prompts identical to V8-V10.
export VIDEO_CKPT="${VIDEO_CKPT:-checkpoints/hf_mobile_ov/neo_exp1_bridge_functional/17108893/neodragon_text_bridge_latest.pt}"
export VIDEO_HF_FILE="${VIDEO_HF_FILE:-neo_exp1_bridge_functional/17108893/neodragon_text_bridge_latest.pt}"
unset RECAPTION_FILE

source scripts/vbench_dreamlite_v4_exp1_64k_1node1gpu.sbatch
