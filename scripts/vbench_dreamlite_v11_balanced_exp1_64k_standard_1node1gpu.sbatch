#!/usr/bin/env bash
# Standard VBench protocol for the strongest current Mobile-OV pipeline:
# raw official prompts, V11-balanced DreamLite bridge, Exp1-64K NeoDragon bridge,
# four independent seeds per prompt, and public QuickSRNet medium 2x before scoring.
# This avoids the known 1x protocol's missing-sample warnings and scores 1024x640 video.
#SBATCH --job-name=mov-vbench-v11std
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=72:00:00
#SBATCH --output=logs/mov-vbench-v11std-%j.out
#SBATCH --error=logs/mov-vbench-v11std-%j.out
#SBATCH --account=berzelius-2026-196
#SBATCH --gres=gpu:1
##SBATCH --gres=gpu:A100-SXM4-80GB:1

set -euo pipefail

export RUN_NAME="${RUN_NAME:-dreamlite_v11_balanced_exp1_64k_vbench4x_quicksr2x}"
export SAMPLES_PER_PROMPT=4
export WITH_QUICKSR=1
export QUICKSR_VARIANT=medium
export QUICKSR_SCALE=2
export QUICKSR_DTYPE=fp16
export QUICKSR_BATCH_FRAMES=4

# Pin the completed V11-balanced checkpoint rather than resolving an arbitrary
# checkpoint with a similar name from the cache or Hugging Face.
export IMAGE_CONFIG="${IMAGE_CONFIG:-configs/mobile_ov_dreamlite_compact_v9.yaml}"
IMAGE_RUN_ROOT="${IMAGE_RUN_ROOT:-output/dreamlite_compact_v11_v10_rates_from_scratch}"
export IMAGE_CKPT="${IMAGE_CKPT:-$(ls -t "${IMAGE_RUN_ROOT}"/*/dreamlite_image_bridge_latest.pt 2>/dev/null | head -n 1)}"
export IMAGE_EXPECTED_STEP=160000

test -n "${IMAGE_CKPT}"
test -f "${IMAGE_CKPT}"
IMAGE_JOB_ID="$(basename "$(dirname "${IMAGE_CKPT}")")"
export IMAGE_HF_FILE="${IMAGE_HF_FILE:-dreamlite_compact_v11_v10_rates_from_scratch/${IMAGE_JOB_ID}/dreamlite_image_bridge_latest.pt}"

# Keep the video branch and raw VBench prompts fixed for a controlled score.
export VIDEO_CKPT="${VIDEO_CKPT:-checkpoints/hf_mobile_ov/neo_exp1_bridge_functional/17108893/neodragon_text_bridge_latest.pt}"
export VIDEO_HF_FILE="${VIDEO_HF_FILE:-neo_exp1_bridge_functional/17108893/neodragon_text_bridge_latest.pt}"
unset RECAPTION_FILE

source scripts/vbench_dreamlite_v4_exp1_64k_1node1gpu.sbatch
