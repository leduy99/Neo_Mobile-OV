#!/usr/bin/env bash
# Official-prompt VBench evaluation: DreamLite V10 + fixed Exp1-64K NeoDragon bridge.
# The local V10 checkpoint is preferred; Hugging Face is only used as a fallback.
# Set MAX_PROMPTS for a generation-only smoke run; the default evaluates all prompts.
#SBATCH --job-name=mov-vbench-v10
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=72:00:00
#SBATCH --output=logs/mov-vbench-v10-%j.out
#SBATCH --error=logs/mov-vbench-v10-%j.out
#SBATCH --account=berzelius-2026-196
#SBATCH --gres=gpu:1
##SBATCH --gres=gpu:A100-SXM4-80GB:1

set -euo pipefail

export RUN_NAME="${RUN_NAME:-dreamlite_v10_exp1_64k_vbench1x}"
export SAMPLES_PER_PROMPT="${SAMPLES_PER_PROMPT:-1}"
export IMAGE_CONFIG="${IMAGE_CONFIG:-configs/mobile_ov_dreamlite_compact_v10.yaml}"
export IMAGE_CKPT="${IMAGE_CKPT:-output/dreamlite_compact_v10_balanced_from_scratch/17280132/dreamlite_image_bridge_latest.pt}"
export IMAGE_HF_FILE="${IMAGE_HF_FILE:-dreamlite_compact_v10_balanced_from_scratch/17280132/dreamlite_image_bridge_latest.pt}"
export IMAGE_EXPECTED_STEP="${IMAGE_EXPECTED_STEP:-160000}"

# Keep the NeoDragon branch fixed so this is directly comparable with V7/V8/V9.
export VIDEO_CKPT="${VIDEO_CKPT:-checkpoints/hf_mobile_ov/neo_exp1_bridge_functional/17108893/neodragon_text_bridge_latest.pt}"
export VIDEO_HF_FILE="${VIDEO_HF_FILE:-neo_exp1_bridge_functional/17108893/neodragon_text_bridge_latest.pt}"

# Prevent an inherited shell variable from accidentally changing this raw-prompt run.
unset RECAPTION_FILE
source scripts/vbench_dreamlite_v4_exp1_64k_1node1gpu.sbatch
