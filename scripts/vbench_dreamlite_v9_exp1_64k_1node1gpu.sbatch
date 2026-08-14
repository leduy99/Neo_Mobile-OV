#!/usr/bin/env bash
# Official-prompt VBench evaluation: DreamLite V9 + fixed Exp1-64K NeoDragon bridge.
# Set MAX_PROMPTS for a generation-only smoke run; the default evaluates all prompts.
#SBATCH --job-name=mov-vbench-v9
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=72:00:00
#SBATCH --output=logs/mov-vbench-v9-%j.out
#SBATCH --error=logs/mov-vbench-v9-%j.out
#SBATCH --account=berzelius-2026-196
#SBATCH --gres=gpu:1
##SBATCH --gres=gpu:A100-SXM4-80GB:1

set -euo pipefail

export RUN_NAME="${RUN_NAME:-dreamlite_v9_exp1_64k_vbench1x}"
export SAMPLES_PER_PROMPT="${SAMPLES_PER_PROMPT:-1}"
export IMAGE_CONFIG="${IMAGE_CONFIG:-configs/mobile_ov_dreamlite_compact_v9.yaml}"
export IMAGE_CKPT="${IMAGE_CKPT:-checkpoints/hf_mobile_ov/dreamlite_compact_v9_grounded_from_scratch/17275939/dreamlite_image_bridge_latest.pt}"
export IMAGE_HF_FILE="${IMAGE_HF_FILE:-dreamlite_compact_v9_grounded_from_scratch/17275939/dreamlite_image_bridge_latest.pt}"
export IMAGE_EXPECTED_STEP="${IMAGE_EXPECTED_STEP:-160000}"

# Keep the NeoDragon branch fixed so this is directly comparable with V7/V8.
export VIDEO_CKPT="${VIDEO_CKPT:-checkpoints/hf_mobile_ov/neo_exp1_bridge_functional/17108893/neodragon_text_bridge_latest.pt}"
export VIDEO_HF_FILE="${VIDEO_HF_FILE:-neo_exp1_bridge_functional/17108893/neodragon_text_bridge_latest.pt}"

# Prevent an inherited shell variable from accidentally changing this official-prompt run.
unset RECAPTION_FILE
source scripts/vbench_dreamlite_v4_exp1_64k_1node1gpu.sbatch
