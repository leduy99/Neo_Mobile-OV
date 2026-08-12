#!/usr/bin/env bash
# Official-prompt VBench evaluation: DreamLite V8-image-only + fixed Exp1-64K NeoDragon bridge.
# This intentionally does not use SmolVLM2 recaptions.
#SBATCH --job-name=mov-vbench-v8i
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=72:00:00
#SBATCH --output=logs/mov-vbench-v8i-%j.out
#SBATCH --error=logs/mov-vbench-v8i-%j.out
#SBATCH --account=berzelius-2026-196
#SBATCH --gres=gpu:1
##SBATCH --gres=gpu:A100-SXM4-80GB:1

set -euo pipefail

export RUN_NAME="${RUN_NAME:-dreamlite_v8_imageonly_exp1_64k_vbench1x}"
export SAMPLES_PER_PROMPT="${SAMPLES_PER_PROMPT:-1}"
export IMAGE_CONFIG="${IMAGE_CONFIG:-configs/mobile_ov_dreamlite_compact_v8.yaml}"
export IMAGE_CKPT="${IMAGE_CKPT:-checkpoints/hf_mobile_ov/dreamlite_compact_v8_imageonly/17245434/dreamlite_image_bridge_latest.pt}"
export IMAGE_HF_FILE="${IMAGE_HF_FILE:-dreamlite_compact_v8_imageonly/17245434/dreamlite_image_bridge_latest.pt}"
export IMAGE_EXPECTED_STEP="${IMAGE_EXPECTED_STEP:-160000}"

# Prevent an inherited shell variable from accidentally changing this official-prompt run.
unset RECAPTION_FILE
source scripts/vbench_dreamlite_v4_exp1_64k_1node1gpu.sbatch
