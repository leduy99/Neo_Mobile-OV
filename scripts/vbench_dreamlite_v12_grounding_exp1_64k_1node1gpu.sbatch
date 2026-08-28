#!/usr/bin/env bash
# Controlled 1-video-per-prompt VBench for V12 grounding-cascade.
# Keep the Exp1-64K video bridge, raw prompts, generation settings, and scoring
# protocol identical to the earlier V8-V11 image-bridge comparisons.
# Submit: sbatch scripts/vbench_dreamlite_v12_grounding_exp1_64k_1node1gpu.sbatch
# Smoke:  MAX_PROMPTS=2 sbatch scripts/vbench_dreamlite_v12_grounding_exp1_64k_1node1gpu.sbatch
#SBATCH --job-name=mov-vbench-v12
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=72:00:00
#SBATCH --output=logs/mov-vbench-v12-%j.out
#SBATCH --error=logs/mov-vbench-v12-%j.out
#SBATCH --account=berzelius-2026-196
#SBATCH --gres=gpu:1
##SBATCH --gres=gpu:A100-SXM4-80GB:1

set -euo pipefail

export RUN_NAME="${RUN_NAME:-dreamlite_v12_grounding_exp1_64k_vbench1x}"
export SAMPLES_PER_PROMPT="${SAMPLES_PER_PROMPT:-1}"
export IMAGE_CONFIG="${IMAGE_CONFIG:-configs/mobile_ov_dreamlite_compact_v12.yaml}"
export IMAGE_CKPT="${IMAGE_CKPT:-checkpoints/hf_mobile_ov/dreamlite_compact_v12_grounding_cascade/17383360/dreamlite_image_bridge_latest.pt}"
export IMAGE_HF_FILE="${IMAGE_HF_FILE:-dreamlite_compact_v12_grounding_cascade/17383360/dreamlite_image_bridge_latest.pt}"
export IMAGE_EXPECTED_STEP="${IMAGE_EXPECTED_STEP:-160000}"

# Isolate the V12 image-bridge effect: keep the previously validated video
# branch and use the original VBench prompts without recaptioning or QuickSR.
export VIDEO_CKPT="${VIDEO_CKPT:-checkpoints/hf_mobile_ov/neo_exp1_bridge_functional/17108893/neodragon_text_bridge_latest.pt}"
export VIDEO_HF_FILE="${VIDEO_HF_FILE:-neo_exp1_bridge_functional/17108893/neodragon_text_bridge_latest.pt}"
export WITH_QUICKSR="${WITH_QUICKSR:-0}"
unset RECAPTION_FILE

source scripts/vbench_dreamlite_v4_exp1_64k_1node1gpu.sbatch
