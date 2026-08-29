#!/usr/bin/env bash
# Full official-prompt VBench for the V11 + V12-grounded 20K pilot.
# Protocol: raw VBench prompts, one video per prompt, fixed Exp1-64K video
# bridge, no QuickSR, and the same generation/scoring path as V11 and V12.
# Submit: sbatch scripts/vbench_dreamlite_v11_plus_v12_pilot20k_exp1_64k_1node1gpu.sbatch
# Smoke:  MAX_PROMPTS=2 sbatch scripts/vbench_dreamlite_v11_plus_v12_pilot20k_exp1_64k_1node1gpu.sbatch
#SBATCH --job-name=mov-vbench-v11v12
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=72:00:00
#SBATCH --output=logs/mov-vbench-v11v12-%j.out
#SBATCH --error=logs/mov-vbench-v11v12-%j.out
#SBATCH --account=berzelius-2026-196
#SBATCH --gres=gpu:1

set -euo pipefail

export RUN_NAME="${RUN_NAME:-dreamlite_v11_plus_v12_pilot20k_exp1_64k_vbench1x}"
export SAMPLES_PER_PROMPT="${SAMPLES_PER_PROMPT:-1}"
export IMAGE_CONFIG="${IMAGE_CONFIG:-configs/mobile_ov_dreamlite_compact_v12.yaml}"
export IMAGE_CKPT="${IMAGE_CKPT:-checkpoints/hf_mobile_ov/dreamlite_v11_plus_v12_grounded_pilot/17421424/dreamlite_image_bridge_latest.pt}"
export IMAGE_HF_FILE="${IMAGE_HF_FILE:-dreamlite_v11_plus_v12_grounded_pilot/17421424/dreamlite_image_bridge_latest.pt}"
export IMAGE_EXPECTED_STEP="${IMAGE_EXPECTED_STEP:-20000}"
export IMAGE_EXPECTED_TRAINING_VERSION="${IMAGE_EXPECTED_TRAINING_VERSION:-v11_plus_v12_grounded_pilot}"
export IMAGE_EXPECTED_SOURCE_STEP="${IMAGE_EXPECTED_SOURCE_STEP:-160000}"
export IMAGE_EXPECTED_SOURCE_TRAINING_VERSION="${IMAGE_EXPECTED_SOURCE_TRAINING_VERSION:-v11_balanced}"
export IMAGE_EXPECTED_BRIDGE_TENSORS="${IMAGE_EXPECTED_BRIDGE_TENSORS:-56}"

# Isolate the image-bridge checkpoint: retain the validated Exp1-64K video
# branch and do not recaption or post-process the official VBench prompts.
export VIDEO_CKPT="${VIDEO_CKPT:-checkpoints/hf_mobile_ov/neo_exp1_bridge_functional/17108893/neodragon_text_bridge_latest.pt}"
export VIDEO_HF_FILE="${VIDEO_HF_FILE:-neo_exp1_bridge_functional/17108893/neodragon_text_bridge_latest.pt}"
export WITH_QUICKSR="${WITH_QUICKSR:-0}"
unset RECAPTION_FILE

source scripts/vbench_dreamlite_v4_exp1_64k_1node1gpu.sbatch
