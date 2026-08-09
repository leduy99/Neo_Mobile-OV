#!/usr/bin/env bash
#SBATCH --job-name=mov-vbench-v7
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=72:00:00
#SBATCH --output=logs/mov-vbench-v7-%j.out
#SBATCH --error=logs/mov-vbench-v7-%j.out
#SBATCH --account=berzelius-2026-196
#SBATCH --gres=gpu:1
##SBATCH --gres=gpu:A100-SXM4-80GB:1

set -euo pipefail

# One independently seeded video per prompt by default. The full wrapper overrides
# this with five samples while keeping the rest of the pipeline identical.
export RUN_NAME="${RUN_NAME:-dreamlite_v7_exp1_64k_vbench1x}"
export SAMPLES_PER_PROMPT="${SAMPLES_PER_PROMPT:-1}"
export IMAGE_CONFIG="${IMAGE_CONFIG:-configs/mobile_ov_dreamlite_compact_v7.yaml}"
export IMAGE_CKPT="${IMAGE_CKPT:-checkpoints/hf_mobile_ov/dreamlite_compact_v7_mobileo/17242707/dreamlite_image_bridge_step120000.pt}"
export IMAGE_HF_FILE="${IMAGE_HF_FILE:-dreamlite_compact_v7_mobileo/17242707/dreamlite_image_bridge_step120000.pt}"
export IMAGE_EXPECTED_STEP="${IMAGE_EXPECTED_STEP:-120000}"

source scripts/vbench_dreamlite_v4_exp1_64k_1node1gpu.sbatch
