#!/usr/bin/env bash
# Train a new DreamLite image head on one shared Exp1 SmolVLM2 forward.
# The Exp1 video bridge is frozen and its NeoDragon condition remains unchanged.
# Submit from the repository root:
#   sbatch scripts/train_dreamlite_shared_smol_v1_1node8gpu.sbatch
# Set RESUME=/path/to/dreamlite_image_bridge_latest.pt only for a previous
# shared-smol run. Do not initialize this run from V1-V7 image checkpoints.
#SBATCH --job-name=mov-dream-shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=72:00:00
#SBATCH --output=logs/mov-dream-shared-%j.out
#SBATCH --error=logs/mov-dream-shared-%j.out
#SBATCH --account=berzelius-2026-196
#SBATCH --gres=gpu:8
##SBATCH --gres=gpu:A100-SXM4-80GB:8

set -euo pipefail

export V8_VARIANT=mix
export TRAINING_VERSION=shared_v1
export OUT="${OUT:-output/dreamlite_shared_smol_v1/${SLURM_JOB_ID:-local}}"
export TARGET_STEP="${TARGET_STEP:-120000}"
export VIDEO_BRIDGE_CKPT="${VIDEO_BRIDGE_CKPT:-checkpoints/hf_mobile_ov/neo_exp1_bridge_functional/17108893/neodragon_text_bridge_latest.pt}"
export SHARED_PROMPT_SUFFIX="${SHARED_PROMPT_SUFFIX:-, cinematic, realistic textures, high detail, natural colours}"
export EXTRA_ARGS="${EXTRA_ARGS:-} --shared-video-bridge-ckpt ${VIDEO_BRIDGE_CKPT}"

source scripts/train_dreamlite_compact_v8_common.sh
