#!/usr/bin/env bash
# Raw-prompt VBench for the deployed joint-v2 pipeline:
# DreamLite V11-balanced first frame + frozen bridge-v2 + monolithic DiT step 10K.
# One independently seeded video per official prompt keeps this directly comparable
# with the prior V7-V11 1x measurements. QuickSR is intentionally disabled.
#SBATCH --job-name=mov-vbench-jv2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=72:00:00
#SBATCH --output=logs/mov-vbench-jv2-%j.out
#SBATCH --error=logs/mov-vbench-jv2-%j.out
#SBATCH --account=berzelius-2026-196
#SBATCH --gres=gpu:1
##SBATCH --gres=gpu:A100-SXM4-80GB:1

set -euo pipefail

export RUN_NAME="${RUN_NAME:-dreamlite_v11_balanced_joint_v2_step10k_vbench1x}"
export SAMPLES_PER_PROMPT="${SAMPLES_PER_PROMPT:-1}"
export IMAGE_CONFIG="${IMAGE_CONFIG:-configs/mobile_ov_dreamlite_compact_v9.yaml}"
export IMAGE_CKPT="${IMAGE_CKPT:-checkpoints/hf_mobile_ov/dreamlite_compact_v11_v10_rates_from_scratch/17291461/dreamlite_image_bridge_latest.pt}"
export IMAGE_HF_FILE="${IMAGE_HF_FILE:-dreamlite_compact_v11_v10_rates_from_scratch/17291461/dreamlite_image_bridge_latest.pt}"
export IMAGE_EXPECTED_STEP="${IMAGE_EXPECTED_STEP:-160000}"

export VIDEO_CKPT="${VIDEO_CKPT:-checkpoints/hf_mobile_ov/mobileov_monolithic_pyramid_cfg_dit/mobileov_monolithic_pyramid_cfg_dit_17346642/neodragon_dit_bridge_step010000.pt}"
export VIDEO_HF_FILE="${VIDEO_HF_FILE:-mobileov_monolithic_pyramid_cfg_dit/mobileov_monolithic_pyramid_cfg_dit_17346642/neodragon_dit_bridge_step010000.pt}"
export VIDEO_JOINT_CHECKPOINT=1
export VIDEO_EXPECTED_STEP=10000

export WITH_QUICKSR=0
unset RECAPTION_FILE

source scripts/vbench_dreamlite_v4_exp1_64k_1node1gpu.sbatch
