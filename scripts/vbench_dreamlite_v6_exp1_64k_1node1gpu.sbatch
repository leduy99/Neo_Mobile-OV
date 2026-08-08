#!/usr/bin/env bash
#SBATCH --job-name=mov-vbench-v6
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=72:00:00
#SBATCH --output=logs/mov-vbench-v6-%j.out
#SBATCH --error=logs/mov-vbench-v6-%j.out
#SBATCH --account=berzelius-2026-196
#SBATCH --gres=gpu:1
##SBATCH --gres=gpu:A100-SXM4-80GB:1

set -euo pipefail

# Keep the V4 and V6 evaluations identical except for the image bridge.
export RUN_NAME="${RUN_NAME:-dreamlite_v6_exp1_64k_vbench1x}"
export IMAGE_CONFIG="${IMAGE_CONFIG:-configs/mobile_ov_dreamlite_compact_v6.yaml}"
export IMAGE_CKPT="${IMAGE_CKPT:-checkpoints/hf_mobile_ov/dreamlite_compact_v6/17223345/dreamlite_image_bridge_latest.pt}"
export IMAGE_HF_FILE="${IMAGE_HF_FILE:-dreamlite_compact_v6/17223345/dreamlite_image_bridge_latest.pt}"
export IMAGE_EXPECTED_STEP="${IMAGE_EXPECTED_STEP:-120000}"

source scripts/vbench_dreamlite_v4_exp1_64k_1node1gpu.sbatch
