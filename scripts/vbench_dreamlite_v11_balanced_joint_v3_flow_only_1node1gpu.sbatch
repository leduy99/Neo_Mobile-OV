#!/usr/bin/env bash
# Full 1x VBench for Joint V3 trained with flow matching only.
#SBATCH --job-name=mov-vb-jv3f
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=72:00:00
#SBATCH --output=logs/mov-vb-jv3f-%j.out
#SBATCH --error=logs/mov-vb-jv3f-%j.out
#SBATCH --account=berzelius-2026-196
#SBATCH --gres=gpu:1

set -euo pipefail

export RUN_NAME="${RUN_NAME:-dreamlite_v11_balanced_joint_v3_flow_only_step4k_vbench1x}"
export SAMPLES_PER_PROMPT="${SAMPLES_PER_PROMPT:-1}"
export IMAGE_CONFIG="${IMAGE_CONFIG:-configs/mobile_ov_dreamlite_compact_v9.yaml}"
export IMAGE_CKPT="${IMAGE_CKPT:-checkpoints/hf_mobile_ov/dreamlite_compact_v11_v10_rates_from_scratch/17291461/dreamlite_image_bridge_latest.pt}"
export IMAGE_HF_FILE="${IMAGE_HF_FILE:-dreamlite_compact_v11_v10_rates_from_scratch/17291461/dreamlite_image_bridge_latest.pt}"
export IMAGE_EXPECTED_STEP="${IMAGE_EXPECTED_STEP:-160000}"
export VIDEO_CKPT="${VIDEO_CKPT:-checkpoints/hf_mobile_ov/mobileov_monolithic_joint_v3/flow_only/joint_v3_flow_only_4k/neodragon_dit_bridge_latest.pt}"
export VIDEO_HF_FILE="${VIDEO_HF_FILE:-mobileov_monolithic_joint_v3/flow_only/joint_v3_flow_only_4k/neodragon_dit_bridge_latest.pt}"
export VIDEO_JOINT_CHECKPOINT=1
export VIDEO_EXPECTED_STEP=4000
export WITH_QUICKSR=0
unset RECAPTION_FILE

source scripts/vbench_dreamlite_v4_exp1_64k_1node1gpu.sbatch

"${PYTHON_BIN}" "${ROOT}/tools/report_vbench_low_prompts.py" \
  --score-dir "${ROOT}/${SCORE_DIR}/vbench" \
  --output-prefix "${ROOT}/${RUN_ROOT}/low_prompt_report" \
  --failure-dir "${ROOT}/${RUN_ROOT}/low_prompt_videos" \
  --top-k "${LOW_PROMPT_TOP_K:-10}"
