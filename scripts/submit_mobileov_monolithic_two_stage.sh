#!/usr/bin/env bash
# Submit bridge re-distillation first, then start joint flow training only if it
# completed successfully. Run this from the repository root on Berzelius.
set -euo pipefail

BRIDGE_JOB_ID=$(sbatch --parsable scripts/train_neodragon_monolithic_cfg_bridge_v2_1node8gpu.sbatch)
BRIDGE_CKPT="output/neo_monolithic_cfg_bridge_v2/monolithic_cfg_bridge_v2_${BRIDGE_JOB_ID}/neodragon_text_bridge_best.pt"
JOINT_JOB_ID=$(
  sbatch \
    --parsable \
    --dependency="afterok:${BRIDGE_JOB_ID}" \
    --export="ALL,BRIDGE_CKPT=${BRIDGE_CKPT}" \
    scripts/train_mobileov_monolithic_joint_flow_1node8gpu.sbatch
)

printf 'Bridge job: %s\n' "${BRIDGE_JOB_ID}"
printf 'Expected bridge checkpoint: %s\n' "${BRIDGE_CKPT}"
printf 'Joint job (afterok dependency): %s\n' "${JOINT_JOB_ID}"
