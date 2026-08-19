#!/usr/bin/env bash
# Submit two independent experiments from the repository root on Berzelius:
# (1) CFG-aware bridge re-distillation and (2) immediate joint monolithic flow
# training from the already available 17318011 bridge.
set -euo pipefail

BRIDGE_JOB_ID=$(sbatch --parsable scripts/train_neodragon_monolithic_cfg_bridge_v2_1node8gpu.sbatch)
JOINT_JOB_ID=$(sbatch --parsable scripts/train_mobileov_monolithic_joint_flow_1node8gpu.sbatch)

printf 'Independent bridge re-distillation job: %s\n' "${BRIDGE_JOB_ID}"
printf 'Independent joint flow job from bridge 17318011: %s\n' "${JOINT_JOB_ID}"
