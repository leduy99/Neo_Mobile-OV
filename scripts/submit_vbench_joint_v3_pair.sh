#!/usr/bin/env bash
set -euo pipefail

distill_job=$(sbatch --parsable scripts/vbench_dreamlite_v11_balanced_joint_v3_distill_1node1gpu.sbatch)
flow_job=$(sbatch --parsable scripts/vbench_dreamlite_v11_balanced_joint_v3_flow_only_1node1gpu.sbatch)

echo "Joint V3 distill VBench job: ${distill_job}"
echo "Joint V3 flow-only VBench job: ${flow_job}"
