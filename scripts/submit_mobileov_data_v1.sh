#!/usr/bin/env bash
set -euo pipefail

# One command submits the complete data build. Training should depend on the
# final freeze job, not on either generation job directly.
video_job=$(sbatch --parsable scripts/build_mobileov_video_cascade_v1_1node8gpu.sbatch)
anchor_job=$(sbatch --parsable --dependency="afterok:${video_job}" scripts/prepare_mobileov_anchor_teacher_v1_1node8gpu.sbatch)
freeze_job=$(sbatch --parsable --dependency="afterok:${anchor_job}" scripts/freeze_mobileov_data_v1_1node1gpu.sbatch)

echo "Video quality cascade job: ${video_job}"
echo "Matched anchor/teacher generation job: ${anchor_job}"
echo "Immutable release + preflight job: ${freeze_job}"
