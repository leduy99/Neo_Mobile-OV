#!/usr/bin/env bash
set -Eeuo pipefail
cd "${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
mkdir -p logs
FIRST="$(sbatch --parsable scripts/vbench_joint_v2_ssd1b_anchor_1node8gpu.sbatch)"
echo "SSD1B -> Joint V2 job: ${FIRST}"
SECOND="$(sbatch --parsable scripts/vbench_native_dreamlite_v11_anchor_1node8gpu.sbatch)"
echo "V11 DreamLite -> Native Hybrid job: ${SECOND}"
