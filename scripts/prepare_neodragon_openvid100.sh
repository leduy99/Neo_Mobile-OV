#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/share_4/users/duy/project/unified_video/New-Mobile-OV}"
cd "$ROOT"

source /share_0/conda/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-/share_4/users/duy/.conda/envs/neo_mobileov}"

export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1

python tools/prepare_neodragon_openvid100.py \
  --source-manifest "${SOURCE_MANIFEST:-/share_4/users/duy/project/unified_video/Omni-Video-smolvlm2/data/openvid_1m/manifests/by_part/part_0111.csv}" \
  --output-dir "${OUT_DIR:-data/neodragon_openvid100}" \
  --num-videos "${NUM_VIDEOS:-100}" \
  --offset "${OFFSET:-0}" \
  --mode "${COPY_MODE:-copy}" \
  ${EXTRA_ARGS:-}
