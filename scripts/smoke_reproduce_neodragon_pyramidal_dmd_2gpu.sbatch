#!/usr/bin/env bash
#SBATCH --job-name=neo-dmd-smoke
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=192G
#SBATCH --time=02:00:00
#SBATCH --output=output/neodragon_pyramidal_dmd_smoke/slurm-%j.out
#SBATCH --error=output/neodragon_pyramidal_dmd_smoke/slurm-%j.out

# Local two-GPU validation of the complete pipeline: synthetic multistep
# generation, DDP fake/student updates, and portable checkpoint writes.
set -euo pipefail

CONDA_ENV="${CONDA_ENV:-/share_4/users/duy/.conda/envs/neo_mobileov}"
TORCHRUN_BIN="${TORCHRUN_BIN:-${CONDA_ENV}/bin/torchrun}"
CONFIG="${CONFIG:-configs/mobile_ov_neodragon.yaml}"
ROOT_OUT="${ROOT_OUT:-output/neodragon_pyramidal_dmd_smoke/${SLURM_JOB_ID:-local}}"
DATA_OUT="${DATA_OUT:-${ROOT_OUT}/data}"
RUN_OUT="${RUN_OUT:-${ROOT_OUT}/run}"
PROMPTS="${PROMPTS:-data/neodragon_openvid100/manifest.csv}"

if [[ ! -f "${PROMPTS}" ]]; then
  echo "Smoke prompt source does not exist: ${PROMPTS}" >&2
  exit 1
fi
mkdir -p "${ROOT_OUT}" "${RUN_OUT}"
export PATH="${CONDA_ENV}/bin:${PATH}"
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export TORCH_DIST_TIMEOUT_MINUTES="${TORCH_DIST_TIMEOUT_MINUTES:-120}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

"${TORCHRUN_BIN}" --standalone --nproc_per_node=2 \
  tools/prepare_neodragon_dmd_synthetic_data.py \
  --config "${CONFIG}" --prompts "${PROMPTS}" --output-dir "${DATA_OUT}" \
  --max-prompts "${SMOKE_PROMPTS:-4}" --first-unit-steps "${FIRST_UNIT_STEPS:-2}" \
  --video-unit-steps "${VIDEO_UNIT_STEPS:-1}" --dtype bf16

"${TORCHRUN_BIN}" --standalone --nproc_per_node=2 \
  tools/train_neodragon_pyramidal_dmd.py \
  --config "${CONFIG}" --manifest "${DATA_OUT}/manifest.csv" --output-dir "${RUN_OUT}" \
  --resume none --steps "${SMOKE_STEPS:-2}" --batch-size 1 --num-workers 0 \
  --fake-updates 2 --save-every 1 --archive-every 1 --log-every 1 --dtype bf16

"${CONDA_ENV}/bin/python" - "${RUN_OUT}" <<'PY'
import json
import sys
from pathlib import Path

import torch

root = Path(sys.argv[1])
payload = torch.load(root / "neodragon_pyramidal_dmd_student_latest.pt", map_location="cpu", weights_only=False)
assert payload["step"] == 2, payload["step"]
assert payload["schedule"] == "hybrid_1-1-1_video_units_only"
assert payload["objective"]["student_fake_update_ratio"] == "1:2"
assert (root / "neodragon_pyramidal_dmd_resume.pt").is_file()
assert (root / "neodragon_pyramidal_dmd_student_step000002.pt").is_file()
summary = {"status": "ok", "step": payload["step"], "checkpoint": str(root / "neodragon_pyramidal_dmd_student_latest.pt")}
(root / "smoke_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
PY
