#!/usr/bin/env bash
#SBATCH --job-name=neo-dmd-v2-audit
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=02:00:00
#SBATCH --output=logs/neo-dmd-v2-audit-%j.out
#SBATCH --error=logs/neo-dmd-v2-audit-%j.out
#SBATCH --account=berzelius-2026-196
#SBATCH --gres=gpu:1

# Native T2V audit for an all-seven-unit DMD-v2 checkpoint.  This deliberately
# uses image=None; SSD1B/DreamLite first-frame injection is a later ablation.
set -euo pipefail

module load buildenv-nvhpc/24.5-cuda12.4

CONDA_ENV="${CONDA_ENV:-/proj/cvl/users/x_fahkh2/envs/neo_mobileov}"
CHECKPOINT="${CHECKPOINT:?Set CHECKPOINT=output/.../neodragon_pyramidal_dmd_student_stepXXXXXX.pt}"
RUN_NAME="${RUN_NAME:-dmd_v2_native_audit_${SLURM_JOB_ID}}"
OUT="${OUT:-output/neodragon_pyramidal_dmd_inference_contract/${RUN_NAME}}"
PROMPT="${PROMPT:-A vintage red car drives along a coastal road at golden hour, the camera tracking beside it.}"

test -f "${CHECKPOINT}"
test -x "${CONDA_ENV}/bin/python"
mkdir -p logs "${OUT}"

export PATH="${CONDA_ENV}/bin:${PATH}"
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1
export HF_HOME="${HF_HOME:-/proj/cvl/users/x_fahkh2/caches}"
export TORCH_HOME="${TORCH_HOME:-/proj/cvl/users/x_fahkh2/caches}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/proj/cvl/users/x_fahkh2/caches}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

"${CONDA_ENV}/bin/python" - <<'PY' "${CHECKPOINT}"
import sys
import torch

payload = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
schedule = payload.get("schedule")
if schedule != "pyramidal_1-1-1_all_native_units":
    raise SystemExit(f"Expected DMD-v2 all-native schedule, received {schedule!r}")
print(f"Validated DMD-v2 checkpoint step={payload.get('step')} schedule={schedule}")
PY

"${CONDA_ENV}/bin/python" -u tools/audit_neodragon_dmd_inference_contract.py \
  --phase generate \
  --output-dir "${OUT}" \
  --dmd-checkpoint "${CHECKPOINT}" \
  --prompt "${PROMPT}" \
  --seed "${SEED:-20260817}" \
  --dtype "${DTYPE:-bf16}"

"${CONDA_ENV}/bin/python" -u tools/audit_neodragon_dmd_inference_contract.py \
  --phase decode \
  --output-dir "${OUT}" \
  --dmd-checkpoint "${CHECKPOINT}" \
  --prompt "${PROMPT}" \
  --seed "${SEED:-20260817}" \
  --dtype "${DTYPE:-bf16}"

echo "Completed DMD-v2 native audit: ${OUT}"
find "${OUT}" -maxdepth 1 -type f -printf '%f %k KB\n' | sort
