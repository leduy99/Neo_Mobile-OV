#!/usr/bin/env bash
# Diagnostic run, not training: localize the DMD student's motion loss against
# the released multistep teacher under controlled anchor/noise/history paths.
#SBATCH --job-name=neo-dmd-motion
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=06:00:00
#SBATCH --output=logs/neo-dmd-motion-%j.out
#SBATCH --error=logs/neo-dmd-motion-%j.out
#SBATCH --account=berzelius-2026-196
#SBATCH --gres=gpu:1

set -euo pipefail
module load buildenv-nvhpc/24.5-cuda12.4

CONDA_ENV="${CONDA_ENV:-/proj/cvl/users/x_fahkh2/envs/neo_mobileov}"
PYTHON_BIN="${CONDA_ENV}/bin/python"
CHECKPOINT="${CHECKPOINT:-output/neodragon_pyramidal_dmd_reproduction/dmd_repro_partial188k_full10k/neodragon_pyramidal_dmd_student_step010000.pt}"
RUN_NAME="${RUN_NAME:-dmd10k_motion_audit}"
OUT="${OUT:-output/neodragon_pyramidal_dmd_motion_audit/${RUN_NAME}}"

test -x "${PYTHON_BIN}"
test -f "${CHECKPOINT}"
mkdir -p logs "${OUT}"
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1
export HF_HOME="${HF_HOME:-/proj/cvl/users/x_fahkh2/caches}"
export TORCH_HOME="${TORCH_HOME:-/proj/cvl/users/x_fahkh2/caches}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/proj/cvl/users/x_fahkh2/caches}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

PROMPT_NAMES=(astronaut car waterfall)
PROMPTS=(
  "An astronaut walks slowly across the red surface of Mars as dust blows behind them."
  "A vintage red car drives along a coastal road at golden hour, the camera tracking beside it."
  "A wide cinematic view of a waterfall flowing into a misty tropical valley, leaves moving in the wind."
)

echo "DMD motion audit: checkpoint=${CHECKPOINT} output=${OUT}"
echo "Five independent process phases per prompt: multistep teacher trace, DMD replay, SSD anchor replay, released-Hybrid control, VAE decode."
nvidia-smi || true

for index in "${!PROMPTS[@]}"; do
  PROMPT_OUT="${OUT}/${PROMPT_NAMES[$index]}"
  PROMPT="${PROMPTS[$index]}"
  echo "=== ${PROMPT_NAMES[$index]}: teacher phase ==="
  "${PYTHON_BIN}" -u tools/audit_neodragon_dmd_motion_staged.py \
    --phase teacher --checkpoint "${CHECKPOINT}" --output-dir "${PROMPT_OUT}" --prompt "${PROMPT}"
  echo "=== ${PROMPT_NAMES[$index]}: DMD replay phase ==="
  "${PYTHON_BIN}" -u tools/audit_neodragon_dmd_motion_staged.py \
    --phase student --checkpoint "${CHECKPOINT}" --output-dir "${PROMPT_OUT}" --prompt "${PROMPT}"
  echo "=== ${PROMPT_NAMES[$index]}: SSD anchor phase ==="
  "${PYTHON_BIN}" -u tools/audit_neodragon_dmd_motion_staged.py \
    --phase ssd --checkpoint "${CHECKPOINT}" --output-dir "${PROMPT_OUT}" --prompt "${PROMPT}"
  echo "=== ${PROMPT_NAMES[$index]}: released Hybrid control phase ==="
  "${PYTHON_BIN}" -u tools/audit_neodragon_dmd_motion_staged.py \
    --phase hybrid --checkpoint "${CHECKPOINT}" --output-dir "${PROMPT_OUT}" --prompt "${PROMPT}"
  echo "=== ${PROMPT_NAMES[$index]}: VAE decode phase ==="
  "${PYTHON_BIN}" -u tools/audit_neodragon_dmd_motion_staged.py \
    --phase decode --checkpoint "${CHECKPOINT}" --output-dir "${PROMPT_OUT}" --prompt "${PROMPT}"
done

"${PYTHON_BIN}" tools/summarize_neodragon_dmd_motion_audit.py \
  --root "${OUT}" --output "${OUT}/aggregate.json"

echo "Completed DMD motion audit."
cat "${OUT}/aggregate.json"
