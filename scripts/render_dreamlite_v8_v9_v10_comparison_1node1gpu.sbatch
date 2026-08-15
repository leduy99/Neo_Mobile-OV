#!/usr/bin/env bash
# Render an exact, side-by-side V8/V9/V10 DreamLite image comparison.
# Submit: sbatch scripts/render_dreamlite_v8_v9_v10_comparison_1node1gpu.sbatch
# Smoke:  MAX_PROMPTS=2 sbatch scripts/render_dreamlite_v8_v9_v10_comparison_1node1gpu.sbatch
# Output: output/dreamlite_v8_v9_v10_visual_compare/<JOBID>/comparison_sheet.png
#SBATCH --job-name=mov-dream-cmp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=08:00:00
#SBATCH --output=logs/mov-dream-cmp-%j.out
#SBATCH --error=logs/mov-dream-cmp-%j.out
#SBATCH --account=berzelius-2026-196
#SBATCH --gres=gpu:1

set -euo pipefail

module load buildenv-nvhpc/24.5-cuda12.4

export PYTHONPATH="./:${PYTHONPATH:-}"
export HF_HOME=/proj/cvl/users/x_fahkh2/caches
export TORCH_HOME=/proj/cvl/users/x_fahkh2/caches
export PIP_CACHE_DIR=/proj/cvl/users/x_fahkh2/caches
export TMPDIR=/proj/cvl/users/x_fahkh2/caches
export TRITON_CACHE_DIR=/proj/cvl/users/x_fahkh2/caches
export TOKENIZERS_PARALLELISM=false
export PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ROOT="${ROOT:-$(pwd -P)}"
CONDA_ENV="${CONDA_ENV:-/proj/cvl/users/x_fahkh2/envs/neo_mobileov}"
PYTHON_BIN="${CONDA_ENV}/bin/python"
HF_REPO="${HF_REPO:-Amshaker/Mobile-OV}"
OUT="${OUT:-output/dreamlite_v8_v9_v10_visual_compare/${SLURM_JOB_ID:-local}}"
MAX_PROMPTS="${MAX_PROMPTS:-10}"
SEED="${SEED:-20260812}"
STOP_FILE="${TMPDIR%/}/mov_dream_cmp_${SLURM_JOB_ID:-local}.stop"

V8_CKPT="${V8_CKPT:-checkpoints/hf_mobile_ov/dreamlite_compact_v8_imageonly/17245434/dreamlite_image_bridge_latest.pt}"
V9_CKPT="${V9_CKPT:-checkpoints/hf_mobile_ov/dreamlite_compact_v9_grounded_from_scratch/17275939/dreamlite_image_bridge_latest.pt}"
V10_CKPT="${V10_CKPT:-checkpoints/hf_mobile_ov/dreamlite_compact_v10_balanced_from_scratch/17280132/dreamlite_image_bridge_latest.pt}"

mkdir -p logs "${OUT}"
test -x "${PYTHON_BIN}"
cd "${ROOT}"

rm -f "${STOP_FILE}"
"${PYTHON_BIN}" tools/utils/gpu_heartbeat.py \
  --all-devices --interval 5 --tensor-mb 4 --work-seconds 1 \
  --stop-file "${STOP_FILE}" --label "mov-dream-cmp-${SLURM_JOB_ID:-local}" &
HEARTBEAT_PID=$!
cleanup_heartbeat() {
  touch "${STOP_FILE}" 2>/dev/null || true
  wait "${HEARTBEAT_PID}" >/dev/null 2>&1 || true
  rm -f "${STOP_FILE}" 2>/dev/null || true
}
trap cleanup_heartbeat EXIT INT TERM

bash scripts/install_dreamlite_dependencies.sh
"${PYTHON_BIN}" tools/prepare_checkpoints.py --config configs/mobile_ov_dreamlite_compact_v10.yaml

download_checkpoint() {
  local target="$1"
  local filename="$2"
  if [[ -f "${target}" ]]; then
    return 0
  fi
  "${PYTHON_BIN}" - "${HF_REPO}" "${filename}" "${target}" <<'PY'
from huggingface_hub import hf_hub_download
from pathlib import Path
import shutil, sys

repo, filename, target = sys.argv[1:]
source = Path(hf_hub_download(repo_id=repo, filename=filename))
destination = Path(target)
destination.parent.mkdir(parents=True, exist_ok=True)
temporary = destination.with_suffix(destination.suffix + ".tmp")
shutil.copy2(source, temporary)
temporary.replace(destination)
print(f"Downloaded {repo}/{filename} -> {destination}")
PY
}

download_checkpoint "${V8_CKPT}" "dreamlite_compact_v8_imageonly/17245434/dreamlite_image_bridge_latest.pt"
download_checkpoint "${V9_CKPT}" "dreamlite_compact_v9_grounded_from_scratch/17275939/dreamlite_image_bridge_latest.pt"
download_checkpoint "${V10_CKPT}" "dreamlite_compact_v10_balanced_from_scratch/17280132/dreamlite_image_bridge_latest.pt"

echo "OUT=${OUT} MAX_PROMPTS=${MAX_PROMPTS} SEED=${SEED}"
nvidia-smi || true
"${PYTHON_BIN}" -u tools/render_dreamlite_bridge_comparison.py \
  --config configs/mobile_ov_dreamlite_compact_v10.yaml \
  --output-dir "${OUT}" \
  --variant "v8_imageonly=${V8_CKPT}" \
  --variant "v9_grounded=${V9_CKPT}" \
  --variant "v10_balanced=${V10_CKPT}" \
  --max-prompts "${MAX_PROMPTS}" \
  --seed "${SEED}" \
  --height 640 --width 1024 --steps 4 --dtype bf16 --overwrite

cleanup_heartbeat
trap - EXIT INT TERM
echo "Completed DreamLite V8/V9/V10 visual comparison."
cat "${OUT}/comparison_manifest.json"
