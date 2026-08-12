#!/usr/bin/env bash
# Controlled 20-prompt matrix: native Qwen vs V8 DreamLite condition crossed
# with native NeoDragon vs Exp1-64K video condition.
# Submit: sbatch scripts/ablate_dreamlite_v8_native_video_20prompt_1node1gpu.sbatch
# Smoke:  MAX_PROMPTS=1 sbatch scripts/ablate_dreamlite_v8_native_video_20prompt_1node1gpu.sbatch
#SBATCH --job-name=mov-v8ablate
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=72:00:00
#SBATCH --output=logs/mov-v8ablate-%j.out
#SBATCH --error=logs/mov-v8ablate-%j.out
#SBATCH --account=berzelius-2026-196
#SBATCH --gres=gpu:1
##SBATCH --gres=gpu:A100-SXM4-80GB:1

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
RUN_NAME="${RUN_NAME:-dreamlite_v8_native_video_control20}"
OUT="${OUT:-output/${RUN_NAME}/${SLURM_JOB_ID:-local}}"
MAX_PROMPTS="${MAX_PROMPTS:-0}"
STOP_FILE="${TMPDIR%/}/mov_v8_ablate_${SLURM_JOB_ID:-local}.stop"

IMAGE_CONFIG="${IMAGE_CONFIG:-configs/mobile_ov_dreamlite_compact_v8.yaml}"
IMAGE_CKPT="${IMAGE_CKPT:-checkpoints/hf_mobile_ov/dreamlite_compact_v8_imageonly/17245434/dreamlite_image_bridge_latest.pt}"
IMAGE_HF_FILE="${IMAGE_HF_FILE:-dreamlite_compact_v8_imageonly/17245434/dreamlite_image_bridge_latest.pt}"
VIDEO_CKPT="${VIDEO_CKPT:-checkpoints/hf_mobile_ov/neo_exp1_bridge_functional/17108893/neodragon_text_bridge_latest.pt}"
VIDEO_HF_FILE="${VIDEO_HF_FILE:-neo_exp1_bridge_functional/17108893/neodragon_text_bridge_latest.pt}"

mkdir -p logs "${OUT}"
test -x "${PYTHON_BIN}"

rm -f "${STOP_FILE}"
"${PYTHON_BIN}" tools/utils/gpu_heartbeat.py \
  --all-devices --interval 5 --tensor-mb 4 --work-seconds 1 \
  --stop-file "${STOP_FILE}" --label "mov-v8-ablate-${SLURM_JOB_ID:-local}" &
HEARTBEAT_PID=$!
cleanup_heartbeat() {
  touch "${STOP_FILE}" 2>/dev/null || true
  wait "${HEARTBEAT_PID}" >/dev/null 2>&1 || true
  rm -f "${STOP_FILE}" 2>/dev/null || true
}
trap cleanup_heartbeat EXIT INT TERM

bash scripts/install_dreamlite_dependencies.sh
"${PYTHON_BIN}" tools/prepare_checkpoints.py --config "${IMAGE_CONFIG}"

download_checkpoint() {
  local target="$1"
  local filename="$2"
  local expected_step="$3"
  if [[ -f "${target}" ]]; then
    local actual_step
    actual_step=$("${PYTHON_BIN}" - "${target}" <<'PY'
import sys, torch
print(int(torch.load(sys.argv[1], map_location="cpu", weights_only=False).get("step", -1)))
PY
)
    if [[ "${actual_step}" == "${expected_step}" ]]; then
      return 0
    fi
    echo "Replacing stale ${target}: step=${actual_step}, expected=${expected_step}"
    rm -f "${target}"
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

download_checkpoint "${IMAGE_CKPT}" "${IMAGE_HF_FILE}" 160000
download_checkpoint "${VIDEO_CKPT}" "${VIDEO_HF_FILE}" 64000

echo "OUT=${OUT} MAX_PROMPTS=${MAX_PROMPTS}"
echo "IMAGE_CKPT=${IMAGE_CKPT}"
echo "VIDEO_CKPT=${VIDEO_CKPT}"
nvidia-smi || true

"${PYTHON_BIN}" tools/ablate_dreamlite_v8_native_video.py \
  --image-config "${IMAGE_CONFIG}" \
  --image-bridge-checkpoint "${IMAGE_CKPT}" \
  --video-bridge-checkpoint "${VIDEO_CKPT}" \
  --output "${OUT}" \
  --max-prompts "${MAX_PROMPTS}"

cleanup_heartbeat
trap - EXIT INT TERM
echo "Completed native/image/video controlled ablation."
cat "${OUT}/summary.json"
