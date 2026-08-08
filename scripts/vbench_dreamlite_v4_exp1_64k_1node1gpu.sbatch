#!/usr/bin/env bash
#SBATCH --job-name=mov-vbench
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=72:00:00
#SBATCH --output=logs/mov-vbench-%j.out
#SBATCH --error=logs/mov-vbench-%j.out
#SBATCH --account=berzelius-2026-196
#SBATCH --gres=gpu:1
##SBATCH --gres=gpu:A100-SXM4-80GB:1

set -euo pipefail

# Detectron2 is compiled against the cu121 PyTorch installed by
# install_vbench_env.sh, so use Berzelius's matching GCC/CUDA toolchain.
module load buildenv-gcccuda/12.1.1-gcc12.3.0

export PYTHONPATH="./:${PYTHONPATH:-}"
export HF_HOME=/proj/cvl/users/x_fahkh2/caches
export TORCH_HOME=/proj/cvl/users/x_fahkh2/caches
export XDG_CACHE_HOME=/proj/cvl/users/x_fahkh2/caches
export PIP_CACHE_DIR=/proj/cvl/users/x_fahkh2/caches
export TMPDIR=/proj/cvl/users/x_fahkh2/caches
export TRITON_CACHE_DIR=/proj/cvl/users/x_fahkh2/caches
export VBENCH_CACHE_DIR=/proj/cvl/users/x_fahkh2/caches/vbench
export TOKENIZERS_PARALLELISM=false
export PYTHONNOUSERSITE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ROOT="${ROOT:-$(pwd -P)}"
CONDA_ENV="${CONDA_ENV:-/proj/cvl/users/x_fahkh2/envs/neo_mobileov}"
PYTHON_BIN="${CONDA_ENV}/bin/python"
VBENCH_ENV="${VBENCH_ENV:-/proj/cvl/users/x_fahkh2/envs/neo_vbench}"
VBENCH_REPO="${VBENCH_REPO:-${ROOT}/checkpoints/vbench_repo}"
RUN_NAME="${RUN_NAME:-dreamlite_v4_exp1_64k_vbench1x}"
RUN_ROOT="${RUN_ROOT:-output/${RUN_NAME}}"
ANCHOR_DIR="${RUN_ROOT}/anchors"
VIDEO_DIR="${RUN_ROOT}/videos"
SCORE_DIR="${RUN_ROOT}/scores"

IMAGE_CONFIG="${IMAGE_CONFIG:-configs/mobile_ov_dreamlite_compact_v4.yaml}"
IMAGE_CKPT="${IMAGE_CKPT:-checkpoints/hf_mobile_ov/dreamlite_compact_v4/17208709/dreamlite_image_bridge_latest.pt}"
VIDEO_CKPT="${VIDEO_CKPT:-checkpoints/hf_mobile_ov/neo_exp1_bridge_functional/17108893/neodragon_text_bridge_latest.pt}"
IMAGE_HF_FILE="${IMAGE_HF_FILE:-dreamlite_compact_v4/17208709/dreamlite_image_bridge_latest.pt}"
VIDEO_HF_FILE="${VIDEO_HF_FILE:-neo_exp1_bridge_functional/17108893/neodragon_text_bridge_latest.pt}"
IMAGE_EXPECTED_STEP="${IMAGE_EXPECTED_STEP:-}"
HF_REPO="${HF_REPO:-Amshaker/Mobile-OV}"
MAX_PROMPTS="${MAX_PROMPTS:-0}"
SAMPLES_PER_PROMPT="${SAMPLES_PER_PROMPT:-1}"
STOP_FILE="${TMPDIR%/}/mov_vbench_setup_${SLURM_JOB_ID:-local}.stop"

mkdir -p logs "${ANCHOR_DIR}" "${VIDEO_DIR}" "${SCORE_DIR}"
test -x "${PYTHON_BIN}"

rm -f "${STOP_FILE}"
"${PYTHON_BIN}" tools/utils/gpu_heartbeat.py \
  --all-devices --interval 5 --tensor-mb 4 --work-seconds 1 \
  --stop-file "${STOP_FILE}" --label "mov-vbench-setup-${SLURM_JOB_ID:-local}" &
HEARTBEAT_PID=$!
cleanup_heartbeat() {
  touch "${STOP_FILE}" 2>/dev/null || true
  wait "${HEARTBEAT_PID}" >/dev/null 2>&1 || true
  rm -f "${STOP_FILE}" 2>/dev/null || true
}
trap cleanup_heartbeat EXIT INT TERM

export ROOT BASE_PYTHON="${PYTHON_BIN}" VBENCH_ENV VBENCH_REPO
bash scripts/install_dreamlite_dependencies.sh
bash scripts/install_vbench_env.sh

download_checkpoint() {
  local target="$1"
  local filename="$2"
  local expected_step="${3:-}"
  if [[ -f "${target}" && -n "${expected_step}" ]]; then
    local actual_step
    actual_step=$("${PYTHON_BIN}" - "${target}" <<'PY'
import sys, torch
payload = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
print(int(payload.get("step", -1)))
PY
)
    if [[ "${actual_step}" != "${expected_step}" ]]; then
      echo "Replacing stale checkpoint ${target}: step=${actual_step}, expected=${expected_step}"
      rm -f "${target}"
    fi
  fi
  [[ -f "${target}" ]] && return 0
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
  if [[ -n "${expected_step}" ]]; then
    "${PYTHON_BIN}" - "${target}" "${expected_step}" <<'PY'
import sys, torch
payload = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
actual = int(payload.get("step", -1))
expected = int(sys.argv[2])
if actual != expected:
    raise SystemExit(f"Checkpoint step mismatch: got {actual}, expected {expected}")
PY
  fi
}
download_checkpoint "${IMAGE_CKPT}" "${IMAGE_HF_FILE}" "${IMAGE_EXPECTED_STEP}"
download_checkpoint "${VIDEO_CKPT}" "${VIDEO_HF_FILE}"
"${PYTHON_BIN}" tools/prepare_checkpoints.py --config "${IMAGE_CONFIG}"

cleanup_heartbeat
trap - EXIT INT TERM

VBENCH_INFO="${VBENCH_REPO}/vbench/VBench_full_info.json"
if [[ ! -f "${VBENCH_INFO}" ]]; then
  VBENCH_INFO="${VBENCH_REPO}/VBench_full_info.json"
fi
test -f "${VBENCH_INFO}"

echo "RUN_ROOT=${RUN_ROOT} MAX_PROMPTS=${MAX_PROMPTS} SAMPLES_PER_PROMPT=${SAMPLES_PER_PROMPT}"
echo "IMAGE_CONFIG=${IMAGE_CONFIG}"
echo "IMAGE_CKPT=${IMAGE_CKPT}"
echo "VIDEO_CKPT=${VIDEO_CKPT}"
nvidia-smi || true

"${PYTHON_BIN}" tools/generate_vbench_dreamlite_exp1.py \
  --vbench-info "${VBENCH_INFO}" \
  --image-config "${IMAGE_CONFIG}" \
  --image-bridge-checkpoint "${IMAGE_CKPT}" \
  --video-bridge-checkpoint "${VIDEO_CKPT}" \
  --anchor-dir "${ANCHOR_DIR}" \
  --video-dir "${VIDEO_DIR}" \
  --summary "${RUN_ROOT}/generation_summary.json" \
  --samples-per-prompt "${SAMPLES_PER_PROMPT}" \
  --max-prompts "${MAX_PROMPTS}"

if [[ "${MAX_PROMPTS}" -gt 0 ]]; then
  echo "Smoke generation completed for MAX_PROMPTS=${MAX_PROMPTS}; skipping full VBench scores."
  exit 0
fi

cd "${VBENCH_REPO}"
"${VBENCH_ENV}/bin/python" "${ROOT}/tools/evaluate_vbench_resume.py" \
  --videos "${ROOT}/${VIDEO_DIR}" \
  --full-info "${VBENCH_INFO}" \
  --output "${ROOT}/${SCORE_DIR}" \
  --expected-samples-per-prompt "${SAMPLES_PER_PROMPT}"

echo "Completed VBench run with ${SAMPLES_PER_PROMPT} independently generated video(s) per prompt."
cat "${ROOT}/${SCORE_DIR}/vbench/summary.json"
