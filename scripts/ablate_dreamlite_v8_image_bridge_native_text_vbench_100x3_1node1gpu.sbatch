#!/usr/bin/env bash
# Final image-bridge ablation: native Qwen3-VL DreamLite conditioning versus
# V8 image-bridge conditioning. Native NeoDragon text conditioning and the
# released Hybrid DiT stay fixed, so any paired score delta is attributable to
# the image bridge plus its DreamLite first-frame effect.
#
# This is a stratified 100-prompt, three-seed comparative VBench run (600
# generated videos), not the canonical 944-prompt VBench score.
# Submit: sbatch scripts/ablate_dreamlite_v8_image_bridge_native_text_vbench_100x3_1node1gpu.sbatch
# Smoke:  PROMPT_COUNT=2 SEEDS=20260812 sbatch scripts/ablate_dreamlite_v8_image_bridge_native_text_vbench_100x3_1node1gpu.sbatch
#SBATCH --job-name=mov-imgablate
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=72:00:00
#SBATCH --output=logs/mov-imgablate-%j.out
#SBATCH --error=logs/mov-imgablate-%j.out
#SBATCH --account=berzelius-2026-196
#SBATCH --gres=gpu:1
##SBATCH --gres=gpu:A100-SXM4-80GB:1

set -euo pipefail

module load buildenv-gcccuda/12.1.1-gcc12.3.0

ROOT="${ROOT:-$(pwd -P)}"
CONDA_ENV="${CONDA_ENV:-/proj/cvl/users/x_fahkh2/envs/neo_mobileov}"
PYTHON_BIN="${CONDA_ENV}/bin/python"
VBENCH_ENV="${VBENCH_ENV:-/proj/cvl/users/x_fahkh2/envs/neo_vbench}"
VBENCH_REPO="${VBENCH_REPO:-${ROOT}/checkpoints/vbench_repo}"
HF_REPO="${HF_REPO:-Amshaker/Mobile-OV}"
RUN_NAME="${RUN_NAME:-dreamlite_v8_image_bridge_native_text_vbench100x3}"
RUN_ROOT="${RUN_ROOT:-output/${RUN_NAME}}"
PROMPT_COUNT="${PROMPT_COUNT:-100}"
SUBSET_SEED="${SUBSET_SEED:-20260812}"
SEEDS="${SEEDS:-20260812,20260912,20261012}"
STOP_FILE="/proj/cvl/users/x_fahkh2/caches/mov_imgablate_${SLURM_JOB_ID:-local}.stop"

IMAGE_CONFIG="${IMAGE_CONFIG:-configs/mobile_ov_dreamlite_compact_v8.yaml}"
IMAGE_CKPT="${IMAGE_CKPT:-checkpoints/hf_mobile_ov/dreamlite_compact_v8_imageonly/17245434/dreamlite_image_bridge_latest.pt}"
IMAGE_HF_FILE="${IMAGE_HF_FILE:-dreamlite_compact_v8_imageonly/17245434/dreamlite_image_bridge_latest.pt}"
IMAGE_EXPECTED_STEP="${IMAGE_EXPECTED_STEP:-160000}"

export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
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

cd "${ROOT}"
RUN_ROOT="$(realpath -m "${RUN_ROOT}")"
mkdir -p logs "${RUN_ROOT}" "$(dirname "${STOP_FILE}")"
test -x "${PYTHON_BIN}"

rm -f "${STOP_FILE}"
"${PYTHON_BIN}" tools/utils/gpu_heartbeat.py \
  --all-devices --interval 5 --tensor-mb 4 --work-seconds 1 \
  --stop-file "${STOP_FILE}" --label "mov-imgablate-${SLURM_JOB_ID:-local}" &
HEARTBEAT_PID=$!
cleanup_heartbeat() {
  touch "${STOP_FILE}" 2>/dev/null || true
  wait "${HEARTBEAT_PID}" >/dev/null 2>&1 || true
  rm -f "${STOP_FILE}" 2>/dev/null || true
}
trap cleanup_heartbeat EXIT INT TERM

bash scripts/install_dreamlite_dependencies.sh
export ROOT BASE_PYTHON="${PYTHON_BIN}" VBENCH_ENV VBENCH_REPO
bash scripts/install_vbench_env.sh
"${PYTHON_BIN}" tools/prepare_checkpoints.py --config "${IMAGE_CONFIG}"

download_checkpoint() {
  local target="$1"
  local filename="$2"
  local expected_step="$3"
  if [[ -f "${target}" ]]; then
    local actual_step
    actual_step=$("${PYTHON_BIN}" - "${target}" <<'PY'
import sys
import torch

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
import shutil
import sys

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

download_checkpoint "${IMAGE_CKPT}" "${IMAGE_HF_FILE}" "${IMAGE_EXPECTED_STEP}"

VBENCH_INFO="${VBENCH_REPO}/vbench/VBench_full_info.json"
[[ -f "${VBENCH_INFO}" ]] || VBENCH_INFO="${VBENCH_REPO}/VBench_full_info.json"
test -f "${VBENCH_INFO}"
SUBSET_INFO="${RUN_ROOT}/vbench_stratified_${PROMPT_COUNT}_info.json"
"${PYTHON_BIN}" tools/prepare_vbench_stratified_subset.py \
  --input "${VBENCH_INFO}" \
  --output "${SUBSET_INFO}" \
  --count "${PROMPT_COUNT}" \
  --seed "${SUBSET_SEED}"
SUBSET_INFO="$(realpath "${SUBSET_INFO}")"

echo "RUN_ROOT=${RUN_ROOT} PROMPT_COUNT=${PROMPT_COUNT} SEEDS=${SEEDS}"
echo "IMAGE_CKPT=${IMAGE_CKPT}"
echo "SUBSET_INFO=${SUBSET_INFO}"
nvidia-smi || true

IFS=',' read -r -a SEED_LIST <<< "${SEEDS}"
SOURCE_ROOTS=()
for seed in "${SEED_LIST[@]}"; do
  seed="${seed//[[:space:]]/}"
  SOURCE_ROOT="${RUN_ROOT}/runs/seed_${seed}"
  SOURCE_ROOTS+=("${SOURCE_ROOT}")
  mkdir -p "${SOURCE_ROOT}"
  COMMON_ARGS=(
    --image-config "${IMAGE_CONFIG}"
    --image-bridge-checkpoint "${IMAGE_CKPT}"
    --output "${SOURCE_ROOT}"
    --vbench-info "${SUBSET_INFO}"
    --max-prompts "${PROMPT_COUNT}"
    --seed "${seed}"
    --video-condition-names native_neodragon
  )
  echo "Generating paired image-only controls for seed=${seed}"
  # Fresh processes ensure Qwen, DreamLite, and NeoDragon never remain loaded
  # together on the one-GPU VBench allocation.
  "${PYTHON_BIN}" -u tools/ablate_dreamlite_v8_native_video.py "${COMMON_ARGS[@]}" --stage image_conditions
  "${PYTHON_BIN}" -u tools/ablate_dreamlite_v8_native_video.py "${COMMON_ARGS[@]}" --stage anchors
  "${PYTHON_BIN}" -u tools/ablate_dreamlite_v8_native_video.py "${COMMON_ARGS[@]}" --stage video_conditions
  "${PYTHON_BIN}" -u tools/ablate_dreamlite_v8_native_video.py "${COMMON_ARGS[@]}" --stage videos
  "${PYTHON_BIN}" -u tools/ablate_dreamlite_v8_native_video.py "${COMMON_ARGS[@]}" --stage summary
done

"${PYTHON_BIN}" tools/prepare_dreamlite_image_bridge_vbench.py \
  --run-roots "${SOURCE_ROOTS[@]}" \
  --vbench-info "${SUBSET_INFO}" \
  --output "${RUN_ROOT}"

for variant in native_qwen v8_imageonly; do
  for seed in "${SEED_LIST[@]}"; do
    seed="${seed//[[:space:]]/}"
    CURRENT_ROOT="${RUN_ROOT}/${variant}/seed_${seed}"
    echo "VBench scoring variant=${variant} seed=${seed}"
    (
      cd "${VBENCH_REPO}"
      "${VBENCH_ENV}/bin/python" "${ROOT}/tools/evaluate_vbench_resume.py" \
        --videos "${CURRENT_ROOT}/videos" \
        --full-info "${SUBSET_INFO}" \
        --output "${CURRENT_ROOT}/scores" \
        --expected-samples-per-prompt 1
    )
  done
done

"${PYTHON_BIN}" tools/summarize_dreamlite_image_bridge_vbench.py \
  --root "${RUN_ROOT}" \
  --output "${RUN_ROOT}/image_bridge_final_ablation_summary.json"

cleanup_heartbeat
trap - EXIT INT TERM
echo "Completed final image-bridge-only VBench ablation."
cat "${RUN_ROOT}/image_bridge_final_ablation_summary.json"
