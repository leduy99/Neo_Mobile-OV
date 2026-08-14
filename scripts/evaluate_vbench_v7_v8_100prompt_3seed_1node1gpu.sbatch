#!/usr/bin/env bash
# Comparative VBench subset: V7, V8 image-only, and V8 mix on identical
# stratified prompts and three independently seeded generations per checkpoint.
# This produces a statistical comparison, not a replacement for full VBench.
# Submit: sbatch scripts/evaluate_vbench_v7_v8_100prompt_3seed_1node1gpu.sbatch
# Smoke:  PROMPT_COUNT=2 SEEDS=20260812 VARIANTS=v8_imageonly sbatch $0
#SBATCH --job-name=mov-vbcmp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=72:00:00
#SBATCH --output=logs/mov-vbcmp-%j.out
#SBATCH --error=logs/mov-vbcmp-%j.out
#SBATCH --account=berzelius-2026-196
#SBATCH --gres=gpu:1
##SBATCH --gres=gpu:A100-SXM4-80GB:1

set -euo pipefail

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
HF_REPO="${HF_REPO:-Amshaker/Mobile-OV}"
RUN_NAME="${RUN_NAME:-vbench_v7_v8_stratified100x3}"
RUN_ROOT="${RUN_ROOT:-output/${RUN_NAME}}"
PROMPT_COUNT="${PROMPT_COUNT:-100}"
SUBSET_SEED="${SUBSET_SEED:-20260812}"
SEEDS="${SEEDS:-20260812,20260912,20261012}"
VARIANTS="${VARIANTS:-v7,v8_imageonly,v8_mix}"
STOP_FILE="${TMPDIR%/}/mov_vbench_compare_${SLURM_JOB_ID:-local}.stop"

mkdir -p logs "${RUN_ROOT}"
test -x "${PYTHON_BIN}"

rm -f "${STOP_FILE}"
"${PYTHON_BIN}" tools/utils/gpu_heartbeat.py \
  --all-devices --interval 5 --tensor-mb 4 --work-seconds 1 \
  --stop-file "${STOP_FILE}" --label "mov-vbench-compare-${SLURM_JOB_ID:-local}" &
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
"${PYTHON_BIN}" tools/prepare_checkpoints.py --config configs/mobile_ov_dreamlite_compact_v8.yaml

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

VIDEO_CKPT="${VIDEO_CKPT:-checkpoints/hf_mobile_ov/neo_exp1_bridge_functional/17108893/neodragon_text_bridge_latest.pt}"
download_checkpoint "${VIDEO_CKPT}" "neo_exp1_bridge_functional/17108893/neodragon_text_bridge_latest.pt" 64000

VBENCH_INFO="${VBENCH_REPO}/vbench/VBench_full_info.json"
[[ -f "${VBENCH_INFO}" ]] || VBENCH_INFO="${VBENCH_REPO}/VBench_full_info.json"
test -f "${VBENCH_INFO}"
SUBSET_INFO="${RUN_ROOT}/vbench_stratified_${PROMPT_COUNT}_info.json"
"${PYTHON_BIN}" tools/prepare_vbench_stratified_subset.py \
  --input "${VBENCH_INFO}" \
  --output "${SUBSET_INFO}" \
  --count "${PROMPT_COUNT}" \
  --seed "${SUBSET_SEED}"
# VBench scoring runs from its own checkout, so keep the subset manifest absolute.
SUBSET_INFO="$(realpath "${SUBSET_INFO}")"

variant_setting() {
  case "$1" in
    v7)
      IMAGE_CONFIG="configs/mobile_ov_dreamlite_compact_v7.yaml"
      IMAGE_CKPT="checkpoints/hf_mobile_ov/dreamlite_compact_v7_mobileo/17242707/dreamlite_image_bridge_step120000.pt"
      IMAGE_HF_FILE="dreamlite_compact_v7_mobileo/17242707/dreamlite_image_bridge_step120000.pt"
      IMAGE_STEP=120000
      ;;
    v8_imageonly)
      IMAGE_CONFIG="configs/mobile_ov_dreamlite_compact_v8.yaml"
      IMAGE_CKPT="checkpoints/hf_mobile_ov/dreamlite_compact_v8_imageonly/17245434/dreamlite_image_bridge_latest.pt"
      IMAGE_HF_FILE="dreamlite_compact_v8_imageonly/17245434/dreamlite_image_bridge_latest.pt"
      IMAGE_STEP=160000
      ;;
    v8_mix)
      IMAGE_CONFIG="configs/mobile_ov_dreamlite_compact_v8.yaml"
      IMAGE_CKPT="checkpoints/hf_mobile_ov/dreamlite_compact_v8_mix/17245433/dreamlite_image_bridge_latest.pt"
      IMAGE_HF_FILE="dreamlite_compact_v8_mix/17245433/dreamlite_image_bridge_latest.pt"
      IMAGE_STEP=160000
      ;;
    v9_grounded)
      IMAGE_CONFIG="configs/mobile_ov_dreamlite_compact_v9.yaml"
      IMAGE_CKPT="checkpoints/hf_mobile_ov/dreamlite_compact_v9_grounded_from_scratch/17275939/dreamlite_image_bridge_latest.pt"
      IMAGE_HF_FILE="dreamlite_compact_v9_grounded_from_scratch/17275939/dreamlite_image_bridge_latest.pt"
      IMAGE_STEP=160000
      ;;
    *) echo "Unknown variant=${1}; expected v7,v8_imageonly,v8_mix,v9_grounded" >&2; exit 2 ;;
  esac
  download_checkpoint "${IMAGE_CKPT}" "${IMAGE_HF_FILE}" "${IMAGE_STEP}"
}

IFS=',' read -r -a VARIANT_LIST <<< "${VARIANTS}"
IFS=',' read -r -a SEED_LIST <<< "${SEEDS}"
for variant in "${VARIANT_LIST[@]}"; do
  variant="${variant//[[:space:]]/}"
  variant_setting "${variant}"
  for seed in "${SEED_LIST[@]}"; do
    seed="${seed//[[:space:]]/}"
    CURRENT_ROOT="${RUN_ROOT}/${variant}/seed_${seed}"
    mkdir -p "${CURRENT_ROOT}/anchors" "${CURRENT_ROOT}/videos" "${CURRENT_ROOT}/scores"
    echo "Running variant=${variant} seed=${seed} root=${CURRENT_ROOT}"
    "${PYTHON_BIN}" tools/generate_vbench_dreamlite_exp1.py \
      --vbench-info "${SUBSET_INFO}" \
      --image-config "${IMAGE_CONFIG}" \
      --image-bridge-checkpoint "${IMAGE_CKPT}" \
      --video-bridge-checkpoint "${VIDEO_CKPT}" \
      --anchor-dir "${CURRENT_ROOT}/anchors" \
      --video-dir "${CURRENT_ROOT}/videos" \
      --summary "${CURRENT_ROOT}/generation_summary.json" \
      --seed "${seed}" \
      --samples-per-prompt 1
    (
      cd "${VBENCH_REPO}"
      "${VBENCH_ENV}/bin/python" "${ROOT}/tools/evaluate_vbench_resume.py" \
        --videos "${ROOT}/${CURRENT_ROOT}/videos" \
        --full-info "${SUBSET_INFO}" \
        --output "${ROOT}/${CURRENT_ROOT}/scores" \
        --expected-samples-per-prompt 1
    )
  done
done

VARIANT_NAMES=$(IFS=','; echo "${VARIANT_LIST[*]}")
SEED_NAMES=$(IFS=','; echo "${SEED_LIST[*]}")
"${PYTHON_BIN}" tools/summarize_vbench_repeated_subset.py \
  --root "${RUN_ROOT}" \
  --variants "${VARIANT_NAMES}" \
  --seeds "${SEED_NAMES}" \
  --output "${RUN_ROOT}/repeated_subset_summary.json"

cleanup_heartbeat
trap - EXIT INT TERM
echo "Completed comparative VBench subset."
cat "${RUN_ROOT}/repeated_subset_summary.json"
