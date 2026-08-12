#!/usr/bin/env bash
# Run the controlled DreamLite V8 / NeoDragon ablation on the local H200 node.
# Each phase uses a fresh Python process so the node never retains Qwen,
# SmolVLM2, DreamLite, and NeoDragon in one CUDA process.
# Smoke: MAX_PROMPTS=1 sbatch scripts/ablate_dreamlite_v8_native_video_local_1gpu.sbatch
# Full:  sbatch scripts/ablate_dreamlite_v8_native_video_local_1gpu.sbatch
#
# This script is intentionally for the current local Slurm cluster. Berzelius
# should continue to use scripts/ablate_dreamlite_v8_native_video_20prompt_1node1gpu.sbatch.
#SBATCH --job-name=mov-v8ab-local
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=logs/mov-v8ab-local-%j.out
#SBATCH --error=logs/mov-v8ab-local-%j.out

set -euo pipefail

ROOT="${ROOT:-$(pwd -P)}"
CONDA_ENV="${CONDA_ENV:-/share_4/users/duy/.conda/envs/neo_mobileov}"
PYTHON_BIN="${PYTHON_BIN:-${CONDA_ENV}/bin/python}"
OUT="${OUT:-output/local_dreamlite_v8_native_video_control20/${SLURM_JOB_ID}}"
MAX_PROMPTS="${MAX_PROMPTS:-0}"
SEED="${SEED:-20260812}"
VBENCH_INFO="${VBENCH_INFO:-}"
REUSE_VBENCH_ROOT="${REUSE_VBENCH_ROOT:-}"
IMAGE_CKPT="${IMAGE_CKPT:-checkpoints/evaluation/dreamlite_v8_imageonly_17245434/dreamlite_compact_v8_imageonly/17245434/dreamlite_image_bridge_latest.pt}"
VIDEO_CKPT="${VIDEO_CKPT:-checkpoints/hf_mobile_ov/neo_exp1_bridge_functional/17108893/neodragon_text_bridge_latest.pt}"

cd "${ROOT}"
mkdir -p logs "${OUT}"
test -x "${PYTHON_BIN}"
test -f "${IMAGE_CKPT}"
test -f "${VIDEO_CKPT}"

export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

COMMON_ARGS=(
  --image-config configs/mobile_ov_dreamlite_compact_v8.yaml
  --image-bridge-checkpoint "${IMAGE_CKPT}"
  --video-bridge-checkpoint "${VIDEO_CKPT}"
  --output "${OUT}"
  --max-prompts "${MAX_PROMPTS}"
  --seed "${SEED}"
)
if [[ -n "${REUSE_VBENCH_ROOT}" ]]; then
  test -n "${VBENCH_INFO}"
  test -f "${VBENCH_INFO}"
  test -f "${REUSE_VBENCH_ROOT}/generation_summary.json"
  COMMON_ARGS+=(
    --vbench-info "${VBENCH_INFO}"
    --reuse-vbench-root "${REUSE_VBENCH_ROOT}"
  )
fi

echo "OUT=${OUT} MAX_PROMPTS=${MAX_PROMPTS} SEED=${SEED}"
echo "REUSE_VBENCH_ROOT=${REUSE_VBENCH_ROOT:-none}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi

"${PYTHON_BIN}" -u tools/ablate_dreamlite_v8_native_video.py "${COMMON_ARGS[@]}" --stage image_conditions
"${PYTHON_BIN}" -u tools/ablate_dreamlite_v8_native_video.py "${COMMON_ARGS[@]}" --stage anchors
"${PYTHON_BIN}" -u tools/ablate_dreamlite_v8_native_video.py "${COMMON_ARGS[@]}" --stage video_conditions
"${PYTHON_BIN}" -u tools/ablate_dreamlite_v8_native_video.py "${COMMON_ARGS[@]}" --stage videos
"${PYTHON_BIN}" -u tools/ablate_dreamlite_v8_native_video.py "${COMMON_ARGS[@]}" --stage summary

echo "Completed isolated local DreamLite V8 / NeoDragon ablation."
cat "${OUT}/summary.json"
