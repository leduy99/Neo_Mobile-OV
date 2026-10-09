#!/usr/bin/env bash
# Trained model ONLY: DreamLite V11 + frozen Bridge V2 + monolithic DiT step 10K.
# 944 official prompts, one video/prompt, shared public QuickSR medium 2x.
#SBATCH --job-name=mov-jv2-qsr1x
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=24
#SBATCH --mem=160G
#SBATCH --time=06:00:00
#SBATCH --output=logs/mov-jv2-qsr1x-%j.out
#SBATCH --error=logs/mov-jv2-qsr1x-%j.out

set -Eeuo pipefail
trap 'rc=$?; echo "ERROR: line=${LINENO} command=${BASH_COMMAND} exit=${rc}" >&2; exit "$rc"' ERR
ROOT="${PROJECT_ROOT:-/share_4/users/duy/project/unified_video/New-Mobile-OV}"
cd "${ROOT}"
PYTHON_BIN="${PYTHON_BIN:-/share_4/users/duy/.conda/envs/neo_mobileov/bin/python}"
VBENCH_ENV="${VBENCH_ENV:-/share_4/users/duy/.conda/envs/vbench_cu121_20260515}"
VBENCH_REPO="${VBENCH_REPO:-${ROOT}/checkpoints/vbench_repo}"
MAX_PROMPTS="${MAX_PROMPTS:-0}"
SUFFIX=""
if [[ "${MAX_PROMPTS}" -gt 0 ]]; then SUFFIX="_smoke${MAX_PROMPTS}"; fi
RUN_ROOT="${RUN_ROOT:-output/vbench_joint_v2_quicksr_local_1x${SUFFIX}}"

export PYTHONPATH="${ROOT}:${VBENCH_REPO}:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_XET=1
export HF_HOME="${HF_HOME:-/share_4/users/duy/.cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-/share_4/users/duy/.cache/torch}"
export VBENCH_CACHE_DIR="${VBENCH_CACHE_DIR:-/share_4/users/duy/.cache/vbench}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
mkdir -p logs "${RUN_ROOT}" "${VBENCH_CACHE_DIR}"
test -x "${PYTHON_BIN}"
test -x "${VBENCH_ENV}/bin/python"
test -s "${VBENCH_REPO}/vbench/VBench_full_info.json"
"${VBENCH_ENV}/bin/python" -c 'import torch, vbench, detectron2; print("VBench environment ready:", torch.__version__)'

echo "RUN_ROOT=${RUN_ROOT}"
echo "Trained Joint V2 only; native NeoDragon is NOT generated/evaluated."
echo "2 GPU workers; 944 prompts x 1 video; 49 frames, 24 FPS, QuickSR 2x, final 1024x640."
echo "Native VAE + DiT architecture config are reused as components, not as a native benchmark run."
srun --ntasks=1 --gres=gpu:2 "${PYTHON_BIN}" -u tools/benchmark_joint_v2_native.py \
  --output-dir "${RUN_ROOT}" \
  --systems joint_v2 \
  --protocol paired1x \
  --samples-per-prompt 1 \
  --workers 2 \
  --max-prompts "${MAX_PROMPTS}" \
  --seed "${SEED:-2026}" \
  --vbench-repo "${VBENCH_REPO}" \
  --vbench-python "${VBENCH_ENV}/bin/python"

if [[ "${MAX_PROMPTS}" -eq 0 ]]; then
  test -s "${RUN_ROOT}/result.json"
  cat "${RUN_ROOT}/result.md"
fi
