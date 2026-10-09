#!/usr/bin/env bash
# Released native baseline, matched to the completed Joint V2 1x QuickSR run.
#SBATCH --job-name=mov-native-qsr1x
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=24
#SBATCH --mem=160G
#SBATCH --time=06:00:00
#SBATCH --output=logs/mov-native-qsr1x-%j.out
#SBATCH --error=logs/mov-native-qsr1x-%j.out

set -Eeuo pipefail
trap 'rc=$?; echo "ERROR: line=${LINENO} command=${BASH_COMMAND} exit=${rc}" >&2; exit "$rc"' ERR
ROOT="${PROJECT_ROOT:-/share_4/users/duy/project/unified_video/New-Mobile-OV}"
cd "${ROOT}"
PYTHON_BIN="${PYTHON_BIN:-/share_4/users/duy/.conda/envs/neo_mobileov/bin/python}"
VBENCH_ENV="${VBENCH_ENV:-/share_4/users/duy/.conda/envs/vbench_cu121_20260515}"
VBENCH_REPO="${VBENCH_REPO:-${ROOT}/checkpoints/vbench_repo}"
RUN_ROOT="${RUN_ROOT:-output/vbench_native_quicksr_local_1x}"
export PYTHONPATH="${ROOT}:${VBENCH_REPO}:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_XET=1
export HF_HOME="${HF_HOME:-/share_4/users/duy/.cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-/share_4/users/duy/.cache/torch}"
export VBENCH_CACHE_DIR="${VBENCH_CACHE_DIR:-/share_4/users/duy/.cache/vbench}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
mkdir -p logs "${RUN_ROOT}"
test -x "${PYTHON_BIN}"
test -x "${VBENCH_ENV}/bin/python"
echo "Native SSD1B + native conditioner + released Hybrid only; same 1x QuickSR protocol as Joint V2."
srun --ntasks=1 --gres=gpu:2 "${PYTHON_BIN}" -u tools/benchmark_joint_v2_native.py \
  --output-dir "${RUN_ROOT}" --systems native_hybrid --protocol paired1x \
  --samples-per-prompt 1 --workers 2 --seed 2026 \
  --vbench-repo "${VBENCH_REPO}" --vbench-python "${VBENCH_ENV}/bin/python"
test -s "${RUN_ROOT}/result.json"
cat "${RUN_ROOT}/result.md"
