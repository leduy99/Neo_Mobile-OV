#!/usr/bin/env bash
# Keep the completed sample-0 clips; generate four more seeds for BOTH systems.
#SBATCH --job-name=mov-extend5x
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=24
#SBATCH --mem=160G
#SBATCH --time=16:00:00
#SBATCH --output=logs/mov-extend5x-%j.out
#SBATCH --error=logs/mov-extend5x-%j.out

set -Eeuo pipefail
trap 'rc=$?; echo "ERROR: line=${LINENO} command=${BASH_COMMAND} exit=${rc}" >&2; exit "$rc"' ERR
ROOT="${PROJECT_ROOT:-/share_4/users/duy/project/unified_video/New-Mobile-OV}"
cd "${ROOT}"
PYTHON_BIN="${PYTHON_BIN:-/share_4/users/duy/.conda/envs/neo_mobileov/bin/python}"
VBENCH_ENV="${VBENCH_ENV:-/share_4/users/duy/.conda/envs/vbench_cu121_20260515}"
VBENCH_REPO="${VBENCH_REPO:-${ROOT}/checkpoints/vbench_repo}"
RUN_ROOT="${RUN_ROOT:-output/vbench_joint_v2_vs_native_quicksr_local_5x_from1x}"
JOINT_1X="${JOINT_1X:-output/vbench_joint_v2_quicksr_local_1x_ready}"
NATIVE_1X="${NATIVE_1X:-output/vbench_native_quicksr_local_1x}"
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
test -s "${JOINT_1X}/generation_summary.json"
test -s "${NATIVE_1X}/generation_summary.json"
echo "5x expansion on ${WORKERS:-2} allocated GPUs: reuse 944 sample-0 videos per model, add 3776 per model."
echo "Same weights, prompts, sampling schedules and QuickSR; full five-sample VBench rescoring."
srun --ntasks=1 "${PYTHON_BIN}" -u tools/benchmark_joint_v2_native.py \
  --output-dir "${RUN_ROOT}" --systems joint_v2 native_hybrid --protocol standard5x \
  --samples-per-prompt 5 --workers "${WORKERS:-2}" --seed 2026 --seed-mapping extendable \
  --reuse-1x-run "joint_v2=${JOINT_1X}" --reuse-1x-run "native_hybrid=${NATIVE_1X}" \
  --vbench-repo "${VBENCH_REPO}" --vbench-python "${VBENCH_ENV}/bin/python"
test -s "${RUN_ROOT}/comparison.json"
cat "${RUN_ROOT}/comparison.md"
