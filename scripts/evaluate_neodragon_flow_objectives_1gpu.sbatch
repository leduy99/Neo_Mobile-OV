#!/usr/bin/env bash
#SBATCH --job-name=neo-pfm-eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=160G
#SBATCH --time=02:00:00
#SBATCH --output=logs/neo-pfm-eval-%j.out
#SBATCH --error=logs/neo-pfm-eval-%j.out

set -euo pipefail

ROOT="/share_4/users/duy/project/unified_video/New-Mobile-OV"
PYTHON_BIN="${PYTHON_BIN:-/share_4/users/duy/.conda/envs/neo_mobileov/bin/python}"
RUN_DIR="${RUN_DIR:?Set RUN_DIR to the completed objective-ablation directory}"
OUTPUT_DIR="${OUTPUT_DIR:-${RUN_DIR}/controlled_rollouts}"

cd "${ROOT}"
mkdir -p logs "${OUTPUT_DIR}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-${ROOT}/checkpoints/hf_cache}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

"${PYTHON_BIN}" tools/evaluate_neodragon_flow_ablation.py \
  --run-dir "${RUN_DIR}" \
  --output-dir "${OUTPUT_DIR}"

test -s "${OUTPUT_DIR}/evaluation_report.json"
find "${OUTPUT_DIR}" -maxdepth 2 -type f \( -name '*.mp4' -o -name '*contact.jpg' -o -name 'evaluation_report.json' \) -print
