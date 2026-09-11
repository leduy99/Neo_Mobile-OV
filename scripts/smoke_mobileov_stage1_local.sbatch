#!/usr/bin/env bash
#SBATCH --job-name=mov-s1-smoke
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=00:40:00
#SBATCH --output=logs/mov-s1-smoke-%j.out
#SBATCH --error=logs/mov-s1-smoke-%j.out

set -Eeuo pipefail
trap 'rc=$?; echo "ERROR: line=${LINENO} command=${BASH_COMMAND} exit=${rc}" >&2; exit "$rc"' ERR
cd "${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?}}"
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2
export HF_HOME="${HF_HOME:-/share_4/users/duy/.cache/huggingface}"
export TMPDIR="${TMPDIR:-/share_4/users/duy/tmp}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PYTHON="${PYTHON_BIN:-/share_4/users/duy/.conda/envs/neo_mobileov/bin/python}"
"${PYTHON}" -s -u tools/train_mobileov_stage1.py \
    --data-root "${DATA_ROOT:-output/alignment_data_checks/stage1_slurm_release}" \
    --output-dir "${OUT:-output/stage1_local_smoke/${SLURM_JOB_ID}}" \
    --steps 3 --accumulation 3 --workers 1 --warmup 1 \
    --save-every 1 --archive-every 3 --validate-every 3 --validation-samples 1 \
    --log-every 1 --verify-frozen
