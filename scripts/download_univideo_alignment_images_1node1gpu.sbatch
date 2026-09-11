#!/usr/bin/env bash
#SBATCH --job-name=mov-align-data
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/mov-align-data-%j.out
#SBATCH --error=logs/mov-align-data-%j.out
#SBATCH --account=berzelius-2026-196
#SBATCH --gres=gpu:1

set -Eeuo pipefail
trap 'rc=$?; echo "ERROR: line=${LINENO} command=${BASH_COMMAND} exit=${rc}" >&2; exit "$rc"' ERR

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit with sbatch from the Neo_Mobile-OV repo}}"
cd "${PROJECT_ROOT}"
module load buildenv-nvhpc/24.5-cuda12.4
exec bash scripts/run_alignment_image_download.sh
