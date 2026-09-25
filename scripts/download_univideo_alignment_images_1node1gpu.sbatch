#!/usr/bin/env bash
# Legacy filename retained; downloading and indexing use CPU resources only.
#SBATCH --job-name=mov-align-data
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/mov-align-data-%j.out
#SBATCH --error=logs/mov-align-data-%j.out
#SBATCH --account=berzelius-2026-196
#SBATCH --partition=berzelius-cpu

set -Eeuo pipefail
trap 'rc=$?; echo "ERROR: line=${LINENO} command=${BASH_COMMAND} exit=${rc}" >&2; exit "$rc"' ERR

PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit with sbatch from the Neo_Mobile-OV repo}}"
cd "${PROJECT_ROOT}"
exec bash scripts/run_alignment_image_download.sh
