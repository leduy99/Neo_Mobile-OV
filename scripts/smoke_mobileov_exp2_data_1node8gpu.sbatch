#!/usr/bin/env bash
# Same real models and worker layout, but a small non-training preview release.
#SBATCH --job-name=mov-exp2-smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --gres=gpu:8
#SBATCH --time=01:00:00
#SBATCH --account=berzelius-2026-196
#SBATCH --output=logs/mov-exp2-smoke-%j.out
#SBATCH --error=logs/mov-exp2-smoke-%j.out

set -Eeuo pipefail
ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the repo root}}"
cd "${ROOT}"
export EXP2_OUT="${EXP2_OUT:-download_data/data/mobileov_exp2_preview/${SLURM_JOB_ID}}"
export EXP2_LIMIT="${EXP2_LIMIT:-16}"
export MAX_RUNTIME_SECONDS="${MAX_RUNTIME_SECONDS:-2700}"
export AUTO_CONTINUE=0
exec bash scripts/prepare_mobileov_exp2_data_1node8gpu.sbatch
