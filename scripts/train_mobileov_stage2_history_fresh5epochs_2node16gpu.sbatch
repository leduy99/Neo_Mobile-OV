#!/usr/bin/env bash
# Fresh Stage 2 from alignment15k + released monolithic DiT, not epoch-3 weights.
#SBATCH --job-name=mov-s2-hist5-16g
#SBATCH --partition=berzelius
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:8
#SBATCH --time=72:00:00
#SBATCH --account=berzelius-2026-196
#SBATCH --output=logs/mov-s2-hist5-16g-%j.out
#SBATCH --error=logs/mov-s2-hist5-16g-%j.out

set -Eeuo pipefail
cd "${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the repo root}}"
export PROJECT_ROOT="${PWD}"
exec bash scripts/run_mobileov_stage2_2node16gpu.sh history
