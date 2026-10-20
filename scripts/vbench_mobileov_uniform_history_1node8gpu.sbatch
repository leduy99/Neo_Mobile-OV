#!/usr/bin/env bash
# Completed uniform/history-corruption continuation; no corruption at inference.
#SBATCH --job-name=mov-vb-history
#SBATCH --partition=berzelius
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH --time=24:00:00
#SBATCH --account=berzelius-2026-196
#SBATCH --output=logs/mov-vb-history-%j.out
#SBATCH --error=logs/mov-vb-history-%j.out
set -Eeuo pipefail
cd "${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the repo root}}"
exec bash scripts/run_vbench_mobileov_continuation.sh uniform_history
