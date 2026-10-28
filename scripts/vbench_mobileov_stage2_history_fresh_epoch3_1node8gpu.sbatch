#!/usr/bin/env bash
# Fresh-history run17892009, epoch 3; distinct from the clean-resume evaluations.
#SBATCH --job-name=mov-hc3
#SBATCH --partition=berzelius
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH --time=24:00:00
#SBATCH --account=berzelius-2026-196
#SBATCH --output=logs/mov-hc3-%j.out
#SBATCH --error=logs/mov-hc3-%j.out

set -Eeuo pipefail
cd "${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the repo root}}"
export EVAL_EPOCH=3
exec bash scripts/run_vbench_mobileov_history_fresh_epoch.sh
