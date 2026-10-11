#!/usr/bin/env bash
# Step 190K is ~2.007 media epochs, not the exact epoch boundary at 189,318.
#SBATCH --job-name=mov-vb-s2-ep2
#SBATCH --partition=berzelius
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH --time=24:00:00
#SBATCH --account=berzelius-2026-196
#SBATCH --output=logs/mov-vb-s2-ep2-%j.out
#SBATCH --error=logs/mov-vb-s2-ep2-%j.out
set -Eeuo pipefail
cd "${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?}}"
export EVAL_EPOCH=2
bash scripts/run_vbench_mobileov_stage2_epoch.sh
