#!/usr/bin/env bash
# Step 100K is ~1.056 media epochs, not the exact epoch boundary at 94,659.
#SBATCH --job-name=mov-vb-s2-ep1
#SBATCH --partition=berzelius
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH --time=24:00:00
#SBATCH --account=berzelius-2026-196
#SBATCH --output=logs/mov-vb-s2-ep1-%j.out
#SBATCH --error=logs/mov-vb-s2-ep1-%j.out
set -Eeuo pipefail
cd "${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?}}"
export EVAL_EPOCH=1
bash scripts/run_vbench_mobileov_stage2_epoch.sh
