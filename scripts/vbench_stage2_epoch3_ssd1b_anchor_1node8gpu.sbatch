#!/usr/bin/env bash
#SBATCH --job-name=mov-s2-ssd-anch
#SBATCH --partition=berzelius
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH --time=24:00:00
#SBATCH --account=berzelius-2026-196
#SBATCH --output=logs/mov-s2-ssd-anch-%j.out
#SBATCH --error=logs/mov-s2-ssd-anch-%j.out
set -Eeuo pipefail
export STAGE2_PHASE=ssd1b_anchor
bash "${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?}}/scripts/run_vbench_stage2_ssd1b_shared.sh"
