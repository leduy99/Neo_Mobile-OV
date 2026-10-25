#!/usr/bin/env bash
# Same protocol as epoch 4, pinned to the completed epoch-5 archive.
#SBATCH --job-name=mov-vb-res-ep5
#SBATCH --partition=berzelius
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH --time=24:00:00
#SBATCH --account=berzelius-2026-196
#SBATCH --output=logs/mov-vb-res-ep5-%j.out
#SBATCH --error=logs/mov-vb-res-ep5-%j.out

set -Eeuo pipefail
cd "${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the repo root}}"
export EVAL_EPOCH=5
exec bash scripts/vbench_mobileov_stage2_resume_epoch4_1node8gpu.sbatch
