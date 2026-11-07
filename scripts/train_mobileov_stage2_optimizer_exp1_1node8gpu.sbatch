#!/usr/bin/env bash
# Same batch64 recipe and media as the 16-GPU job; eight accumulation rounds.
#SBATCH --job-name=mov-s2-opt8-exp1
#SBATCH --partition=berzelius
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:8
#SBATCH --time=72:00:00
#SBATCH --account=berzelius-2026-196
#SBATCH --output=logs/mov-s2-opt8-exp1-%j.out
#SBATCH --error=logs/mov-s2-opt8-exp1-%j.out

set -Eeuo pipefail
cd "${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the repo root}}"
export PROJECT_ROOT="${PWD}"
export EXP1_NODES=1 EXP1_ACCUMULATION=8
export EXP1_SUBMIT_SCRIPT=scripts/train_mobileov_stage2_optimizer_exp1_1node8gpu.sbatch
# Preserve the 16-GPU job's sixteen validation examples per task.
export VALIDATION_SAMPLES="${VALIDATION_SAMPLES:-2}"
exec bash scripts/run_mobileov_stage2_optimizer_exp1_16gpu.sh
