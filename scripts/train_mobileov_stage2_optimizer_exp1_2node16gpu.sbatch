#!/usr/bin/env bash
# Original epoch-3 weights + Adam; 20K NEW updates at global batch 64.
#SBATCH --job-name=mov-s2-opt-exp1
#SBATCH --partition=berzelius
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gres=gpu:8
#SBATCH --time=72:00:00
#SBATCH --account=berzelius-2026-196
#SBATCH --output=logs/mov-s2-opt-exp1-%j.out
#SBATCH --error=logs/mov-s2-opt-exp1-%j.out

set -Eeuo pipefail
cd "${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the repo root}}"
export PROJECT_ROOT="${PWD}"
export EXP1_NODES=2 EXP1_ACCUMULATION=4
export EXP1_SUBMIT_SCRIPT=scripts/train_mobileov_stage2_optimizer_exp1_2node16gpu.sbatch
exec bash scripts/run_mobileov_stage2_optimizer_exp1_16gpu.sh
