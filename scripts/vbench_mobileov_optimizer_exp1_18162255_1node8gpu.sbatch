#!/usr/bin/env bash
# Completed 16-GPU training run; four RAW/EMA archives, each scored raw + QuickSR.
#SBATCH --job-name=mov-vb-opt18162255
#SBATCH --partition=berzelius
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH --time=72:00:00
#SBATCH --account=berzelius-2026-196
#SBATCH --output=logs/mov-vb-opt18162255-%j.out
#SBATCH --error=logs/mov-vb-opt18162255-%j.out

set -Eeuo pipefail
cd "${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the repo root}}"
export PROJECT_ROOT="${PWD}"
export EXP1_RUN=output/mobileov_stage2_optimizer_exp1/18162255
export RUN_ROOT="${RUN_ROOT:-output/vbench_mobileov_optimizer_exp1_18162255_5x_dual}"
export EVAL_UPDATES="10000 20000" REQUIRE_EXP1_COMPLETE=1 EXPECTED_TRAIN_WORLD_SIZE=16
exec bash scripts/vbench_mobileov_stage2_optimizer_exp1_1node8gpu.sbatch
