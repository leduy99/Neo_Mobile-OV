#!/usr/bin/env bash
# Completed Job B: trained LLM LoRA + MCP + monolithic DiT, paired raw/QuickSR.
#SBATCH --job-name=mov-vb-jobb
#SBATCH --partition=berzelius
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH --time=24:00:00
#SBATCH --account=berzelius-2026-196
#SBATCH --output=logs/mov-vb-jobb-%j.out
#SBATCH --error=logs/mov-vb-jobb-%j.out
set -Eeuo pipefail
cd "${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the repo root}}"
exec bash scripts/run_vbench_mobileov_continuation.sh job_b
