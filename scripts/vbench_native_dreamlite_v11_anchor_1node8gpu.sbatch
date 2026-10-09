#!/usr/bin/env bash
# Native Hybrid video stack, only first image changed from SSD1B to original V11/DreamLite PNG.
#SBATCH --job-name=mov-nat-dlv11
#SBATCH --partition=berzelius
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH --time=24:00:00
#SBATCH --account=berzelius-2026-196
#SBATCH --output=logs/mov-nat-dlv11-%j.out
#SBATCH --error=logs/mov-nat-dlv11-%j.out
set -Eeuo pipefail
cd "${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?}}"
export SWAP_CASE=native_hybrid_dreamlite_v11
bash scripts/run_vbench_first_frame_swap.sh
