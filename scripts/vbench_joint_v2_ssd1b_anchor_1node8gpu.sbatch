#!/usr/bin/env bash
# Joint V2 video stack, only first image changed from V11/DreamLite to original SSD1B PNG.
#SBATCH --job-name=mov-jv2-ssd
#SBATCH --partition=berzelius
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:8
#SBATCH --time=24:00:00
#SBATCH --account=berzelius-2026-196
#SBATCH --output=logs/mov-jv2-ssd-%j.out
#SBATCH --error=logs/mov-jv2-ssd-%j.out
set -Eeuo pipefail
cd "${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?}}"
export SWAP_CASE=joint_v2_ssd1b
bash scripts/run_vbench_first_frame_swap.sh
