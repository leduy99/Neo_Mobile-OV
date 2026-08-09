#!/usr/bin/env bash
#SBATCH --job-name=mov-vbench-v7-full
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=72:00:00
#SBATCH --output=logs/mov-vbench-v7-full-%j.out
#SBATCH --error=logs/mov-vbench-v7-full-%j.out
#SBATCH --account=berzelius-2026-196
#SBATCH --gres=gpu:1
##SBATCH --gres=gpu:A100-SXM4-80GB:1

set -euo pipefail

# Full VBench protocol: five independently seeded videos for every prompt.
export RUN_NAME="${RUN_NAME:-dreamlite_v7_exp1_64k_vbench5x}"
export SAMPLES_PER_PROMPT=5

source scripts/vbench_dreamlite_v7_exp1_64k_1node1gpu.sbatch
