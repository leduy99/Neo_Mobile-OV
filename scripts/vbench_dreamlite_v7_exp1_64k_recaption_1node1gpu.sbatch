#!/usr/bin/env bash
#SBATCH --job-name=mov-vbench-v7r
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=72:00:00
#SBATCH --output=logs/mov-vbench-v7r-%j.out
#SBATCH --error=logs/mov-vbench-v7r-%j.out
#SBATCH --account=berzelius-2026-196
#SBATCH --gres=gpu:1
##SBATCH --gres=gpu:A100-SXM4-80GB:1

set -euo pipefail

export RUN_NAME="${RUN_NAME:-dreamlite_v7_exp1_64k_smolrecap_vbench1x}"
export SAMPLES_PER_PROMPT="${SAMPLES_PER_PROMPT:-1}"
export RECAPTION_FILE="${RECAPTION_FILE:-output/vbench_smolvlm2_recaptions/vbench_full_smolvlm2_recap.json}"

source scripts/vbench_dreamlite_v7_exp1_64k_1node1gpu.sbatch
