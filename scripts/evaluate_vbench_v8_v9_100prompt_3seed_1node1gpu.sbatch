#!/usr/bin/env bash
# Controlled checkpoint comparison: V8 image-only versus V9 grounded.
# Both use the same 100 stratified VBench prompts, three seeds, and Exp1-64K
# NeoDragon video bridge. Submit: sbatch scripts/evaluate_vbench_v8_v9_100prompt_3seed_1node1gpu.sbatch
# Smoke: PROMPT_COUNT=2 SEEDS=20260812 sbatch scripts/evaluate_vbench_v8_v9_100prompt_3seed_1node1gpu.sbatch
#SBATCH --job-name=mov-v8v9
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=72:00:00
#SBATCH --output=logs/mov-v8v9-%j.out
#SBATCH --error=logs/mov-v8v9-%j.out
#SBATCH --account=berzelius-2026-196
#SBATCH --gres=gpu:1
##SBATCH --gres=gpu:A100-SXM4-80GB:1

set -euo pipefail

export RUN_NAME="${RUN_NAME:-vbench_v8_imageonly_v9_grounded_stratified100x3}"
export PROMPT_COUNT="${PROMPT_COUNT:-100}"
export SUBSET_SEED="${SUBSET_SEED:-20260812}"
export SEEDS="${SEEDS:-20260812,20260912,20261012}"
export VARIANTS="${VARIANTS:-v8_imageonly,v9_grounded}"

source scripts/evaluate_vbench_v7_v8_100prompt_3seed_1node1gpu.sbatch
