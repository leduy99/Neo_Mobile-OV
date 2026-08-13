#!/usr/bin/env bash
#SBATCH --job-name=mov-dream-v9
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=72:00:00
#SBATCH --output=logs/mov-dream-v9-%j.out
#SBATCH --error=logs/mov-dream-v9-%j.out
#SBATCH --account=berzelius-2026-196
#SBATCH --gres=gpu:8
##SBATCH --gres=gpu:A100-SXM4-80GB:8

set -euo pipefail

source scripts/train_dreamlite_compact_v9_grounded_from_scratch_common.sh
