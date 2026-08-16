#!/usr/bin/env bash
# V11 control: exact V9 curriculum with an explicit content-aware objective.
#SBATCH --job-name=mov-dream-v11c
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=72:00:00
#SBATCH --output=logs/mov-dream-v11c-%j.out
#SBATCH --error=logs/mov-dream-v11c-%j.out
#SBATCH --account=berzelius-2026-196
#SBATCH --gres=gpu:8
##SBATCH --gres=gpu:A100-SXM4-80GB:8

set -euo pipefail

export V11_VARIANT=control
source scripts/train_dreamlite_compact_v11_common.sh
