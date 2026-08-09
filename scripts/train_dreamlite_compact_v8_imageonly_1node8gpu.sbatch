#!/usr/bin/env bash
#SBATCH --job-name=mov-dream-v8-image
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=72:00:00
#SBATCH --output=logs/mov-dream-v8-image-%j.out
#SBATCH --error=logs/mov-dream-v8-image-%j.out
#SBATCH --account=berzelius-2026-196
#SBATCH --gres=gpu:8
##SBATCH --gres=gpu:A100-SXM4-80GB:8

set -euo pipefail

export V8_VARIANT=imageonly
source scripts/train_dreamlite_compact_v8_common.sh
