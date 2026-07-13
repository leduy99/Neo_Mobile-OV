#!/usr/bin/env bash
#SBATCH --job-name=mobile-ov
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --time=72:00:00
#SBATCH --output=logs/slurmm1.0-%j.out
#SBATCH --error=logs/slurmm1.0-%j.out
#SBATCH --account=berzelius-2026-196
#SBATCH --gres=gpu:8
##SBATCH --gres=gpu:A100-SXM4-80GB:8

set -euo pipefail

# Backward-compatible alias. Prefer:
#   sbatch scripts/encode_neodragon_vae_latents.sbatch

bash scripts/encode_neodragon_vae_latents.sbatch
