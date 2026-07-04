#!/usr/bin/env bash
#SBATCH --job-name=neo-vae-enc
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=900G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.out

set -euo pipefail

# Backward-compatible alias. Prefer:
#   sbatch scripts/encode_neodragon_vae_latents.sbatch

bash scripts/encode_neodragon_vae_latents.sbatch
