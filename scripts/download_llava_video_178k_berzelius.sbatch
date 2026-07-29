#!/usr/bin/env bash
#SBATCH --job-name=mov-dl-llava-video
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --output=logs/mov-dl-llava-video-%j.out
#SBATCH --error=logs/mov-dl-llava-video-%j.out
#SBATCH --account=berzelius-2026-196
#SBATCH --gres=gpu:1
##SBATCH --gres=gpu:A100-SXM4-80GB:1

set -euo pipefail

module load buildenv-nvhpc/24.5-cuda12.4
#module load buildenv-gcccuda/12.9.1-gcc11

export PYTHONPATH="./:${PYTHONPATH:-}"
export HF_HOME=/proj/cvl/users/x_fahkh2/caches
export TORCH_HOME=/proj/cvl/users/x_fahkh2/caches
export PIP_CACHE_DIR=/proj/cvl/users/x_fahkh2/caches
export TMPDIR=/proj/cvl/users/x_fahkh2/caches
export TRITON_CACHE_DIR=/proj/cvl/users/x_fahkh2/caches

export DATASET_REPO_ID="lmms-lab/LLaVA-Video-178K"
export DATASET_OUTPUT_DIR="${DATASET_OUTPUT_DIR:-download_data/data/llava_video_178k}"
export DOWNLOAD_WORKERS="${DOWNLOAD_WORKERS:-8}"

bash scripts/run_hf_dataset_download_with_heartbeat.sh
