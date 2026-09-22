#!/usr/bin/env bash
# Add Vchitect shards; reuse BLIP3o and old videos without overwriting the old release.
# Resubmit unchanged after interruption. This job does not train or pre-encode VAE latents.
#SBATCH --job-name=mov-t2v-expand
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=logs/mov-t2v-expand-%j.out
#SBATCH --error=logs/mov-t2v-expand-%j.out
#SBATCH --account=berzelius-2026-196
#SBATCH --gres=gpu:1

set -Eeuo pipefail
trap 'rc=$?; echo "ERROR: line=${LINENO} command=${BASH_COMMAND} exit=${rc}" >&2; exit "$rc"' ERR
PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the Neo_Mobile-OV repo}}"
cd "${PROJECT_ROOT}"
module load buildenv-nvhpc/24.5-cuda12.4
export ALIGNMENT_SCOPE=stage1-expand
export DATASET_OUTPUT_DIR="${DATASET_OUTPUT_DIR:-download_data/data/univideo_stage1_expanded650}"
export TOTAL_VIDEO_SHARDS="${TOTAL_VIDEO_SHARDS:-650}"
export MIN_TRAIN_VIDEOS="${MIN_TRAIN_VIDEOS:-500000}"
export MAX_TOTAL_VIDEO_GIB="${MAX_TOTAL_VIDEO_GIB:-2500}"
echo "Expand T2V: total shards=${TOTAL_VIDEO_SHARDS}, minimum TRAIN videos=${MIN_TRAIN_VIDEOS}"
echo "Old+new video archive cap=${MAX_TOTAL_VIDEO_GIB} GiB; images/latents/checkpoints are additional."
echo "Only additional shards are downloaded; original task manifests and validation stay unchanged."
exec bash scripts/run_alignment_image_download.sh
