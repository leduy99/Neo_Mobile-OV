#!/usr/bin/env bash
# Explicitly submit download/preparation, then train only after successful data completion.
set -Eeuo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
mkdir -p logs
DATA_JOB="$(sbatch --parsable scripts/download_mobileov_job_b_data_1node1gpu.sbatch)"
echo "Job-B GPU data/preparation job: ${DATA_JOB}"
TRAIN_JOB="$(sbatch --parsable --dependency="afterok:${DATA_JOB}" --kill-on-invalid-dep=yes \
    scripts/train_mobileov_stage2_job_b_lora_1node8gpu.sbatch)"
echo "Job-B LoRA training job: ${TRAIN_JOB} (after successful data preparation)"
