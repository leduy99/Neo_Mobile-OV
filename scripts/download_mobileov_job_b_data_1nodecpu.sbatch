#!/usr/bin/env bash
# Bounded understanding source files only; no GPU, VAE encoding or training.
#SBATCH --job-name=mov-jobb-data
#SBATCH --partition=berzelius-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --account=berzelius-2026-196
#SBATCH --output=logs/mov-jobb-data-%j.out
#SBATCH --error=logs/mov-jobb-data-%j.out

set -Eeuo pipefail
trap 'rc=$?; echo "ERROR: line=${LINENO} command=${BASH_COMMAND} exit=${rc}" >&2; exit "$rc"' ERR
cd "${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the Neo_Mobile-OV repository root}}"
PYTHON="${PYTHON:-/proj/cvl/users/x_fahkh2/envs/neo_mobileov/bin/python}"
export HF_HOME="${HF_HOME:-/proj/cvl/users/x_fahkh2/caches}"
export HF_HUB_DISABLE_XET=1 PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false
test -x "${PYTHON}"
mkdir -p logs
echo "Job B: Cauldron image QA + Smol-SmolTalk train subset + bounded LLaVA-Video QA/media."
echo "About 8.05 GiB source files; no Mobile-O archives or generation data."
echo "Download completion is NOT training readiness; no training job will be submitted."
exec srun --ntasks=1 "${PYTHON}" -s -u tools/download_mobileov_job_b_data.py \
    --plan configs/data_sources/mobileov_job_b_understanding_sources.json \
    --output-dir "${JOB_B_DATA_ROOT:-download_data/data/mobileov_job_b_understanding_sources}" \
    --workers "${DOWNLOAD_WORKERS:-4}" --retries 3 \
    --max-download-gib "${MAX_DOWNLOAD_GIB:-9}" --disk-margin-gib 2
