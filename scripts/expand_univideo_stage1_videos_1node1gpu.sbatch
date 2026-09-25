#!/usr/bin/env bash
# Compatibility entry point. Despite its legacy name, this job requests CPU only.
# Prefer expand_univideo_stage1_videos_1nodecpu.sbatch for new submissions.
#SBATCH --job-name=mov-t2v-expand
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --output=logs/mov-t2v-expand-%j.out
#SBATCH --error=logs/mov-t2v-expand-%j.out
#SBATCH --account=berzelius-2026-196
#SBATCH --partition=berzelius-cpu

set -Eeuo pipefail
trap 'rc=$?; echo "ERROR: line=${LINENO} command=${BASH_COMMAND} exit=${rc}" >&2; exit "$rc"' ERR
PROJECT_ROOT="${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the Neo_Mobile-OV repo}}"
cd "${PROJECT_ROOT}"
exec bash scripts/expand_univideo_stage1_videos_1nodecpu.sbatch
