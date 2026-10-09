#!/usr/bin/env bash
# Exercise the production generator, resume and actual VBench environment.
#SBATCH --job-name=mov-jv2-vb-smoke
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=24
#SBATCH --mem=160G
#SBATCH --time=00:30:00
#SBATCH --output=logs/mov-jv2-vb-smoke-%j.out
#SBATCH --error=logs/mov-jv2-vb-smoke-%j.out

set -Eeuo pipefail
ROOT="${PROJECT_ROOT:-/share_4/users/duy/project/unified_video/New-Mobile-OV}"
cd "${ROOT}"
export MAX_PROMPTS=2
export RUN_ROOT="${RUN_ROOT:-output/paired_vbench_preflight/gpu_import_fix}"
bash scripts/vbench_joint_v2_quicksr_local_2gpu.sbatch
bash scripts/vbench_joint_v2_quicksr_local_2gpu.sbatch
export PYTHONPATH="${ROOT}:${ROOT}/checkpoints/vbench_repo:${PYTHONPATH:-}"
export VBENCH_CACHE_DIR="${VBENCH_CACHE_DIR:-/share_4/users/duy/.cache/vbench}"
/share_4/users/duy/.conda/envs/vbench_cu121_20260515/bin/python tools/evaluate_vbench_resume.py \
  --videos "${RUN_ROOT}/joint_v2/videos" \
  --full-info "${RUN_ROOT}/vbench_info.json" \
  --output "${RUN_ROOT}/smoke_scores" \
  --expected-samples-per-prompt 1 --start 6 --end 7
echo "GPU generation, resume and real VBench temporal-flickering smoke passed. Not a full benchmark score."
