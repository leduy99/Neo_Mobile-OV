#!/usr/bin/env bash
# Functional test on the available 130K intermediate checkpoint, never a final-model benchmark.
#SBATCH --job-name=mov-s2dual-smoke
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=00:40:00
#SBATCH --output=logs/mov-s2dual-smoke-%j.out
#SBATCH --error=logs/mov-s2dual-smoke-%j.out

set -Eeuo pipefail
ROOT="${PROJECT_ROOT:-/share_4/users/duy/project/unified_video/New-Mobile-OV}"
cd "${ROOT}"
PYTHON="${PYTHON_BIN:-/share_4/users/duy/.conda/envs/neo_mobileov/bin/python}"
VBENCH_PYTHON="${VBENCH_PYTHON:-/share_4/users/duy/.conda/envs/vbench_cu121_20260515/bin/python}"
VBENCH_REPO="${VBENCH_REPO:-${ROOT}/checkpoints/vbench_repo}"
OUT="${RUN_ROOT:-output/stage2_media3epochs_dual_smoke/${SLURM_JOB_ID}}"
CHECKPOINT="${CHECKPOINT:-checkpoints/hf_mobile_ov/mobileov_stage2_media3epochs/17641364/stage2_dit_connector_latest.pt}"
export PYTHONPATH="${ROOT}:${VBENCH_REPO}:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-/share_4/users/duy/.cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-/share_4/users/duy/.cache/torch}"
export VBENCH_CACHE_DIR="${VBENCH_CACHE_DIR:-/share_4/users/duy/.cache/vbench}"
export PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_XET=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ARGS=(--checkpoint "${CHECKPOINT}" --output-dir "${OUT}" --workers 1 --samples-per-prompt 1
      --max-prompts 2 --allow-incomplete-checkpoint --vbench-repo "${VBENCH_REPO}" --vbench-python "${VBENCH_PYTHON}")
"${PYTHON}" -u tools/benchmark_mobileov_stage2_media3epochs_dual.py "${ARGS[@]}"
"${PYTHON}" -u tools/benchmark_mobileov_stage2_media3epochs_dual.py "${ARGS[@]}"
for variant in quicksr no_quicksr; do
  "${VBENCH_PYTHON}" tools/evaluate_vbench_resume.py \
    --videos "${OUT}/${variant}/videos" --full-info "${OUT}/vbench_info.json" \
    --output "${OUT}/${variant}/smoke_scores" --expected-samples-per-prompt 1 --start 6 --end 7
done
echo "Stage-2 real GPU generation, raw/QuickSR exports, resume and both VBench flickering checks passed."
echo "Intermediate checkpoint functional test ONLY; final model and full score were not evaluated."
