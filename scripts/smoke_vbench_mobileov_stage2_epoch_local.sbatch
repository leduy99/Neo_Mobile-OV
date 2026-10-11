#!/usr/bin/env bash
# Real GPU functional test on the locally available 130K checkpoint, not epoch-1/2 results.
#SBATCH --job-name=mov-epdual-smoke
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=00:40:00
#SBATCH --output=logs/mov-epdual-smoke-%j.out
#SBATCH --error=logs/mov-epdual-smoke-%j.out
set -Eeuo pipefail
ROOT="${PROJECT_ROOT:-/share_4/users/duy/project/unified_video/New-Mobile-OV}"
cd "${ROOT}"
PYTHON="${PYTHON_BIN:-/share_4/users/duy/.conda/envs/neo_mobileov/bin/python}"
VBENCH_PYTHON="${VBENCH_PYTHON:-/share_4/users/duy/.conda/envs/vbench_cu121_20260515/bin/python}"
VBENCH_REPO="${ROOT}/checkpoints/vbench_repo"
OUT="${RUN_ROOT:-output/stage2_epoch_dual_smoke/${SLURM_JOB_ID}}"
CHECKPOINT="checkpoints/hf_mobile_ov/mobileov_stage2_media3epochs/17641364/stage2_dit_connector_latest.pt"
export PYTHONPATH="${ROOT}:${VBENCH_REPO}:${PYTHONPATH:-}" PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false
export HF_HOME="/share_4/users/duy/.cache/huggingface" TORCH_HOME="/share_4/users/duy/.cache/torch"
export VBENCH_CACHE_DIR="/share_4/users/duy/.cache/vbench" HF_HUB_DISABLE_XET=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ARGS=(--checkpoint "${CHECKPOINT}" --expected-step 130000 --epoch 1 --output-dir "${OUT}" --workers 1
      --samples-per-prompt 1 --max-prompts 2 --vbench-repo "${VBENCH_REPO}" --vbench-python "${VBENCH_PYTHON}")
"${PYTHON}" -u tools/benchmark_mobileov_stage2_epoch_dual.py "${ARGS[@]}"
"${PYTHON}" -u tools/benchmark_mobileov_stage2_epoch_dual.py "${ARGS[@]}"
for VARIANT in quicksr no_quicksr; do
  "${VBENCH_PYTHON}" tools/evaluate_vbench_resume.py --videos "${OUT}/${VARIANT}/videos" \
    --full-info "${OUT}/vbench_info.json" --output "${OUT}/${VARIANT}/smoke_scores" \
    --expected-samples-per-prompt 1 --start 6 --end 7
done
echo "Pinned intermediate checkpoint + real GPU generation + raw/QuickSR + resume + metric checks passed."
echo "130K functional smoke ONLY; 100K/190K weights are on Berzelius and were not evaluated here."
