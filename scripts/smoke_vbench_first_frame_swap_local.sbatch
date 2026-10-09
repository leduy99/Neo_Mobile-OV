#!/usr/bin/env bash
# Re-create only two matched baseline prompts, since old local media was intentionally pruned.
#SBATCH --job-name=mov-swap-smoke
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=160G
#SBATCH --gres=gpu:2
#SBATCH --time=00:40:00
#SBATCH --output=logs/mov-swap-smoke-%j.out
#SBATCH --error=logs/mov-swap-smoke-%j.out
set -Eeuo pipefail
ROOT="${PROJECT_ROOT:-/share_4/users/duy/project/unified_video/New-Mobile-OV}"
cd "${ROOT}"
PYTHON="${PYTHON_BIN:-/share_4/users/duy/.conda/envs/neo_mobileov/bin/python}"
VBENCH_PYTHON="${VBENCH_PYTHON:-/share_4/users/duy/.conda/envs/vbench_cu121_20260515/bin/python}"
VBENCH_REPO="${ROOT}/checkpoints/vbench_repo"
OUT="${RUN_ROOT:-output/first_frame_swap_smoke/${SLURM_JOB_ID}}"
export PYTHONPATH="${ROOT}:${VBENCH_REPO}:${PYTHONPATH:-}" PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false
export HF_HOME="/share_4/users/duy/.cache/huggingface" TORCH_HOME="/share_4/users/duy/.cache/torch"
export VBENCH_CACHE_DIR="/share_4/users/duy/.cache/vbench" HF_HUB_DISABLE_XET=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ARGS=(--workers 2 --max-prompts 2 --samples-per-prompt 1 --seed 2026 --seed-mapping extendable
      --vbench-repo "${VBENCH_REPO}" --vbench-python "${VBENCH_PYTHON}")
"${PYTHON}" -u tools/benchmark_joint_v2_quicksr_ablation.py --output-dir "${OUT}/baseline" "${ARGS[@]}"
for CASE in joint_v2_ssd1b native_hybrid_dreamlite_v11; do
  "${PYTHON}" -u tools/benchmark_first_frame_swap.py --swap "${CASE}" --baseline-run "${OUT}/baseline" \
    --output-dir "${OUT}/${CASE}" "${ARGS[@]}"
  "${PYTHON}" -u tools/benchmark_first_frame_swap.py --swap "${CASE}" --baseline-run "${OUT}/baseline" \
    --output-dir "${OUT}/${CASE}" "${ARGS[@]}"
  if [[ "${CASE}" == joint_v2_ssd1b ]]; then SYSTEM=joint_v2; else SYSTEM=native_hybrid; fi
  for VARIANT in quicksr no_quicksr; do
    "${VBENCH_PYTHON}" tools/evaluate_vbench_resume.py \
      --videos "${OUT}/${CASE}/${VARIANT}/${SYSTEM}/videos" \
      --full-info "${OUT}/${CASE}/${VARIANT}/vbench_info.json" \
      --output "${OUT}/${CASE}/${VARIANT}/${SYSTEM}/smoke_scores" \
      --expected-samples-per-prompt 1 --start 6 --end 7
  done
done
echo "Real GPU anchor swap + both exports + resume + VBench flickering passed for both cross settings."
echo "Two-prompt functional test only; no full benchmark score claimed."
