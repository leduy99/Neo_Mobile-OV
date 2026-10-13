#!/usr/bin/env bash
# Real GPU regression: shared SSD1B, both routes, dual export, cache resume, one metric.
#SBATCH --job-name=mov-s2-shared-test
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --output=logs/mov-s2-shared-test-%j.out
#SBATCH --error=logs/mov-s2-shared-test-%j.out
set -Eeuo pipefail
cd "${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the project root}}"
export PYTHONPATH="${PWD}:${PWD}/checkpoints/vbench_repo:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export HF_HOME="${HF_HOME:-/share_4/users/duy/.cache/huggingface}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TMPDIR="${TMPDIR:-/share_4/users/duy/tmp}"
mkdir -p "${TMPDIR}"
PYTHON="${PYTHON_BIN:-/share_4/users/duy/.conda/envs/neo_mobileov/bin/python}"
VBENCH_PYTHON="${VBENCH_PYTHON:-/share_4/users/duy/.conda/envs/vbench_cu121_20260515/bin/python}"
CHECKPOINT="${CHECKPOINT:-checkpoints/hf_mobile_ov/mobileov_stage2_media3epochs/17641364/stage2_dit_connector_step283977.pt}"
OUT="${OUT:-output/stage2_ssd1b_conditioning_smoke/${SLURM_JOB_ID}}"
COMMON=(--checkpoint "${CHECKPOINT}"
  --expected-sha256 a74fd3069d17437b674415268c5157a85d94223509121220572e2d2fcdb68053
  --workers 1 --max-prompts 2 --samples-per-prompt 1 --seed 2026)
"${PYTHON}" -u tools/prepare_vbench_stage2_ssd1b_frames.py "${COMMON[@]}" --output-dir "${OUT}/shared_ssd1b"
"${PYTHON}" -u tools/prepare_vbench_stage2_ssd1b_frames.py "${COMMON[@]}" --output-dir "${OUT}/shared_ssd1b"
for MODE in ssd1b_anchor image_condition; do
  "${PYTHON}" -u tools/benchmark_stage2_ssd1b_conditioning_dual.py "${COMMON[@]}" \
    --mode "${MODE}" --frames-root "${OUT}/shared_ssd1b" --output-dir "${OUT}/${MODE}" --vbench-python "${VBENCH_PYTHON}"
  "${PYTHON}" -u tools/benchmark_stage2_ssd1b_conditioning_dual.py "${COMMON[@]}" \
    --mode "${MODE}" --frames-root "${OUT}/shared_ssd1b" --output-dir "${OUT}/${MODE}" --vbench-python "${VBENCH_PYTHON}"
  for VARIANT in no_quicksr quicksr; do
    "${VBENCH_PYTHON}" -u tools/evaluate_vbench_resume.py --videos "${OUT}/${MODE}/${VARIANT}/videos" \
      --full-info "${OUT}/${MODE}/vbench_info.json" --output "${OUT}/${MODE}/${VARIANT}/smoke_scores" \
      --expected-samples-per-prompt 1 --start 6 --end 7
  done
done
echo "Completed real GPU smoke: ${OUT}; subset only, not a full VBench score."
