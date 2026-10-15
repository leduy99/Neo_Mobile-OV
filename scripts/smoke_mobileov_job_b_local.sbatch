#!/usr/bin/env bash
# Actual 1-GPU NeoDragon + LoRA pause/resume correctness test, not quality training.
#SBATCH --job-name=mov-jobb-smoke
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --output=logs/mov-jobb-smoke-%j.out
#SBATCH --error=logs/mov-jobb-smoke-%j.out
set -Eeuo pipefail
trap 'rc=$?; echo "ERROR: line=${LINENO} command=${BASH_COMMAND} exit=${rc}" >&2; exit "$rc"' ERR
cd "${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?}}"
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2
export HF_HOME="${HF_HOME:-/share_4/users/duy/.cache/huggingface}"
export TMPDIR="${TMPDIR:-/share_4/users/duy/tmp}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PYTHON="${PYTHON_BIN:-/share_4/users/duy/.conda/envs/neo_mobileov/bin/python}"
OUT="${OUT:-output/job_b_gpu_smoke/${SLURM_JOB_ID}}"
PARENT="checkpoints/hf_mobile_ov/mobileov_stage2_media3epochs/17641364/stage2_dit_connector_step283977.pt"
mkdir -p "${TMPDIR}"
"${PYTHON}" -s tools/prepare_mobileov_job_b_smoke.py \
    --source output/alignment_data_checks/stage1_slurm_release --output "${OUT}"
COMMON=(--phase 2 --init-stage2 "${PARENT}" --expected-stage2-step 283977
        --expected-stage2-sha256 a74fd3069d17437b674415268c5157a85d94223509121220572e2d2fcdb68053
        --media-epochs 1 --media-start-epoch 3 --t2v-first-probability .5 --reconstruction-weight .2
        --accumulation 2 --data-root "${OUT}/data" --qa-root "${OUT}/qa" --output-dir "${OUT}/train"
        --integration-smoke --workers 0 --lr 2e-5 --warmup 50 --log-every 1 --qa-every 1
        --validation-samples 1 --validate-every 5000 --save-every 10000 --archive-every 10000)
"${PYTHON}" -s tools/train_mobileov_job_b.py "${COMMON[@]}" --stop-after 283978
"${PYTHON}" -s tools/train_mobileov_job_b.py "${COMMON[@]}" --stop-after 283979 \
    --resume "${OUT}/train/job_b_resume.pt"
echo "Actual Job-B 1-GPU fork/gradient/checkpoint/resume test completed: ${OUT}"
echo "This checkpoint is integration-only, not quality training."
