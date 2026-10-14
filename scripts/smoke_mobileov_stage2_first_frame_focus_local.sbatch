#!/usr/bin/env bash
#SBATCH --job-name=mov-s2-ff-smoke
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=160G
#SBATCH --gres=gpu:2
#SBATCH --time=01:00:00
#SBATCH --output=logs/mov-s2-ff-smoke-%j.out
#SBATCH --error=logs/mov-s2-ff-smoke-%j.out
set -Eeuo pipefail
trap 'rc=$?; echo "ERROR: line=${LINENO} command=${BASH_COMMAND} exit=${rc}" >&2; exit "$rc"' ERR
cd "${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?}}"
export PYTHONPATH="${PWD}:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2
export HF_HOME="${HF_HOME:-/share_4/users/duy/.cache/huggingface}"
export TMPDIR="${TMPDIR:-/share_4/users/duy/tmp}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 TORCH_DIST_INIT_DEVICE_ID=1 TORCH_DIST_TIMEOUT_MINUTES=3
PYTHON="${PYTHON_BIN:-/share_4/users/duy/.conda/envs/neo_mobileov/bin/python}"
OUT="${OUT:-output/stage2_first_frame_focus_smoke/${SLURM_JOB_ID}}"
PARENT="checkpoints/hf_mobile_ov/mobileov_stage2_media3epochs/17641364/stage2_dit_connector_step283977.pt"
mkdir -p "${TMPDIR}"
"${PYTHON}" -s tools/prepare_stage2_media_smoke.py \
    --source output/alignment_data_checks/stage1_slurm_release --output "${OUT}/data"
"${PYTHON}" -s tools/plan_mobileov_stage2_first_frame.py \
    --data-root "${OUT}/data" --parent "${PARENT}" --world-size 2 --accumulation 2 \
    --min-train-videos 1 --integration-smoke --audit-samples 2 --output "${OUT}/epoch_plan.json"
COMMON=(--phase 2 --init-stage2 "${PARENT}" --expected-stage2-step 283977
        --expected-stage2-sha256 a74fd3069d17437b674415268c5157a85d94223509121220572e2d2fcdb68053
        --media-epochs 1 --media-start-epoch 3 --t2v-first-probability .5 --reconstruction-weight .2
        --accumulation 2 --data-root "${OUT}/data" --output-dir "${OUT}/train"
        --integration-smoke --workers 1 --lr 2e-5 --warmup 50 --log-every 1
        --validation-samples 1 --validate-every 5000 --save-every 10000 --archive-every 10000
        --verify-frozen --cuda-memory-limit-gib 60)
"${PYTHON}" -s -m torch.distributed.run --standalone --nproc_per_node=2 \
    tools/train_mobileov_stage1.py "${COMMON[@]}" --stop-after 283978
"${PYTHON}" -s -m torch.distributed.run --standalone --nproc_per_node=2 \
    tools/train_mobileov_stage1.py "${COMMON[@]}" --stop-after 283980 --resume "${OUT}/train/stage2_resume.pt"
"${PYTHON}" -s tools/infer_mobileov_stage1.py \
    --checkpoint "${OUT}/train/stage2_dit_connector_latest.pt" \
    --validation-task t2i --data-root "${OUT}/data" --output-dir "${OUT}/t2i"
"${PYTHON}" -s tools/infer_mobileov_stage1.py \
    --checkpoint "${OUT}/train/stage2_dit_connector_latest.pt" \
    --validation-task t2v --data-root "${OUT}/data" --output-dir "${OUT}/t2v"
echo "Actual two-GPU fork/pause/resume + T2I/T2V inference smoke completed: ${OUT}"
echo "This is a correctness test, NOT a quality-training checkpoint."
