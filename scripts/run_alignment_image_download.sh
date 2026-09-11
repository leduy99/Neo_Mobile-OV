#!/usr/bin/env bash
# Public alignment data: no caption generation, VAE, or DMD.
set -Eeuo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
PYTHON_BIN="${PYTHON_BIN:-/proj/cvl/users/x_fahkh2/envs/neo_mobileov/bin/python}"
if [[ "${ALIGNMENT_SCOPE:-images}" == "stage1" ]]; then
    DATASET_OUTPUT_DIR="${DATASET_OUTPUT_DIR:-download_data/data/univideo_stage1}"
else
    DATASET_OUTPUT_DIR="${DATASET_OUTPUT_DIR:-download_data/data/univideo_alignment_images}"
fi
export HF_HOME="${HF_HOME:-/proj/cvl/users/x_fahkh2/caches}"
export TMPDIR="${TMPDIR:-${HF_HOME}/tmp}"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-3600}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-60}"
export HF_XET_NUM_CONCURRENT_RANGE_GETS="${HF_XET_NUM_CONCURRENT_RANGE_GETS:-4}"
export TOKENIZERS_PARALLELISM=false
HEARTBEAT_PID=""
DOWNLOAD_PID=""
STOP_FILE=""

cleanup() {
    rc=$?
    trap - EXIT INT TERM
    if [[ -n "${DOWNLOAD_PID}" ]] && kill -0 "${DOWNLOAD_PID}" 2>/dev/null; then
        kill "${DOWNLOAD_PID}" 2>/dev/null || true
        wait "${DOWNLOAD_PID}" 2>/dev/null || true
    fi
    if [[ -n "${HEARTBEAT_PID}" ]]; then
        touch "${STOP_FILE}" 2>/dev/null || true
        for _ in {1..15}; do
            kill -0 "${HEARTBEAT_PID}" 2>/dev/null || break
            sleep 1
        done
        if kill -0 "${HEARTBEAT_PID}" 2>/dev/null; then
            kill "${HEARTBEAT_PID}" 2>/dev/null || true
        fi
        wait "${HEARTBEAT_PID}" 2>/dev/null || true
    fi
    [[ -z "${STOP_FILE}" ]] || rm -f -- "${STOP_FILE}"
    exit "${rc}"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'rc=$?; echo "ERROR: line=${LINENO} command=${BASH_COMMAND} exit=${rc}" >&2; exit "$rc"' ERR

if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Missing Python executable: ${PYTHON_BIN}" >&2
    exit 1
fi
if [[ "${DRY_RUN:-0}" != "1" && -z "${SLURM_JOB_ID:-}" ]]; then
    echo "Use sbatch/srun for the GPU heartbeat. Only DRY_RUN=1 may run without SLURM." >&2
    exit 1
fi
mkdir -p logs "${DATASET_OUTPUT_DIR}" "${HF_HOME}" "${TMPDIR}"
"${PYTHON_BIN}" -c 'import huggingface_hub, PIL; print("Dependencies: huggingface_hub=" + huggingface_hub.__version__ + " Pillow=" + PIL.__version__)'

ARGS=(
    --output-dir "${DATASET_OUTPUT_DIR}"
    --revision "${DATASET_REVISION:-main}"
    --num-shards "${NUM_SHARDS:-100}"
    --seed "${DATASET_SEED:-20260911}"
    --workers "${DOWNLOAD_WORKERS:-8}"
    --retries "${DOWNLOAD_RETRIES:-8}"
    --max-download-gib "${MAX_DOWNLOAD_GIB:-80}"
    --disk-margin-gib "${DISK_MARGIN_GIB:-10}"
)
TOOL=tools/data_prepare/download_alignment_images.py
if [[ "${ALIGNMENT_SCOPE:-images}" == "stage1" ]]; then
    "${PYTHON_BIN}" -c 'import av; print("Video decoder: PyAV=" + av.__version__)'
    TOOL=tools/data_prepare/prepare_stage1_alignment.py
    ARGS=(
        --image-root "${IMAGE_OUTPUT_DIR:-download_data/data/univideo_alignment_images}"
        --video-root "${VIDEO_OUTPUT_DIR:-download_data/data/univideo_alignment_videos}"
        --output-dir "${DATASET_OUTPUT_DIR}"
        --image-shards "${IMAGE_SHARDS:-100}" --video-shards "${VIDEO_SHARDS:-100}"
        --image-revision "${DATASET_REVISION:-main}" --video-revision "${VIDEO_REVISION:-main}"
        --seed "${DATASET_SEED:-20260911}" --workers "${DOWNLOAD_WORKERS:-8}"
        --retries "${DOWNLOAD_RETRIES:-8}"
        --max-image-gib "${MAX_IMAGE_GIB:-80}" --max-video-gib "${MAX_VIDEO_GIB:-450}"
        --disk-margin-gib "${DISK_MARGIN_GIB:-20}"
        --min-video-seconds "${MIN_VIDEO_SECONDS:-2}" --max-video-seconds "${MAX_VIDEO_SECONDS:-12}"
        --min-video-side "${MIN_VIDEO_SIDE:-256}" --min-video-frames "${MIN_VIDEO_FRAMES:-49}"
        --validation-fraction "${VALIDATION_FRACTION:-0.02}"
    )
    if [[ "${STAGE1_BUILD_ONLY:-0}" == "1" ]]; then
        ARGS+=(--skip-download)
    fi
    if [[ -n "${EXCLUDE_PROMPTS:-}" ]]; then
        ARGS+=(--exclude-prompts "${EXCLUDE_PROMPTS}")
    fi
elif [[ "${ALIGNMENT_SCOPE:-images}" != "images" ]]; then
    echo "Unknown ALIGNMENT_SCOPE: ${ALIGNMENT_SCOPE}" >&2
    exit 1
fi
if [[ "${DRY_RUN:-0}" == "1" ]]; then
    "${PYTHON_BIN}" "${TOOL}" "${ARGS[@]}" --dry-run
    exit 0
fi

echo "Alignment data job=${SLURM_JOB_ID} scope=${ALIGNMENT_SCOPE:-images} output=${DATASET_OUTPUT_DIR}"
echo "Counts are measured after indexing and filtering; shard counts are not exact sample quotas."
df -h "${DATASET_OUTPUT_DIR}"
STOP_FILE="$(mktemp "${TMPDIR%/}/alignment-data-${SLURM_JOB_ID}.XXXXXX")"
rm -f -- "${STOP_FILE}"
# A parent interactive srun can export a CPU mask outside this smaller step.
srun --overlap --nodes=1 --ntasks=1 --cpus-per-task=1 --cpu-bind=none --gpus-per-task=1 --gpu-bind=single:1 \
    "${PYTHON_BIN}" tools/utils/gpu_heartbeat.py \
    --devices all --interval "${HEARTBEAT_INTERVAL:-5}" \
    --tensor-mb 4 --work-seconds "${HEARTBEAT_WORK_SECONDS:-1}" \
    --stop-file "${STOP_FILE}" --label "alignment-data-${SLURM_JOB_ID}" &
HEARTBEAT_PID=$!
sleep 3
if ! kill -0 "${HEARTBEAT_PID}" 2>/dev/null; then
    echo "GPU heartbeat failed to start; refusing to download." >&2
    exit 1
fi
"${PYTHON_BIN}" "${TOOL}" "${ARGS[@]}" &
DOWNLOAD_PID=$!
while kill -0 "${DOWNLOAD_PID}" 2>/dev/null; do
    if ! kill -0 "${HEARTBEAT_PID}" 2>/dev/null; then
        echo "GPU heartbeat exited during download; stopping. Resubmit to resume." >&2
        exit 1
    fi
    sleep 5
done
if wait "${DOWNLOAD_PID}"; then
    DOWNLOAD_PID=""
else
    rc=$?
    DOWNLOAD_PID=""
    exit "${rc}"
fi
echo "Download and raw-data checks completed. No VAE latents have been encoded and no training was launched."
if [[ "${ALIGNMENT_SCOPE:-images}" == "stage1" ]]; then
    echo "Summary: ${DATASET_OUTPUT_DIR}/stage1_summary.json"
    echo "Tasks: train/{t2i,t2v,image_reconstruction}.jsonl and validation equivalents."
else
    echo "Summary: ${DATASET_OUTPUT_DIR}/download_summary.json"
    echo "Index: ${DATASET_OUTPUT_DIR}/samples.jsonl (archive paths relative to ${DATASET_OUTPUT_DIR})"
fi
