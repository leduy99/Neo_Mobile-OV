#!/usr/bin/env bash
# GPU allocation requested by the project; safe handles CPU/network-heavy work.
# The heartbeat reports CUDA status, not artificial GPU load.
#SBATCH --job-name=mov-jobb-data-gpu
#SBATCH --partition=berzelius
#SBATCH --reservation=safe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --account=berzelius-2026-196
#SBATCH --output=logs/mov-jobb-data-%j.out
#SBATCH --error=logs/mov-jobb-data-%j.out

set -Eeuo pipefail
cd "${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the Neo_Mobile-OV repository root}}"
if [[ -z "${SLURM_JOB_ID:-}" || "${SLURM_JOB_RESERVATION:-}" != safe ]]; then
    echo "Submit this GPU download job with --reservation=safe; status heartbeat does not prevent idle-GPU cancellation." >&2
    exit 1
fi
PYTHON="${PYTHON:-/proj/cvl/users/x_fahkh2/envs/neo_mobileov/bin/python}"
export PYTHON
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export DOWNLOAD_WORKERS="${DOWNLOAD_WORKERS:-8}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-60}"
test -x "${PYTHON}"
mkdir -p logs
STATE="$(mktemp -d "${TMPDIR:-/tmp}/mov-jobb-heartbeat.XXXXXX")"
READY="${STATE}/ready"
HEARTBEAT_PID=""
DOWNLOAD_PID=""

cleanup() {
    rc=$?
    trap - EXIT INT TERM
    for pid in "${DOWNLOAD_PID}" "${HEARTBEAT_PID}"; do
        if [[ -n "${pid}" ]]; then
            kill -TERM "${pid}" 2>/dev/null || true
            wait "${pid}" 2>/dev/null || true
        fi
    done
    rm -f "${READY}" "${STATE}/ready.tmp"
    rmdir "${STATE}" 2>/dev/null || true
    exit "${rc}"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'rc=$?; echo "ERROR: line=${LINENO} command=${BASH_COMMAND} exit=${rc}" >&2; exit "$rc"' ERR

echo "Job B: 1 allocated GPU, reservation=safe, DOWNLOAD_WORKERS=${DOWNLOAD_WORKERS}."
echo "CUDA heartbeat reports health only; it does not generate dummy GPU work."
"${PYTHON}" -s -u tools/utils/gpu_status_heartbeat.py \
    --interval "${HEARTBEAT_INTERVAL:-30}" --ready-file "${READY}" \
    --label "mov-jobb-data-${SLURM_JOB_ID}" &
HEARTBEAT_PID=$!
deadline=$((SECONDS + 120))
while [[ ! -f "${READY}" ]]; do
    if ! kill -0 "${HEARTBEAT_PID}" 2>/dev/null; then
        wait "${HEARTBEAT_PID}" || true
        echo "CUDA heartbeat failed to initialize; download not started." >&2
        exit 1
    fi
    if (( SECONDS >= deadline )); then
        echo "Timed out waiting for CUDA heartbeat; download not started." >&2
        exit 1
    fi
    sleep 0.2
done

# Execute the existing downloader in this allocation; do not submit another job.
bash scripts/download_mobileov_job_b_data_1nodecpu.sbatch &
DOWNLOAD_PID=$!
while kill -0 "${DOWNLOAD_PID}" 2>/dev/null; do
    if ! kill -0 "${HEARTBEAT_PID}" 2>/dev/null; then
        echo "CUDA heartbeat exited unexpectedly; stopping download (resubmit to resume)." >&2
        exit 1
    fi
    sleep 1
done
wait "${DOWNLOAD_PID}"
DOWNLOAD_PID=""
echo "Job-B source download finished; training has not been launched."
