#!/usr/bin/env bash
# Same VBench protocol as Stage 2 100k; only the checkpoint and output directory differ.
#SBATCH --job-name=mov-vb-s2rec
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --time=72:00:00
#SBATCH --account=berzelius-2026-196
#SBATCH --output=logs/mov-vb-s2rec-%j.out
#SBATCH --error=logs/mov-vb-s2rec-%j.out

set -Eeuo pipefail
trap 'rc=$?; echo "ERROR: line=${LINENO} command=${BASH_COMMAND} exit=${rc}" >&2; exit "$rc"' ERR
cd "${PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?Submit from the Neo_Mobile-OV repo root}}"
export PROJECT_ROOT="${PWD}"
export HF_REPO=Amshaker/Mobile-OV
export HF_FILE=mobileov_stage2_reconstruction150k/17558065/stage2_dit_connector_latest.pt
export HF_REVISION=dde3f1b64e3ca35f066b100469407d146ca44c99
export EXPECTED_STEP=150000
export EXPECTED_SHA256=2d53da4aa25285eeb2e4c05fd019398396f8434121248bbff028254abe7fd263
export SAMPLES_PER_PROMPT="${SAMPLES_PER_PROMPT:-1}"
export RUN_ROOT="${RUN_ROOT:-output/vbench_mobileov_stage2_reconstruction150k_${SAMPLES_PER_PROMPT}x}"
echo "Reconstruction 150k benchmark: SmolVLM2 + trained connector/monolithic DiT; all frames generated."
echo "No DreamLite/SSD1B, no native text conditioner, no reconstruction image provided at inference."
bash scripts/vbench_mobileov_stage2_100k_1node1gpu.sbatch
