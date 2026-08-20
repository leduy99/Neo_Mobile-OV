#!/usr/bin/env bash
# Two-phase, low-cost validation of the corrected Mobile-OV/NeoDragon recipe.
# Phase 1 adapts the DiT while freezing the bridge; phase 2 opens the bridge
# only after the model has recovered on the correct Pyramidal-Flow contract.
#SBATCH --job-name=neo-pfm-joint
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=24
#SBATCH --mem=240G
#SBATCH --time=08:00:00
#SBATCH --output=logs/neo-pfm-joint-%j.out
#SBATCH --error=logs/neo-pfm-joint-%j.out

set -euo pipefail

ROOT="/share_4/users/duy/project/unified_video/New-Mobile-OV"
PYTHON_BIN="${PYTHON_BIN:-/share_4/users/duy/.conda/envs/neo_mobileov/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-/share_4/users/duy/.conda/envs/neo_mobileov/bin/torchrun}"
MANIFEST="${MANIFEST:-data/neodragon_flow_ablation_latents/latent_manifest.csv}"
BRIDGE_CKPT="${BRIDGE_CKPT:-checkpoints/hf_mobile_ov/neo_monolithic_cfg_bridge_v2/monolithic_cfg_bridge_v2_17327720/neodragon_text_bridge_latest.pt}"
RUN_NAME="${RUN_NAME:-monolithic_pyramid_pilot_${SLURM_JOB_ID}}"
OUT="${OUT:-output/mobileov_monolithic_pyramid_recipe/${RUN_NAME}}"
MAX_SAMPLES="${MAX_SAMPLES:-100}"
PHASE1_STEPS="${PHASE1_STEPS:-300}"
PHASE2_STEPS="${PHASE2_STEPS:-700}"
BATCH_SIZE="${BATCH_SIZE:-2}"

cd "${ROOT}"
mkdir -p logs "${OUT}/phase1" "${OUT}/phase2"
test -s "${MANIFEST}"
test -s "${BRIDGE_CKPT}"
test -x "${PYTHON_BIN}"
test -x "${TORCHRUN_BIN}"

export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-${ROOT}/checkpoints/hf_cache}"
export TOKENIZERS_PARALLELISM=false
export FSDP_USE_ORIG_PARAMS=true
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"

COMMON_ARGS=(
  --target-stack multistep
  --manifest "${MANIFEST}"
  --bridge-ckpt "${BRIDGE_CKPT}"
  --batch-size "${BATCH_SIZE}"
  --max-samples "${MAX_SAMPLES}"
  --use-precomputed-latents
  --train-last-n-blocks 0
  --parallel fsdp
  --dit-trainable-fp32
  --flow-contract pyramid
  --stage-sampling-ratio 1,1,1
  --history-corrupt-max 0.3333333333333333
  --flow-weight 1.0
  --distill-weight 1.0
  --distill-cos-weight 0.1
  --preservation-weight 0.5
  --preservation-cos-weight 0.05
  --preservation-every 1
  --dit-warmup-steps 50
  --clip-grad-norm 1.0
  --log-every 10
)

echo "Phase 1: correct Pyramidal Flow + teacher preservation, bridge frozen"
"${TORCHRUN_BIN}" --standalone --nproc_per_node=2 \
  tools/train_neodragon_dit_bridge.py \
  "${COMMON_ARGS[@]}" \
  --output-dir "${OUT}/phase1" \
  --steps "${PHASE1_STEPS}" \
  --lr "${PHASE1_LR:-3e-5}" \
  --no-train-bridge \
  --save-every "${PHASE1_STEPS}" \
  --archive-every "${PHASE1_STEPS}"

PHASE1_CKPT="${OUT}/phase1/neodragon_dit_bridge_latest.pt"
test -s "${PHASE1_CKPT}"

echo "Phase 2: protected joint bridge + DiT adaptation"
"${TORCHRUN_BIN}" --standalone --nproc_per_node=2 \
  tools/train_neodragon_dit_bridge.py \
  "${COMMON_ARGS[@]}" \
  --init-joint-ckpt "${PHASE1_CKPT}" \
  --output-dir "${OUT}/phase2" \
  --steps "${PHASE2_STEPS}" \
  --lr "${PHASE2_DIT_LR:-3e-6}" \
  --bridge-lr "${PHASE2_BRIDGE_LR:-1e-6}" \
  --train-bridge \
  --bridge-trainable-fp32 \
  --bridge-start-step 0 \
  --bridge-warmup-steps 100 \
  --bridge-repr-weight 0.25 \
  --bridge-functional-weight 0.5 \
  --bridge-functional-cos-weight 0.05 \
  --bridge-functional-every 1 \
  --save-every "${PHASE2_STEPS}" \
  --archive-every "${PHASE2_STEPS}"

find "${OUT}" -maxdepth 2 -type f -print -exec ls -lh {} \;
