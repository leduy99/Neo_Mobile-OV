#!/usr/bin/env bash
# Local H200 smoke for the fixed-bridge production recipe.
#SBATCH --job-name=neo-pfm-cfg-smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=24
#SBATCH --mem=240G
#SBATCH --time=01:00:00
#SBATCH --output=logs/neo-pfm-cfg-smoke-%j.out
#SBATCH --error=logs/neo-pfm-cfg-smoke-%j.out

set -euo pipefail

ROOT="/share_4/users/duy/project/unified_video/New-Mobile-OV"
PYTHON_BIN="${PYTHON_BIN:-/share_4/users/duy/.conda/envs/neo_mobileov/bin/python}"
TORCHRUN_BIN="${TORCHRUN_BIN:-/share_4/users/duy/.conda/envs/neo_mobileov/bin/torchrun}"
MANIFEST="${MANIFEST:-data/neodragon_flow_ablation_latents/latent_manifest.csv}"
BRIDGE_CKPT="${BRIDGE_CKPT:-checkpoints/hf_mobile_ov/neo_monolithic_cfg_bridge_v2/monolithic_cfg_bridge_v2_17327720/neodragon_text_bridge_latest.pt}"
INIT_JOINT_CKPT="${INIT_JOINT_CKPT:-checkpoints/hf_mobile_ov/mobileov_monolithic_pyramid_recipe/mobileov_monolithic_pyramid_17344858/phase1/neodragon_dit_bridge_step000300.pt}"
OUT="${OUT:-output/mobileov_monolithic_pyramid_cfg_dit/smoke_${SLURM_JOB_ID}}"

cd "${ROOT}"
test -x "${PYTHON_BIN}"
test -x "${TORCHRUN_BIN}"
test -s "${MANIFEST}"
test -s "${BRIDGE_CKPT}"
test -s "${INIT_JOINT_CKPT}"
mkdir -p logs "${OUT}"

export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export FSDP_USE_ORIG_PARAMS=true
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export HF_HOME="${HF_HOME:-${ROOT}/checkpoints/hf_cache}"
export TORCH_HOME="${TORCH_HOME:-/share_4/users/duy/.cache/torch}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/share_4/users/duy/.cache/triton}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"

"${TORCHRUN_BIN}" --standalone --nnodes=1 --nproc_per_node=2 \
  tools/train_neodragon_dit_bridge.py \
  --target-stack multistep \
  --manifest "${MANIFEST}" \
  --bridge-ckpt "${BRIDGE_CKPT}" \
  --init-joint-ckpt "${INIT_JOINT_CKPT}" \
  --output-dir "${OUT}" \
  --steps "${STEPS:-2}" \
  --batch-size "${BATCH_SIZE:-2}" \
  --max-samples "${MAX_SAMPLES:-8}" \
  --use-precomputed-latents \
  --train-last-n-blocks 0 \
  --parallel fsdp \
  --dit-trainable-fp32 \
  --no-train-bridge \
  --flow-contract pyramid \
  --stage-sampling-ratio 1,1,1 \
  --history-corrupt-max 0.3333333333333333 \
  --flow-weight 1.0 \
  --distill-weight 1.0 \
  --distill-cos-weight 0.1 \
  --cfg-distill-weight 0.5 \
  --cfg-distill-cos-weight 0.1 \
  --cfg-every 1 \
  --cfg-scale 5.0 \
  --preservation-weight 0.5 \
  --preservation-cos-weight 0.05 \
  --preservation-every 1 \
  --lr 3e-6 \
  --dit-warmup-steps 2 \
  --clip-grad-norm 1.0 \
  --diagnostic-every 1 \
  --log-every 1 \
  --save-every "${STEPS:-2}" \
  --archive-every 0

"${PYTHON_BIN}" - "${OUT}/neodragon_dit_bridge_latest.pt" <<'PY'
import sys
import torch

payload = torch.load(sys.argv[1], map_location="cpu", mmap=True, weights_only=False)
assert payload["objective"]["flow_contract"] == "pyramid"
assert payload["objective"]["joint_bridge_training"] is False
assert payload["objective"]["cfg_cross_distillation"] is True
assert payload["objective"]["teacher_condition_preservation"] is True
assert payload["args"]["cfg_every"] == 1
assert payload["args"]["cfg_scale"] == 5.0
assert all(row["bridge_active"] == 0.0 for row in payload["history"])
assert all(row["bridge_grad_norm"] == 0.0 for row in payload["history"])
assert any(row["cfg_distill_loss"] > 0.0 for row in payload["history"])
print("Validated fixed bridge, active PFM/CFG distillation, and native preservation.")
PY
