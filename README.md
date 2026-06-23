# New Mobile-OV

Self-contained research repo for the next Mobile-OV text-to-video generation branch.

The repo keeps the existing Mobile-OV understanding branch and Bridge v2 text side, but separates generation into three options:

1. `mobile_ov_current`: reference option matching the current SmolVLM2 + Bridge v2 + SANA-video direction.
2. `mobile_o_sana_0_5b`: lightweight anchor-generator option based on Mobile-O 0.5B / SANA-image components.
3. `mobile_ov_neodragon`: Neodragon video generation branch while keeping Mobile-OV understanding + Bridge v2 intact.

The new experiment is not full T2V first. It starts with the feasibility test:

```text
WanVAE(video) -> Z_gt
z0_gt = Z_gt[:, :, 0]
SmolVLM2 + Bridge v2(prompt) -> prompt tokens -> pooled text condition
LatentMotionWeaver(z0_gt, text) -> Z_pred
compare Z_pred against Z_copy = repeat(z0_gt)
```

## Repository Layout

```text
new_mobile_ov/
  bridge/        SmolVLM2 + Bridge v2 wrapper
  generation/    backend interfaces for current Mobile-OV, Mobile-O/SANA-0.5B, and Neodragon
  motion/        Latent Motion Weaver
  training/      latent dataset and losses
  smolvlm2/      vendored SmolVLM2 loader/model code
third_party/
  sana/          vendored SANA source used by the current Mobile-OV branch
  mobileo/       vendored Mobile-O source used by option 2
configs/
  mobile_ov_current.yaml
  mobile_o_sana_0_5b.yaml
  mobile_ov_neodragon.yaml
```

Weights are intentionally not committed to git, but the repo can prepare the required runtime assets under
`./checkpoints/` automatically when they are missing.

For implementation status, distributed training notes, and today's FSDP/NCCL fixes, see
[`Docs/DEVELOPMENT_NOTE.md`](Docs/DEVELOPMENT_NOTE.md).

## Environment

The main environment for this repo is:

```bash
conda activate /share_4/users/duy/.conda/envs/neo_mobileov
```

To recreate it from scratch:

```bash
conda env create -f environment/neo_mobileov.yaml
conda activate neo_mobileov
```

All SLURM scripts default to `/share_4/users/duy/.conda/envs/neo_mobileov`, but you can override it with `CONDA_ENV=/path/to/env`.

## Checkpoints And Auto-Download

The default checkpoint layout is:

```text
checkpoints/
  smolvlm2_500m/
    smolvlm2_500m.pt          # converted SmolVLM2 checkpoint
  neodragon_repo/             # cloned Neodragon source repo
  neodragon/                  # Hugging Face cache for karnewar/Neodragon
```

The bridge loader automatically prepares `checkpoints/smolvlm2_500m/smolvlm2_500m.pt` if it is missing:

1. If `MOBILEOV_SMOLVLM2_CKPT_SOURCE` is set, copy/download that converted `.pt` file.
2. Else if `MOBILEOV_SMOLVLM2_CKPT_HF_REPO` and `MOBILEOV_SMOLVLM2_CKPT_HF_FILE` are set, download that converted `.pt` file from Hugging Face.
3. Else convert raw `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` weights into the local `.pt` checkpoint.

Neodragon assets are also prepared automatically for the Neodragon branch:

```text
repo_url: https://github.com/Qualcomm-AI-research/neodragon.git
model_id: karnewar/Neodragon
```

You can prepare everything explicitly before training:

```bash
python tools/prepare_checkpoints.py --config configs/mobile_ov_neodragon.yaml
```

Multi-GPU/multi-node runs use file locks during asset preparation, so only one rank downloads/converts while the other ranks wait.

## Generation Options

### Option 1: Current Mobile-OV

Use when we want parity with the current 135k checkpoint and SANA-video reference branch. This is mostly a teacher/reference backend while we develop the anchor + motion weaver branch.

```yaml
backend:
  name: mobile_ov_current
  checkpoint_path: /path/to/mobile_ov_135k.pt
```

### Option 2: Mobile-O / SANA-0.5B Anchor Branch

Use when we want a lighter image/anchor generation branch. Mobile-O source is vendored under `third_party/mobileo`; its checkpoint path is configured separately:

```yaml
backend:
  name: mobile_o_sana_0_5b
  model_path: /path/to/Mobile-O-0.5B
```

Important: Mobile-O image latents and SANA-video/WanVAE video latents may not live in the same latent space. The first safe target is therefore `z0_gt` from WanVAE video latents. If Mobile-O anchor latents mismatch, add an adapter or train a direct anchor head into WanVAE `z0_gt`.

### Option 3: Mobile-OV + Neodragon

Use when we want to replace the SANA-video generation branch with Neodragon while preserving the Mobile-OV understanding branch and Bridge v2.

```yaml
backend:
  name: mobile_ov_neodragon
  extra:
    repo_path: /path/to/neodragon
    cache_dir: /path/to/neodragon/models
    model_id: karnewar/Neodragon
    mode: hybrid
```

The current smoke inference runs both pieces deliberately:

```text
prompt -> SmolVLM2 + Bridge v2 -> bridge tokens [B, 128, 1536]   # verified/logged
prompt -> Neodragon native text stack -> Neodragon video          # current video output
```

To make Neodragon fully bridge-conditioned, this repo uses a Neodragon-shaped bridge output:

```text
SmolVLM2 + Bridge v2
  -> prompt_embeds [B, 128, 1536]
  -> pooled_prompt_embeds [B, 2048]
  -> Neodragon DiT
```

The bridge is trained by distilling frozen Neodragon `TextEncoderBundle + ContextAdapter` token conditions and CLIP pooled projections from the same prompts. This lets us keep the understanding branch unchanged while removing Neodragon's text-conditioning stack from the new model path.

Smoke inference:

```bash
sbatch scripts/infer_mobile_ov_neodragon_smoke.sbatch
```

Smoke bridge distillation:

```bash
sbatch scripts/train_neodragon_text_bridge_smoke.sbatch
```

Smoke bridge-conditioned inference:

```bash
CONDITION_SOURCE=bridge \
BRIDGE_CKPT=output/neodragon_text_bridge_smoke/bridge/neodragon_text_bridge_latest.pt \
sbatch scripts/infer_mobile_ov_neodragon_smoke.sbatch
```

Smoke Neodragon DiT training with bridge condition:

```bash
sbatch scripts/train_neodragon_dit_bridge_smoke.sbatch
```

## First Feasibility Test

Prepare a manifest of cached WanVAE latents:

```csv
latent_path,prompt
sample_000000.pkl,A dog runs through grass.
```

Each pickle must contain:

```python
{
  "latent_feature": Tensor[16, 21, H, W],
  "prompt": "...",
}
```

Then run a shape smoke:

```bash
python tools/smoke_shapes.py --config configs/mobile_ov_current.yaml --device cpu
```

Train LMW on cached latents:

```bash
python tools/train_lmw_projection.py --config configs/mobile_ov_current.yaml
```

For fast debugging without loading SmolVLM2:

```bash
python tools/train_lmw_projection.py --config configs/mobile_ov_current.yaml --random-text
```

## Loss

The anchor slice is fixed and not penalized:

```text
L_lat    = L1(Z_pred[:, :, 1:] - Z_gt[:, :, 1:])
L_motion = L1(diff_t(Z_pred) - diff_t(Z_gt))
L        = L_lat + 0.5 * L_motion
```

The required baseline is:

```text
Z_copy = repeat(z0_gt, T)
```

LMW only passes if it beats `Z_copy` in latent metrics and decoded videos show more motion than copy baseline.
