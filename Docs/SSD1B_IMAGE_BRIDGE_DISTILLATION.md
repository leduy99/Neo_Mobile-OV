# SSD1B Image Bridge Distillation

## Objective

The Image Bridge replaces both native SSD1B CLIP text encoders at deployment
time while preserving the exact condition tensors expected by the frozen
SSD1B UNet. The native CLIP models remain frozen teachers during training.

```text
One frozen SmolVLM2 text forward
        |
        v
last-four semantic layers + hidden-0 lexical residual
        |
        v
mask-aware 77-token + 1-global-query resampler
        |
        +--> CLIP-L head     [B, 77, 768]
        +--> CLIP-bigG head  [B, 77, 1280]
        |                    concatenate -> [B, 77, 2048]
        +--> pooled head     [B, 1280]
                                  |
                                  v
                            frozen SSD1B UNet
```

The Video Bridge is unchanged. The final dual-bridge model will therefore
have three token heads:

1. Image CLIP-L token head.
2. Image CLIP-bigG token head.
3. Existing NeoDragon video token head.

The image and video branches also retain their own pooled heads.

## Why Two Image Token Heads

SSD1B is an SDXL-style model. Its native prompt tensor is not produced by one
2048-dimensional encoder. It concatenates the penultimate hidden states of:

- CLIP-L: `[B, 77, 768]`
- CLIP-bigG: `[B, 77, 1280]`

The final pooled condition `[B, 1280]` comes from CLIP-bigG. Reproducing these
three native teacher outputs separately gives a well-defined distillation
target and avoids asking one unconstrained 2048-dimensional projection to
discover the two-stream decomposition implicitly.

The resampler uses 77 learned token queries because the frozen SSD1B UNet was
trained with a fixed 77-token SDXL condition. A separate global query produces
the pooled condition. Source attention uses the full 512-token SmolVLM2 window
and its attention mask, so information is compressed only after semantic and
lexical features are available.

## Losses

Representation alignment is always active:

```text
L_repr =
    0.25 * L_CLIP-L
  + 0.25 * L_CLIP-bigG
  + 1.00 * L_pooled
  + 0.50 * L_geometry
  + 0.10 * L_norm
```

Each token-stream loss combines normalized token MSE and cosine distance.
The pooled loss combines raw MSE and cosine distance. Norm alignment preserves
teacher magnitude. Geometry alignment preserves pairwise prompt similarities
over the global distributed batch, preventing a low-variance bridge from
reducing pointwise loss through semantic collapse.

From step 5,001, the trainer also compares the response of the frozen SSD1B
UNet under native and bridge conditions:

```text
same latent + same timestep + same SDXL time IDs
  -> frozen UNet(native CLIP condition)  = teacher response
  -> frozen UNet(Image Bridge condition) = student response
```

The functional weight ramps to full strength over 5,000 steps.

From step 10,001, every eighth optimization step uses the native four-call LCM
rollout:

```text
timesteps = [999, 749, 499, 249]
```

The Image Bridge runs once and its condition is reused by all four UNet calls.
The student latent trajectory is never detached, so losses at later calls
backpropagate through earlier scheduler updates. At each call, the frozen
teacher is evaluated at the detached student state. Teacher and student
scheduler transitions receive identical random noise.

## Lessons Carried Over From The Video Bridge

- A low embedding loss is not sufficient evidence of good generation.
- A one-call functional match does not fully supervise a sequential sampler.
- Trainable bridge parameters use FP32 master weights even when frozen-model
  inference uses BF16. This prevents small updates from rounding to zero.
- Short, medium, and long captions are sampled uniformly.
- Every loss component is logged separately.
- The latest checkpoint is not automatically considered the best checkpoint;
  generated-image parity must be evaluated throughout training.
- The native conditioning contract is preserved instead of modifying the
  frozen generation model.

## Data

This experiment distills text conditioning and does not require decoded images
or SSD1B VAE latents. The default Berzelius job reads:

```text
download_data/data/openvid/manifests/openvid_all_recaptions_merged.csv
```

It randomly selects one of `caption_short`, `caption_medium`, and
`caption_long`. The native NeoDragon prompt modifier is appended because the
SSD1B branch is used as the first-frame generator in hybrid inference.

## Berzelius Training

Submit from the repository root:

```bash
sbatch scripts/train_ssd1b_image_bridge_distill_1node8gpu.sbatch
```

Defaults:

- 1 node, 8 GPUs.
- Per-GPU batch size 4; global batch size 32.
- 100,000 optimizer steps.
- BF16 frozen teacher/UNet inference.
- FP32 Image Bridge master parameters.
- Latest checkpoint every 5,000 steps.
- Archived checkpoint every 10,000 steps.
- Automatic resume from the stable output directory.

Outputs:

```text
output/ssd1b_image_bridge_distill/
  ssd1b_image_bridge_latest.pt
  ssd1b_image_bridge_step010000.pt
  ssd1b_image_bridge_step020000.pt
  ...
  history.json
```

The checkpoint stores only the Image Bridge state and optimizer state. The
frozen SmolVLM2 checkpoint is loaded from `./checkpoints/` and is not duplicated
inside every Image Bridge checkpoint.

## Validation Completed

The implementation was smoke-tested through SLURM with two GPUs:

- NCCL/DDP initialization on both ranks.
- Native CLIP-L and CLIP-bigG teacher loading.
- Frozen SSD1B UNet loading.
- Exact output shapes `[B,77,768]`, `[B,77,1280]`, and `[B,1280]`.
- Representation loss backward.
- One-step functional UNet distillation.
- Full four-call differentiable rollout.
- Optimizer update with FP32 bridge parameters.
- Latest and archive checkpoint writes.
- Resume from the saved checkpoint on two GPUs.

The production eight-GPU Berzelius run uses the same `torchrun` path with
`--nproc_per_node=8`.

## 100K Evaluation and V2

This document describes the original V1 implementation. The measured 100K
evaluation found padding-dominated token supervision, compressed condition
rank, weak prompt retrieval, and compounding four-step trajectory drift. The
controlled objective-only V2 and its exact metrics are documented in
[`SSD1B_IMAGE_BRIDGE_100K_EVALUATION_AND_V2.md`](SSD1B_IMAGE_BRIDGE_100K_EVALUATION_AND_V2.md).
