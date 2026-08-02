# Self-Contained Mobile-OV DreamLite Branch

## Scope

This branch replaces the SSD1B image generator with DreamLite-mobile while
keeping the existing SmolVLM2 understanding model and NeoDragon video branch.
No DreamLite source repository is required after cloning New-Mobile-OV.
DreamLite's maintained implementation is provided by the pinned Diffusers
package, while model weights are downloaded into `./checkpoints/dreamlite-mobile`.

The deployed image path is:

```text
generation:
prompt -> frozen SmolVLM2 -> DreamLite bridge -> [B, 128, 2048]
       -> frozen DreamLite UNet (4 calls) -> Tiny VAE -> image

editing:
source image + instruction -> frozen SmolVLM2 -> same bridge -> [B, 128, 2048]
source image -> frozen Tiny VAE -> source latent
noise latent || source latent -> frozen DreamLite UNet (4 calls) -> edited image
```

Only the bridge is trained. Qwen3-VL is a BF16 training teacher and is not
needed after distillation.

The released DreamLite teacher is not quantized: its Qwen3-VL safetensors are
about 4.25 GB in BF16. Keeping this teacher in its released precision avoids
distilling quantization error. The complete local DreamLite checkpoint is
about 4.8 GB; deployment removes the Qwen teacher and retains only the UNet,
Tiny VAE, scheduler, SmolVLM2, and the trained bridge.

## Why This Objective

The bridge does not use direct raw token MSE because Qwen3-VL and SmolVLM2 have
different tokenizers and different visual-token layouts. The teacher sequence
is mask-aware and resampled into the bridge's fixed query slots. Representation
supervision then combines normalized token alignment, cosine alignment, norm,
global pooled alignment, batch geometry, and variance preservation.

Representation matching alone was insufficient in earlier Mobile-OV
experiments. The trainer therefore adds:

1. A same-state functional objective using the frozen DreamLite UNet.
2. A native four-call closed-loop objective with independent teacher and
   student trajectories from the same noise.
3. Prediction, transition, and terminal-latent matching across the complete
   deployed denoising schedule.

All bridge parameters and optimizer updates stay in FP32. DreamLite, Qwen3-VL,
and SmolVLM2 run in BF16. Gradient flows through every frozen student UNet call
to the bridge, but no teacher or generator weights are updated.

The eight-GPU job uses DDP rather than FSDP. Only about 10.5M bridge parameters
are trainable, while every large component is frozen and fits independently on
an A100 80 GB. Sharding this small optimizer state would add complexity without
meaningful memory savings. Each rank receives a different data shard, while
generation/edit mode selection is synchronized across ranks so all workers run
the same objective branch on each step.

## Berzelius Setup

DreamLite weights are gated. Accept the Hugging Face model terms and expose a
valid `HF_TOKEN` before submitting the download job.

```bash
sbatch scripts/download_dreamlite_checkpoint_berzelius.sbatch
```

This one-GPU job keeps a CUDA heartbeat active while it updates the existing
`neo_mobileov` environment and downloads `carlofkl/DreamLite-mobile` revision
`diffusers`. It does not clone an auxiliary repository.

Generation-only distillation on one node and eight GPUs:

```bash
sbatch scripts/train_dreamlite_image_bridge_1node8gpu.sbatch
```

Generation and editing distillation:

```bash
EDIT_MANIFEST=/absolute/path/to/edit_manifest.csv \
EDIT_PROBABILITY=0.25 \
sbatch scripts/train_dreamlite_image_bridge_1node8gpu.sbatch
```

The edit manifest requires `source_image` and `instruction` columns. Relative
image paths are resolved against the manifest directory. A paired target image
is not required for bridge distillation because native Qwen3-VL conditions and
native DreamLite responses provide the targets. Paired targets should still be
used for downstream edit-quality evaluation.

Useful overrides include:

```bash
GENERATION_PROMPTS=/absolute/path/to/captions.csv \
OUT=output/dreamlite_bridge_run \
TARGET_STEP=100000 \
BATCH_SIZE=4 \
sbatch scripts/train_dreamlite_image_bridge_1node8gpu.sbatch
```

The generation manifest supports `caption_short`, `caption_medium`, and
`caption_long`; one available caption is sampled per row with equal default
probability. Checkpoints are saved as `dreamlite_image_bridge_latest.pt` every
5K steps and archived every 10K steps. `RESUME=auto` resumes the latest file in
the same output directory.

## Inference

```bash
python tools/infer_dreamlite_image_bridge.py \
  --bridge-checkpoint output/dreamlite_image_bridge_latest.pt \
  --prompt "A red panda reading beside a window" \
  --output output/red_panda.png
```

Editing uses the identical bridge checkpoint:

```bash
python tools/infer_dreamlite_image_bridge.py \
  --bridge-checkpoint output/dreamlite_image_bridge_latest.pt \
  --source-image input.png \
  --prompt "turn the sky into a warm sunset" \
  --output output/edited.png
```

## Licensing

The integration imports DreamLite from Diffusers and does not vendor or commit
DreamLite weights. DreamLite code is Apache-2.0, but the released weights use a
non-commercial research license. Confirm license compatibility before any
commercial or product deployment.

## Validation Performed

The local validation suite covers the exact contracts used on Berzelius:

- DreamLite and Qwen3-VL import from the pinned environment without an external
  DreamLite checkout.
- SmolVLM2 generation and multimodal edit forwards both produce finite
  `[B, 128, 2048]` conditions, and all 56 trainable bridge tensors receive
  gradients.
- Native Qwen teacher outputs were verified for generation and editing.
- A frozen DreamLite UNet call propagates finite gradients to the condition but
  creates no UNet parameter gradients.
- The complete four-call independent closed loop propagates through all four
  student calls and reports prediction, transition, and terminal losses.
- A two-process DDP training smoke completed representation and functional
  steps, wrote checkpoints, and resumed from the saved optimizer state.
- The repository unit suite passes 33 tests. A two-GPU NCCL smoke script is
  provided in `scripts/smoke_dreamlite_bridge_2gpu.sbatch` for cluster-level
  validation of generation and edit branches.
