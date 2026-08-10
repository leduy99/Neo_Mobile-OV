# Shared SmolVLM2 Conditioning

## Goal

The previous end-to-end Mobile-OV pipeline loaded two frozen SmolVLM2-500M
instances and ran two text forwards for every prompt:

1. The DreamLite image bridge used its own `"[Generate]:"` prompt template and
   a 256-token window to create the first-frame condition.
2. The Exp1 NeoDragon bridge used the raw caption plus NeoDragon suffix, a
   512-token window, and strict 128-token selection to create the video
   condition.

This is unnecessary duplicate compute and duplicate model memory. More
importantly, it gives the image and video branches different language contexts
for what should be one shared generation request.

## Design

`SharedMobileOVGenerationConditioner` owns exactly one frozen Exp1 video bridge
and an image head without a local feature provider.

```text
prompt + NeoDragon suffix
          |
          v
  frozen SmolVLM2, one forward
          |
          +------------------------------+
          |                              |
          v                              v
DreamLite image head                frozen Exp1 MCP head
variable-length 2048 condition      [B, 128, 1536] NeoDragon condition
          |                              |
      DreamLite                      NeoDragon Hybrid DiT
      first frame                    video continuation
```

The NeoDragon branch calls the exact existing Exp1 projector on the shared
hidden states. This keeps the released 64K video bridge unchanged. The new
DreamLite image head receives the same selected hidden-layer sequence and is
re-distilled from the frozen Qwen3-VL/DreamLite teacher.

The canonical input contract is explicit and saved in every shared checkpoint:

- prompt: `prompt + ", cinematic, realistic textures, high detail, natural colours"`
- SmolVLM2 window: `512`
- token selection: existing Exp1 strict selection to `128` tokens
- DreamLite teacher target: the original prompt, without the NeoDragon suffix

The last distinction matters: the suffix aligns the shared student features
with deployed Exp1 video inference, while the image teacher remains supervised
to generate the image requested by the user.

## Compatibility

V1-V7 DreamLite checkpoints are **not visually deployable** through this path.
They were trained on a different SmolVLM2 prompt/token contract. Their learned
DreamLite head can be loaded structurally, which is useful for shape and
integration tests, but its output must not be used for image quality claims.

The shared trainer therefore creates a new image head from scratch and only
allows resume from a prior shared-SmolVLM2 run. The video bridge is loaded from
the successful Exp1 64K checkpoint and stays frozen.

This version is generation-only. DreamLite editing includes source-image tokens,
while the Exp1 NeoDragon bridge is text-only; forcing both through this shared
text sequence would discard edit information. Editing remains a separate,
multimodal path until it has its own compatible shared contract.

## Validation

All tests ran on an allocated H200 GPU on 10 August 2026.

| Check | Result |
| --- | --- |
| Exp1 video condition parity | `0.0` max absolute error for prompt embeddings, mask, and pooled vector |
| Shared DreamLite condition | finite `[1, 21, 2048]` condition and `[1, 21]` mask |
| Legacy encoder count | 2 SmolVLM2 instances |
| Shared encoder count | 1 SmolVLM2 instance |
| Condition latency, 10 warm runs | `46.49 ms` legacy versus `25.21 ms` shared |
| Condition saving | `21.28 ms/prompt` (`1.84x` condition-stage speedup) |
| Representation smoke | one full forward, backward, optimizer step, and checkpoint save passed |
| Functional smoke | frozen DreamLite 4-call functional loss, backward, optimizer step, and checkpoint save passed |
| End-to-end route | a shared checkpoint completed DreamLite first-frame generation and NeoDragon Hybrid video generation |

The end-to-end smoke checkpoint was trained for one step, so it validates
wiring and latency only. It is not a visual-quality benchmark. A direct
legacy/shared warm comparison also had only two generated samples; report the
measured `~22 ms` condition saving rather than attributing its larger total-time
variation to the redesign.

## Run On Berzelius

Train a quality-valid shared image head on eight GPUs:

```bash
sbatch scripts/train_dreamlite_shared_smol_v1_1node8gpu.sbatch
```

The job uses the V8 mixed prompt curriculum, trains for `120000` steps by
default, saves `latest` every `5000` steps and an archive every `10000` steps.
Its checkpoint records `shared_smolvlm2.enabled=true`, the frozen Exp1
checkpoint path, the NeoDragon config, and the canonical suffix.

After training, benchmark the actual shared full pipeline:

```bash
python tools/benchmark_full_mobileov_vs_native_neodragon.py \
  --image-config configs/mobile_ov_dreamlite_compact_v8.yaml \
  --video-config configs/mobile_ov_neodragon.yaml \
  --image-bridge-ckpt output/dreamlite_shared_smol_v1/<JOB_ID>/dreamlite_image_bridge_latest.pt \
  --video-bridge-ckpt checkpoints/hf_mobile_ov/neo_exp1_bridge_functional/17108893/neodragon_text_bridge_latest.pt \
  --shared-smolvlm2 --warmup 2 --runs 10
```

`--shared-smolvlm2` rejects legacy V1-V7 image checkpoints deliberately. This
prevents accidentally evaluating an image head on the wrong feature contract.
