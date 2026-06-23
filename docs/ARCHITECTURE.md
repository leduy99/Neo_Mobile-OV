# New Mobile-OV Architecture

## Goal

Build a mobile-oriented text-to-video generation branch without forcing a full 2B video diffusion model onto device.

The proposed decomposition is:

```text
prompt
  -> SmolVLM2 text stack
  -> Bridge v2 lexical-gated projector
  -> anchor generation branch
  -> z0 anchor latent
  -> Latent Motion Weaver
  -> video latent trajectory
  -> video decoder
```

## Shared Text Side

All generation options reuse the current Mobile-OV text side:

```text
SmolVLM2 hidden states
  -> MCP lexical-gated bridge
  -> prompt tokens [B, 300, 2304]
  -> masked mean pool for LMW v0 [B, 2304]
```

The full token sequence is preserved so later LMW variants can use lightweight cross-attention instead of only pooled FiLM.

## Option 1: Current Mobile-OV

`backend.name = mobile_ov_current`

This is the reference branch:

```text
SmolVLM2 + Bridge v2 -> SANA-video condition -> SANA-video/WanVAE generation
```

For this new repo it is kept as a teacher/reference path. The first LMW experiment should not depend on sampling a full video from this backend.

## Option 2: Mobile-O / SANA-0.5B

`backend.name = mobile_o_sana_0_5b`

This is the lightweight branch:

```text
SmolVLM2 + Bridge v2 -> lightweight image/anchor generator -> z0 anchor
```

Mobile-O source is vendored in `third_party/mobileo`. Its checkpoint is configured by path, not copied into the repo.

Important caveat: Mobile-O's image/DC-AE latent is not guaranteed to match SANA-video WanVAE latent slices. If the latent spaces differ, the correct target remains:

```text
z0_gt = WanVAE(video)[:, :, 0]
```

Then we either:

1. train a direct anchor head into WanVAE `z0_gt`, or
2. add an adapter from Mobile-O image latent to WanVAE anchor latent.

## Option 3: Mobile-OV + Neodragon

`backend.name = mobile_ov_neodragon`

This branch replaces the SANA-video generator with Neodragon while keeping the Mobile-OV understanding branch and Bridge v2 unchanged.

Current smoke inference path:

```text
prompt
  -> SmolVLM2 + Bridge v2
  -> bridge tokens [B, 300, 2304] and pooled text [B, 2304]
  -> logged for shape/norm verification

prompt
  -> Neodragon native CLIP/T5 text stack
  -> Neodragon context adapter
  -> Neodragon DiT + VAE
  -> video
```

This confirms that the old understanding branch still loads and produces valid bridge embeddings, and that Neodragon can generate inside the same Mobile-OV repo/runtime.

Bridge-conditioned training path:

```text
prompt
  -> frozen SmolVLM2 + Neodragon-shaped Bridge v2
  -> direct Neodragon DiT condition:
       prompt_embeds         [B, 128, 1536]
       prompt_attention_mask [B, 128]
       pooled_prompt_embeds  [B, 2048]
  -> Neodragon DiT
```

The stage-1 teacher target is produced by frozen Neodragon text modules:

```text
prompt + Neodragon prompt modifier
  -> frozen Neodragon TextEncoderBundle
  -> frozen Neodragon ContextAdapter for token target only
  -> target token condition [B, 128, 1536]
  -> target pooled condition [B, 2048]
```

Bridge distillation loss:

```text
L_token  = masked MSE(pred_tokens, target_tokens)
L_pool   = MSE(pred_pooled, target_pooled)
L_cos    = masked cosine loss(pred_tokens, target_tokens)
L        = L_token + 0.25 * L_pool + 0.05 * L_cos
```

After this bridge is trained well enough, Neodragon inference can replace its native text stack with:

```text
SmolVLM2 + Neodragon-shaped Bridge v2 -> Neodragon DiT
```

Stage-2 DiT training uses bridge-conditioned flow matching:

```text
video -> Neodragon VAE -> scaled latent Z
Z_t = sigma * noise + (1 - sigma) * Z_clean
target_flow = noise - Z_clean
Neodragon DiT(Z_t, bridge_condition, past_gt_latents) -> target_flow
```

## Latent Motion Weaver

Target WanVAE shape for SANA-video 480p:

```text
RGB video:     [B, 3, 81, 480, 832]
WanVAE latent: [B, 16, 21, 60, 104]
anchor z0:     [B, 16, 60, 104]
```

LMW v0:

```text
LatentMotionWeaver(z0_gt, pooled_text) -> Z_pred [B, 16, 21, H, W]
```

The first slice is fixed:

```text
Z_pred[:, :, 0] = z0_gt
```

Loss:

```text
L_lat    = L1(Z_pred[:, :, 1:] - Z_gt[:, :, 1:])
L_motion = L1(diff_t(Z_pred) - diff_t(Z_gt))
L        = L_lat + 0.5 * L_motion
```

Required baseline:

```text
Z_copy = repeat(z0_gt, T=21)
```

The direction only passes if `Z_pred` beats `Z_copy` clearly in latent metrics and decoded videos show real motion instead of only static reconstruction.
