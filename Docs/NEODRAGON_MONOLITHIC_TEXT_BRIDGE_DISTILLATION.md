# NeoDragon Monolithic Text-Bridge Distillation

## Goal

This experiment trains the existing `MobileOVNeodragonTextBridge` to replace
the released **multi-step (monolithic)** NeoDragon text stack.  It is a
bridge-only experiment:

- SmolVLM2 remains frozen.
- The released `TextEncoderBundle`, multi-step `ContextAdapter`, and multi-step
  `PyramidMMDiT` remain frozen.
- The Mobile-OV MCP bridge is the only trainable component.
- No DiT parameter, DMD student parameter, image model, or first-frame model
  is updated.

The bridge output remains exactly the existing NeoDragon direct-condition
contract: token embeddings `[B, 128, 1536]`, token mask `[B, 128]`, and pooled
projection `[B, 2048]`.  This deliberately avoids an architecture change.

## Why A Separate Monolithic Bridge

The successful original Exp1 bridge was supervised against the released
**hybrid** NeoDragon text adapter and DiT.  A new DMD student is instead
initialized from the released **multi-step** DiT and is trained against the
multi-step text/context contract.  Although both stacks accept tensors with the
same shapes, equal shapes do not prove equal conditioning behavior.  The bridge
therefore needs to match the multi-step adapter and frozen multi-step DiT before
it is evaluated with DMD.

## Training Protocol

The core losses and weights are intentionally the successful Exp1 recipe:

| Component | Weight |
| --- | ---: |
| Raw token MSE | 0.25 |
| Normalized token MSE | 1.00 |
| Token cosine distance | 0.50 |
| Token norm alignment | 0.10 |
| Pooled MSE | 0.25 |
| Pooled cosine distance | 0.20 |
| Relational cosine loss | 0.10 |
| Frozen-DiT functional MSE | 1.00 |
| Frozen-DiT functional cosine distance | 0.10 |

Training uses OpenVid recaptions with equal sampling from `caption_short`,
`caption_medium`, and `caption_long`, global batch 32 (8 GPUs x 4), learning
rate `5e-5`, and a 2,000-step functional-loss ramp.  Trainable bridge
parameters use FP32 master weights; frozen NeoDragon modules continue in BF16.

The only functional-supervision upgrade is required by the corrected DMD-v2
protocol.  Original Exp1 sampled only video units `1..6`, because its deployed
hybrid path assumed an external first-frame anchor.  The new DMD-v2 student
generates native T2V units `0..6`.  The new bridge therefore cycles one frozen
multi-step-DiT functional call through `unit 0..6` on every seven updates.
Each call still samples one of NeoDragon's three Pyramidal stages and a native
training noise level.  This adds coverage without changing the bridge, loss
family, or DiT training objective.

## Checkpoint And Selection Policy

The default run is 64,000 steps, matching the known successful Exp1 exposure
budget.  It writes:

- `neodragon_text_bridge_latest.pt` every 5,000 steps.
- Archive checkpoints every 10,000 steps.
- A held-out 256-prompt functional-validation `best` checkpoint every 2,000
  steps.

Each checkpoint records `teacher_stack=multistep`, the exact context-adapter
and DiT IDs, FP32-master status, and the all-seven-unit functional policy.  A
hybrid-target checkpoint cannot be accidentally selected as the monolithic
bridge by the compatibility audit.

## DMD Compatibility Gate

After both the bridge and a corrected all-native-unit DMD-v2 checkpoint finish,
run `scripts/audit_neodragon_monolithic_bridge_dmd_v2_1node1gpu.sbatch`.  It
rejects checkpoints unless both explicitly target:

- `context_adapter_multistep_t2v`
- `diffusion_transformer_320p_multistep_t2v`
- DMD schedule `pyramidal_1-1-1_all_native_units`

For each prompt the audit measures:

1. Native multi-step post-context tokens and pooled projection versus bridge
   output.
2. The DMD student's conditional flow response under native versus bridge
   condition at the same state for every `7 x 3 = 21` deployed calls.
3. Two full native T2V DMD one-step rollouts from the same random seed, then
   their final latent and per-unit/per-stage drift.

This gate is stronger than checking embedding cosine alone: it directly tests
the condition interface consumed by the newly distilled mobile DiT.  Good bridge
compatibility does **not** prove that the DMD student itself has sufficient
video quality; DMD quality and condition compatibility remain separate gates.

## Commands

Train the bridge on Berzelius:

```bash
sbatch scripts/train_neodragon_monolithic_text_bridge_1node8gpu.sbatch
```

Audit a completed bridge against a completed DMD-v2 checkpoint:

```bash
DMD_CHECKPOINT=output/neodragon_pyramidal_dmd_reproduction/dmd_repro_v2_all7_10k/neodragon_pyramidal_dmd_student_step010000.pt \
BRIDGE_CHECKPOINT=output/neo_monolithic_text_bridge/<run>/neodragon_text_bridge_best.pt \
sbatch scripts/audit_neodragon_monolithic_bridge_dmd_v2_1node1gpu.sbatch
```

## Local Smoke Evidence

The local H200 smoke ran through SLURM with one GPU and completed seven updates.
It verified that the checkpoint is labeled `multistep`, trainable parameters
are FP32, and functional unit indices are exactly `[0, 1, 2, 3, 4, 5, 6]`.
The separate DMD compatibility smoke also completed all 21 flow calls and both
full native rollouts.  Those smoke checkpoints are intentionally not quality
results; they validate code paths and checkpoint contracts only.
