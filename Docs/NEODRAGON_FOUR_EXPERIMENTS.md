# NeoDragon Bridge and DiT Training: Four Controlled Experiments

Last updated: 2026-07-20

## 1. Design Rule

All four experiments use the original Mobile-OV NeoDragon bridge architecture.
No sequence translator, condition head, mask head, LoRA, or additional inference
module is introduced by these experiments. This keeps the existing 200k bridge
checkpoint directly compatible and makes the four runs comparable.

The shared condition path is:

```text
prompt
  -> frozen SmolVLM2-500M
  -> MCP lexical-gated projector and refinement
  -> token condition [B, 128, 1536]
  -> original pooled head [B, 2048]
  -> NeoDragon DiT
```

The teacher path is:

```text
prompt
  -> frozen NeoDragon TextEncoderBundle
  -> frozen NeoDragon ContextAdapter
  -> teacher token condition [B, 128, 1536]
  -> teacher pooled condition [B, 2048]
```

The change is therefore in the training objectives, not the model architecture.

## 2. Why Stronger Losses Are Added

The previous bridge run showed that a numerically decreasing embedding loss did
not guarantee useful video generation. Two condition tensors can be close under
one average MSE while differing in direction, magnitude, sample geometry, or
their effect on the nonlinear NeoDragon DiT.

The new trainers keep the architecture fixed and add complementary supervision:

### Raw token MSE

Matches the absolute post-ContextAdapter teacher values on valid teacher tokens.
It preserves the direct numeric contract expected by NeoDragon cross-attention.

### Normalized token MSE

Layer-normalizes student and teacher tokens before MSE. It emphasizes feature
shape when absolute scale is still imperfect.

### Token cosine distance

Matches feature direction. This is useful because attention projections are
sensitive to direction even when absolute MSE appears moderate.

### Token norm alignment

Explicitly matches average valid-token magnitude. It complements cosine loss,
which does not constrain scale.

### Pooled MSE and pooled cosine

Supervise the original `[B, 2048]` pooled branch consumed by NeoDragon. Token
alignment alone does not train this global condition sufficiently.

### Relational similarity

Matches the pairwise cosine geometry between samples in a distributed batch.
This discourages semantically different prompts from collapsing to similar
conditions. Exp 1 and Exp 3 enable this term.

### Frozen-DiT functional distillation

Feeds teacher and student conditions through the same frozen NeoDragon DiT at
the same synthetic scheduler-valid state. It minimizes differences between the
resulting vector fields. This directly asks whether the student condition has
the same downstream effect, not merely a similar embedding.

### Native flow matching

Uses real precomputed OpenVid VAE latents. For clean latent `x`, noise `eps`, and
scheduler sigma `s`:

```text
noisy       = s * eps + (1 - s) * x
target_flow = eps - x
L_flow      = MSE(DiT(noisy, condition, timestep), target_flow)
```

This trains the full DiT to generate video from Mobile-OV conditions.

### Cross-response distillation

At the same real noisy latent state, it matches the trainable Mobile-OV+DiT
prediction to the frozen native NeoDragon teacher prediction.

### Teacher-condition preservation

Runs the trainable DiT with the original teacher condition and compares it to
the frozen teacher DiT. This discourages catastrophic drift of the pretrained,
already-pruned NeoDragon model.

There is no mask BCE objective because the original bridge has no trainable mask
head. Mask behavior remains part of the original bridge contract.

## 3. Shared Data and Distributed Setup

All production scripts request:

```text
nodes:               1
GPUs:                8
torchrun processes:  8
parallel backend:    FSDP
dtype:               bf16
```

Exp 1 reads augmented captions from:

```text
download_data/data/openvid/manifests/openvid_all_recaptions_merged.csv
```

Exp 2, Exp 3, and Exp 4 read offline NeoDragon VAE latents from:

```text
data/openvid_neodragon_2s_latents/latent_manifest.csv
```

The latent manifest includes `caption_short`, `caption_medium`, and
`caption_long`. Joint runs sample these variants with weights `5,4,1`.

## 4. Experiment 1: Bridge-Only Distillation From Scratch

Script:

```text
scripts/exp1_train_neodragon_bridge_functional_distill_1node8gpu.sbatch
```

Initialization and gradients:

| Component | Initialization | State |
| --- | --- | --- |
| SmolVLM2 | released/converted checkpoint | frozen, eval mode |
| Original MCP bridge and pooled head | random | trainable |
| NeoDragon text bundle and ContextAdapter | released | frozen |
| NeoDragon DiT used for functional loss | released | frozen |

Loss:

```text
L_exp1 =
    0.25 * raw_token_mse
  + 1.00 * normalized_token_mse
  + 0.50 * token_cosine
  + 0.10 * token_norm
  + 0.25 * pooled_mse
  + 0.20 * pooled_cosine
  + 0.10 * relational
  + functional_scale * (1.00 * functional_mse + 0.10 * functional_cosine)
```

The functional scale ramps over the first 2,000 steps. Exp 1 does not use video
latents and does not update the DiT.

Defaults:

```text
steps:             24,000
batch per GPU:     4
global batch:      32
log every:         20
checkpoint every: 1,000
```

Submit:

```bash
sbatch scripts/exp1_train_neodragon_bridge_functional_distill_1node8gpu.sbatch
```

## 5. Experiment 2: Continue the Existing Bridge and Train Full DiT

Script:

```text
scripts/exp2_train_neodragon_joint_flow_distill_1node8gpu.sbatch
```

The default bridge checkpoint is the existing aligned run:

```text
output/neo_bridge_8gpu_200k/17002251/neodragon_text_bridge_latest.pt
```

On Berzelius its absolute path is:

```text
/proj/cvl/users/x_fahkh2/Neo_Mobile-OV/output/neo_bridge_8gpu_200k/17002251/neodragon_text_bridge_latest.pt
```

Initialization and gradients:

| Component | Initialization | State |
| --- | --- | --- |
| SmolVLM2 | checkpoint embedded in bridge contract | frozen, eval mode |
| Original bridge and pooled head | existing 200k/latest checkpoint | trainable |
| Trainable NeoDragon DiT | released pretrained weights | fully trainable |
| Teacher text stack and teacher DiT | released pretrained weights | frozen |

Exp 2 combines native flow matching, teacher-response distillation, teacher-
condition preservation, and a light representation anchor. It intentionally
does not introduce any new bridge layer.

Defaults:

```text
steps:             28,000
batch per GPU:     1
global batch:      8
DiT LR:            3e-6
bridge LR:         1e-5
log every:         20
checkpoint every: 1,000
```

Submit with the default checkpoint:

```bash
sbatch scripts/exp2_train_neodragon_joint_flow_distill_1node8gpu.sbatch
```

Override the checkpoint when needed:

```bash
sbatch --export=ALL,BRIDGE_CKPT=/absolute/path/to/neodragon_text_bridge_latest.pt \
  scripts/exp2_train_neodragon_joint_flow_distill_1node8gpu.sbatch
```

## 6. Experiment 3: Joint Training From a Random Original Bridge

Script:

```text
scripts/exp3_train_neodragon_joint_from_scratch_1node8gpu.sbatch
```

Exp 3 uses the same original bridge architecture as Exp 2, but initializes the
bridge randomly. The released NeoDragon DiT initializes the trainable and frozen
teacher copies.

Active objective groups:

```text
native flow matching
complete bridge representation distillation
frozen-DiT bridge functional distillation
cross-response DiT distillation
teacher-condition DiT preservation
```

Flow weight ramps from `0.05` to `0.30`, then cools to `0.10`. Representation
and functional weights are stronger early and reduce near the end. This gives
the random bridge an alignment signal while gradually prioritizing OpenVid flow.

Defaults:

```text
steps:             200,000
batch per GPU:     1
global batch:      8
log every:         100
checkpoint every: 10,000
```

Submit:

```bash
sbatch scripts/exp3_train_neodragon_joint_from_scratch_1node8gpu.sbatch
```

## 7. Experiment 4: Matched Flow-Only Baseline

Script:

```text
scripts/exp4_train_neodragon_flow_only_from_scratch_1node8gpu.sbatch
```

Exp 4 matches Exp 3 in bridge architecture, random seed, bridge initialization,
released DiT initialization, data, caption sampling, learning rates, flow
schedule, and checkpoint schedule. Its only optimized objective is native flow
matching. It does not load the teacher text stack or frozen teacher DiT.

This baseline answers whether the additional teacher objectives in Exp 3 are
actually necessary.

Submit:

```bash
sbatch scripts/exp4_train_neodragon_flow_only_from_scratch_1node8gpu.sbatch
```

## 8. Logging and Outputs

SLURM logs:

```text
logs/neo-exp1-<JOBID>.out
logs/neo-exp2-<JOBID>.out
logs/neo-exp3-<JOBID>.out
logs/neo-exp4-<JOBID>.out
```

Output roots:

```text
output/neo_exp1_bridge_functional/<JOBID>/
output/neo_exp2_joint_flow_distill/<JOBID>/
output/neo_exp3_joint_from_scratch/<JOBID>/
output/neo_exp4_flow_only/<JOBID>/
```

Exp 1 saves `neodragon_text_bridge_latest.pt`. Exp 2-4 save
`neodragon_dit_bridge_latest.pt`, containing both bridge and DiT states.
`history.json` contains all logged loss components, gradient norms, sampled
latent unit/stage, world size, and peak allocated GPU memory.

## 9. Interpretation

| Comparison | Question |
| --- | --- |
| Existing bridge vs Exp 1 | Do the stronger representation and functional losses improve alignment from scratch? |
| Exp 2 vs original inference | Can an already aligned bridge and full DiT adapt without losing native behavior? |
| Exp 3 vs Exp 4 | Are teacher losses necessary when bridge and DiT train jointly from a random bridge? |
| Exp 2 vs Exp 3 | How much does the existing bridge checkpoint help compared with one-pass joint training? |

The controlled variable is the training objective and initialization. The bridge
architecture itself remains unchanged across all four experiments.

## 10. Current Limitations

- Checkpoints save model weights and logged history but not full optimizer state.
- Exact optimizer-state resume after preemption is not yet implemented.
- Exp 1 synthetic functional states test condition equivalence but do not replace
  real-video flow training.
- The current weights and schedules are informed starting points and still need
  empirical validation on the full eight-GPU runs.
