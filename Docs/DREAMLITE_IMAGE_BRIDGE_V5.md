# DreamLite Image Bridge V5

## Scope

V5 trains the existing 5.73M-parameter compact, variable-length DreamLite bridge from scratch. It does not change the bridge architecture, unfreeze SmolVLM2 or DreamLite, train editing, or reuse a V4 checkpoint.

The design addresses the two measured V4 bottlenecks:

1. Pooled alignment was strong, but prompt-content token alignment and rare/compositional semantics remained weak.
2. Teacher-state functional predictions were reasonable, but free-running errors accumulated because deployment eventually reaches student-induced states.

## V5 objectives

### Content-aware representation

DreamLite retains three generation-wrapper tokens before prompt content and five suffix tokens after it. V5 splits those regions explicitly and gives prompt content full weight while retaining only 0.25x supervision on wrapper tokens. The same objective is applied before and after DreamLite's frozen condition projection.

V5 also adds a batch-level contrastive objective. Student prompt `i` must be closer to teacher prompt `i` than to other prompts in the batch. This discourages generic conditions that match average statistics but lose object identity, count, color, or relations.

### Mixed same-state functional distillation

V5 never compares teacher and student predictions from different latent states.

For teacher-state samples, a frozen native-Qwen prefix creates `x_k`. For student-state samples, a detached bridge prefix creates `x_k`. Teacher and student are then evaluated on that same `x_k`, timestep, source latent, resolution, and logical `time_ids`.

The student-state probability ramps from zero to 25% after step 30K. This exposes the bridge to deployment-like states without restoring the invalid V3 closed-loop objective.

### Transition supervision

In addition to matching the DreamLite UNet prediction, V5 compares the next scheduler states produced from the same input state:

```text
x_teacher_next = scheduler.step(teacher_prediction, x_k)
x_student_next = scheduler.step(student_prediction, x_k)
```

This directly penalizes condition errors according to how much they perturb the next denoising state.

### Prompt curriculum

OpenVid short, medium, and long captions remain 80% of training. The remaining 20% is a deterministic synthetic image-prompt curriculum covering rare nouns, counts, colors, spatial relations, visible text, materials, styles, and wide compositions. It is generated inside the dataset and contains no VBench prompts.

## Schedule

```text
0-10K:    content-aware representation and contrastive alignment
10K-30K:  ramp teacher-state functional and transition losses
30K-50K:  ramp student-state same-input supervision to 25%
50K-120K: full mixed-state V5 objective
```

Call sampling is `40% / 20% / 20% / 20%` for DreamLite calls 0-3 because call 0 had the largest measured error and seeds the remaining trajectory.

The V4 multi-resolution curriculum is preserved exactly, including separate actual render grids and logical `time_ids`.

## Berzelius

Submit from the repository root:

```bash
sbatch scripts/train_dreamlite_compact_v5_1node8gpu.sbatch
```

Inspect the log after submission:

```bash
tail -n 20 logs/mov-dream-v5-<JOBID>.out
```

V5 defaults to `--resume none`; a new job therefore starts from random bridge weights even if an older V5 output exists.
