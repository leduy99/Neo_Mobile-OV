# DreamLite V10 Postmortem and V11 Controlled Runs

## Why V10 Is Not the Intended Balanced Ablation

V10 completed 160K optimizer steps and its checkpoint is valid. The failure is
not a failed job, missing gradients, or an invalid VBench run. Its problem is
that the launched objective differed from the documented V10 objective.

The launcher passed `--training-version v10` and content-aware loss weights,
but the trainer only recognized V5 through V9 as content-aware versions. V10
therefore selected direct uniform token matching. This omitted the intended
content-token weighting and semantic contrastive loss, even though their flags
appeared in the command line.

V10 is consequently an accidental but useful ablation:

```text
uniform direct representation matching
+ four-call frozen-UNet functional and transition distillation
+ V10's reduced semantic and grounded sampling rates
```

It is not evidence that reducing the V9 curriculum is harmful, because two
variables changed together.

## Guardrail

`tools/train_dreamlite_image_bridge.py` now has an explicit
`--representation-objective` argument. New experiments must set
`content_aware`, rather than depending on `--training-version` to select a
loss. The run log and `history.jsonl` record the resolved objective. Enabling
`--global-contrastive` with a non-content objective fails before training.

The historical `auto` behavior remains available for prior launchers. It now
also maps V10 to content-aware if it is ever re-run.

## V11 Protocol

Both runs preserve the same compact 5.725M-parameter bridge, frozen Qwen3-VL
teacher and DreamLite UNet, V8 image-only main distribution, 4-call functional
and transition losses, resolution buckets, optimizer, LR schedule, batch size,
and 160K target. Both start from random bridge weights.

| Setting | V11 control | V11 balanced |
|---|---:|---:|
| Representation objective | content-aware | content-aware |
| Main prompts | JourneyDB 71.43%, short-caption 28.57% | Same |
| Synthetic semantic prompt probability | 0.15 | 0.12 |
| Dedicated verified grounded-batch probability | 0.15 | 0.10 |
| Grounded functional batch | 2 | 2 |
| Grounded functional multiplier | 0.5 | 0.5 |
| Functional start/ramp | 10K / 20K | Same |
| Closed loop | Disabled | Disabled |

`V11-control` is an explicit, reproducible V9 control. `V11-balanced` isolates
the only intended V10 difference: less targeted semantic and grounded
curriculum. Any score difference can therefore be attributed to those rates,
rather than an accidental representation-loss change.

## Submission

```bash
sbatch scripts/train_dreamlite_compact_v11_control_1node8gpu.sbatch
sbatch scripts/train_dreamlite_compact_v11_balanced_1node8gpu.sbatch
```

Expected output roots:

```text
output/dreamlite_compact_v11_v9_control_from_scratch/<JOBID>/
output/dreamlite_compact_v11_v10_rates_from_scratch/<JOBID>/
```

At startup, each log must report:

```text
representation_objective=content_aware
```

After the first training log, `history.jsonl` must contain
`"representation_objective": "content_aware"` and a nonzero
`semantic_batch_size` (32 for the global representation batch). After step
10K, selected grounded updates must record `dedicated_grounded_batch=true` and
`grounded_images=2`.
