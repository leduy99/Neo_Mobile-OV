# NeoDragon Flow-Matching Audit

## Decision

Plain flow matching is valid for NeoDragon only when it follows the complete
Pyramidal-Flow training contract. The former Mobile-OV joint trainer did not:
it trained every spatial stage on an independent noise-to-clean path. A short
controlled experiment shows that this mismatch can destroy generation after
only 300 updates.

For Mobile-OV adaptation, the recommended objective is therefore:

1. correct data Pyramidal Flow Matching;
2. Gaussian corruption of autoregressive ground-truth history;
3. released-teacher response preservation on the same state;
4. a short DiT-only recovery phase before opening the bridge.

This is not evidence that more losses are always better. It is evidence that
the data path must first be correct, after which teacher supervision is a useful
trust region.

## What The Papers Specify

Pyramidal Flow does not use the ordinary full-resolution path at every stage.
For each low-to-high spatial stage, it constructs a stage-specific start and
end distribution using correlated pyramid noise. It trains only the current
autoregressive unit while conditioning on lower-resolution history. The
official implementation supports configurable per-stage sample ratios, with an
equal `1:1:1` default, and corrupts causal history by up to one third Gaussian
noise.

NeoDragon retains this contract during pruning recovery. Its Stage 1 uses
approximately 350K videos, Adam at a fixed 3e-5 learning rate, global batch 16,
and only 300 iterations. Stage 2 combines data FM, teacher-predicted FM, and
teacher feature matching. The paper reports:

| Stage-2 supervision | VBench total | Quality | Semantic | Note |
|---|---:|---:|---:|---|
| Data FM only | 78.39 | 81.58 | 65.63 | Stage-1 result |
| Teacher FM only | 80.00 | 83.52 | 65.89 | Strong recovery |
| Teacher + data FM | 80.04 | 83.52 | 66.11 | Stable without feature loss |
| Feature + teacher + data, simple mapping | 80.21 | 83.54 | 66.90 | Selected deployment model |
| Feature + data, next mapping | 80.35 | 83.82 | 66.48 | Higher score but artifacts and black videos |

The curriculum is important: NeoDragon reports that applying Stage 2 directly
after pruning did not work as well as Stage 1 followed by Stage 2.

## The Previous Mobile-OV Mismatch

For all stages, the former trainer used:

```text
x_t = sigma * independent_noise + (1 - sigma) * clean_current_stage
target = independent_noise - clean_current_stage
```

That is correct only for a conventional single-scale noise-to-data path. It
omits four parts of NeoDragon's actual contract:

- the higher-stage start point based on the upsampled previous clean scale;
- the non-final-stage noisy endpoint;
- correlated noise across spatial scales;
- corrupted autoregressive history and explicit stage-balanced sampling.

The proper target and the legacy target have only about 0.362 cosine similarity
on the same samples. Consequently, teacher distillation in previous joint runs
was also evaluated on states outside the released model's intended path.

## Controlled Ablation

Setup: released multistep NeoDragon, native text conditions, 100 cached OpenVid
latents at 49 frames and 320x512, 80 training samples, four fixed holdout
samples, 300 updates, LR 3e-5. Native text was used to remove the bridge as a
confounder. This pilot used a fixed `1:2:1` low/mid/high stage mixture to spend
more probes on the middle transition; this was an ablation choice, not a
reported NeoDragon setting. The production pilot now defaults to the official
Pyramidal-Flow implementation's equal `1:1:1` mixture.

| Objective | Holdout student/teacher rel. MSE | Cosine | Output norm ratio | Parameter drift rel. L2 |
|---|---:|---:|---:|---:|
| Legacy data FM | 2.2402 | 0.9840 | 2.2321 | 0.00815 |
| Correct PFM, clean history | 0.0131 | 0.9935 | 0.9364 | 0.00668 |
| Correct PFM, corrupted history | 0.0138 | 0.9932 | 0.9386 | 0.00681 |
| Correct PFM + corrupted history + teacher | **0.0097** | **0.9952** | **0.9413** | **0.00652** |

One-call teacher-forced measurements alone are insufficient. Before this
training ablation, the old joint V1/V2 checkpoints still showed approximately
0.944 one-call cosine under proper states, despite severe deployed-rollout
failure. Therefore, every checkpoint was also evaluated through a controlled
autoregressive rollout with identical text, initial noise, and transition
noise.

| System | Latent RMS | Motion-energy ratio | Optical-flow mean | Sharpness | Latent cosine vs released |
|---|---:|---:|---:|---:|---:|
| Released multistep | 1.064 | 0.129 | 0.529 | 517 | 1.000 |
| Legacy FM | 4.029 | 0.091 | 0.139 | 14 | 0.595 |
| Correct PFM, clean history | 0.959 | 0.147 | 0.424 | 514 | 0.865 |
| Correct PFM, corrupted history | 0.968 | 0.235 | 0.603 | 539 | 0.866 |
| Correct PFM + history + teacher | 1.020 | 0.151 | 0.430 | 452 | **0.894** |

The legacy cell becomes color blobs or nearly static video after only 300
steps. All correct-PFM cells retain recognizable subjects and scenes. History
corruption increases motion but can overshoot; teacher preservation gives the
closest overall trajectory to the released model.

## Interpretation

The old joint failures do not prove that flow matching is unsuitable for the
monolithic NeoDragon model. They primarily prove that ordinary per-stage
noise-to-clean flow matching is incompatible with its pyramidal scheduler.

Pure, correct PFM is appropriate for a short recovery stage. For longer
Mobile-OV adaptation, data FM plus teacher response matching is safer because
the OpenVid objective is underdetermined: many velocity fields can reduce local
MSE while changing the complete autoregressive rollout. The teacher term limits
that drift. Feature matching may be useful later, as in NeoDragon Stage 2, but
it should not be added until the corrected two-loss baseline passes controlled
rollout evaluation.

## New Training Recipe

Phase 1, DiT recovery:

- initialize from released multistep NeoDragon and the validated bridge;
- freeze the bridge;
- use correct PFM, equal `1:1:1` stage sampling, and history corruption in
  [0, 1/3];
- use data FM plus teacher response MSE/cosine on the identical noisy state;
- begin with 300 updates and evaluate before extending.

Phase 2, protected joint adaptation:

- initialize from Phase 1 rather than the released checkpoint again;
- lower DiT LR by approximately 10x;
- open the bridge gradually;
- retain data FM, teacher response preservation, bridge representation, and
  frozen-teacher bridge functional losses;
- select checkpoints by controlled rollout drift and video metrics, not the
  training loss alone.

Only after the adapted multistep model is validated should it be distilled to
the Hybrid 1-1-1 deployment schedule. Monolithic flow training and Hybrid DMD
distillation solve different problems and should remain separate stages.

## Reproducibility And Limits

Implemented artifacts:

- `new_mobile_ov/training/neodragon_pyramid_flow.py`: corrected stage paths,
  correlated noise, history corruption, and stage allocation;
- `tools/audit_neodragon_flow_contract.py`: no-update contract diagnostics;
- `tools/ablate_neodragon_flow_training.py`: short controlled objective cells;
- `tools/evaluate_neodragon_flow_ablation.py`: matched autoregressive rollout;
- `tools/train_neodragon_dit_bridge.py`: production `--flow-contract pyramid`;
- `scripts/train_mobileov_monolithic_pyramid_recipe_2gpu.sbatch`: two-phase
  low-cost training pilot.
- `scripts/train_mobileov_monolithic_pyramid_recipe_1node8gpu.sbatch`:
  Berzelius 8-GPU run with global batch 16 and the same two-phase curriculum.

The released multistep checkpoint is used as the teacher in our pilot. This is
a preservation teacher, not the undisclosed full internal teacher used by the
NeoDragon authors. The paper also does not disclose Stage-2 duration, learning
rate, exact loss weights, or all teacher feature taps. Therefore, the corrected
PFM contract is reproducible, but the complete proprietary Stage-2 recipe is
not claimed to be reproduced exactly.

## References

- [NeoDragon paper](https://arxiv.org/html/2511.06055v1)
- [Pyramidal Flow Matching paper](https://arxiv.org/html/2410.05954)
- [Official Pyramidal Flow implementation](https://github.com/jy0205/Pyramid-Flow/blob/main/pyramid_dit/pyramid_dit_for_video_gen_pipeline.py)
