# VBench: First-Frame Swaps and Stage-2 Three-Epoch Results

- **Recorded:** 2026-10-12 (Asia/Dubai)
- **Status:** measured-score record; checkpoint-curve results are still pending
- **Scope:** recent Berzelius evaluations, not mobile latency measurements

## 1. Main Findings

The first-frame stack strongly affects prompt following. With the same SSD1B
first-frame images, Joint V2 approaches the matched native Hybrid's semantic
score: **76.81 vs. 77.07 with QuickSR**. However, its dynamic-degree score remains
much lower: **45.56 vs. 75.83**. Improving the first frame alone does not eliminate
the video-stack gap.

The single-generator, UniVideo-style Stage-2 system improved substantially over
the earlier Stage-2 runs. Its raw total score is **75.93**, compared with **72.33**
for the earlier reconstruction-150K run. It still trails the two-generator
systems, especially in scene depiction and visual appearance.

These observations do not establish that fine-tuning alone caused the remaining
gap. Joint V2 and native Hybrid use different video weights, conditioners, and
denoising schedules. The newer Stage-2 run also changed the data and task mixture,
not only the number of steps.

## 2. Sources and Status

The numerical tables below were transcribed from Markdown reports pasted by the
user from Berzelius. The benchmark code and local cached training contract were
checked separately. The remote JSON receipts, checkpoint hashes, per-video
scores, and confidence intervals have **not** been independently retrieved for
these latest completed runs.

All paths in this note are relative to the repository root. The Berzelius root
is `/proj/cvl/users/x_fahkh2/Neo_Mobile-OV`.

| Report | Output root | Evidence/status |
| --- | --- | --- |
| Matched Joint V2 and native baseline | `output/vbench_joint_v2_vs_native_quicksr_berzelius_5x_dual/` | User supplied `quicksr_ablation.md`; full score table available |
| SSD1B first frame -> Joint V2 | `output/vbench_joint_v2_ssd1b_5x_dual/` | User supplied `anchor_effect.md`; intended job `17728808` |
| V11/DreamLite first frame -> native Hybrid | `output/vbench_native_hybrid_dreamlite_v11_5x_dual/` | User supplied `anchor_effect.md`; submitted job `17728809` |
| Stage-2 three-media-epoch checkpoint | `output/vbench_mobileov_stage2_media3epochs_5x_dual/` | User supplied `quicksr_ablation.md`; benchmark job ID not independently confirmed |
| Separate native-release reproduction | `output/vbench_native_neodragon_release_5x/` | Earlier user-supplied report; a different generation protocol |

Submission `17728910` duplicated the SSD1B -> Joint V2 job. Cancellation was
recommended, but no cancellation receipt was supplied. The table above records
submission history, not a verified Slurm exit-status audit.

Relevant implementation commits:

- `3d4924e`: paired raw/QuickSR benchmark.
- `243f019`: separate native-release-protocol reproduction.
- `f2e29d5`: final Stage-2 three-epoch benchmark.
- `d6ee5a8`: first-frame cross-swap jobs.
- `42fee10`: pinned intermediate-checkpoint evaluation at 100K and 190K.

No weights, videos, or optimizer states are included in this documentation
commit. Checkpoint uploads were deferred because HF storage was exhausted.

## 3. Evaluation Settings

The new full runs use **944 official unique prompts and five independently
generated videos per prompt**, or **4,720 generated clips per setting**. Each
VBench dimension evaluates its own official prompt subset; not every dimension
evaluates all 944 prompts.

Each clip is exported twice from identical generated RGB frames:

| Variant | Output | Postprocessing |
| --- | --- | --- |
| Without QuickSR / raw | 512x320, 49 frames, 24 FPS | Direct export; no resizing |
| With QuickSR | 1024x640, 49 frames, 24 FPS | Public QuickSRNet Medium 2x, FP16 |

This measures the **combined resolution and super-resolution effect**, not
super-resolution at fixed output resolution. The raw version is not downsampled
from an already-upscaled MP4.

Aggregate quality, semantic, and total scores are percentages. Dimension rows
are the original dimension scores multiplied by 100, not the normalized
`scaled` values used internally for aggregation. Changes are percentage points
(pp), not relative percentage changes.

Dynamic degree is the fraction of clips meeting VBench's optical-flow movement
criterion. It is not a measure of correct actions, realistic motion, or average
motion speed. More movement does not automatically mean a better video.

### 3.1 Systems

| System | Conditioning and generator | First frame |
| --- | --- | --- |
| Joint V2 + V11/DreamLite | Frozen SmolVLM2 + frozen video Bridge V2 + monolithic DiT fine-tuned for 10K steps; configured 20/10 steps per stage and CFG 7/5 | V11 image bridge + DreamLite, four-step image generation |
| Joint V2 + native SSD1B | Exactly the same Joint V2 video stack | Original SSD1B PNGs from the matched native baseline |
| Native Hybrid + native SSD1B | Native text encoders/context adapter + released Hybrid, 1-1-1 and no CFG | Native SSD1B |
| Native Hybrid + V11/DreamLite | Exactly the same native Hybrid video stack | Original V11/DreamLite PNGs from the matched Joint V2 baseline |
| Stage 2, three media epochs | Frozen SmolVLM2 + trained multimodal connector + trained monolithic DiT; 20/10 steps per stage and CFG 7/5 | DiT generates frame zero and all subsequent frames; no external image generator |

Here, monolithic means the released multi-step NeoDragon backbone, not an
unpruned foundation model. Hybrid is the released few-step policy. These
benchmarks do not establish equal inference compute or mobile latency.

With a supplied first-frame anchor, the anchored systems do not ask their video
DiT to generate frame zero. The 20/10 values above describe the configured
first-frame/video sampling schedules, not a claim that the first-frame schedule
is executed when an image is supplied.

The native Hybrid swap directly passes the supplied image to the generation
helper. It does not use the Hybrid pipeline entry point that would regenerate
SSD1B and ignore the supplied image.

### 3.2 Matched Cross-Swap Controls

For each cross-swap, only the original first-frame PNG is replaced. Video
weights, text conditioning, prompt modifier, video seed, and sampling schedule
remain unchanged. Source PNG hashes and sample metadata are verified; images
are not regenerated or selected as best-of-five samples.

The paired experiments therefore identify the effect of changing the
**first-frame stack and its compatibility with each video stack**. They do not
isolate image-bridge error from the capabilities of SSD1B versus DreamLite.

The matched anchor experiments use the cinematic prompt modifier and reset
image/video RNG independently. Stage 2 uses the literal prompt, a trained empty
negative condition, no modifier, and seeds reset within its generation helper.
Stage 2 is a separate system comparison, not another first-frame-only swap.

### 3.3 Configured Weight Identities

The anchored benchmark pins the following identities in its
[protocol constants](../tools/vbench_joint_native_protocol.py). They specify the
intended weights; the latest remote artifact hashes have not been independently
downloaded for this note.

| Component | Checkpoint identity |
| --- | --- |
| Joint V2 DiT + frozen Bridge V2 | `mobileov_monolithic_pyramid_cfg_dit/mobileov_monolithic_pyramid_cfg_dit_17346642/neodragon_dit_bridge_step010000.pt`; SHA256 `f70909cccdcfc6f40beb5fafb11acdadcdd7b546e6fff2347182ec0a2ebaa14f` |
| V11 image bridge | `dreamlite_compact_v11_v10_rates_from_scratch/17291461/dreamlite_image_bridge_latest.pt`; SHA256 `371a393cfee257436124ea3c1cc0919f2ade28f7928608a1afafd74fe5052126` |
| Native default snapshot | `karnewar/Neodragon`, revision `5f296764e442d7261c53fcf9367a6b0f01e22f7f`; preserve the existing cache |
| New Stage-2 training run | `output/mobileov_stage2_media3epochs/17641364/`; final expected step 283,977 |

The anchored checkpoints are from HF repo `Amshaker/Mobile-OV`. Joint V2 may be
restored under `output/` or `checkpoints/hf_mobile_ov/`; the benchmark verifies
content hashes rather than assuming a matching filename identifies the weights.

## 4. Aggregate Results

All rows below are the new five-video-per-prompt reports.

| Setting | QuickSR | Quality | Semantic | Total | Dynamic degree |
| --- | --- | ---: | ---: | ---: | ---: |
| Joint V2 + V11/DreamLite | Without | 82.73 | 69.31 | 80.04 | 65.00 |
| Joint V2 + V11/DreamLite | With | 83.06 | 72.32 | 80.91 | 64.72 |
| Joint V2 + native SSD1B | Without | 82.29 | 71.99 | 80.23 | 46.11 |
| Joint V2 + native SSD1B | With | 82.68 | 76.81 | 81.51 | 45.56 |
| Native Hybrid + native SSD1B | Without | 83.85 | 72.56 | 81.59 | 76.39 |
| Native Hybrid + native SSD1B | With | 84.30 | 77.07 | 82.85 | 75.83 |
| Native Hybrid + V11/DreamLite | Without | 83.74 | 69.98 | 80.99 | 87.22 |
| Native Hybrid + V11/DreamLite | With | 84.06 | 73.25 | 81.90 | 86.11 |
| Stage 2, three media epochs | Without | 79.83 | 60.32 | 75.93 | 66.94 |
| Stage 2, three media epochs | With | 79.77 | 62.00 | 76.22 | 64.72 |

### 4.1 Same-Image Video-Stack Gaps

Native minus Joint V2, calculated from the displayed QuickSR scores:

| Shared first-frame source | Quality gap (pp) | Semantic gap (pp) | Total gap (pp) | Dynamic-degree gap (pp) |
| --- | ---: | ---: | ---: | ---: |
| V11/DreamLite | 1.00 | 0.93 | 0.99 | 21.39 |
| Native SSD1B | 1.62 | 0.26 | 1.34 | 30.27 |

The semantic gap is small once the first-frame source is shared. A substantial
dynamic-degree gap persists with either first-frame source. This cannot yet be
assigned solely to DiT fine-tuning: the conditioner and monolithic/Hybrid
sampling policy also differ.

Within these tested systems, Joint V2 + SSD1B + QuickSR has the highest
Mobile-OV-trained system total (81.51), but it uses native SSD1B conditioning.
It is a diagnostic reference, not a solution that removes every native encoder.
Joint V2 + V11/DreamLite + QuickSR remains the reported 80.91 system using the
SmolVLM2-based image and video bridges, with separate image/video generators.

## 5. First-Frame Effects

These are the **changes printed in the original reports**, rather than changes
recomputed from their rounded absolute scores. A difference of 0.01 pp between
a printed change and subtracting displayed values is expected from rounding.

| Score/dimension | Joint: DreamLite -> SSD1B, raw | Joint: DreamLite -> SSD1B, QuickSR | Native: SSD1B -> DreamLite, raw | Native: SSD1B -> DreamLite, QuickSR |
| --- | ---: | ---: | ---: | ---: |
| quality_score | -0.44 | -0.38 | -0.11 | -0.24 |
| semantic_score | +2.68 | +4.50 | -2.59 | -3.82 |
| total_score | +0.19 | +0.59 | -0.61 | -0.96 |
| subject_consistency | +0.38 | +0.41 | -0.72 | -0.84 |
| background_consistency | +0.38 | +0.89 | -0.11 | -0.61 |
| motion_smoothness | -0.06 | -0.09 | +0.01 | +0.03 |
| dynamic_degree | -18.89 | -19.17 | +10.83 | +10.28 |
| aesthetic_quality | +0.28 | +0.48 | -0.17 | -0.34 |
| imaging_quality | +5.59 | +5.34 | -5.36 | -5.10 |
| temporal_flickering | -0.01 | -0.04 | +0.14 | +0.17 |
| object_class | +2.85 | +3.05 | -2.75 | -2.47 |
| multiple_objects | -3.54 | -0.53 | +2.61 | +1.05 |
| color | -3.46 | -2.96 | +4.04 | +5.32 |
| spatial_relationship | -0.27 | +9.33 | -0.57 | -9.93 |
| scene | +4.72 | +5.07 | -4.40 | -4.43 |
| temporal_style | +2.21 | +2.18 | -1.71 | -1.67 |
| overall_consistency | +1.40 | +1.36 | -1.15 | -1.25 |
| human_action | +10.00 | +12.60 | -10.40 | -12.00 |
| appearance_style | +0.83 | +0.87 | -0.85 | -0.83 |

Two observations recur in both swap directions:

1. SSD1B first frames improve aggregate prompt-following and imaging scores;
   V11/DreamLite first frames produce lower values on these measures.
2. V11/DreamLite first frames produce a larger fraction of clips passing the
   movement criterion for both video stacks.

Neither source wins every dimension. Color and multiple-object scores can favor
V11/DreamLite. In particular, native Hybrid gains 10.28 pp in dynamic degree
with DreamLite but loses 12.00 pp in human-action score. More qualifying motion
is therefore not evidence of better prompt-matched actions; these dimensions
also use different prompt subsets.

Spatial-score changes are especially sensitive to output processing. For Joint
V2, switching to SSD1B changes raw spatial score by -0.27 pp but QuickSR spatial
score by +9.33 pp. Do not describe this as a change in the model's learned
spatial reasoning: the video weights did not change.

## 6. Stage-2 Training and Progress

The three-epoch run is `output/mobileov_stage2_media3epochs/17641364/`.
It starts from the 15K Stage-1 connector and released multistep DiT, **not**
by continuing the old reconstruction-150K checkpoint.

The following plan/settings are available in the locally cached 130K
checkpoint's training contract. They describe the planned run, not an
independent audit of the final remote checkpoint's weight hash.

| Item | Value |
| --- | --- |
| Data | BLIP3o image-caption pairs + expanded Vchitect T2V DataVerse; public substitutes, not the original UniVideo dataset |
| Training images | 984,636 |
| Training videos | 529,902 |
| Global media batch | 16, from eight GPUs x accumulation two |
| Steps per media epoch | 94,659 |
| Planned final step | 283,977 |
| Media passes | Three |
| Image presentations | 2,953,908, each paired with reconstruction |
| Video presentations | 1,589,706 |
| Presentations per video mode | 529,902; each video appears once in each mode |
| Video modes | T2V, image-to-VLM reference, VAE first-frame anchor |
| Trainable components | Connector and DiT |
| Frozen components | SmolVLM2 and VAE; no LLM LoRA in this run |
| Objective | Pyramidal flow-matching MSE; balanced image/video losses plus paired reconstruction weight 0.1 |
| Learning rate | 2e-5, constant after 50 warmup steps |
| History | Ground-truth previous latent units during training; generated history during inference |

**Three media epochs are not three T2V epochs.** Each video appears once as T2V,
once as image-to-VLM reference, and once as VAE-anchor conditioning. Each media
presentation samples a training unit and pyramid stage; it is not a full
denoising rollout through every video frame/stage.

Videos are sampled uniformly across their whole source clip into 49 frames,
rather than preserving native FPS. Caption correctness, motion coverage, and
the consequences of time normalization remain data-audit questions. Decode
checks do not prove that captions accurately describe the visual content.

### 6.1 Older Stage-2 Reports

All these systems generate frame zero without DreamLite/SSD1B. The older runs
used one video per prompt and a different seed plan; the new run uses five.
The data and task mixture also changed. This is a historical comparison, not a
controlled learning curve or an isolated test of more training steps.

| Run | Videos/prompt | QuickSR | Quality | Semantic | Total |
| --- | ---: | --- | ---: | ---: | ---: |
| Old Stage 2, 100K | 1 | Without | 77.33 | 46.53 | 71.17 |
| Old Stage 2 + reconstruction, 150K | 1 | Without | 77.99 | 49.67 | 72.33 |
| New Stage 2, three media epochs | 5 | Without | 79.83 | 60.32 | 75.93 |
| New Stage 2, three media epochs | 5 | With | 79.77 | 62.00 | 76.22 |

The older 100K model is from run `17519025`; the 150K reconstruction continuation
is from run `17558065`. They must not be confused with the 100K/190K archives of
the new run `17641364`.

Selected raw-score changes from old reconstruction-150K to new three-epoch run:

| Measure | Old 150K | New three-epoch raw | Difference (pp) |
| --- | ---: | ---: | ---: |
| Quality | 77.99 | 79.83 | +1.84 |
| Semantic | 49.67 | 60.32 | +10.65 |
| Total | 72.33 | 75.93 | +3.60 |
| Object class | 61.87 | 79.08 | +17.21 |
| Multiple objects | 15.93 | 41.62 | +25.69 |
| Human action | 44.00 | 68.20 | +24.20 |
| Scene | 6.47 | 14.91 | +8.44 |
| Dynamic degree | 66.67 | 66.94 | +0.27 |

The displayed differences use the original unrounded old report values where
available, so a few entries differ by 0.01 pp from rounded-column subtraction.

The new recipe has a substantially better reported semantic result. Its main
remaining weaknesses include scene score (14.01 with QuickSR), aesthetic score
(51.56), and multiple-object score (48.20). Dynamic degree is not the primary
source of improvement over the old 150K result.

QuickSR adds 1.68 pp in semantic score and 0.29 pp in total to the new run, while
aggregate quality changes by -0.06 pp. The training-system improvement is
visible before super-resolution.

At the complete-system level, Stage 2 + QuickSR (76.22) trails Joint V2 +
DreamLite (80.91) by 4.69 pp, Joint V2 + SSD1B (81.51) by 5.29 pp, and the matched
native Hybrid (82.85) by 6.63 pp. Those systems have externally generated first
frames and different sampling/conditioning policies, so these gaps do not
isolate the effect of the Stage-2 training objective.

## 7. Why Native 82.85 and Native 83.35 Must Stay Separate

The native-release reproduction is not the native column of the matched swap
experiment. Both use native SSD1B and native Hybrid weights, and **both have a
first-frame anchor**.

| Native protocol | QuickSR | Quality | Semantic | Total |
| --- | --- | ---: | ---: | ---: |
| Matched comparison | Without | 83.85 | 72.56 | 81.59 |
| Matched comparison | With | 84.30 | 77.07 | 82.85 |
| Separate released-CLI reproduction | Without | 84.00 | 73.04 | 81.81 |
| Separate released-CLI reproduction | With | 84.54 | 78.59 | 83.35 |

Confirmed implementation differences:

| Item | Matched comparison | Separate released-CLI reproduction |
| --- | --- | --- |
| Prompt modifier | `, cinematic, realistic textures, high detail, natural colours` | `, hyper quality, Ultra HD, 8K` |
| RNG policy | Reset independently before image and video generation | Reset once before the complete SSD1B -> Hybrid call |
| Entry point | Explicit image generation, then video helper with saved PNG | Released Hybrid pipeline call |

These differences make the two runs distinct experimental conditions. Which
factor explains how much of the 0.50 pp total gap has not been isolated.
The gap is not evidence that our training damaged the native model: neither
native run uses our fine-tuned DiT.

Do not subtract 0.50 pp, or any per-dimension offset, from other model scores to
claim a corrected paper-equivalent result. Report measured values and their
protocols separately. Neither this note nor the reproduction code claims exact
equivalence to the paper's quantized mobile implementation.

## 8. Conclusions and Open Questions

**Supported by the supplied measurements:**

- First-frame-stack choice is a substantial contributor to semantic and imaging scores in the anchored systems.
- Better aggregate semantic score with SSD1B does not eliminate the video-stack dynamic-degree gap.
- The new Stage-2 recipe has better reported scores than older Stage-2 runs; the improvement is visible without QuickSR.
- QuickSR/resolution changes materially affect some semantic measurements and must be kept consistent in comparisons.

**Not established by these measurements:**

- That the image bridge alone causes the full semantic gap; the image generator also changes.
- That first-frame anchors are useless; none of the four cross-swap settings removes the anchor.
- That Joint V2 fine-tuning alone causes weaker motion; a matched pre/post monolithic comparison is missing.
- That additional steps alone, adding LoRA alone, or a particular loss change will close the gap.
- That these GPU VBench runs establish mobile latency, unified understanding improvement, or exact paper reproduction.

The next checkpoint comparisons should keep the new run, prompt/seed plan,
inference code, and raw/QuickSR evaluation fixed. They can determine whether the
new run improves, plateaus, or degrades across training. They cannot, by
themselves, separate data quality from task-mixture effects.

## 9. Pending Checkpoint Comparisons

Exact media-epoch boundaries are 94,659; 189,318; and 283,977 steps. Archiving was
configured every 10K steps, so the chosen intermediate checkpoints are slightly
after epoch boundaries.

| Checkpoint of run 17641364 | Approximate media epochs | Measurement status |
| --- | ---: | --- |
| `stage2_dit_connector_step100000.pt` | 1.056 | Evaluation scripts prepared; no score supplied yet |
| `stage2_dit_connector_step190000.pt` | 2.007 | Evaluation scripts prepared; no score supplied yet |
| `stage2_dit_connector_step283977.pt` / final `latest` | 3.000 | Remote full VBench report supplied above |
| Locally cached `stage2_dit_connector_latest.pt` | 1.373, step 130K | Intermediate only; not the final three-epoch weight |

The new epoch jobs verify the payload step and training contract, compare their
inference settings with the completed epoch-3 report, and write
`epoch_comparison.md` for raw and QuickSR. They do not regenerate epoch 3.

Local jobs `6323`, `6324`, `6325`, and `6408` are functional smoke tests of
generation/export/resume and selected metric paths. They are not full VBench
results for 100K, 190K, or the final three-epoch model. In particular, the
Stage-2 local smoke weights were the 130K checkpoint.

## Appendix A. All Matched and Stage-2 Dimensions

Column abbreviations refer to the five systems in Section 3.1. All rows are
dimension percentages. Aggregate scores are in Section 4.

### A.1 With QuickSR

| Dimension (%) | Joint/DreamLite | Joint/SSD1B | Native/SSD1B | Native/DreamLite | Stage 2, 3 epochs |
| --- | ---: | ---: | ---: | ---: | ---: |
| subject_consistency | 96.47 | 96.88 | 95.02 | 94.18 | 91.88 |
| background_consistency | 95.02 | 95.92 | 95.85 | 95.24 | 94.23 |
| motion_smoothness | 99.31 | 99.23 | 99.05 | 99.08 | 98.70 |
| dynamic_degree | 64.72 | 45.56 | 75.83 | 86.11 | 64.72 |
| aesthetic_quality | 61.69 | 62.16 | 62.24 | 61.90 | 51.56 |
| imaging_quality | 59.67 | 65.01 | 63.86 | 58.76 | 59.25 |
| temporal_flickering | 99.47 | 99.42 | 99.18 | 99.34 | 98.62 |
| object_class | 90.09 | 93.15 | 92.90 | 90.43 | 78.09 |
| multiple_objects | 65.34 | 64.80 | 64.27 | 65.32 | 48.20 |
| color | 91.52 | 88.56 | 87.38 | 92.70 | 93.20 |
| spatial_relationship | 61.27 | 70.60 | 71.65 | 61.73 | 58.26 |
| scene | 45.10 | 50.17 | 50.84 | 46.41 | 14.01 |
| temporal_style | 22.67 | 24.86 | 24.75 | 23.08 | 20.70 |
| overall_consistency | 26.72 | 28.08 | 28.04 | 26.80 | 22.72 |
| human_action | 74.80 | 87.40 | 91.20 | 79.20 | 69.20 |
| appearance_style | 22.08 | 22.95 | 22.67 | 21.84 | 21.35 |

### A.2 Without QuickSR

| Dimension (%) | Joint/DreamLite | Joint/SSD1B | Native/SSD1B | Native/DreamLite | Stage 2, 3 epochs |
| --- | ---: | ---: | ---: | ---: | ---: |
| subject_consistency | 96.66 | 97.05 | 95.31 | 94.59 | 92.17 |
| background_consistency | 95.56 | 95.94 | 95.78 | 95.66 | 94.38 |
| motion_smoothness | 99.36 | 99.30 | 99.14 | 99.15 | 98.84 |
| dynamic_degree | 65.00 | 46.11 | 76.39 | 87.22 | 66.94 |
| aesthetic_quality | 60.04 | 60.32 | 60.34 | 60.17 | 51.07 |
| imaging_quality | 57.81 | 63.41 | 61.84 | 56.48 | 57.65 |
| temporal_flickering | 99.50 | 99.49 | 99.25 | 99.38 | 98.75 |
| object_class | 89.43 | 92.28 | 92.26 | 89.51 | 79.08 |
| multiple_objects | 57.24 | 53.70 | 53.87 | 56.48 | 41.62 |
| color | 90.99 | 87.52 | 87.94 | 91.98 | 92.78 |
| spatial_relationship | 40.75 | 40.48 | 40.40 | 39.83 | 48.27 |
| scene | 46.25 | 50.97 | 51.96 | 47.56 | 14.91 |
| temporal_style | 22.88 | 25.10 | 25.06 | 23.35 | 20.87 |
| overall_consistency | 26.72 | 28.12 | 28.05 | 26.89 | 22.79 |
| human_action | 76.00 | 86.00 | 91.00 | 80.60 | 68.20 |
| appearance_style | 21.95 | 22.78 | 22.42 | 21.58 | 21.39 |

## Appendix B. Reproducibility References

The source score reports remain on Berzelius under the output roots in Section
2. For a full independent audit, retain their `run_contract.json`,
`generation_summary.json`, `quicksr_ablation.json`, `score_assets.json`,
`checkpoint_audit.json` where present, and per-video metric records. Preserve the
original baseline PNGs and `anchor_records` for future first-frame comparisons.
Markdown alone does not establish checkpoint/file hashes or statistical
significance.

Relevant code, with original final-epoch inference intentionally unchanged by
the intermediate-checkpoint wrapper:

- [Matched raw/QuickSR driver](../tools/benchmark_joint_v2_quicksr_ablation.py)
- [First-frame swap driver](../tools/benchmark_first_frame_swap.py)
- [Separate native-release reproduction](../tools/reproduce_native_neodragon_vbench.py)
- [Final Stage-2 benchmark](../tools/benchmark_mobileov_stage2_media3epochs_dual.py)
- [Intermediate Stage-2 checkpoint benchmark](../tools/benchmark_mobileov_stage2_epoch_dual.py)
- [Media-epoch sampling contract](../new_mobile_ov/training/stage2_media_epochs.py)
- [Frozen Stage-1/2 SmolVLM2 encoder](../new_mobile_ov/bridge/stage1_conditioner.py)
