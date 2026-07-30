# Part 5 - Conclusions, Decisions, and Roadmap

## 1. Evidence hierarchy

The project has produced many checkpoints, but they do not carry equal evidential value.

### Strong evidence

- Native Hybrid versus Monolithic benchmark under the same local protocol.
- Exp1 64K versus Exp1 200K and rollout continuations using controlled prompts.
- Exp2/3/4 old-versus-final comparisons showing no clear recovery with more training.
- Image Bridge V1/V2 controlled prompt and modifier ablations.
- Stage/unit condition analysis and fixed-anchor video comparisons.

### Medium evidence

- Six-prompt qualitative comparisons.
- Latent endpoint, CLIP, sharpness, and optical-flow diagnostics.
- Training-loss trends when interpreted together with generated videos.

### Weak or non-comparable evidence

- Checkpoint file size as a model-size proxy.
- Single-prompt visual impressions.
- Cross-model speed comparisons with different frame counts and resolutions.
- One-video duplicated VBench scores as a public leaderboard result.
- A low local flow-matching loss without preserved Hybrid behavior.

This hierarchy is important because several early decisions looked promising in loss space but failed under generation.

---

## 2. What has been established

### 2.1 The frozen Hybrid NeoDragon generator is valuable

Native NeoDragon Hybrid is fast because the released 1-1-1 schedule runs only 18 DiT calls in the measured pipeline. Its behavior is not equivalent to a normal pretrained flow model that can be freely fine-tuned with a generic flow-matching objective.

### 2.2 Bridge alignment can work

Exp1 64K showed that SmolVLM2 can replace the native video text path while the NeoDragon DiT remains frozen. The successful result depended on:

- Correct teacher targets after the native ContextEmbedder/ContextAdapter path.
- Multiple representation and functional losses.
- FP32 master weights for the small bridge.
- A balanced short/medium/long caption distribution.
- Frequent inference-based validation.

### 2.3 More steps are not automatically better

The old 200K bridge distillation and the BF16 continuation demonstrated that step count cannot compensate for the wrong target, weak optimization, or numerically ineffective updates. Exp1 64K was visibly better because the supervision and update path were better, not because 64K is a universally optimal number.

### 2.4 Generic DiT flow fine-tuning is unsafe

Exp2, Exp3, Exp4, and Exp5 all showed variants of the same failure:

- Flow loss could decrease.
- Some local latent or sharpness metrics could improve.
- Generated semantics, temporal naturalness, or native Hybrid behavior could still degrade.

Exp5 was especially informative because it started from a useful Exp1 bridge yet became worse after DiT training. This isolates DiT adaptation as the main risk rather than bridge initialization alone.

### 2.5 Full rollout distillation is necessary but not sufficient

The 18-call rollout experiment corrected the one-call supervision gap. It showed that matching the complete Hybrid trajectory is technically feasible and improved the trained rollout objective, but it did not reliably exceed Exp1 64K or native NeoDragon in visual quality.

This means the missing information is not only "more calls." The training distribution, sensitivity of specific stages/units, and preservation of the released distilled policy also matter.

### 2.6 Image conditioning remains unresolved

Image Bridge V1 and V2 proved that the output contracts are implementable:

```text
[B, 77, 768]
[B, 77, 1280]
[B, 1280]
```

However, low condition-space or functional loss did not guarantee native SSD1B image quality. V2 improved some trajectory behavior but regressed visibly on difficult prompts. Prompt modifiers sometimes helped, which points to a teacher-distribution mismatch rather than a simple architecture-shape problem.

---

## 3. Failed hypotheses that should not be repeated

| Hypothesis | Evidence against it | Revised rule |
|---|---|---|
| More bridge steps will eventually fix quality | Old 200K and BF16 continuation did not beat Exp1 64K | Validate effective updates and generated outputs |
| Low one-call DiT loss is enough | Motion remained weaker than native | Supervise the deployed rollout, not an isolated call |
| Generic flow matching can safely adapt Hybrid | Exp2/3/4 degraded despite training | Treat Hybrid as a distilled policy |
| A good bridge initialization protects the DiT | Exp5 regressed from Exp1 | Freeze or tightly constrain the DiT |
| Matching CLIP tensors is enough for SSD1B | Image Bridge V1/V2 had prompt-specific failures | Add output-level and distribution-aware validation |
| A smaller checkpoint means a complete smaller model | Image Bridge excludes frozen teachers; Exp5 contains optimizer state | Inventory deployed unique weights explicitly |
| Parameter count predicts generation speed | Hybrid is larger but 4.37x faster than Monolithic | Count executed calls and FLOPs |

---

## 4. Best current baseline

The safest current system is:

```text
Understanding:
  SmolVLM2

Image anchor:
  Native SSD1B dual CLIP conditioning
  Native SSD1B UNet and VAE

Video condition:
  Exp1-style Mobile-OV video bridge

Video generation:
  Frozen released NeoDragon Hybrid 1-1-1 DiT
  Native video VAE
```

For bridge checkpoint selection, Exp1 64K remains the strongest demonstrated reference. The 80K-100K rollout checkpoints should only replace it after a broader held-out evaluation demonstrates a consistent gain.

![Exp1 64K and 200K comparison](assets/exp1_64k_vs_200k.jpg)

![Rollout checkpoints and native reference](assets/rollout_80k_100k_native.jpg)

This baseline does not fully remove duplicated text encoders, but it protects the strongest current image and video behavior while preserving the unified understanding backbone for future capability work.

---

## 5. Recommended next experiment

The next experiment should not be another unrestricted full-DiT flow run. It should be a targeted, stage-aware control experiment around a frozen Hybrid DiT.

### Step 1: measure causal condition sensitivity

For every Hybrid stage and unit:

1. Run native teacher conditioning.
2. Replace only that stage/unit condition with the student condition.
3. Measure endpoint latent deviation, decoded quality, semantics, and motion.
4. Repeat with interpolation between teacher and student conditions.

The result should be a sensitivity map:

```text
stage/unit -> semantic sensitivity
stage/unit -> motion sensitivity
stage/unit -> visual-quality sensitivity
```

This tells us where an adapter can have useful control and where modification is dangerous.

### Step 2: add a minimal zero-initialized control adapter

The adapter should:

- Leave the released Hybrid DiT weights frozen.
- Produce small residual condition corrections.
- Be stage-aware and, only where measurements justify it, unit-aware.
- Start at exact identity through zero initialization.
- Use the existing bridge tokens rather than introduce another large encoder.

A simple form is:

```text
c_control(s, u) = c_exp1 + alpha(s, u) * Delta(c_exp1, task, visual_condition)
```

where `alpha` begins at zero. This is safer than rewriting the DiT because the initial model is exactly the known baseline.

### Step 3: train on the deployed trajectory

The objective should combine:

1. **Native behavior preservation**: keep the student trajectory close to the released Hybrid teacher where no new capability is requested.
2. **Endpoint/trajectory matching**: supervise multiple calls or the complete rollout at sensitive stages.
3. **Text sensitivity**: ensure prompt changes cause appropriate output changes.
4. **Task loss**: add T2V, I2V, or editing supervision only for the relevant task.
5. **Residual regularization**: penalize unnecessarily large control updates.

The adapter should not optimize generic flow matching as its only objective.

---

## 6. Image Bridge redesign

The image bridge remains useful as a research direction, but it should be retrained only after the data and evaluation protocol are corrected.

### Data changes

- Use prompts closer to the SSD1B training/inference distribution.
- Include style, composition, photography, rendering, and quality modifiers.
- Keep short, medium, and long captions, but do not assume OpenVid captions alone represent image-generation prompts.
- Add hard-negative prompt pairs that differ in object, count, color, relation, and action.

### Loss and validation changes

- Preserve representation alignment for both sequence outputs and pooled output.
- Keep multi-call functional/trajectory supervision.
- Add contrastive or retrieval-oriented separation loss.
- Measure effective rank and pairwise prompt separation.
- Validate decoded images at every checkpoint interval.
- Use a fixed difficult prompt suite, including red panda, text rendering, counting, spatial relations, and multi-object composition.

### Acceptance gate

Do not remove native SSD1B CLIPs unless the bridge:

- Preserves prompt retrieval/separation.
- Matches native first-frame semantics across held-out prompts.
- Does not regress hard prompts under the same seed and generation settings.
- Provides a meaningful package-size or runtime-memory benefit after conversion.

![Image Bridge V1 example](assets/image_bridge_v1.jpg)

![Image Bridge V2 example](assets/image_bridge_v2.jpg)

![Red-panda failure analysis](assets/image_bridge_v2_red_panda.jpg)

---

## 7. Capability roadmap

### Phase A: lock the T2V baseline

- Freeze native SSD1B and NeoDragon Hybrid.
- Select the best Exp1-family bridge on a larger held-out prompt set.
- Run a standard full VBench evaluation with genuine samples rather than duplicated outputs.
- Record exact seeds, anchors, schedules, and prompt variants.

### Phase B: controlled I2V

- Add visual-condition tokens from the understanding branch.
- Use the zero-initialized stage-aware adapter.
- Train only the adapter and condition projection first.
- Preserve native text-only T2V through teacher regularization and mixed-task batches.

### Phase C: video and image editing

- Add explicit task tokens and source visual conditions.
- Train edit reconstruction, instruction adherence, and identity/content preservation.
- Reuse the same understanding backbone, but keep task-specific lightweight heads if their contracts differ.

### Phase D: image bridge consolidation

- Revisit SSD1B CLIP removal only after the improved bridge passes its acceptance gate.
- Quantize or cache the native CLIPs as an interim deployment optimization.

### Phase E: mobile conversion

- Export model-only weights.
- Quantize condition encoders first.
- Profile sequential loading versus resident loading.
- Measure package size, peak RAM, first-token/first-frame latency, full-video latency, power, and thermal throttling on the target device.

---

## 8. Evaluation gates for future training

Every future experiment should pass the following gates before it is scaled:

| Gate | Required evidence |
|---|---|
| Numerical | Non-zero effective parameter updates; no BF16 update collapse |
| Contract | Exact tensor shapes, masks, sequence lengths, and dtype |
| Representation | Cosine, norm, token distribution, effective rank, and prompt separation |
| Functional | Teacher/student response under the actual downstream model |
| Rollout | Endpoint and intermediate trajectory under the deployed schedule |
| Visual | Fixed-seed images/videos against native and best previous checkpoint |
| Semantic | Prompt adherence and hard-negative discrimination |
| Temporal | Motion magnitude, smoothness, flicker, and identity preservation |
| Efficiency | Warm latency, peak memory, executed calls, and package size |
| Regression | Native T2V behavior remains acceptable when adding I2V/editing |

Loss curves alone cannot approve a checkpoint.

---

## 9. Open research questions

1. Which Hybrid stages and units are causally responsible for text semantics, motion, and anchor preservation?
2. Can a small stage-aware residual adapter add I2V/editing without modifying the distilled DiT?
3. Does Exp1 plateau because of bridge capacity, teacher ambiguity, or prompt-distribution mismatch?
4. Can SSD1B prompt-distribution data close the image-bridge gap without expanding the architecture?
5. Which understanding-layer mixture best balances semantic diversity and downstream condition compatibility?
6. How much can native dual CLIPs be quantized before first-frame quality drops?
7. Can task-conditioned control share one adapter, or are separate lightweight image/video heads more stable?

These questions are narrower and more testable than another broad full-model training run.

---

## 10. Final recommendation

The project should retain NeoDragon Hybrid as a fast, protected generation policy and build controlled capability modules around it. The experiments do not support replacing its training recipe with ordinary flow matching, nor do they yet support removing SSD1B's native CLIPs.

The most defensible direction is:

```text
shared SmolVLM2 understanding
        |
        +-- validated video bridge
        +-- improved image bridge later
        +-- task/visual control projection
                     |
          zero-initialized stage-aware adapter
                     |
          frozen NeoDragon Hybrid 1-1-1 DiT
```

This direction directly follows the evidence:

- Exp1 proves bridge replacement can work.
- Exp2/3/4 prove unrestricted flow training is risky.
- Exp5 proves good initialization alone does not protect the DiT.
- Rollout training proves schedule-aware supervision is necessary.
- Exp6 shows a short recovery run is not enough to reconstruct the released policy.
- Image Bridge V1/V2 show that shape compatibility and low loss are not equal to generative equivalence.

The next contribution should therefore be precise control of a strong efficient generator, not another attempt to relearn it from scratch.
