# NeoDragon DMD-V2 External-Anchor Alternative

## Question

DMD-V1 produced acceptable individual frames but weak motion. DMD-V2 trained
all seven native units and degraded the first frame, structure, semantics, and
motion. The two results were not directly comparable because V1 was evaluated
after a strong external first-frame anchor while V2 generated unit zero itself.

This experiment isolates that difference:

> Does DMD-V2 recover V1-like visual quality when unit zero is removed from
> optimization and deployment again starts from an external first frame?

## Controlled DMD Change

The alternative keeps the V2 teacher trajectories, model initialization,
optimizer, learning rates, student-to-fake update ratio, DMD loss, Cauchy loss,
CFG for video units, global batch, and 10k-update budget. It changes only the
unit protocol:

```text
DMD-V2:      optimize units 0..6 x stages 0..2
DMD-V2-alt:  optimize units 1..6 x stages 0..2
```

During training, stored multistep teacher unit zero remains causal history for
later units. At deployment, unit zero is replaced by an SSD1B, DreamLite, or
source-image anchor. The existing 188k trajectories therefore remain usable;
this experiment does not claim that they were generated from external anchors.

The checkpoint schedule is deliberately distinct:

```text
pyramidal_1-1-1_external_anchor_video_units
```

It cannot be resumed as V1 legacy or V2 all-native training.

## Bridge Redesign

The bridge remains fully separate from DMD training. It keeps the successful
Exp1 representation losses against the native multistep text/context stack,
and its functional loss is evaluated by the frozen released **monolithic
multistep DiT**. Functional sampling cycles only units `1..6`.

```text
SmolVLM2 bridge condition ----+
                              +--> frozen monolithic DiT response match
native multistep condition ---+
```

No DMD checkpoint is loaded or updated by the bridge job. This preserves the
intended procedure: first learn the monolithic conditioning interface, then
independently distill the monolithic DiT into a fast student, and finally test
whether that unchanged interface transfers to DMD-V2-alt.

## Commands

Train DMD-V2-alt on Berzelius:

```bash
sbatch scripts/reproduce_neodragon_pyramidal_dmd_v2_anchor_alt_1node8gpu.sbatch
```

Train the bridge from scratch. This job is independent and can run in parallel
with DMD-V2-alt:

```bash
sbatch scripts/train_neodragon_monolithic_video_units_text_bridge_1node8gpu.sbatch
```

## Decision Rule

The final comparison must reuse the same external first frame, native text
condition, random noise, transition noise, prompt, and six-unit rollout for
V1 and DMD-V2-alt.

- If DMD-V2-alt returns to V1 quality, unit-zero optimization/interference was
  the main reason V2 looked globally worse.
- If DMD-V2-alt remains worse, the all-unit inference protocol was not the
  complete explanation and the remaining DMD stage/history objective must be
  corrected before further scaling.
- The bridge is evaluated only after native-condition DMD-V2-alt passes the
  quality gate. A bridge cannot repair a poor DMD student.

This experiment intentionally does not change sigma construction or rebalance
DMD versus Cauchy gradients. Those are separate ablations; changing them here
would prevent attribution to unit-zero removal.
