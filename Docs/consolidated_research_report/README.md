# Mobile-OV Consolidated Research Report

**Coverage:** experiments and measurements completed through 2026-07-30

**Repository:** `New-Mobile-OV`

**Status:** research record, not a claim that the current system is production-ready

This folder consolidates the Mobile-OV work that had previously been distributed
across experiment notes, logs, evaluation outputs, and design discussions. It
does not simply reproduce the older documents. It reorganizes the evidence by
research question, adds the later experiments, and separates measured results
from interpretations and proposed next steps.

## Reading Guide

| Part | Main question |
| --- | --- |
| [Part 1: System, Data, and Methods](PART_1_SYSTEM_DATA_AND_METHODS.md) | What is the system, what data was prepared, and how were experiments controlled? |
| [Part 2: Video Bridge and DiT Experiments](PART_2_VIDEO_BRIDGE_AND_DIT_EXPERIMENTS.md) | What happened in Exp1-Exp6, and why did several apparently reasonable objectives fail? |
| [Part 3: Image Bridge and End-to-End Results](PART_3_IMAGE_BRIDGE_AND_END_TO_END.md) | Can SmolVLM2 replace SSD1B's dual CLIP stack, and can the image and video branches work together? |
| [Part 4: Model, Speed, Memory, and Quality Measurements](PART_4_MEASUREMENTS_AND_DEPLOYMENT.md) | How large and fast are the components, and what do the measured diagnostics actually mean? |
| [Part 5: Conclusions and Roadmap](PART_5_CONCLUSIONS_AND_ROADMAP.md) | Which findings are confirmed, what should not be repeated, and what is the lowest-risk next direction? |
| [Appendix: Evidence Index](APPENDIX_EVIDENCE_INDEX.md) | Which script, checkpoint family, metric file, and visual artifact supports each result? |

## Executive Result

The strongest verified design principle is to preserve the released NeoDragon
Hybrid generator and learn a compatible condition interface around it. Exp1
showed that a 5.56M-parameter Mobile-OV video bridge can produce semantically
valid videos when trained with full representation losses and frozen-DiT
functional supervision. The old 200K bridge did not fail because it saw too few
optimizer steps; it had weaker supervision and a smaller effective sample
exposure. Its later BF16 continuation was also nearly a no-op.

The negative result is equally important. Exp2, Exp3, Exp4, and Exp5 showed that
ordinary flow matching and local response matching can damage the released
Hybrid DiT even when training loss decreases. NeoDragon Hybrid is already
pruned and step-distilled for a causal `1-1-1` schedule. It is not equivalent to
a monolithic flow model that can safely be fine-tuned with a generic
teacher-forced flow objective.

Full 18-call rollout distillation improved the training objective but did not
close the motion gap to native NeoDragon. At 100K, its corrected optical-flow
proxy was `0.2788` pixels at `256x160`, compared with `0.5457` for native
NeoDragon. The 80K and 100K checkpoints were nearly tied. This says that
full-rollout supervision is more faithful than one-call supervision, but it is
not sufficient on its own.

The SSD1B Image Bridge reached the correct tensor contract and good average
frozen-UNet parity with only 11.15M trainable parameters. It nevertheless
compressed prompt diversity and failed important prompts such as `red panda`.
V2 improved average UNet parity but did not consistently improve semantic image
quality. The native dual CLIP encoders therefore remain the quality baseline.

The current practical baseline is:

```text
understanding:
  SmolVLM2

image anchor:
  native SSD1B dual CLIP + SSD1B generator

video condition:
  Exp1 bridge, selected by controlled inference

video generation:
  released frozen NeoDragon Hybrid DiT + video VAE
```

## Evidence Labels

The report uses the following labels implicitly:

- **Measured:** directly read from saved metrics, checkpoints, or benchmark
  logs.
- **Observed:** qualitative conclusion from controlled contact sheets or
  videos with shared seeds and anchors.
- **Inferred:** a mechanism consistent with measurements but not isolated by a
  fully controlled experiment.
- **Proposed:** a future experiment or architecture direction.

This distinction matters. For example, a higher optical-flow magnitude is a
measured motion proxy, not proof of better motion quality. A lower training
loss is not evidence of better video generation unless it survives controlled
inference.

## Primary Source Documents

The detailed source notes remain useful when implementation-level context is
needed:

- `Docs/NEODRAGON_EXPERIMENTS_AND_EXP5.md`
- `Docs/NEODRAGON_EXP2_EXP3_EXP4_FAILURE_POSTMORTEM.md`
- `Docs/NEODRAGON_FULL_ROLLOUT_DISTILLATION.md`
- `Docs/NEODRAGON_HYBRID_RECOVERY_EXP6.md`
- `Docs/SSD1B_IMAGE_BRIDGE_DISTILLATION.md`
- `Docs/SSD1B_IMAGE_BRIDGE_100K_EVALUATION_AND_V2.md`
- `Docs/IMAGE_BRIDGE_V1_V2_AND_EXP6_RESULTS.md`
- `Docs/DEVELOPMENT_NOTE.md`

The consolidated report should be used for orientation and decision-making.
The source documents should be used for exact command-line and implementation
details.
