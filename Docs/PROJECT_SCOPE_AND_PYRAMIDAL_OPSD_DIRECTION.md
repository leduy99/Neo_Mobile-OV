# Mobile-OV Project Scope and Pyramidal OPSD Direction

**Last updated:** 2026-09-03  
**Status:** research decision note  
**Purpose:** preserve the reasoning behind the experiments and prevent distinct
research problems from being collapsed into one training objective.

Evidence language in this note is deliberate:

- **Measured** means a value exists in a saved log or evaluation artifact.
- **Observed** means a controlled visual comparison supports the statement.
- **Inferred** means the explanation fits the evidence but has not been isolated
  by one causal ablation.
- **Proposed** means the method has not yet demonstrated an end-to-end gain.

## 1. Central diagnosis

Mobile-OV is not currently one model-training problem. It is a stack of at
least seven coupled research problems:

1. replacing DreamLite's native Qwen3-VL condition with a compact SmolVLM2
   image bridge;
2. replacing NeoDragon's native text condition with a compact SmolVLM2 video
   bridge;
3. adapting a monolithic video DiT to the new condition without losing its
   pretrained semantics, quality, and motion;
4. distilling a multistep monolithic model into NeoDragon's one-step Pyramidal
   Hybrid policy;
5. correcting causal errors caused by generated video history;
6. building aligned image, video, first-frame, caption, and instruction data;
7. preserving the latency, memory, precision, and quantization constraints of
   an on-device system.

Each problem can fail independently and hide progress in the others. A valid
bridge can be paired with a damaged DiT. A good video DiT can preserve a wrong
first-frame identity. A DMD student can produce sharp frames while losing
motion. A lower training loss can therefore coexist with a worse deployed
pipeline.

The project has often tried to improve several of these components in the same
run. That made some experiments useful as system stress tests, but difficult to
interpret as clean research evidence.

## 2. Why this is broader than Mobile-O

Mobile-O primarily co-adapts one MCP and one image DiT before applying curated
SFT and multimodal training. Its reported recipe starts with approximately 9M
text-image pairs, followed by roughly 105K curated prompt-image pairs and 105K
multimodal quadruplets. The image DiT is allowed to learn the MCP's condition
space jointly.

Mobile-OV adds several constraints that Mobile-O does not need to solve at the
same time:

- two generation branches and two generator latent spaces;
- a generated first frame that causally anchors the entire video;
- an autoregressive video generator with multiscale latent history;
- a pruned and step-distilled `1-1-1` Hybrid policy;
- monolithic-to-Hybrid Pyramidal DMD;
- separate image and video condition contracts;
- eventual image/video understanding and generation SFT;
- a requirement to remove the native generation encoders, not merely add a new
  interface beside them.

Mobile-O is evidence that joint MCP-generator learning can work. It is not
evidence that caption-only representation matching can replace the native
condition of an already frozen, compressed image or video generator.

## 3. Experimental ledger and what it established

| Track | What was tested | Strongest conclusion | Unresolved problem |
| --- | --- | --- | --- |
| DreamLite image bridge V3-V12 | Representation, same-state functional, rollout, real-image grounding, data mixtures, and controlled continuations | The bridge can satisfy the tensor and local response contract, but more steps or grounded samples under the same objective do not guarantee better images | Native prompt semantics, composition, and first-frame identity are not fully preserved |
| NeoDragon video bridge | Frozen-Hybrid representation and functional distillation, longer training, and full-rollout variants | Exp1-64K is a useful compact condition baseline; simply extending training did not close the native gap | The bridge condition remains less semantically separated than the native condition |
| Hybrid and monolithic joint fine-tuning | Generic flow matching, preservation, local response distillation, corrected Pyramidal Flow, fixed-bridge and jointly trained variants | A decreasing local objective does not prove that a compressed autoregressive policy remains good in closed-loop generation | How to learn new semantics before compression without damaging native motion |
| Pyramidal DMD V1-V3 | Synthetic multistep trajectories, one-step stage targets, fake-model distribution matching, all-unit corrections, and rollout-aware variants | Local one-step endpoints and sharp frames are learnable, but self-history can still collapse motion or generation | Stable own-history credit assignment across all Pyramidal calls |
| OPSD-Neo Exp7 | Trainable Hybrid and bridge, frozen references, student-generated history, and privileged clean history | The temporal training path is executable and slightly improves local native-trajectory agreement | It did not improve semantic separation and slightly reduced the measured motion proxy |
| Data and SFT | OpenVid latents/captions, image grounding cascades, short/medium/long captions, and proposed unified records | Data quality and contract alignment matter more than raw row count | No single release yet supervises anchor semantics, video dynamics, both teacher conditions, and unified instructions together |

### 3.1 Image bridge evidence

DreamLite V11-balanced remains the strongest demonstrated compact image-bridge
baseline in the standard four-video VBench protocol (job `17295023`): quality
`0.8151`, semantic `0.6915`, and total `0.7904`. The published NeoDragon result
`0.8161` is a contextual target rather than a matched component swap.

The later V12 grounding run (job `17420997`) reported quality `0.8091`, semantic
`0.6701`, and total `0.7813` under the one-video protocol. Continuing V11 with
V12 grounded data for 20K steps (job `17421568`) reported quality `0.8135`,
semantic `0.6653`, and total `0.7839`. These runs did not beat V11. The result
does not prove that grounded data is useless; it proves that adding more
grounded data and training steps to the current objective is not, by itself,
the missing solution.

A controlled native-Qwen versus bridge comparison localized a substantial
residual semantic penalty to the image condition. This matters twice: the
DreamLite image is an output, and the same image becomes the causal first-frame
anchor for NeoDragon. Video motion cannot repair an object, count, relation, or
scene that is already wrong in frame one.

### 3.2 Video bridge evidence

Exp1 demonstrated that a small SmolVLM2 bridge can control the released frozen
Hybrid well enough to generate coherent video. Its success depended on the
correct post-adapter teacher target, multiple representation losses,
same-state functional supervision, FP32 master weights, and controlled
short/medium/long prompts.

Longer training was not automatically better. The nominal 200K continuation
and later rollout variants did not reliably exceed Exp1-64K. One diagnostic
shows pooled prompt-to-prompt off-diagonal cosine around `0.917` for the bridge
versus `0.375` for the native condition. The compact condition therefore keeps
different prompts much closer together than the native condition does.

This does not mean raw embedding equality is the only goal. It means a teacher
must explicitly supervise whether prompt differences still produce the right
generator response.

### 3.3 Joint fine-tuning evidence

Multiple runs updated a model that was already pruned and/or specialized for a
few-step deployment schedule. Flow matching and local response losses could
decrease while generated motion, semantics, or native visual behavior became
worse. Exp5 was especially useful because a good Exp1 bridge survived while
the updated DiT degraded, localizing the main failure to DiT adaptation.

The corrected monolithic Pyramidal Flow recipe later produced valid videos,
but longer joint variants did not establish a consistent improvement over the
fixed-bridge baseline. The resulting research lesson is not that flow matching
is invalid. It is that a compressed deployment policy is a poor place to ask
the model to learn a large new semantic interface. Capability learning should
ideally happen in a sufficiently expressive foundation model, followed by
contract-preserving pruning and own-history distillation.

### 3.4 Pyramidal DMD evidence

The corrected native text-to-video DMD topology contains seven generated
latent units and three stages, or 21 student calls. An external-anchor Hybrid
deployment instead treats frame one as given and generates six video units,
or 18 calls. These protocols must not be mixed silently.

The DMD experiments showed that good local targets, stable optimization, and
sharp decoded frames do not guarantee a good causal rollout. Teacher-forced
history preserved substantially more motion than student self-history, which
localized the main failure to accumulated history error rather than simple
condition incompatibility. A bridge being callable by a DMD checkpoint is not
evidence that the DMD video quality is acceptable.

### 3.5 OPSD-Neo Exp7 evidence

Exp7 trained both the released Hybrid DiT and the Exp1 bridge for 20K updates.
The bridge was frozen for the first 1K updates, ramped for 2K, and then trained
at a small learning rate. The frozen teacher used a frozen Exp1 bridge and a
frozen released Hybrid, while privileged OpenVid history replaced older
student-generated history where that replacement was meaningful.

In the controlled six-prompt evaluation, full OPSD changed the optical-flow
proxy from `0.4661` to `0.4294`, sharpness from `183.40` to `184.15`, and CLIP
text score from `0.3355` to `0.3351`. Native-flow cosine improved slightly from
`0.9311` to `0.9335`, and native-endpoint cosine from `0.9289` to `0.9317`.
Prompt-separation cosine remained effectively unchanged (`0.9173` versus
`0.9171`).

The correct interpretation is narrow: Exp7 slightly improved local trajectory
agreement and preserved sharpness, but did not improve semantics and slightly
reduced measured motion. The bridge received gradients, but the teacher used
the old bridge too. It could preserve or adapt that condition; it could not
teach the bridge information that only the native NeoDragon text condition
contained.

## 4. The objective-separation rule

Two teacher concepts must be kept distinct:

- **Native Hybrid teacher** means the frozen released Hybrid DiT weights.
- **Native condition teacher** means NeoDragon's native text-conditioning
  output for the same prompt.

Changing condition and history in one teacher target mixes semantic error with
temporal error. A lower combined loss would not reveal which problem improved.
The next method should therefore separate the objectives.

### 4.1 Semantic functional distillation for the bridge

Use the same student-visited noisy state, timestep, Pyramidal stage, unit, and
generated history in both branches. Change only the condition:

```text
teacher = FrozenHybrid(x_student, native_condition, generated_history)
student = FrozenHybrid(x_student, bridge_condition, generated_history)
```

Freeze the Hybrid and update only the bridge. The main target is the Hybrid
velocity or transition response, with representation losses as auxiliaries.
This is same-state semantic functional distillation. Calling it OPSD-V would be
misleading because its purpose is encoder replacement, not history correction.

### 4.2 Temporal OPSD for the DiT

Use the same detached MCP condition in both branches. Change only the history
and model weights:

```text
teacher = FrozenNativeHybrid(x_student, bridge_condition, clean_history)
student = TrainableHybrid(x_student, bridge_condition, generated_history)
```

Freeze or detach the bridge and update only the DiT. The teacher must evaluate
the student's own noisy state and exact deployed timestep. This objective asks
one question only: can the trainable Hybrid behave like the released Hybrid
would behave if its old causal context were cleaner?

The frozen released Hybrid is a suitable preservation and temporal-correction
teacher. It is unlikely to teach quality beyond its own capability ceiling. A
stronger monolithic or higher-step teacher is required for a claim that exceeds
native Hybrid quality.

### 4.3 Optional joint consolidation

Joint bridge-DiT training should happen only after the isolated bridge and DiT
stages pass their own generation gates. It should use small learning rates and
retain both teacher axes. Joint training is a consolidation stage, not the
place where an unverified bridge and an unstable DiT are expected to repair
each other.

## 5. Candidate focused contribution: Pyramidal OPSD

The complete unified edge model remains the system vision, but it is too broad
to serve as one clean experimental claim. A narrower potential contribution is
stage-aware OPSD for a Pyramidal autoregressive generator.

The research question is:

> Can privileged clean latent history teach a one-step Pyramidal Hybrid to
> resist self-history drift while preserving its released semantics, motion,
> and mobile inference contract?

A defensible method would include:

1. exact student on-policy states rather than teacher trajectory states;
2. exact `1-1-1` deployed stage timesteps;
3. unit- and stage-balanced training across all 18 external-anchor calls;
4. clean-history replacement only where older causal history exists;
5. frozen-Hybrid preservation for early calls without useful privileged
   history;
6. teacher-advantage gating so clean history is used only when it provides a
   better local target;
7. separate semantic bridge supervision rather than asking temporal OPSD to
   replace the text encoder;
8. evaluation with real and generated first-frame anchors;
9. no inference-time module or additional denoising call.

This direction is still a proposal, not a validated contribution. Exp7 proves
that the adapted training path runs; it does not yet prove a quality or motion
gain. A paper claim becomes credible only if the stage-aware method improves
closed-loop motion or long-horizon consistency without degrading prompt
semantics and native visual quality.

## 6. Data and SFT roles

The data problem should also be decomposed instead of solved by one enormous
mixture.

### Bridge data

Bridge learning needs diverse prompts and paired teacher conditions. Hard
groups should cover object identity, multiple objects, count, color, spatial
relations, scene, style, and action. Real images are useful when they provide a
generator-level target, but adding image rows without a stronger teacher
objective is not sufficient.

### Temporal OPSD data

Temporal training needs videos long enough to expose history drift, accurate
motion captions, clean latent histories, and deployment-like generated
histories. The existing 49-frame OpenVid samples provide only a small number of
useful later units. They are sufficient for implementation and early ablation,
but weaker than the long-video setting that motivates OPSD-V.

### Anchor-distribution data

Deployment uses a DreamLite-generated first frame, while much training and
evaluation has used real OpenVid or native SSD1B anchors. Training should mix
verified real anchors with DreamLite-generated anchors so the video model sees
the errors it will receive in deployment. Anchor mismatch is a secondary
problem relative to incorrect bridge and DiT objectives, but it must be
controlled in final evaluation.

### Unified SFT data

The eventual unified record should connect:

```text
video
+ verified first frame
+ static first-frame caption
+ dynamic full-video caption
+ native DreamLite teacher condition
+ native NeoDragon teacher condition
+ understanding/generation instruction
```

SFT should be applied after the image condition, video condition, and video
generator each pass independent quality gates. SFT can organize capabilities;
it cannot reliably repair a collapsed DMD policy or recover semantic
information that a bridge never learned.

## 7. Scope decision

Trying to solve image-bridge replacement, video-bridge replacement,
monolithic capability learning, pruning, DMD, temporal OPSD, unified data, SFT,
and deployment in one contribution creates too many uncontrolled variables.
The full system should remain the long-term goal, but the research claim should
focus on one bottleneck.

Pyramidal OPSD is a plausible focused contribution because it addresses a
specific failure observed repeatedly: local one-step agreement can coexist
with global self-history motion collapse. It also has a clear baseline,
mechanistic ablations, and a no-overhead deployment goal. If it cannot beat the
released-Hybrid preservation baseline under controlled anchors and seeds, the
direction should be stopped rather than hidden inside a larger joint run.

A rigorous solution to this one bottleneck can be more valuable than a broad
unified demo in which every component is partially reliable. The unified
Mobile-OV system can then be presented as the motivating application and
future integration path.

## 8. Required evidence gates

Future runs should be promoted only after passing all relevant gates:

1. **Bridge gate:** native-condition versus bridge-condition response parity on
   the same student states, plus held-out semantic generation.
2. **Temporal gate:** better self-history motion/consistency than Exp7 without
   worse prompt alignment or sharpness.
3. **Stage gate:** no unit or Pyramidal stage silently regresses while aggregate
   loss improves.
4. **Anchor gate:** conclusions hold under the same real anchor and under the
   same DreamLite-generated anchor.
5. **Closed-loop gate:** decoded rollout and VBench improve; local velocity,
   endpoint, or training loss alone is insufficient.
6. **Deployment gate:** the method retains the Hybrid call count and adds no
   required inference-time teacher or module.

## 9. Evidence map

The detailed implementation and historical evidence remain in:

- [DreamLite image-bridge training](TRAINING_IMAGE_BRIDGE_DREAMLITE.md)
- [DreamLite V7-V9 and controlled ablations](DREAMLITE_IMAGE_BRIDGE_V7_TO_V9.md)
- [DreamLite V10 postmortem and V11 design](DREAMLITE_V10_POSTMORTEM_AND_V11.md)
- [Video bridge and joint-training experiments](NEODRAGON_EXPERIMENTS_AND_EXP5.md)
- [Full-rollout bridge distillation](NEODRAGON_FULL_ROLLOUT_DISTILLATION.md)
- [Exp6 decision matrix](EXP6_DECISION_MATRIX_96_PROMPT_ABLATION.md)
- [Pyramidal Flow training audit](NEODRAGON_FLOW_MATCHING_AUDIT_20260821.md)
- [Pyramidal DMD reproduction](NEODRAGON_PYRAMIDAL_DMD_REPRODUCTION.md)
- [OPSD-Neo Exp7](OPSD_NEO_JOINT_FINE_TUNING.md)
- [Consolidated research report](consolidated_research_report/README.md)
- [A* research review slides](slides/MOBILE_OV_ASTAR_RESEARCH_REVIEW.pdf)

The six-prompt OPSD measurements are stored under
`output/_archive/pre_monolithic_cfg_ablation_20260819/neo_opsd_2x2_ablation_20260809/summary.json`.
The full OPSD checkpoint evaluation is stored under
`output/_archive/pre_monolithic_cfg_ablation_20260819/neo_opsd_joint_step20k_eval_20260809/summary.json`.
