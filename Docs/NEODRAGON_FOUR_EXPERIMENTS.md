# NeoDragon Bridge and DiT Training: Four Controlled Experiments

Created: 2026-07-14  
Last updated: 2026-07-16

## 1. Purpose of This Document

This document describes the four training experiments implemented for the
Mobile-OV + NeoDragon branch. It is intended to answer four questions in enough
detail that another engineer can audit, run, or port the implementation:

1. What was changed in the model and training code?
2. What exactly is trained and frozen in each experiment?
3. Why does each loss exist, and where do its gradients go?
4. Why is this four-experiment design safer than directly fine-tuning the pruned
   NeoDragon DiT with a newly initialized bridge?

Experiments 1 and 2 form the conservative sequential path. Experiments 3 and 4
form a controlled from-scratch joint-training comparison:

```text
Experiment 1
  captions only
  -> train Mobile-OV bridge v2
  -> match NeoDragon condition tensors
  -> match the frozen NeoDragon DiT response
  -> save bridge checkpoint

Experiment 2
  Experiment 1 bridge checkpoint + offline OpenVid VAE latents
  -> unfreeze bridge and full NeoDragon DiT
  -> train flow matching on real video latents
  -> retain teacher behavior with two functional distillation paths
  -> save one checkpoint containing bridge + DiT

Experiment 3
  randomly initialized bridge + released pretrained NeoDragon DiT
  -> jointly train bridge and the full DiT from step 1
  -> native flow matching on OpenVid latents
  -> full post-ContextAdapter bridge representation distillation
  -> frozen-teacher bridge functional distillation
  -> cross-system teacher-response distillation
  -> teacher-condition DiT preservation
  -> save one checkpoint containing bridge + DiT

Experiment 4
  same random bridge + same released pretrained NeoDragon DiT
  -> jointly train bridge and the full DiT from step 1
  -> native flow matching only
  -> no teacher text encoder, ContextAdapter, or teacher DiT is loaded
  -> log text-sensitivity diagnostics without backpropagating them
  -> save one checkpoint containing bridge + DiT
```

The current hyperparameters are a carefully chosen starting point, not a claim
that an ablation study has already established a global optimum. The design is
the strongest current compromise between adaptation, teacher preservation,
training cost, and the need to separate gains caused by teacher preservation
from gains caused by ordinary flow matching.

## 2. The Original Compatibility Problem

NeoDragon and Mobile-OV do not naturally produce the same text condition.
Changing only the final tensor width is not enough.

### 2.1 NeoDragon's native conditioning stack

The released NeoDragon `TextEncoderBundle` contains three text encoders:

| Component | Purpose | Output used by DiT |
| --- | --- | --- |
| CLIP text encoder 1 | Global prompt semantics | First half of pooled projection |
| CLIP text encoder 2 | Global prompt semantics | Second half of pooled projection |
| Distilled T5 encoder | Token-level prompt representation | 128 token features before ContextAdapter |

The two CLIP projections are concatenated into:

```text
teacher_pooled: [B, 2048]
```

The distilled T5 path creates a maximum 128-token sequence. NeoDragon's released
`ContextAdapter` is a skip-connected MLP that maps the T5 feature dimension into
the condition dimension consumed by the DiT:

```text
teacher_tokens_before_CA
  -> ContextAdapter
  -> teacher_tokens: [B, 128, 1536]

teacher_mask: [B, 128]
```

The DiT therefore sees the output after the ContextAdapter, not the raw T5
hidden state. This is why the teacher target in both experiments is:

```text
TextEncoderBundle(prompt).token_features
  -> ContextAdapter
  -> final teacher token condition
```

Distilling a tensor before the ContextAdapter would optimize the wrong interface
unless we retained the original ContextAdapter during inference. The new model
is intended to replace the complete NeoDragon text-conditioning stack, so the
student must directly reproduce the post-adapter condition.

### 2.2 Mobile-OV's student conditioning stack

The student starts from the existing Mobile-OV components:

```text
prompt
  -> frozen SmolVLM2-500M
  -> hidden states from multiple language layers
  -> MCP lexical-gated projector
  -> 128 features with width 1536
```

SmolVLM2 and NeoDragon T5 use different tokenizers and different internal
representations. Token position `i` in the Smol sequence is not guaranteed to
mean the same thing as token position `i` in the T5 sequence. A direct linear
projection can match shapes while still producing a poor sequence for DiT
cross-attention.

### 2.3 Why the previous embedding-only result could have low loss but bad video

The old bridge objective mostly optimized token MSE, token cosine, and pooled
MSE. This is useful, but insufficient for four reasons:

1. A small average MSE can hide large errors on a few semantically important
   tokens.
2. Cosine similarity matches direction but ignores feature magnitude.
3. Tokenwise losses assume stronger positional correspondence than two different
   tokenizers can guarantee.
4. The NeoDragon DiT is nonlinear. Two conditions that look close under an
   embedding metric can produce different cross-attention keys/values and very
   different denoising predictions.

The practical symptom is exactly what we observed: a distillation loss that
appeared numerically reasonable did not guarantee useful bridge-conditioned
video generation.

The new design therefore aligns both the representation and the function that
consumes that representation.

## 3. New Shared Bridge v2 Architecture

The architecture is enabled only by:

```text
configs/mobile_ov_neodragon_bridge_v2.yaml
```

The legacy config keeps `neodragon_v2_conditioning: false`, so old inference and
old bridge checkpoints do not silently instantiate new random layers.

The implementation lives in:

```text
new_mobile_ov/bridge/neodragon_text_bridge.py
```

### 3.1 End-to-end tensor contract

| Tensor | Shape | Consumer |
| --- | --- | --- |
| Smol/MCP base tokens | `[B, 128, 1536]` | Sequence translator and pooled head |
| Smol/MCP base mask | `[B, 128]` | Cross-attention and mask-length head |
| Student token condition | `[B, 128, 1536]` | NeoDragon DiT cross-attention |
| Student condition mask | `[B, 128]` | NeoDragon DiT cross-attention mask |
| Student pooled condition | `[B, 2048]` | NeoDragon DiT pooled conditioning path |

### 3.2 Frozen SmolVLM2 backbone

`SanaPromptBridge` loads the converted SmolVLM2 checkpoint and disables gradients
for the backbone. The new `MobileOVNeodragonTextBridge.train()` override also
forces SmolVLM2 to remain in evaluation mode even while the bridge heads train.

This matters because `requires_grad=False` alone does not disable dropout. If a
parent module calls `train()`, a frozen child is normally switched to train mode
too. Stochastic hidden states would make the bridge target change across steps
and ranks. Keeping the backbone in `eval()` gives a stable deterministic input to
the trainable bridge.

### 3.3 MCP lexical-gated projector

The existing `mcp_lexical_gated` projector remains the primary feature mapper.
It fuses multiple SmolVLM2 hidden layers, maps the 960-dimensional Smol features
to 1536 dimensions, and adds a gated lexical path initialized at `0.2`.

Why keep MCP instead of replacing it:

- It is the bridge design that worked best in the previous Mobile-OV branch.
- Multi-layer fusion preserves both early lexical diversity and later semantic
  information.
- The lexical gate gives a controlled path for token identity while the fused
  semantic path learns the teacher space.
- Reusing it reduces the number of simultaneous architecture changes.

### 3.4 `NeoDragonSequenceTranslator`

The new translator is a position-aware cross-attention block:

```text
queries = base_tokens + learned_position_queries
keys    = LayerNorm(base_tokens)
values  = LayerNorm(base_tokens)

translated = queries + CrossAttention(queries, keys, values)
translated = translated + FFN(LayerNorm(translated))
```

Configuration:

```text
sequence length: 128
feature width:   1536
attention heads: 12
FFN width:       3072
```

The final FFN projection is zero-initialized. At initialization, this keeps the
new block close to a residual mapping rather than applying two uncontrolled
transformations at once.

Why this is better than only changing output width:

- It can redistribute information across token positions.
- It does not require exact Smol-token to T5-token positional correspondence.
- It preserves the fixed 128-token NeoDragon interface.
- Cross-attention can learn that one teacher token should depend on multiple
  Smol tokens.
- The residual path makes optimization safer than a fully replacement-style
  resampler initialized from scratch.

The module validates token and mask shapes and fails immediately if it does not
receive the fixed sequence length expected by NeoDragon.

### 3.5 `NeoDragonConditionHead`

The translated tokens pass through:

```text
normalized = LayerNorm(translated)
residual   = Linear(1536 -> 768) -> SiLU -> Linear(768 -> 1536)
output     = normalized * learned_channel_scale
           + learned_channel_bias
           + residual
```

The residual output projection is zero-initialized. The channel scale starts at
`0.78`, which gives an initial expected token norm of approximately:

```text
sqrt(1536) * 0.78 = 30.57
```

Observed teacher token norms in earlier runs were roughly 28 to 31, while a
plain unit LayerNorm output has norm `sqrt(1536) = 39.19`. The old scale mismatch
forced training to spend capacity correcting a global distribution error before
learning semantics.

There is intentionally no final unit LayerNorm after this head. NeoDragon was
trained on raw ContextAdapter outputs, so forcing every student token back to
unit variance would remove the magnitude information we explicitly need to
match.

### 3.6 Pooled-condition head

NeoDragon also needs a 2048-dimensional pooled condition produced natively by
two CLIP encoders. The student predicts it from masked mean-pooled MCP tokens:

```text
MCP tokens + MCP mask
  -> masked mean pool
  -> LayerNorm
  -> Linear(1536 -> 2048)
  -> student_pooled
```

The pooled path is trained independently from the token sequence because the
NeoDragon DiT consumes both interfaces.

### 3.7 Learned contiguous mask

The Smol tokenizer mask cannot be copied blindly as the T5 mask. The new bridge
predicts a corrected sequence length:

```text
base_length = sum(Smol mask)
delta       = tanh(mask_length_head(pooled_source)) * 32
pred_length = clamp(base_length + delta, 1, 128)
```

The maximum correction is `128 * 0.25 = 32` tokens. Soft logits are computed for
all positions and optimized with binary cross entropy against the T5 mask. A
hard contiguous mask is used for the DiT forward pass.

This design is preferable to predicting 128 unrelated binary decisions because
normal padded language masks are contiguous. The one-dimensional length model
has the right inductive bias, fewer parameters, and cannot create holes such as
`[1, 1, 0, 1, 0, ...]`.

### 3.8 Size of the new architecture

The sequence translator and condition head add approximately 21.46 million
parameters. This count excludes the reused MCP projector, pooled head, and tiny
mask head. The addition is large enough to correct tokenizer/condition mismatch
but still small relative to the video DiT being adapted.

## 4. Shared Distillation Objectives

Reusable loss functions are implemented in:

```text
new_mobile_ov/training/neodragon_objectives.py
```

### 4.1 Why a composite loss is necessary

The bridge is not learning an ordinary regression target. It is replacing an
entire pretrained text-conditioning system while the downstream NeoDragon DiT
expects a very specific interface. A useful student condition must satisfy at
least four different constraints:

1. **Coordinate compatibility.** The numerical values must be meaningful to the
   existing DiT projection matrices. A rotated or rescaled embedding space may
   preserve semantic similarity while still being invalid for the frozen DiT.
2. **Directional and relational geometry.** Similar prompts must remain close,
   different prompts must remain distinguishable, and individual token vectors
   must point in useful directions.
3. **Magnitude and mask calibration.** Token norms and valid-token positions
   affect attention logits, residual magnitudes, and which condition slots the
   DiT can read.
4. **Functional equivalence.** Ultimately, replacing the teacher condition with
   the student condition should produce the same NeoDragon vector field on the
   same noisy latent state.

No single loss enforces all four properties. For example:

```text
cosine only:
  correct direction is possible with a completely wrong magnitude

raw MSE only:
  large scale errors can dominate optimization before semantic structure is
  learned, and average error does not explicitly protect batch geometry

token losses only:
  can look good numerically without proving that the frozen DiT responds in the
  same way

functional loss only:
  can be weakly prompt-sensitive at some noise states and can therefore hide
  semantic collapse
```

The objective is consequently designed as a hierarchy:

```text
representation shape and direction
  + absolute coordinate and scale calibration
  + prompt-to-prompt geometry
  + mask and pooled-path supervision
  + downstream DiT functional supervision
```

The losses are complementary rather than duplicate measurements of the same
property.

### 4.2 Teacher-mask convention used by all token losses

The teacher T5 attention mask is used to select valid token positions:

```text
M_teacher[b, i] in {0, 1}
```

This choice is important. The student also predicts a mask, but using the
student mask inside the representation losses would create an easy loophole:
the student could mark difficult positions as padding and avoid their loss.
Using the fixed teacher mask means every teacher-valid target must be learned,
regardless of the student's current mask prediction.

The common denominator is based on the number of teacher-valid tokens rather
than the full 128-token window. This prevents short captions from having their
loss artificially diluted by padding and keeps examples with different caption
lengths more comparable.

The mask convention also means that token losses and mask loss have separate
roles:

```text
token losses:
  teach the content at every teacher-valid position

mask BCE:
  teaches the student which positions should be exposed to the DiT
```

### 4.3 Masked raw token MSE: absolute interface compatibility

Implementation:

```text
L_raw = sum(M_teacher * (C_student - C_teacher)^2)
        / (sum(M_teacher) * 1536)
```

where both tensors have shape `[B, 128, 1536]` after the student's condition
head and the teacher's native ContextAdapter.

#### Failure mode addressed

The NeoDragon DiT was trained with fixed linear projections of the teacher
condition. Suppose the student reproduced the same semantic space but applied a
feature rotation `R`:

```text
C_student = C_teacher @ R
```

Pairwise cosine geometry could remain almost unchanged if `R` were orthogonal,
but the pretrained DiT cross-attention weights would not apply the inverse
rotation. The resulting keys and values would be wrong. Raw MSE anchors the
student to the teacher's actual coordinate basis, not merely an equivalent
abstract embedding space.

Raw MSE also fixes per-dimension bias and scale errors that cosine distance
cannot see.

#### Gradient destination

This loss updates the trainable token-producing bridge modules: the MCP
projector, sequence translator, and final condition head. It does not update the
frozen teacher, frozen SmolVLM2 backbone, pooled-only head, or DiT.

#### Why its Exp 1 weight is `0.25`, not `1.0`

Absolute MSE can be numerically dominated by early scale mismatch. If it were
the only dominant term, optimization could spend many steps shrinking a few
large coordinates without first learning the useful normalized feature shape.
The `0.25` coefficient keeps coordinate matching active as a hard compatibility
constraint while allowing normalized MSE and cosine losses to establish
geometry early.

The coefficient is not a statement that raw compatibility is only 25 percent
as important. Loss coefficients multiply quantities with different natural
scales, so their weighted gradient norms, not their literal coefficients,
determine influence.

#### What this loss does not guarantee

Low raw MSE averaged over 128 x 1536 values does not by itself guarantee:

- that prompt-to-prompt differences are preserved;
- that the predicted mask is correct;
- that the pooled 2048-dimensional condition is correct;
- or that the remaining error lies in dimensions irrelevant to the DiT.

Those gaps motivate the other losses.

### 4.4 Masked normalized token MSE: feature-shape alignment

The implementation independently applies LayerNorm over the 1536 features of
each token before computing MSE:

```text
S_hat[b, i] = LN(C_student[b, i])
T_hat[b, i] = LN(C_teacher[b, i])

L_norm_mse = masked_MSE(S_hat, T_hat)
```

LayerNorm removes each token's scalar mean and standard deviation. The loss is
therefore much less sensitive to an early global offset or magnitude error and
more sensitive to the relative feature pattern within a token.

#### Failure mode addressed

At random initialization, the student condition head may have the wrong norm.
Raw MSE then mixes two learning problems:

```text
problem A: learn which feature coordinates should be high or low
problem B: calibrate the absolute output magnitude
```

Normalized MSE makes problem A learnable before problem B is solved. This gives
the bridge a stable semantic-shape signal even while the condition head scale is
still moving.

#### Why this is the dominant representation term (`1.0`)

The previous bridge experiment reached a seemingly acceptable aggregate loss
but generated poor videos. One likely cause is that absolute error alone did
not sufficiently preserve the teacher's structured token features. Making
normalized MSE the largest representation coefficient prioritizes the shape of
the teacher representation, while raw MSE and norm alignment separately restore
the absolute interface.

#### Difference from cosine loss

Normalized MSE and cosine are related but not identical:

- LayerNorm first subtracts the token mean and divides by its standard
  deviation, so it compares centered feature patterns.
- Cosine compares the original vectors after L2 normalization and does not
  remove the feature mean.
- Normalized MSE penalizes coordinate-wise disagreement after normalization;
  cosine collapses disagreement into one angle.

Using both gives a denser coordinate-level gradient and an explicit global
direction constraint.

#### Limitation

Layer-normalized tokens can match perfectly while having the wrong absolute
norm or offset. This is why normalized MSE cannot replace raw MSE and norm
alignment.

### 4.5 Masked token cosine distance: directional alignment

```text
L_token_cos = 1
              - sum(M_teacher * cos(C_student, C_teacher))
                / sum(M_teacher)
```

#### Failure mode addressed

Cosine distance directly asks whether every student token points in the same
feature-space direction as its teacher target. This is valuable because
attention projections often preserve directional semantic structure even when
the student has not yet matched magnitude.

It also creates an easily interpretable diagnostic:

```text
cosine loss near 0.0  -> average valid-token direction is well aligned
cosine loss near 1.0  -> average direction is approximately unrelated
```

#### Why it is weighted `0.50`

The direction signal should be strong, but cosine is scale-invariant. Giving it
the same or greater effective priority than all absolute losses could allow the
bridge to find the right directions with norms too small or too large for the
DiT. The `0.50` coefficient makes it a major auxiliary constraint while raw MSE
and norm alignment retain scale authority.

#### Gradient destination and limitation

The gradient updates the token-producing bridge path. It does not train the
pooled head or mask head. It also cannot detect:

```text
C_student = 0.01 * C_teacher
```

as a serious scale error, because the cosine is still approximately one.

### 4.6 Token norm alignment: explicit scale calibration

The implementation does not compare every token norm independently. It first
computes the mean valid-token norm for each sample:

```text
n_student[b] = mean_valid_i(||C_student[b, i]||_2)
n_teacher[b] = mean_valid_i(||C_teacher[b, i]||_2)

L_token_norm = mean_b(
    ((n_student[b] - n_teacher[b]) / max(n_teacher[b], 1e-4))^2
)
```

#### Failure mode addressed

Even with correct directions, a norm mismatch changes the magnitude of
cross-attention keys and values after the DiT's learned projections. That can
change attention sharpness, residual scale, and ultimately the denoising vector
field. The old low-loss-but-bad-generation behavior makes this calibration
particularly important.

The relative error is used instead of absolute error so a sample with a larger
teacher norm does not automatically dominate a sample with a smaller one.

#### Why it is weighted `0.10`

Norm is a one-dimensional summary per sample, while raw and normalized token
losses supervise hundreds of thousands of feature values. A small explicit
coefficient is enough to act as a scale guardrail without encouraging the model
to match norm while ignoring direction.

#### Limitation

Matching the mean norm does not guarantee that each token has the correct norm
distribution. It is intentionally a low-cost calibration term, not a complete
distribution-matching objective. The raw token loss handles finer errors.

### 4.7 Pooled MSE and pooled cosine: supervising the global branch

NeoDragon supplies the DiT with a separate pooled condition:

```text
P_teacher: [B, 2048]
```

It is formed from the two native CLIP text projections. Mobile-OV predicts this
vector through a separate pooled head, so token losses cannot train it.

The two pooled losses are:

```text
L_pool_mse = MSE(P_student, P_teacher)

L_pool_cos = 1 - mean_b(cos(P_student[b], P_teacher[b]))
```

#### Why both are required

The same absolute-versus-direction argument applies here:

- pooled MSE anchors the exact coordinate basis and magnitude expected by the
  DiT's pooled projection path;
- pooled cosine protects global semantic direction while scale is still being
  calibrated.

#### Why the Exp 1 weights are `0.25` and `0.20`

The token condition is the richer 128 x 1536 interface and receives most of the
representation budget. The pooled branch is important but lower-dimensional
and should not dominate the bridge. Similar, moderately sized MSE and cosine
weights ensure it receives direct supervision without overwhelming token
learning.

#### Failure if these losses are removed

The token path could converge while the pooled vector remains random. The DiT
would then receive contradictory local and global prompt signals. Generation
might show partial object semantics but poor overall style, composition, or
prompt consistency.

### 4.8 Batch relational loss: anti-collapse geometry

For each sample, valid token features are mean-pooled and L2-normalized. The
implementation then compares the within-batch cosine Gram matrices:

```text
u_student[b] = normalize(mean_valid(C_student[b]))
u_teacher[b] = normalize(mean_valid(C_teacher[b]))

G_student = u_student @ u_student.T
G_teacher = u_teacher @ u_teacher.T

L_rel = MSE(G_student, G_teacher)
```

#### Failure mode addressed

Per-sample losses ask whether each sample is near its target, but they do not
explicitly emphasize diversity across the batch. A weak bridge can reduce
average loss by predicting a generic condition near the center of many teacher
targets. This is especially dangerous for video generation because such a
bridge may produce visually plausible but nearly prompt-independent videos.

If all student prompts collapse to one vector, then all off-diagonal entries of
`G_student` approach one. Teacher prompts with different semantics have lower
pairwise similarities, so the relational loss exposes the collapse.

#### Why it is weighted `0.10`

Relational geometry is a guardrail, not the absolute target interface. A high
weight could preserve pairwise relationships while allowing a globally rotated
space that the DiT cannot consume. The small coefficient reinforces diversity
after coordinate-level losses establish the teacher basis.

#### Distributed-training detail

The current implementation computes this matrix within each rank's local
batch; it does not all-gather embeddings across ranks. With the Exp 1 batch size
of four per GPU, each rank contributes six non-trivial prompt pairs. If local
batch size is one, the function intentionally returns zero because no pairwise
relationship exists.

### 4.9 Mask BCE: learning a differentiable length decision

The student bridge predicts soft logits for all 128 mask positions. The target
is the teacher T5 mask:

```text
L_mask = BCEWithLogits(mask_logits, M_teacher)
```

The DiT itself consumes a hard mask. Thresholding or converting a predicted
length to a Boolean mask is non-differentiable, so generation loss cannot
reliably train the mask decision directly. BCE provides a smooth gradient to
the mask head.

#### Failure mode addressed

An overly short mask hides useful condition tokens from cross-attention. An
overly long mask exposes padding-like or untrained positions. Either error can
make a numerically good token tensor functionally wrong.

#### Why it is weighted `0.05`

Mask prediction is a small auxiliary classification problem. A larger weight
could make the bridge optimize token count at the expense of token content. The
teacher mask is also tokenizer-dependent, so exact mask replication should be
a useful compatibility signal, not the dominant semantic objective.

#### Separation from hard-mask diagnostics

`mask_loss` trains the soft logits. `mask_accuracy` in the logs compares the
eventual hard student mask with the hard teacher mask. It is possible for BCE to
decrease before hard accuracy changes because logits may move toward the
threshold without crossing it.

### 4.10 Functional DiT-response MSE: downstream equivalence

Representation matching assumes that closeness in condition space translates
to closeness in generated behavior. The functional objective tests that
assumption directly.

For the same scheduler-valid noisy pyramid state `x_t`, stage, continuation
unit, timestep, and pooled/mask structure:

```text
v_teacher = DiT_frozen(x_t, t, C_teacher, M_teacher, P_teacher)
v_student = DiT_frozen(x_t, t, C_student, M_student, P_student)

L_func_mse = MSE(v_student, v_teacher)
```

Only the condition changes between calls. Therefore, output disagreement is
attributable to the student conditioning path rather than a different latent or
timestep.

#### Failure mode addressed

Not every condition feature dimension matters equally to the DiT. A small error
in a high-sensitivity direction may damage generation more than a larger error
in an ignored direction. Plain condition MSE weights both errors equally.

Backpropagating through the frozen DiT implicitly weights condition errors by
the DiT's Jacobian:

```text
dL_func / dC_student
  = (dL_func / dv_student) * (dv_student / dC_student)
```

The bridge therefore receives larger gradients for condition differences that
actually change the NeoDragon vector field. This is the main reason Exp 1 is
stronger than the previous embedding-only distillation.

#### Gradient destination

Teacher output is computed under `torch.no_grad()`. The DiT parameters have
`requires_grad=False`, but the student call is not wrapped in `no_grad`, so
autograd traverses the frozen DiT and reaches the bridge condition. No DiT
weight is updated in Exp 1.

#### Why its final weight is `1.0`

Functional behavior is the final purpose of the condition, so once the bridge
has entered a reasonable representation region this is the strongest direct
task-level term. It is ramped rather than weakened permanently; Section 5.9
explains the ramp.

#### Limitation

At some noisy states, the DiT output may be dominated by noise dynamics and be
only weakly sensitive to text. A collapsed condition could then obtain a
deceptively modest functional loss. Representation, pooled, relational, and
mask losses prevent the model from exploiting that ambiguity.

### 4.11 Functional response cosine: vector-field direction

The DiT output is flattened per sample and compared by cosine distance:

```text
L_func_cos = 1 - mean_b(cos(flat(v_student), flat(v_teacher)))
```

#### Why response MSE is not enough

Flow and diffusion samplers repeatedly integrate the model prediction. The
direction of that prediction determines where the latent trajectory moves,
while magnitude affects step size. Response MSE supervises both but can be
dominated by high-magnitude regions. Cosine explicitly protects global update
direction.

#### Why cosine is only `0.10`

The teacher's absolute vector-field magnitude is meaningful to its scheduler,
so scale-invariant cosine must not replace MSE. The smaller cosine term acts as
a directional safeguard while MSE remains the primary functional target.

#### Limitation

Flattened cosine is a global statistic. It does not guarantee that every
spatial location or channel matches. Functional MSE supplies the dense local
supervision.

### 4.12 Why these loss weights should not be read as percentages

The Exp 1 coefficients are:

```text
raw token MSE:          0.25
normalized token MSE:   1.00
token cosine:           0.50
token norm:             0.10
pooled MSE:             0.25
pooled cosine:          0.20
batch relational:       0.10
mask BCE:               0.05
functional MSE:         ramp to 1.00
functional cosine:      ramp to 0.10
```

These numbers are not a percentage allocation. For example, a raw loss of
`0.20` with weight `0.25` contributes `0.05`, while a cosine loss of `0.02` with
weight `0.50` contributes `0.01`. More importantly, equal scalar contributions
can still produce different gradient norms in different modules.

The current values encode the following priority order:

```text
1. establish normalized token structure;
2. make the frozen DiT respond correctly;
3. preserve token direction;
4. calibrate absolute coordinates, pooled condition, norm, mask, and diversity;
5. do not let any auxiliary invariant replace exact teacher compatibility.
```

They are reasoned defaults for a costly first production run, not weights proven
optimal by ablation. The correct first-run audit is to inspect both unweighted
losses and their weighted contributions. If one weighted term is orders of
magnitude larger throughout training, the intended balance is not being
realized and the coefficients should be revisited.

### 4.13 Loss-removal failure matrix

| Removed term | Most likely hidden failure |
| --- | --- |
| Raw token MSE | Semantically similar but DiT-incompatible coordinate system |
| Normalized token MSE | Slow or unstable early learning under scale mismatch |
| Token cosine | Weak directional semantic alignment |
| Token norm | Correct directions with invalid cross-attention magnitude |
| Pooled losses | Token condition works but global prompt branch stays random |
| Relational loss | Prompt embeddings become insufficiently diverse |
| Mask BCE | Good token values are exposed through the wrong valid-token mask |
| Functional MSE | Low embedding loss does not translate into correct DiT behavior |
| Functional cosine | Similar average magnitude but wrong global denoising direction |

## 5. Experiment 1: Bridge Representation and Functional Distillation

### 5.1 Goal

Train a bridge from scratch that is compatible with the released frozen
NeoDragon DiT before allowing any DiT weight to move.

This experiment answers:

```text
Can SmolVLM2 + bridge v2 replace TextEncoderBundle + ContextAdapter while the
released NeoDragon denoiser remains unchanged?
```

### 5.2 Code entry points

| Role | File |
| --- | --- |
| Trainer | `tools/train_neodragon_text_bridge.py` |
| Bridge model | `new_mobile_ov/bridge/neodragon_text_bridge.py` |
| Shared losses | `new_mobile_ov/training/neodragon_objectives.py` |
| Model config | `configs/mobile_ov_neodragon_bridge_v2.yaml` |
| Berzelius job | `scripts/exp1_train_neodragon_bridge_functional_distill_1node8gpu.sbatch` |

### 5.3 Data input

Experiment 1 requires only text. The default manifest is:

```text
download_data/data/openvid/manifests/openvid_all_recaptions_merged.csv
```

Expected caption columns:

```text
caption_short
caption_medium
caption_long
```

One available caption is selected randomly for each sample. The default Exp 1
weights are `1:1:1`, so all granularities are equally likely.

Why text-only is appropriate here:

- Representation targets come from frozen text encoders.
- Functional targets can be queried from a frozen DiT on synthetic but
  scheduler-valid latent states.
- We avoid expensive video decoding and VAE encoding while solving the text
  compatibility problem.
- Caption diversity, not video reconstruction, is the main data requirement.

The native NeoDragon prompt modifier is appended to the prompt by default for
both teacher and student. This preserves inference-time prompt formatting parity.

### 5.4 Trainable and frozen modules

| Module | State | Reason |
| --- | --- | --- |
| SmolVLM2 backbone | Frozen and eval | Preserve understanding model and deterministic features |
| MCP projector | Trainable | Map Smol hidden layers into NeoDragon condition space |
| Sequence translator | Trainable | Correct tokenizer and positional mismatch |
| Condition head | Trainable | Match raw ContextAdapter distribution |
| Pooled head | Trainable | Replace two native CLIP pooled projections |
| Mask-length head | Trainable | Match native T5 attention mask length |
| TextEncoderBundle | Frozen and eval | Teacher only |
| ContextAdapter | Frozen and eval | Defines actual token target seen by DiT |
| NeoDragon DiT | Frozen and eval | Defines functional target |

### 5.5 Teacher representation path

```text
prompt
  -> NeoDragon TextEncoderBundle
  -> T5 token features, T5 mask, two CLIP pooled projections
  -> ContextAdapter(T5 token features)
  -> teacher_tokens [B, 128, 1536]
  -> teacher_mask   [B, 128]
  -> teacher_pooled [B, 2048]
```

The complete path runs under `torch.no_grad()`.

### 5.6 Student representation path

```text
prompt
  -> frozen SmolVLM2
  -> MCP lexical-gated projector
  -> base_tokens/base_mask
  -> sequence translator
  -> condition head
  -> student_tokens [B, 128, 1536]

base_tokens/base_mask
  -> pooled head
  -> student_pooled [B, 2048]

base mask length + pooled source
  -> mask-length head
  -> student_mask [B, 128]
```

### 5.7 Functional latent sampling

Functional distillation must compare teacher and student conditions under the
same DiT state. The trainer creates that state using NeoDragon's own pyramid
utilities and scheduler.

For the configured 49 pixel frames:

```text
latent temporal length = ((49 - 1) // 8) + 1 = 7
base latent shape       = [B, 16, 7, 40, 64]
```

For each functional step:

1. Sample a random clean latent tensor.
2. Sample a continuation unit index from `1..6`.
3. Build causal past-condition latents from units before that index.
4. Sample one of NeoDragon's three pyramid stages.
5. Downsample the current clean unit to the selected stage.
6. Sample Gaussian noise and a scheduler timestep.
7. Construct the valid flow-matching state:

```text
x_t = sigma * noise + (1 - sigma) * clean_stage
```

The same `x_t`, past context, pyramid stage, and timestep are reused in both DiT
calls. This is essential. If teacher and student saw different noise, their
output difference would not isolate conditioning error.

### 5.8 Functional teacher and student calls

Teacher response:

```text
frozen_DiT(x_t, teacher_tokens, teacher_mask, teacher_pooled, timestep)
  -> teacher_prediction
```

Student-conditioned response:

```text
same frozen_DiT(x_t, student_tokens, student_mask, student_pooled, timestep)
  -> student_prediction
```

The teacher call runs under `no_grad`. The student-conditioned call does not
update DiT parameters because all DiT weights have `requires_grad=False`, but
autograd is retained through the DiT operations back to the student condition.
This is how the bridge learns from the DiT response without training the DiT.

### 5.9 Complete Exp 1 objective

The default job uses:

```text
L_repr = 0.25 * L_raw_token_mse
       + 1.00 * L_normalized_token_mse
       + 0.50 * L_token_cosine
       + 0.10 * L_token_norm
       + 0.25 * L_pooled_mse
       + 0.20 * L_pooled_cosine
       + 0.10 * L_batch_relational
       + 0.05 * L_mask_bce

L_func = 1.00 * L_dit_response_mse
       + 0.10 * L_dit_response_cosine

L_exp1 = L_repr + functional_ramp(step) * L_func
```

`functional_ramp` rises linearly over the first 2,000 steps.

Why ramp functional loss:

- At initialization, student condition scale and mask can be far from teacher.
- A full-strength DiT response gradient through many nonlinear blocks can be
  noisy before the representation becomes usable.
- Representation losses first establish the correct local condition manifold.
- Functional loss then optimizes the part of that manifold that matters to DiT.

The ramp starts at `1 / 2000` on step 1 and reaches `1.0` on step 2,000. It does
not create a hard phase boundary. This is deliberate: the bridge receives a
small downstream signal immediately, but the signal cannot overwhelm basic
condition alignment during the most unstable initialization period.

#### 5.9.1 Gradient routing in Exp 1

| Loss group | Bridge token path | Bridge pooled path | Bridge mask head | SmolVLM2 | NeoDragon DiT |
| --- | --- | --- | --- | --- | --- |
| Token MSE/cos/norm | Updated | No direct gradient | No direct gradient | Frozen | Not used |
| Pooled MSE/cos | Shared bridge features plus pooled head | Updated | No direct gradient | Frozen | Not used |
| Relational | Updated | No direct gradient | No direct gradient | Frozen | Not used |
| Mask BCE | Shared bridge features plus mask head | No direct gradient | Updated | Frozen | Not used |
| Functional MSE/cos | Updated through DiT | Updated through DiT | No direct gradient because the DiT receives a hard mask | Frozen | Frozen weights, autograd-through-input only |

This routing is important because Exp 1 has a fixed consumer. The bridge cannot
reduce functional loss by asking the DiT to adapt to a bad private condition
space. The only solution is to produce a condition that the released DiT already
understands.

The mask-head exception is intentional and worth making explicit. The functional
call receives `pred_mask`, which is created by thresholding `mask_logits >= 0`.
That comparison has no useful derivative. Functional loss can update token and
pooled conditions, but `L_mask_bce` is the direct training signal for
`mask_length_head`. This is why mask BCE cannot be removed merely because a
functional loss is present.

#### 5.9.2 Why representation and functional objectives are used together

The two objectives cover each other's blind spots:

```text
representation objective asks:
  "Did the bridge reconstruct the teacher interface?"

functional objective asks:
  "Does the reconstructed interface cause the frozen DiT to behave correctly?"
```

Representation loss alone treats all condition dimensions according to a
hand-designed metric. It does not know which dimensions the DiT is sensitive
to. Functional loss supplies that sensitivity through the DiT Jacobian.

Functional loss alone is also insufficient. On a high-noise latent, two
different prompts may produce similar vector fields because noise removal
dominates semantics. A student could therefore obtain low response loss while
using weakly differentiated prompt conditions. Direct token, pooled, mask, and
relational targets keep the student tied to the complete teacher interface.

The combination is intentionally redundant at two abstraction levels:

```text
condition-space correctness
  AND
downstream behavior correctness
```

That redundancy is desirable because the previous experiment demonstrated that
a single low aggregate alignment loss was not a reliable quality certificate.

#### 5.9.3 Why functional distillation uses only one sample per rank

The default representation batch is four prompts per GPU, but
`--functional-batch-size 1` sends only one prompt per rank through the frozen
DiT teacher/student pair. Two full DiT forward passes are substantially more
expensive than the bridge losses. Restricting the functional sub-batch keeps
Exp 1 affordable while eight ranks still provide up to eight functional samples
per optimization step.

All four local prompts still train the representation objective, including the
relational loss. Prompt order changes with the shuffled data iterator, so the
functional subset is not permanently tied to a fixed caption group.

If compute permits, increasing functional batch size is safer than increasing
functional weight: it reduces gradient variance without changing the intended
loss balance.

#### 5.9.4 Why Exp 1 intentionally has no flow-matching loss

Exp 1 consumes captions only and freezes the DiT. Its purpose is to answer one
isolated question: can Mobile-OV replace NeoDragon's native text bundle without
changing the generator?

Adding flow matching here would require video latents and an unfrozen DiT. That
would make a failure ambiguous:

```text
did the bridge fail to reproduce the condition?
or did the DiT move while learning OpenVid?
```

Exp 1 avoids that ambiguity and produces a bridge checkpoint that can be tested
with the released DiT before any generative weights are modified.

### 5.10 Default training configuration

```text
steps:                    24,000
GPUs:                     8
parallelism:              FSDP
batch size per GPU:       4 captions
global caption batch:     32
optimizer:                AdamW
learning rate:            5e-5
functional batch per GPU: 1
functional frequency:     every step
checkpoint interval:      1,000 steps
dtype:                    bf16
```

Functional batch size is smaller than representation batch size because the
two DiT forwards dominate activation memory. All captions still contribute to
the cheaper representation losses.

### 5.11 Why Exp 1 is separated from DiT fine-tuning

If a random bridge and DiT are trained together immediately, the system is
underdetermined:

```text
bad bridge + compensating DiT change
```

can reduce training loss without producing a reusable or teacher-compatible
condition. It also makes diagnosis difficult because failure may come from the
bridge, the DiT, or both.

Freezing the DiT in Exp 1 creates a fixed functional contract. The bridge must
adapt to the released generator instead of forcing the generator to adapt to a
bad condition. This is especially important for NeoDragon because its pruned
architecture has less redundant capacity than the original larger teacher and
may be more sensitive to destructive updates.

### 5.12 Logging and expected behavior

`history.json` and the progress bar expose:

```text
raw_token_loss
normalized_token_loss
pooled_loss
cos_loss
norm_loss
pooled_cos_loss
relational_loss
mask_loss
mask_accuracy
functional_loss
functional_cos_loss
functional_scale
functional_stage
functional_unit
pred_norm
target_norm
target_mask_tokens
```

Healthy behavior should include:

- `pred_norm` approaching `target_norm` instead of remaining around 39.
- Token and pooled cosine losses decreasing.
- `mask_accuracy` increasing without predicted masks collapsing to all ones.
- Functional response loss decreasing after the ramp begins.
- Functional loss remaining finite across all three pyramid stages.

A low token loss alone is not a pass criterion. The important downstream check
is that frozen-DiT response loss decreases and bridge-conditioned generation
improves.

### 5.13 Exp 1 checkpoint contract

Main output:

```text
output/neo_exp1_bridge_functional/<job_id>/neodragon_text_bridge_latest.pt
```

Step snapshots:

```text
neodragon_text_bridge_step001000.pt
neodragon_text_bridge_step002000.pt
...
```

Checkpoint payload:

```text
step
bridge
config
args
history
target
architecture
parallel
shapes
```

The `bridge` state is a consolidated full state dictionary gathered on rank 0.
It is the required initialization for Experiment 2.

### 5.14 Run Exp 1

```bash
mkdir -p logs
sbatch scripts/exp1_train_neodragon_bridge_functional_distill_1node8gpu.sbatch
```

Useful overrides:

```bash
PROMPTS=/path/to/openvid_all_recaptions_merged.csv \
OUT=output/my_exp1 \
STEPS=24000 \
BATCH_SIZE=4 \
LR=5e-5 \
sbatch scripts/exp1_train_neodragon_bridge_functional_distill_1node8gpu.sbatch
```

## 6. Experiment 2: Joint Flow Matching and Teacher Distillation

### 6.1 Goal

Initialize the bridge from Experiment 1, initialize the DiT from the released
NeoDragon checkpoint, then jointly adapt both modules to OpenVid video latents.

This experiment answers:

```text
Can the complete Mobile-OV condition + NeoDragon DiT system learn real OpenVid
flow targets without losing the useful behavior of the released pruned model?
```

### 6.2 Code entry points

| Role | File |
| --- | --- |
| Trainer | `tools/train_neodragon_dit_bridge.py` |
| Offline VAE encoder | `tools/data_prepare/encode_neodragon_vae_latents.py` |
| Shared losses | `new_mobile_ov/training/neodragon_objectives.py` |
| Model config | `configs/mobile_ov_neodragon_bridge_v2.yaml` |
| Berzelius job | `scripts/exp2_train_neodragon_joint_flow_distill_1node8gpu.sbatch` |

### 6.3 Required inputs

Bridge initialization:

```text
BRIDGE_CKPT=output/neo_exp1_bridge_functional/<job_id>/neodragon_text_bridge_latest.pt
```

Default latent manifest:

```text
data/openvid_neodragon_2s_latents/latent_manifest.csv
```

Required manifest information:

```text
latent_path
caption or prompt
caption_short
caption_medium
caption_long
```

The trainer auto-detects `latent_path` and skips online VAE loading when
precomputed latents are available.

### 6.4 Offline latent contract

The offline encoder reads 49 frames at 24 FPS, center-crops/resizes to 320x512,
encodes them with the released NeoDragon VAE, and applies the same scale/shift
rules used by NeoDragon generation.

The first latent slice and later video slices use NeoDragon's corresponding
scale and shift constants before being saved. This prevents the training script
from accidentally learning on raw VAE latents while inference uses scaled
latents.

Typical stored latent shape:

```text
[16, 7, 40, 64]
```

The exact temporal length is checked dynamically by the trainer.

Why offline VAE encoding is preferable for the large run:

- The same deterministic latent target is reused across epochs.
- Expensive VAE work is removed from every training step.
- GPU memory is reserved for student and teacher DiT forwards.
- Video decoding failures are handled during preprocessing rather than in the
  middle of distributed training.
- Throughput is more predictable on Berzelius.

### 6.5 Caption augmentation

Experiment 2 randomly samples caption granularities with weights:

```text
short:  5
medium: 4
long:   1
```

The model therefore sees short and medium prompts most often while still
receiving long descriptions. This is a conservative response to the observed
sensitivity of generation quality to prompt style. It also keeps the training
distribution closer to short benchmark prompts without discarding detailed
semantic supervision.

These weights are configurable and should be treated as a starting policy, not
an immutable architectural requirement.

### 6.6 Initialization and trainable modules

| Module | Initialization | State in Exp 2 |
| --- | --- | --- |
| SmolVLM2 | Existing Mobile-OV checkpoint | Frozen and eval |
| Bridge v2 | Experiment 1 checkpoint | Trainable |
| Student NeoDragon DiT | Released NeoDragon DiT | Fully trainable |
| Teacher TextEncoderBundle | Released NeoDragon | Frozen and eval |
| Teacher ContextAdapter | Released NeoDragon | Frozen and eval |
| Teacher NeoDragon DiT | Released NeoDragon DiT | Frozen and eval |
| NeoDragon VAE | Offline preprocessing only | Not loaded for latent training |

`--train-last-n-blocks 0` means the full student DiT remains trainable. It does
not mean zero blocks are trained. Positive values freeze the DiT except for the
requested final blocks, but the provided Exp 2 job intentionally uses full
fine-tuning.

### 6.7 Flow-matching sample construction

For each real OpenVid latent:

1. Select a continuation unit index from `1..T_latent-1`.
2. Convert all previous clean units into NeoDragon causal pyramid context.
3. Randomly select one of the three pyramid stages.
4. Downsample the selected clean unit to that stage.
5. Sample Gaussian noise and one of 1,000 training timestep indices.
6. Read the corresponding stage-specific `sigma` and timestep ratio.
7. Construct:

```text
x_t         = sigma * noise + (1 - sigma) * clean_stage
target_flow = noise - clean_stage
```

8. Predict the flow using the student bridge condition and student DiT.

```text
student_prediction = student_DiT(
    past_context + x_t,
    student_tokens,
    student_mask,
    student_pooled,
    timestep,
)

L_flow = MSE(student_prediction, target_flow)
```

This follows NeoDragon's stage-specific flow-matching parameterization rather
than applying a generic DDPM noise target.

### 6.8 Current continuation-only scope

The current trainer samples unit indices starting from `1`, not `0`. Therefore
Exp 2 trains the causal video-continuation/motion units with at least one clean
past unit available.

This matches the current system decomposition in which first-frame/anchor
generation is handled separately and NeoDragon is used as the video-motion
branch. It also avoids mixing first-frame generation behavior with continuation
behavior in the first joint experiment.

This is an explicit limitation if the future goal changes to making the
NeoDragon DiT independently generate the first latent unit from pure noise. In
that case, unit `0` must be sampled with a controlled probability and evaluated
as a separate ablation.

### 6.9 Teacher-response distillation

The first teacher signal compares complete teacher and student systems:

```text
student_prediction = student_DiT(x_t, student_condition)
teacher_prediction = frozen_teacher_DiT(x_t, teacher_condition)

L_cross_mse = MSE(student_prediction, teacher_prediction)
L_cross_cos = 1 - cosine(student_prediction, teacher_prediction)
```

The noise, clean latent, past context, pyramid stage, and timestep are identical
for both predictions.

Gradients from this loss update both the student bridge and student DiT.

Purpose:

- Keep the complete student system near released NeoDragon behavior.
- Let the DiT compensate for residual bridge mismatch in a controlled way.
- Distill behavior on actual OpenVid latent states rather than only synthetic
  states.

Limitation:

This loss alone cannot tell whether an error came from the bridge or the student
DiT because both differ from the teacher path. The next two losses address that
ambiguity.

### 6.10 Teacher-condition DiT preservation

The same teacher condition is fed to both DiTs:

```text
student_on_teacher = student_DiT(x_t, teacher_condition)
teacher_prediction = frozen_teacher_DiT(x_t, teacher_condition)

L_preserve_mse = MSE(student_on_teacher, teacher_prediction)
L_preserve_cos = 1 - cosine(student_on_teacher, teacher_prediction)
```

Because the condition is identical, this path isolates DiT drift. The bridge is
not responsible for any difference.

Why this is particularly important for a pruned model:

- The released pruned weights encode behavior recovered by NeoDragon's original
  training/distillation pipeline.
- Full fine-tuning on a new dataset can overwrite this behavior quickly.
- A pruned model may have less spare capacity to absorb a large distribution
  shift without forgetting.
- Preservation gives the student permission to adapt through flow matching but
  anchors its vector field to the released model.

The preservation forward runs every four steps to reduce compute. Its loss is
multiplied by four on active steps, preserving approximately the requested
expected weight over time.

### 6.11 Bridge representation anchor during joint training

Experiment 2 continues a smaller version of representation distillation:

```text
L_bridge_repr = 1.00 * normalized_token_mse
              + 0.50 * token_cosine
              + 0.10 * token_norm
              + 0.25 * pooled_mse
              + 0.20 * pooled_cosine
              + 0.05 * mask_bce
```

The complete term is multiplied by `0.10` in the Exp 2 total objective.

Purpose:

- Prevent the bridge from drifting into an arbitrary private code understood
  only by the jointly changing DiT.
- Preserve compatibility with the teacher condition interface.
- Retain a meaningful standalone bridge checkpoint.
- Reduce the chance that text semantics collapse while flow loss still improves.

### 6.12 Complete Exp 2 objective

```text
L_exp2 = w_flow(step) * L_flow
       + 1.00 * L_cross_mse
       + 0.10 * L_cross_cos
       + frequency_scale * (
             0.50 * L_preserve_mse
           + 0.05 * L_preserve_cos
         )
       + 0.10 * L_bridge_repr
```

This objective contains both flow matching and distillation. Exp 2 is not a
distillation-only phase.

The losses serve different roles:

| Loss | What it adapts or protects |
| --- | --- |
| Flow matching | Learns real OpenVid video latent dynamics |
| Cross-system teacher response | Keeps bridge + DiT behavior near NeoDragon |
| Teacher-condition preservation | Prevents standalone DiT catastrophic drift |
| Bridge representation anchor | Prevents condition-space collapse or private coding |

The brief table above hides an important design point: the three teacher terms
are not interchangeable. They separate two sources of change that a single
end-to-end distillation loss cannot identify.

#### 6.12.1 Formal decomposition of the joint-training problem

Define:

```text
theta_0 = released, frozen NeoDragon DiT parameters
theta   = trainable Exp 2 DiT parameters

c_T = native NeoDragon teacher condition
c_S = Mobile-OV bridge condition

f(theta, x_t, t, c) = NeoDragon vector-field prediction
```

At the beginning of Exp 2, two things can differ from the released system:

```text
condition replacement error:
  c_S != c_T

generator adaptation error or drift:
  theta != theta_0
```

The end-to-end output difference is:

```text
f(theta, x_t, t, c_S) - f(theta_0, x_t, t, c_T)
```

That difference mixes bridge error and DiT drift. Cross-system distillation
alone can reduce the total difference without telling us which component
changed. For example, the DiT could learn to compensate for an increasingly
non-teacher-like bridge. The complete objective adds two isolating constraints:

```text
bridge representation anchor:
  directly constrains c_S toward c_T

teacher-condition preservation:
  directly constrains f(theta, x_t, t, c_T)
                    toward f(theta_0, x_t, t, c_T)
```

Together, the three teacher signals form a triangle:

```text
                 cross-system response
  f(theta,c_S) -------------------------- f(theta_0,c_T)
       |                                         |
       | bridge anchor                           | fixed teacher
       |                                         |
  f(theta,c_T) -------- preservation ------------+
```

The diagram is conceptual: bridge anchoring acts in condition space, while the
other two terms act in DiT output space. The purpose is to prevent bridge and
DiT errors from silently canceling each other.

#### 6.12.2 Flow-matching loss: learning from real OpenVid targets

For one clean latent pyramid unit `x_0` and Gaussian noise `epsilon`, the code
uses NeoDragon's scheduler value `sigma_t`:

```text
x_t = sigma_t * epsilon + (1 - sigma_t) * x_0

v_target = epsilon - x_0

v_student = f(theta, x_t, t, c_S)

L_flow = MSE(v_student, v_target)
```

The target is the derivative of the linear interpolation path with respect to
`sigma`. It is not a teacher pseudo-label. It comes from the actual OpenVid
latent and sampled noise, so it is the only objective that teaches the model to
fit the new training data rather than merely reproduce released NeoDragon.

##### Why MSE is appropriate here

Flow matching is a vector-field regression problem. The sampler requires both
direction and magnitude of the velocity. MSE is the standard dense objective
for this target because it penalizes every channel and spatial location and has
the correct optimum at the conditional expectation of the target velocity.

A cosine-only flow loss would leave velocity magnitude underdetermined and
would therefore be incompatible with the scheduler's integration scale.

##### Where its gradients go

`L_flow` updates both the full trainable DiT and the bridge. The DiT learns the
OpenVid latent dynamics, while the bridge learns which condition representation
helps predict those dynamics.

##### Why flow is weighted `0.30` initially

The released pruned DiT already contains useful generation behavior. The goal
is adaptation, not training a generator from scratch. A unit flow coefficient
could let a large real-data gradient immediately overwrite the pretrained
vector field, especially while the bridge is still imperfect. `0.30` keeps a
strong data signal but makes the fixed teacher response the dominant behavioral
reference early in the run.

The coefficient decreases to `0.10` during cooldown, not because data stops
being important, but because final stabilization should favor retaining useful
teacher behavior after the model has already adapted to OpenVid.

#### 6.12.3 Cross-system response distillation: end-to-end compatibility

The student and teacher receive the same `x_t`, timestep, stage, and past
condition latents:

```text
v_cross_student = f(theta,   x_t, t, c_S)
v_teacher       = f(theta_0, x_t, t, c_T)

L_cross_mse = MSE(v_cross_student, v_teacher)
L_cross_cos = 1 - cos(flat(v_cross_student), flat(v_teacher))
```

##### Failure mode addressed

Flow matching can improve dataset loss while damaging the behavior inherited
from NeoDragon. It can also encourage the bridge and DiT to invent a private
condition code that works only on training captions. Cross-system distillation
anchors the complete new path to the released end-to-end model.

Unlike Exp 1, this loss updates both bridge and DiT. It says that the combined
student system should remain behaviorally close to NeoDragon while the flow
objective introduces real-data adaptation.

##### Why MSE weight is `1.0` and cosine weight is `0.10`

The released teacher is our strongest available prior for a pruned DiT that
already generates coherent video. Absolute MSE is therefore the main
preservation target. Cosine is lower because scheduler-compatible prediction
magnitude must remain anchored; direction alone is insufficient.

The fact that the cross MSE coefficient is larger than the flow coefficient
does not prove its gradient is larger. Their raw scales differ. The intended
meaning is that teacher behavior is the baseline and OpenVid flow is a
controlled adaptation force.

##### Limitation

This term alone cannot reveal whether low loss comes from a correct bridge, a
preserved DiT, or two mutually compensating errors. That is why the next two
terms are retained.

#### 6.12.4 Teacher-condition preservation: isolating DiT drift

The trainable and frozen DiTs are both evaluated with the exact same native
teacher condition:

```text
v_preserve_student = f(theta,   x_t, t, c_T)
v_teacher          = f(theta_0, x_t, t, c_T)

L_preserve_mse = MSE(v_preserve_student, v_teacher)
L_preserve_cos = 1 - cos(flat(v_preserve_student), flat(v_teacher))
```

Because condition is held fixed at `c_T`, any difference is caused by DiT
parameter drift rather than bridge error. The gradient updates the trainable DiT
only; the bridge is not used in this call.

##### Why this is different from cross-system distillation

Cross-system loss allows this compensation:

```text
bridge changes in one direction
DiT changes in the opposite direction
combined output remains close to teacher
```

Preservation loss removes that degree of freedom. The DiT must still understand
the original teacher interface, which protects its pretrained/pruned behavior
and makes debugging more interpretable.

##### Why the weights are `0.50` MSE and `0.05` cosine

Preservation is important but should not freeze the DiT in practice. If it were
as strong as the cross-system objective on every step, the model would have
little freedom to improve on OpenVid. Half-strength MSE and a proportionally
smaller cosine term create an elastic constraint: drift is allowed when the
data signal consistently supports it, but unconstrained catastrophic movement
is penalized.

##### Why it runs every four steps and is multiplied by four

The preservation call requires an additional forward pass through the
trainable DiT and is expensive. It runs once every `K=4` steps. On active steps,
the code multiplies the term by `K`:

```text
active-step coefficient = K * configured coefficient
inactive-step coefficient = 0
```

If steps are sampled uniformly, the expected average contribution is:

```text
(1 / K) * K * coefficient = coefficient
```

This preserves the intended average loss strength while reducing compute. The
tradeoff is a burstier preservation gradient, which is one reason gradient
clipping and a low DiT learning rate remain useful.

#### 6.12.5 Bridge representation anchor: preventing private coding

During joint training the bridge receives a lighter representation objective:

```text
L_bridge_repr = 1.00 * L_normalized_token_mse
              + 0.50 * L_token_cosine
              + 0.10 * L_token_norm
              + 0.25 * L_pooled_mse
              + 0.20 * L_pooled_cosine
              + 0.05 * L_mask_bce

total contribution = 0.10 * L_bridge_repr
```

##### Failure mode addressed

When bridge and DiT are both trainable, flow loss can be reduced by creating an
arbitrary private code between them. Such a code may memorize training-caption
regularities but lose teacher semantics, prompt diversity, or compatibility
with the released NeoDragon interface.

The representation anchor keeps the bridge near the post-ContextAdapter teacher
manifold while still allowing small task-driven changes.

##### Why the outer weight is only `0.10`

Exp 1 already performed strong direct alignment. Exp 2 should preserve that
alignment, not repeat Exp 1 at full strength. A large representation weight
could prevent useful co-adaptation to OpenVid and make full DiT unfreezing
pointless. `0.10` acts as a tether: weak enough to permit adaptation, strong
enough to make large condition drift costly.

##### Why raw token MSE and relational loss are omitted here

The joint-stage anchor intentionally emphasizes normalized shape, direction,
norm, pooled condition, and mask rather than exact raw-token reconstruction.
This gives the bridge limited freedom to shift coordinates if flow matching and
the trainable DiT consistently benefit.

Exact end-to-end compatibility is still supervised by cross-system response
MSE, and DiT drift is isolated by preservation loss. Relational loss is omitted
to keep the joint objective and distributed cost smaller; Exp 1 has already
established prompt geometry. This is a design tradeoff, not proof that raw and
relational losses would be harmful in all settings.

#### 6.12.6 Why all four Exp 2 objective groups are needed

| Objective retained | What the model can learn | What remains uncontrolled |
| --- | --- | --- |
| Flow only | OpenVid training vector field | Teacher behavior, bridge semantics, DiT drift |
| Cross distillation only | Released end-to-end behavior | No new data adaptation; bridge/DiT errors can cancel |
| Flow + cross | Adaptation with an output anchor | Bridge private code and isolated DiT drift remain ambiguous |
| Flow + cross + preservation | Adaptation while protecting DiT | Bridge can still leave teacher representation space |
| Full objective | Data adaptation, end-to-end compatibility, DiT preservation, and bridge anchoring | Hyperparameter balance still requires measurement |

This is the central reason the objective looks more complicated than a standard
flow-matching fine-tune. We are changing both sides of a pretrained interface at
the same time. Each added term removes one degree of freedom that could hide a
bad solution.

#### 6.12.7 How to interpret competing gradients

Some disagreement between flow and teacher losses is expected:

```text
flow gradient:
  move toward the OpenVid conditional vector field

teacher gradients:
  remain near released NeoDragon behavior
```

If they agreed perfectly, fine-tuning would be unnecessary. The goal is not to
make every term zero simultaneously, but to find a controlled compromise.

Healthy behavior is:

- flow loss decreases from its initial level;
- cross and preservation losses remain bounded rather than exploding;
- bridge representation loss does not trend upward without limit;
- fixed-seed samples retain NeoDragon coherence while improving prompt/data
  alignment.

Warning patterns are:

```text
flow down, cross/preserve sharply up:
  catastrophic teacher forgetting is likely

flow down, bridge_repr sharply up, preserve stable:
  bridge is moving toward a private condition code

cross down, flow flat:
  training is mostly copying the teacher and not learning OpenVid

all scalar losses down, samples prompt-independent:
  inspect mask, relational behavior, caption sampling, and fixed-seed semantic
  response; scalar averages can still hide collapse
```

### 6.13 Learning-rate and loss schedules

Default schedule:

```text
total steps:       28,000
DiT LR:            3e-6
bridge LR:         1e-5
warmup:            2,000 steps
cooldown:          final 4,000 steps
initial flow weight: 0.3
final flow weight:   0.1
```

The DiT LR is lower because it starts from a valuable released checkpoint and
contains the pretrained/pruned generation behavior we want to preserve. The
bridge LR is higher because it must continue adapting to the student DiT and
OpenVid captions.

During warmup, both parameter-group learning rates increase linearly. During the
final 4,000 steps:

- Flow weight decreases linearly from `0.3` to `0.1`.
- Learning rates decrease to 10 percent of their base values.

Flow weight does not reach zero. Real-data flow matching remains active through
the final step, while the lower LR and stronger relative teacher contribution
stabilize the final checkpoint.

### 6.14 Why not use only flow matching?

Pure flow training is simpler and cheaper, but it has two major risks:

1. The student DiT may over-adapt to OpenVid and lose the released NeoDragon
   vector field.
2. The bridge and DiT may invent a private condition representation that reduces
   training loss but generalizes poorly to unseen prompts.

This risk is more serious because the bridge is new and the DiT is fully
unfrozen. Teacher losses constrain both failure modes.

### 6.15 Why not use only self-distillation?

Distillation alone can at best reproduce the released teacher. It does not teach
the model the new OpenVid latent distribution or improve caption-video
alignment on the prepared dataset.

Flow matching supplies the ground-truth data signal. Distillation is a
regularizer and behavior-preservation mechanism, not a replacement for the
generative objective.

### 6.16 Why not combine Exp 1 and Exp 2 from step zero?

Combining everything from a random bridge is possible but less reliable:

- Early bridge conditions are out of distribution for the DiT.
- The easiest short-term response is for DiT weights to chase those bad
  conditions.
- Flow loss, condition loss, and teacher response loss can send conflicting
  gradients before the condition interface is established.
- A final failure gives no clean checkpoint for determining whether bridge
  alignment ever worked.

The two-experiment sequence creates a curriculum:

```text
first learn the interface with a fixed consumer
then adapt the interface and consumer together under preservation constraints
```

This is the safest design when there is limited budget for repeated large runs.

### 6.17 Distributed training implementation

The supplied job uses FSDP on eight GPUs.

Important implementation details:

- `use_orig_params=True` allows frozen and trainable parameters to coexist.
- `sync_module_states=True` broadcasts rank-0 initialization so all ranks start
  from identical random bridge layers and pretrained DiT weights.
- The bridge and DiT are wrapped separately and optimized with separate learning
  rates.
- Scalar logs are averaged across ranks.
- Full state dictionaries are collectively gathered and offloaded to CPU on
  rank 0 for portable checkpoints.
- Joint bridge + DiT training intentionally rejects DeepSpeed in the current
  implementation because two independently wrapped trainable modules and one
  consolidated checkpoint path have only been designed for FSDP/DDP.

Why state synchronization was changed:

The original FSDP wrapper used `sync_module_states=False`. This is unsafe for
new random bridge layers because separate `torchrun` processes may initialize
them differently. Sharding inconsistent initial models would make rank updates
semantically invalid even if NCCL runs without error.

### 6.18 Logging and expected behavior

Exp 2 logs:

```text
loss
diff_loss
flow_weight
distill_loss
distill_cos_loss
preservation_loss
preservation_cos_loss
bridge_repr_loss
dit_lr_scale
unit_index
stage
latent_t
trainable_params
trainable_bridge_params
world_size
```

Interpretation:

- `diff_loss` measures real-data flow matching and should trend down over a
  sufficiently long window.
- `distill_loss` measures total student-system deviation from the teacher.
- `preservation_loss` specifically measures DiT drift under teacher condition.
- `bridge_repr_loss` measures condition-space drift.
- `stage` should vary among all three pyramid stages.
- `unit_index` should vary among continuation units.

Warning signs:

- Falling flow loss with rapidly increasing preservation loss indicates
  destructive DiT adaptation.
- Falling distillation loss with stagnant flow loss indicates excessive teacher
  constraint and weak OpenVid learning.
- Low total loss with rising bridge representation loss may indicate a private
  bridge/DiT code.
- NaN in one stage but not others usually points to stage-specific scaling or
  precision problems.

### 6.19 Exp 2 checkpoint contract

Main output:

```text
output/neo_exp2_joint_flow_distill/<job_id>/neodragon_dit_bridge_latest.pt
```

Step snapshots:

```text
neodragon_dit_bridge_step001000.pt
neodragon_dit_bridge_step002000.pt
...
```

Checkpoint payload:

```text
step
dit
bridge
bridge_ckpt
config
args
history
objective
parallel
```

The final checkpoint contains both trainable model components needed by the new
generation path. It does not contain the frozen teacher text encoders or frozen
teacher DiT.

Important current limitation: optimizer and LR-scheduler states are not saved.
The step checkpoints are model snapshots and audit points, but they do not yet
support mathematically exact optimizer-state resume. Exact resume should be
implemented before relying on multiple 72-hour SLURM segments for one run.

### 6.20 Run Exp 2

```bash
BRIDGE_CKPT=output/neo_exp1_bridge_functional/<job_id>/neodragon_text_bridge_latest.pt \
  sbatch scripts/exp2_train_neodragon_joint_flow_distill_1node8gpu.sbatch
```

Useful overrides:

```bash
BRIDGE_CKPT=/path/to/neodragon_text_bridge_latest.pt \
MANIFEST=/path/to/latent_manifest.csv \
OUT=output/my_exp2 \
STEPS=28000 \
DIT_LR=3e-6 \
BRIDGE_LR=1e-5 \
FLOW_FINAL_WEIGHT=0.1 \
sbatch scripts/exp2_train_neodragon_joint_flow_distill_1node8gpu.sbatch
```

## 7. Fail-Fast and Backward-Compatibility Changes

### 7.1 Opt-in architecture

New config fields default to the legacy behavior:

```text
neodragon_v2_conditioning: false
```

Only the new v2 YAML enables the translator, condition head, and mask head. This
prevents old inference from changing simply because the code was updated.

### 7.2 Strict v2 checkpoint loading

Experiment 2 checks missing and unexpected bridge keys. If the v2 config is used
with an old checkpoint, training fails rather than silently leaving v2 layers
randomly initialized.

### 7.3 Argument validation

Both trainers validate loss weights, step counts, intervals, warmup/cooldown,
functional frequency, and supported distributed mode before expensive model
loading.

### 7.4 Shape validation

The sequence translator checks fixed token length and mask shape. The latent
loader verifies `[C,T,H,W]` or a single-sample `[1,C,T,H,W]` payload.

### 7.5 Stable frozen modules

Teacher modules are set to eval and have gradients disabled. SmolVLM2 is forced
back to eval whenever the bridge is switched into train mode.

## 8. Why This Is the Best Current One-Pass Strategy

The design is recommended for the current constraints because it balances four
competing objectives:

| Requirement | Design response |
| --- | --- |
| Replace the expensive native text stack | Bridge directly emits all DiT condition tensors |
| Preserve released NeoDragon quality | Frozen-DiT functional loss in Exp 1 and preservation loss in Exp 2 |
| Learn OpenVid video dynamics | Real-latent flow matching remains active throughout Exp 2 |
| Avoid semantic collapse | Token, pooled, relational, mask, and bridge-anchor losses |
| Respect pruned-model fragility | Low DiT LR, warmup, cooldown, teacher-condition preservation |
| Keep diagnosis possible | Bridge-only checkpoint exists before joint training |
| Fit 8-GPU training | Offline VAE latents, bf16, FSDP, reduced functional batch |

It is better than embedding-only distillation because it optimizes the frozen
DiT behavior directly. It is safer than pure full-DiT fine-tuning because the
released vector field remains an explicit target. It is more useful than pure
self-distillation because flow matching still learns from real videos.

The word "best" here means best justified by the current evidence and resource
constraints. It does not mean the loss weights or step counts have been proven
optimal by controlled ablation.

## 9. Known Limitations and Open Questions

### 9.1 Two-GPU FSDP smoke passed; eight-GPU scale remains untested

On 2026-07-15, SLURM job `1938` ran both experiments sequentially with two H200
GPUs and the real model components. Both ranks initialized NCCL, completed two
forward/backward/optimizer steps, gathered full FSDP state dictionaries, and
saved reloadable checkpoints.

The smoke covered:

```text
Exp 1:
  all representation losses
  functional teacher/student DiT response losses
  FSDP bridge update
  full bridge checkpoint save

Exp 2:
  precomputed real NeoDragon VAE latents
  full trainable DiT + trainable bridge
  flow matching
  cross-system response distillation
  teacher-condition preservation
  bridge representation anchoring
  full bridge + DiT checkpoint save
```

All logged losses and every floating-point checkpoint tensor were finite. The
Exp 1 mask head moved away from its zero initialization, and 31 trainable bridge
tensors changed again during Exp 2, confirming that optimizer updates occurred.

The run emitted an FSDP warning that rank 0 had no local gradients for one
wrapper during gradient clipping. With `use_orig_params=True`, most bridge
parameters frozen, and only a small set of trainable tensors, one rank can own
no local trainable gradient shard. FSDP still all-reduced the global norm and
the rank owning gradients updated its shards. The nonzero checkpoint deltas
confirm this warning was not a missing-backward failure.

This smoke validates the two-GPU execution path, not convergence or eight-GPU
scaling. The Berzelius production run should still be watched through its first
checkpoint for rank synchronization, memory use, throughput, and finite losses.

### 9.2 Exact training resume is not implemented

Model checkpoints are saved, but optimizer state is not. A resumed run would
not preserve Adam moments or exact LR schedule state.

### 9.3 Exp 2 does not train latent unit zero

The current objective is continuation-focused. A future full text-to-video DiT
experiment should include unit zero with an explicit sampling probability.

### 9.4 Synthetic states in Exp 1 have no video semantics

Functional distillation in Exp 1 uses scheduler-valid random latent states. This
is efficient and isolates condition behavior, but it does not replace Exp 2 on
real OpenVid latents.

### 9.5 Hyperparameters still require empirical validation

The relative scales of flow, cross-system distillation, preservation, and bridge
anchoring are reasoned defaults. Their gradient magnitudes should be monitored
in the first production logs.

### 9.6 No EMA checkpoint

The trainer saves current weights only. An EMA model could improve evaluation
stability but would increase memory and checkpoint complexity.

## 10. Verification Completed So Far

The implementation has passed:

```text
8 unit tests for config opt-in, translator shape/gradient, condition scale,
complete loss gradients, both ramp implementations, and invalid token length

Python bytecode compilation for both trainers and new modules

bash syntax checks for both Berzelius experiment scripts

Ruff checks under the repository's local-import policy

git diff whitespace checks

CPU sampler check using NeoDragon's real scheduler and pyramid utilities

Full-width translator/head gradient check at [1,128,1536]
```

The full-width architecture check measured an initial mean condition norm of
approximately `30.57`, as intended by the `0.78` scale initialization.

## 11. Recommended Evaluation Sequence

After Exp 1:

1. Plot representation losses, mask accuracy, norm ratio, and functional loss.
2. Run bridge-conditioned inference with the released frozen DiT.
3. Compare against native NeoDragon condition on the same prompts and seeds.
4. Reject the checkpoint if functional loss falls but generations remain
   semantically collapsed.

After Exp 2:

1. Compare OpenVid flow loss and teacher-preservation loss over time.
2. Generate with short, medium, and long prompts using fixed seeds.
3. Compare against the Exp 1 bridge + released DiT baseline.
4. Inspect motion, prompt alignment, flicker, and first-frame consistency.
5. Run VBench only after fixed-seed qualitative sanity passes.

This sequence prevents an expensive benchmark from hiding a basic condition or
generation failure.

## 12. Four-Experiment Matrix

The four runs answer different questions and should not be treated as four
interchangeable hyperparameter variants.

| Experiment | Bridge initialization | DiT initialization | Trainable modules | Video flow loss | Teacher losses | Primary question |
| --- | --- | --- | --- | --- | --- | --- |
| Exp 1 | Random | Released, frozen | Bridge only | No | Representation + frozen-DiT functional | Can the bridge learn the released NeoDragon condition contract before video training? |
| Exp 2 | Exp 1 checkpoint | Released | Bridge + full DiT | Yes | Light representation + cross-response + preservation | Can an already aligned bridge and DiT adapt jointly without forgetting the released model? |
| Exp 3 | Random | Released | Bridge + full DiT | Yes | Complete representation + bridge functional + cross-response + preservation | Can one joint run learn alignment and video generation simultaneously? |
| Exp 4 | Random | Released | Bridge + full DiT | Yes | None | What does native flow matching alone learn from exactly the same initialization and data? |

"From scratch" in Exp 3 and Exp 4 refers only to the Mobile-OV bridge. It does
not mean that SmolVLM2 or NeoDragon is randomly initialized. Both runs use:

```text
frozen pretrained SmolVLM2 understanding backbone
randomly initialized trainable bridge/projector heads
released pretrained NeoDragon DiT, fully trainable from step 1
released pretrained NeoDragon VAE latents prepared offline
```

This distinction matters. Training a pruned NeoDragon DiT from random weights
would discard the released model's learned generative prior and is not what the
new scripts do.

## 13. Experiment 3: Joint Training From a Random Bridge With Full Distillation

### 13.1 Purpose

Exp 3 is the strongest single-run option when a separate bridge pretraining run
is not available. It combines alignment, functional preservation, and real
OpenVid flow matching from the beginning.

The script is:

```text
scripts/exp3_train_neodragon_joint_from_scratch_1node8gpu.sbatch
```

The trainer is shared with Exp 2 and Exp 4:

```text
tools/train_neodragon_dit_bridge.py
```

No `--bridge-ckpt` is passed. The trainer records:

```text
bridge_initialization: random
objective.mode: joint-distill
```

in every checkpoint, so the run cannot later be mistaken for Exp 2.

### 13.2 Forward graph

```text
caption
  |-----------------------------------------------|
  v                                               v
frozen SmolVLM2                              frozen NeoDragon text bundle
  v                                               v
random trainable Mobile-OV bridge             frozen ContextAdapter
  v                                               v
c_student = (tokens, mask, pooled)            c_teacher = (tokens, mask, pooled)
  |                    |                          |
  |                    |                          |
  |                    v                          v
  |             frozen teacher DiT <--------- same noisy latent state
  |                    |                          |
  v                    v                          v
trainable full DiT   teacher response under    teacher response under
  |                  student condition         teacher condition
  v
flow prediction on real OpenVid latent
```

The same offline latent sample, pyramid stage, temporal continuation unit,
noise, and timestep are reused for all DiT comparisons in a step. This removes
sampling variance from the teacher-response losses.

### 13.3 Loss A: complete post-ContextAdapter representation alignment

The student does not imitate raw T5. It imitates the tensor after the released
NeoDragon ContextAdapter because that is the actual DiT interface.

The composite is:

```text
L_repr =
    0.25 L_raw_token_MSE
  + 1.00 L_normalized_token_MSE
  + 0.50 L_token_cosine
  + 0.10 L_token_norm
  + 0.25 L_pooled_MSE
  + 0.20 L_pooled_cosine
  + 0.10 L_relational
  + 0.05 L_mask_BCE
```

Each term addresses a distinct failure mode:

| Term | What it constrains | Why it is retained |
| --- | --- | --- |
| Raw token MSE | Absolute post-adapter values and scale | Cross-attention projections are sensitive to magnitude, not only direction |
| Normalized token MSE | Feature pattern independent of global scale | Prevents a few high-variance channels from dominating raw MSE |
| Token cosine | Direction of each valid token | Preserves semantic orientation when norms are still adapting |
| Token norm | Mean valid-token magnitude | Prevents cosine-good but amplitude-wrong conditions |
| Pooled MSE | Absolute 2048-D global condition | NeoDragon consumes pooled projections separately from token conditions |
| Pooled cosine | Global semantic direction | Stabilizes pooled alignment when its scale changes |
| Relational | Pairwise similarities between prompts | Penalizes semantic collapse where many prompts map to the same condition |
| Mask BCE | Effective sequence length | A wrong mask changes which token positions cross-attention can read |

With one sample per GPU, a local relational loss would always be zero. The joint
trainer therefore gathers bridge and teacher conditions across distributed
ranks with an autograd-aware all-gather. On eight GPUs with batch size one, the
relational term sees a global batch of eight prompts and its gradient still
returns to the local bridge shard.

The outer representation weight starts at `1.0` and cools to `0.1`. Alignment
is therefore strongest while the bridge is random, but remains as an anchor
after real video flow training becomes reliable.

### 13.4 Loss B: frozen-teacher bridge functional distillation

Representation similarity does not guarantee that NeoDragon reacts similarly.
Exp 3 therefore asks the released frozen teacher DiT to process the same noisy
latent twice:

```text
y_teacher = frozen_DiT(x_t, c_teacher)
y_bridge  = frozen_DiT(x_t, c_student)

L_bridge_func = MSE(y_bridge, y_teacher)
              + 0.1 cosine_distance(y_bridge, y_teacher)
```

All frozen teacher-DiT parameters have `requires_grad=False`, but the student
condition is not detached. Autograd passes through the frozen DiT operations
and updates only the bridge. This is intentionally different from the
cross-system loss below.

This loss prevents a dangerous shortcut: if only the trainable DiT consumed a
bad bridge, the DiT could adapt to that private representation and lower the
training loss without making the bridge compatible with released NeoDragon
behavior. A fixed teacher consumer cannot move to accommodate the bridge.

The functional path uses one sample per rank and runs every four steps by
default. On active steps it is multiplied by four, preserving approximately the
requested expected contribution while reducing teacher-DiT compute. Its scale
ramps from zero during the first 2,000 steps because a completely random bridge
can otherwise produce a very noisy functional gradient. It cools to `0.1` near
the end instead of disappearing.

### 13.5 Loss C: native NeoDragon flow matching

For one sampled temporal continuation unit and pyramid stage:

```text
x_t = sigma * epsilon + (1 - sigma) * z_clean
v_target = epsilon - z_clean
v_student = trainable_DiT(x_t, c_student)

L_flow = MSE(v_student, v_target)
```

This is the actual generative learning objective. The teacher losses can make
the replacement condition compatible, but they cannot teach the model to fit
the OpenVid latent distribution. Flow matching updates both the bridge through
its condition tensors and every trainable NeoDragon DiT parameter.

The flow coefficient ramps from `0.05` to `0.30` over 2,000 steps. This keeps
the real video objective active from step 1 while avoiding a large early update
from meaningless random conditions. It cools to `0.10` during the final 10,000
steps together with the learning rate.

### 13.6 Loss D: cross-system teacher-response distillation

```text
y_student = trainable_DiT(x_t, c_student)
y_teacher = frozen_DiT(x_t, c_teacher)

L_cross = MSE(y_student, y_teacher)
        + 0.1 cosine_distance(y_student, y_teacher)
```

Unlike bridge functional distillation, `L_cross` updates both the bridge and the
trainable DiT. It gives the entire new Mobile-OV + NeoDragon system a behavior
target while flow matching adapts it to OpenVid.

MSE is primary because flow velocity magnitude matters to the scheduler.
Cosine distance is secondary because it protects response direction when the
absolute scale is temporarily changing.

### 13.7 Loss E: teacher-condition DiT preservation

The trainable DiT is also evaluated using the correct native teacher condition:

```text
y_preserve = trainable_DiT(x_t, c_teacher)
y_teacher  = frozen_DiT(x_t, c_teacher)

L_preserve = 0.5 MSE(y_preserve, y_teacher)
           + 0.05 cosine_distance(y_preserve, y_teacher)
```

This isolates DiT drift from bridge error. If the trainable DiT forgets released
NeoDragon behavior, it cannot blame a poor student condition because both sides
receive the same teacher condition.

The preservation forward runs every four steps and is frequency-scaled by four.
It updates the trainable DiT but not the bridge because `c_teacher` is detached.

### 13.8 Complete Exp 3 objective and gradient destinations

```text
L_exp3 =
    w_flow(t) * L_flow
  + L_cross
  + frequency_preserve * L_preserve
  + w_repr(t) * L_repr
  + frequency_bridge_func * w_bridge_func(t) * L_bridge_func
```

| Objective | Updates bridge | Updates trainable DiT | Uses frozen teacher DiT |
| --- | --- | --- | --- |
| `L_repr` | Yes | No | No |
| `L_bridge_func` | Yes | No | Yes, with gradient only to condition input |
| `L_flow` | Yes | Yes | No |
| `L_cross` | Yes | Yes | Yes, target under `no_grad` |
| `L_preserve` | No | Yes | Yes, target under `no_grad` |

This combination is deliberately redundant. Every major failure has an
independent signal: representation mismatch, function mismatch, video-data
mismatch, whole-system teacher mismatch, and DiT forgetting.

### 13.9 Why full-DiT training is still reasonable for a pruned model

NeoDragon's released DiT is already pruned and trained. Pruning reduces its
capacity but does not make its remaining weights untouchable. Exp 3 starts from
those released weights, uses a small DiT learning rate (`3e-6`), warms it up,
and continuously regularizes its response against a frozen copy. The objective
is adaptation of the surviving network, not relearning the original model from
random weights.

The main risk is still catastrophic drift because the random bridge produces an
out-of-distribution condition early. The low initial flow weight, LR warmup,
direct representation targets, frozen-consumer functional loss, and
teacher-condition preservation all specifically reduce this risk.

## 14. Experiment 4: Matched Joint Flow-Only Baseline

### 14.1 Purpose

Exp 4 answers whether the full teacher machinery is actually necessary. It is
not a weaker implementation of Exp 3; it is the controlled ablation needed to
interpret Exp 3.

The script is:

```text
scripts/exp4_train_neodragon_flow_only_from_scratch_1node8gpu.sbatch
```

It uses:

```text
--objective-mode flow-only
```

The trainer rejects the run before model loading if any representation,
functional, cross-response, or preservation teacher weight is nonzero. It then
does not instantiate the NeoDragon TextEncoderBundle, ContextAdapter, or frozen
teacher DiT. This saves memory and guarantees that the experiment is pure.

### 14.2 Matched initialization and schedule

Exp 3 and Exp 4 use the same defaults for:

```text
seed = 2026
random bridge architecture and initialization
released pretrained NeoDragon DiT initialization
full-DiT unfreezing from step 1
OpenVid offline latent manifest
short/medium/long caption sampling weights = 5/4/1
batch size per GPU = 1
DiT LR = 3e-6
bridge LR = 1e-5
warmup = 2,000 steps
flow coefficient = 0.05 -> 0.30 -> 0.10
cooldown = final 10,000 steps
default total = 200,000 steps
```

Consequently, the intended independent variable is teacher supervision. If the
two experiments are launched from the same code revision and latent manifest,
their comparison is meaningful.

### 14.3 Only optimized objective

```text
L_exp4 = w_flow(t) * MSE(
    trainable_DiT(x_t, bridge(prompt)),
    epsilon - z_clean
)
```

Both the bridge and full DiT receive gradients. There is no embedding MSE,
teacher output loss, preservation forward, or teacher mask target.

This baseline may learn a useful private condition language between bridge and
DiT. It may also collapse text semantics while still reducing average flow
loss. That ambiguity is why diagnostics are required.

### 14.4 Diagnostics that do not alter training

Every 100 steps by default, the trainer gathers detached conditions across
ranks and shifts them so each latent receives a different rank's caption. It
then runs two no-gradient DiT forwards:

```text
correct = DiT(x_t, c_correct)
wrong   = DiT(x_t, c_from_another_prompt)
```

The log records:

| Metric | Interpretation |
| --- | --- |
| `diagnostic_correct_flow_loss` | Post-update flow error under the correct condition |
| `diagnostic_shuffled_flow_loss` | Flow error under a mismatched caption condition |
| `diagnostic_text_sensitivity` | MSE between correct-condition and shuffled-condition outputs |
| `diagnostic_condition_offdiag_cos` | Average cosine similarity between different prompt embeddings |

None of these values is added to `loss`. They cannot improve Exp 4 directly and
therefore do not compromise the flow-only ablation.

A healthy text-conditioned model should eventually show a positive shuffled
caption penalty and nonzero output sensitivity. A low flow loss together with
near-zero sensitivity is evidence that the DiT is ignoring the bridge.

## 15. Why Exp 3 and Exp 4 Share One Trainer

Duplicating the joint training loop would make the ablation unreliable. Small
differences in noise sampling, latent stage selection, scheduler math, caption
augmentation, optimizer construction, or FSDP wrapping could explain a score
difference.

Both scripts therefore call the same trainer. `--objective-mode` controls only
teacher availability and validates the loss contract. The native flow path,
data loader, random seeding, optimizer, FSDP wrappers, checkpoint writer, and
diagnostics are shared.

The random seed is applied in two stages:

```text
before model construction:
  identical seed on all ranks
  -> reproducible random bridge initialization

after FSDP synchronization:
  seed + global rank
  -> distinct caption choice per rank

dedicated per-rank torch.Generator:
  seed + global rank + 100,000
  -> noise, timestep, stage, and temporal unit sampling
```

This avoids identical distributed samples without allowing each rank to start
from a different bridge. The dedicated sampling generator also prevents the
extra teacher forwards in Exp 3 from changing future latent/noise samples
relative to Exp 4 if a model implementation consumes the global RNG.

## 16. Distributed Training and Checkpoint Contract

Exp 3 and Exp 4 use two separate FSDP roots, one for the DiT and one for the
bridge. `use_orig_params=True` preserves mixed frozen/trainable parameter state
inside the bridge, where SmolVLM2 is frozen but projector heads are trainable.

All ranks participate in full-state gathering. Rank 0 writes:

```text
neodragon_dit_bridge_latest.pt
neodragon_dit_bridge_stepXXXXXX.pt
history.json
```

The checkpoint contains:

```text
dit
bridge
bridge_ckpt or null
bridge_initialization = random or checkpoint
objective.mode = joint-distill or flow-only
objective flags for every teacher path
teacher_modules_loaded
config
all CLI arguments
loss history
parallel backend and world size
```

Exp 3 and Exp 4 currently save model weights but not Adam optimizer state. They
are checkpointable for inference and weight initialization, but not exact
optimizer-state resume. This limitation is unchanged from Exp 2 and should be
fixed before relying on preemption recovery in a long production run.

## 17. Running Exp 3 and Exp 4 on Berzelius

Both scripts default to the repository-local offline latent manifest:

```text
data/openvid_neodragon_2s_latents/latent_manifest.csv
```

Exp 3:

```bash
sbatch scripts/exp3_train_neodragon_joint_from_scratch_1node8gpu.sbatch
```

Exp 4:

```bash
sbatch scripts/exp4_train_neodragon_flow_only_from_scratch_1node8gpu.sbatch
```

Override examples:

```bash
MANIFEST=/path/to/latent_manifest.csv \
STEPS=200000 \
SAVE_EVERY=10000 \
sbatch scripts/exp3_train_neodragon_joint_from_scratch_1node8gpu.sbatch
```

```bash
MANIFEST=/path/to/latent_manifest.csv \
STEPS=200000 \
SAVE_EVERY=10000 \
sbatch scripts/exp4_train_neodragon_flow_only_from_scratch_1node8gpu.sbatch
```

The default production output roots are:

```text
output/neo_exp3_joint_from_scratch/<SLURM_JOB_ID>/
output/neo_exp4_flow_only/<SLURM_JOB_ID>/
```

## 18. Decision Rules After Training

Exp 3 is preferred if it preserves native prompt behavior, improves video flow
fit, and remains semantically sensitive. Exp 4 is valuable even if it performs
worse because it quantifies how much teacher supervision contributes.

The comparison should include:

1. Fixed-seed prompt generations from short, medium, and long captions.
2. Native NeoDragon generation with the same prompt and seed.
3. Correct-vs-shuffled prompt sensitivity.
4. Flow loss on a held-out latent subset.
5. VBench after qualitative generation passes.
6. Bridge condition diversity and mask statistics.
7. DiT response drift under native teacher conditions.

Interpretation:

| Outcome | Meaning |
| --- | --- |
| Exp 3 good, Exp 4 collapsed | Teacher alignment and preservation are necessary |
| Both good | Flow matching can establish a useful private bridge-DiT interface; teacher losses mainly improve compatibility |
| Exp 4 good, Exp 3 worse | Teacher weights or schedules over-constrain OpenVid adaptation |
| Both poor | Random-bridge joint training is too hard, the latent objective/data path is wrong, or a separate Exp 1 stage is necessary |

The first production checkpoint should be treated as a gate. Continue only if
all losses are finite, all ranks advance, bridge parameters change, shuffled
condition sensitivity is nonzero, and fixed-seed generations do not regress to
obvious semantic collapse.

## 19. Exp 3/4 Implementation Verification on 2026-07-16

The following checks passed after implementing Exp 3 and Exp 4:

```text
Python bytecode compilation
8/8 unit tests
Ruff checks, excluding the repository's intentional local-import E402 pattern
Bash syntax for all four production scripts
git diff whitespace validation
flow-only fail-fast contract with a deliberately nonzero teacher weight
two-rank Gloo test for differentiable all-gather and shifted-condition gather
one-H200, batch-size-two, two-step real-model preflight for Exp 3 and Exp 4
```

The one-GPU preflight was SLURM job `2109`. It completed both experiments in 34
seconds, wrote consolidated bridge + DiT checkpoints, and returned exit code 0.
The test used four previously encoded real NeoDragon latent samples.

Exp 3 verification:

```text
bridge_initialization = random
objective.mode = joint-distill
teacher_modules_loaded = true
trainable DiT parameters = 1,512,155,200
trainable bridge parameters = 27,018,951
flow, cross-response, complete representation, and bridge-functional losses finite
all saved bridge and DiT tensors finite
```

Exp 4 verification:

```text
bridge_initialization = random
objective.mode = flow-only
teacher_modules_loaded = false
trainable DiT parameters = 1,512,155,200
trainable bridge parameters = 27,018,951
all teacher and functional losses exactly zero
flow loss finite
all saved bridge and DiT tensors finite
```

The preflight's four rows intentionally share smoke-test captions, so its text
sensitivity values validate diagnostic execution but are not meaningful model
quality measurements. Production OpenVid rows contain diverse captions.

The two-H200 FSDP smoke passed as SLURM job `2108`:

```text
state = COMPLETED
exit code = 0
elapsed = 54 seconds
world size = 2
Exp 3 checkpoint = 4.0 GB
Exp 4 checkpoint = 4.0 GB
Exp 3 peak allocated memory per rank = 22.8 GiB
Exp 4 peak allocated memory per rank = 15.0 GiB
```

Both NCCL ranks initialized, completed two forward/backward/optimizer steps,
executed shuffled-condition diagnostics, gathered full FSDP bridge and DiT
states, and exited cleanly. Exp 3 produced finite cross-response,
representation, relational, and bridge-functional losses. Exp 4 did not load
teacher modules and every teacher loss remained exactly zero. Every one of the
533 bridge tensors and 585 DiT tensors in each saved checkpoint was finite.

A final one-H200 gradient-logging preflight passed as SLURM job `2112`. It
confirmed nonzero DiT and bridge gradient norms in both modes:

```text
Exp 3 step 1: DiT grad norm = 15.1875, bridge grad norm = 266.0
Exp 3 step 2: DiT grad norm =  7.5938, bridge grad norm =  34.5
Exp 4 step 1: DiT grad norm =  4.2812, bridge grad norm =  40.0
Exp 4 step 2: DiT grad norm =  0.8984, bridge grad norm =   3.625
```

These are pre-clipping norms from a two-step smoke, not convergence metrics.
Their purpose is to verify that both the full DiT and random bridge receive a
real training signal in Exp 3 and in the pure flow-only Exp 4 path.
