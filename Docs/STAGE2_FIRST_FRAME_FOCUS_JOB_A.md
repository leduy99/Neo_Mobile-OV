# Job A: first-frame-focused Stage-2 continuation

Recipe agreed on 2026-10-15. This replaces the earlier Job-A proposal that
added history corruption and EMA. It is not an exact UniVideo reproduction,
an isolated reconstruction ablation, or a guarantee of quality improvement.

## Fixed recipe

- Start from completed Stage 2, step 283977, both trained connector and DiT.
- Parent SHA256: `a74fd3069d17437b674415268c5157a85d94223509121220572e2d2fcdb68053`.
- Train MCP + monolithic DiT; freeze SmolVLM2 and VAE. No teacher, LoRA, EMA,
  new image generator, synthetic data or history corruption.
- Native pyramidal flow MSE: `0.5 mean(T2I) + 0.5 mean(video) + 0.2 mean(reconstruction)`.
- Reconstruction reuses the paired T2I image and its VAE sample. The VLM sees
  the image only; the DiT never receives the clean image latent as condition.
- T2V selects unit zero with probability 0.5; otherwise uniformly selects a
  future unit. Reference mode stays uniform over units 0-6; anchor mode over
  units 1-6. T2V never sends its target image/vision tokens into the VLM.
- Sample pyramid stages uniformly, unchanged. Keep geometry, 49 whole-clip
  frames, caption policy, VAE window 8 and condition dropout 0.1 unchanged.
- AdamW FP32 masters, BF16 compute, LR `2e-5`, local 50-update warmup, betas
  `(0.9, 0.95)`, eps `1e-8`, zero weight decay, gradient clipping one.

The optimizer is NEW at the fork. This is not an exact optimizer continuation
from epoch 3. Subsequent pauses resume this new optimizer and per-rank RNG.
FP32 parent weights are loaded without an intermediate BF16 rounding step.

## Data and exposure

Use `download_data/data/univideo_stage1_expanded650`, unchanged and identity
matched to the parent: 984636 training images, 529902 training videos.
The raw image source is BLIP3o long-caption; videos are Vchitect DataVerse,
public task substitutes rather than UniVideo's original internal corpus.

| Budget | Value |
| --- | ---: |
| GPUs / accumulation / global media batch | 8 / 2 / 16 |
| Additional media passes | 1, absolute epoch 4 |
| Additional optimizer updates | 94659 |
| Final absolute step | 378636 |
| Additional paired reconstructions | 984636 |
| Additional videos, all three modes combined | 529902 |

Every media record is used once more. A video receives ONE of the rotating
modes in this pass, not all three. Sampling order, mode rotation and private
noise seeds start at the fourth-pass offset, not epoch zero. Padding has zero
loss and is not counted as exposure.

Decode/hash checks do not certify caption semantics. The CPU preflight writes
representative training previews and complete captions to
`OUT/data_audit/audit.md`. Inspect whether captioned objects/scenes occur in
the first frame; no automatic filtering or relabeling is performed. A changed
release is rejected rather than silently changing the experiment or budget.

## Berzelius launch

After this code is available on Berzelius, submit from the repository root:

```bash
mkdir -p logs
sbatch scripts/train_mobileov_stage2_first_frame_focus_1node8gpu.sbatch
```

Default parent:
`output/mobileov_stage2_media3epochs/17641364/stage2_dit_connector_latest.pt`.
If absent, the launcher tries the verified local HF cache of step283977. It
never falls back to the older step130000 checkpoint and does not download one.

Output: `output/mobileov_stage2_first_frame_focus/JOBID`.
Log: `logs/mov-s2-ffocus-JOBID.out`.
Reserve approximately 110 GiB of output space for ten inference snapshots,
the larger resume checkpoint and temporary atomic saves; quotas also matter.

Save/archive every 10000 NEW updates; validate every 5000. Filenames retain
absolute steps, e.g. first archive step293977 and final step378636. Inference
snapshots exclude optimizer/RNG/SmolVLM2/VAE. `stage2_resume.pt` includes the
new optimizer and per-rank RNG. The launcher pauses before the time limit and
queues an `afterok` continuation with the same output directory by default.

## Verification and evaluation

History logs include `optimizer_step`, absolute `step`, epoch 4, separate
`t2v.first`/`t2v.future` means and counts, reconstruction and gradient norms.
Missing categories remain null, not zero. Validation retains per-stage
first/last-unit losses. Loss reduction alone is not evidence of better images.

Evaluate text-only first-frame/T2I quality first, then full T2V, image-condition
and latent-anchor routes with matched raw/QuickSR protocols. Anchor scores
alone can conceal a persistent text-to-first-frame problem.

Targeted CPU tests passed (170), including native small pyramidal blocks,
two-rank DDP/global-gradient equivalence, legacy regression, FP32 init, sampler
offset, guard failures and exact stochastic pause/resume. The actual frozen
SmolVLM2 tensor/processor identities match the epoch-3 parent on CPU.

The actual two-GPU smoke launcher is
`scripts/smoke_mobileov_stage2_first_frame_focus_local.sbatch`. It is explicitly
limited to at most ten updates on tiny local data, tests fork/pause/resume and
T2I/T2V inference, and marks outputs as NOT quality-training checkpoints.
At implementation time, local job 6653 was pending GPU availability; these
CPU results must not be described as a completed actual-GPU smoke.
