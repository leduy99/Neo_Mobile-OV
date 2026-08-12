# NeoDragon Pyramidal-DMD Reproduction

## Scope and Name

This work is deliberately separate from every previous Mobile-OV bridge,
OpenVid flow-matching, OPSD, and image-conditioning experiment. Its name in
code, checkpoints, and Slurm jobs is **NeoDragon Pyramidal-DMD Reproduction**.
It targets one question only:

> Can the released multi-step NeoDragon DiT supervise a new conditional
> one-step 1-1-1 video DiT through the Pyramidal Distribution Matching
> Distillation procedure described in the NeoDragon paper?

The first frame is not part of this DiT distillation. A released Hybrid
NeoDragon video begins from an external first-frame latent and generates six
video latent units. Each unit has three Pyramidal stages, so the deployed
student has exactly `6 x 3 x 1 = 18` conditional DiT calls.

## What Is Implemented

The pipeline has three explicit stages:

1. **Synthetic teacher data**: `prepare_neodragon_dmd_synthetic_data.py` takes
   prompts only. It uses the released native multi-step model at `20/10` steps
   with native CFG and writes seven latent units per prompt. Unit zero is the
   teacher anchor; the other six are video targets. Raw OpenVid pixels or VAE
   latents are not used as the DMD target distribution.
2. **Pyramidal DMD training**: `train_neodragon_pyramidal_dmd.py` loads three
   copies of the released multi-step DiT. The frozen copy is the teacher. The
   student and fake model both initialize from the same multi-step checkpoint.
   No Mobile-OV bridge is loaded or optimized.
3. **Controlled student rollout**: `evaluate_neodragon_pyramidal_dmd.py` keeps
   a synthetic teacher anchor fixed and runs the student for all six video
   units with the `1-1-1` schedule. This isolates the distilled DiT from any
   first-frame generator while checking the deployed call topology.

The primary implementation files are:

- `new_mobile_ov/training/neodragon_pyramidal_dmd.py`
- `tools/prepare_neodragon_dmd_synthetic_data.py`
- `tools/train_neodragon_pyramidal_dmd.py`
- `tools/evaluate_neodragon_pyramidal_dmd.py`

## Training Mechanics

For a clean teacher endpoint `z`, stage `i` has released scheduler endpoints
`sigma_start[i]` and `sigma_end[i]`. The trainer constructs the Pyramidal
stage endpoints with the same noise sample `epsilon`:

```text
y_start = (1 - sigma_start) * Up(Down(z, 2), 2) + sigma_start * epsilon
y_end   = (1 - sigma_end)   * z                    + sigma_end * epsilon
```

At stage zero, `Up(Down(z))` is replaced by zero, so the stage starts from
pure noise. The one-step student predicts a velocity from `y_start`, then its
endpoint is the release scheduler's one Euler update:

```text
z_student = y_start - D_student(y_start, t_start)
```

The student itself is conditional-only: it makes one DiT call at deployment.
The frozen multi-step teacher uses native video CFG `5.0` to transfer the
teacher's guided distribution into that single conditional call. The synthetic
anchor already exists, so none of the six DMD targets is the monolithic
first-image call that normally uses CFG `7.0`.

For a local stage noise `tau`, the endpoint is re-noised:

```text
z_tau = (1 - tau) * y_end(z_student) + tau * y_start(z_student)
```

The fake model learns ordinary Pyramidal Flow Matching on this **detached
current student endpoint distribution**:

```text
L_fake = MSE(D_fake(z_tau, tau), y_start(z_student) - y_end(z_student))
```

The fake optimizer runs twice per student update, matching the paper's `1:2`
student:fake ratio. The student receives the Distribution Matching direction
from the difference between frozen teacher and fake velocities. The scalar
surrogate is only a mechanism to inject this gradient, so it is logged as
`dmd_surrogate` and must not be interpreted as an ordinary decreasing loss.

```text
g_DMD = w * (D_teacher(z_tau, tau) - D_fake(z_tau, tau))
w     = clamp(1 / L1(D_teacher, y_start - y_end))
```

The supervised endpoint stabilizer is the Cauchy term described in the paper:

```text
L_cauchy = log(1 + ||z_student - z||_2^2)
L_student = 1.0 * L_DMD + 0.5 * L_cauchy
```

Student `tau` cycles through four fixed values: `0.125`, `0.375`, `0.625`, and
`0.875`. Fake `tau` is sampled uniformly in `[0, 1]`. Every rank cycles the
same six-unit by three-stage schedule, preventing DDP ranks from taking
different DiT branches.

## Important Reproduction Limits

This is a faithful implementation of the public algorithm and released
components, but it cannot yet be called a numerically exact paper
reproduction:

- The paper uses a lower-resolution Pyramidal-Flow checkpoint and a curated
  approximately 350k prompt collection. Neither is released with the public
  NeoDragon code or model snapshot.
- This implementation therefore uses the released `320p` multi-step DiT and
  a deterministic, de-duplicated sample from a supplied prompt CSV. The prompt
  source is recorded in `metadata.json`.
- The context adapter remains the frozen released multi-step context adapter.
  The student checkpoint records that adapter ID. It must not silently be
  paired with the released Hybrid context adapter during evaluation.
- The procedure distills the video DiT only. A complete Hybrid model still
  needs a first-frame generator trained or aligned separately.

These differences are explicit so a positive or negative result is not
mistaken for a direct comparison with the paper's unpublished data and
lower-resolution teacher.

## Berzelius Commands

Run from the repository root. The synthetic-data job resumes at individual
latent files, so resubmitting after the 72-hour walltime is safe. The manifest
is written after all requested samples have completed.

```bash
# 1. Generate the paper-scale teacher data. Re-run the same command if the
#    72-hour allocation ends before all 350k samples have completed.
MAX_PROMPTS=350000 sbatch scripts/prepare_neodragon_pyramidal_dmd_synthetic_1node8gpu.sbatch

# 2. Train the standalone DMD student/fake pair for the paper's 5k iterations.
RUN_NAME=dmd_repro_v1 sbatch scripts/reproduce_neodragon_pyramidal_dmd_1node8gpu.sbatch

# 3. Evaluate the student DiT on a fixed synthetic anchor.
CHECKPOINT=output/neodragon_pyramidal_dmd_reproduction/dmd_repro_v1/neodragon_pyramidal_dmd_student_latest.pt \
sbatch scripts/evaluate_neodragon_pyramidal_dmd_1gpu.sbatch
```

For a small end-to-end DDP validation before the full data run:

```bash
sbatch scripts/smoke_reproduce_neodragon_pyramidal_dmd_2gpu.sbatch
```

The full job defaults are global batch `8`, student learning rate `1e-6`, fake
learning rate `1e-6`, `1:2` student:fake updates, and `5000` training steps.
It overwrites a deployable student checkpoint and resumable state every `500`
steps while retaining a student archive at step `5000`. The resumable state is
intentionally much larger because it contains the student, fake model, and
both optimizer states.

## Metrics and Decision Criteria

The trainer writes `history.json` with the following values:

- `endpoint_mse`: supervised error from student endpoint to the teacher
  synthetic endpoint. It should remain finite and generally decrease, but it
  is not sufficient to prove distribution matching.
- `cauchy`: stabilizing endpoint term. This should remain finite; a collapse to
  near zero while rollout quality fails is a warning that this term dominates.
- `fake_mse`: Pyramidal Flow Matching error of the fake model. It should not
  diverge and should track the current student distribution.
- `teacher_pf_error`: frozen teacher error on the re-noised student endpoint.
  This checks that DMD probes remain in a region where the teacher provides a
  meaningful flow direction.
- `direction_rms` and `mean_sample_weight`: DMD gradient-health diagnostics.
  A near-zero direction for the whole run means the distribution term is not
  contributing; runaway weights indicate an unstable inverse-error region.
- `dmd_surrogate`: do not compare this as a conventional loss. Its sign and
  magnitude depend on the gradient-injection construction.

The first quality gate is the controlled evaluator: it must execute all 18
student calls, produce finite `[B,16,7,40,64]` latents, and beat a randomly
initialized student on held-out synthetic teacher samples. The second gate is
visual/video evaluation with a fixed first-frame protocol. Only after these
two gates should the student be considered for a full first-frame integration
or mobile pipeline benchmark.

## Local Validation Performed

The implementation was tested before this document was written:

| Check | Result |
| --- | --- |
| Static compilation | Passed for all generator, trainer, evaluator, and Slurm scripts. |
| Unit tests | `3 passed` for stage construction, scheduler time orientation, DMD gradient injection, and Cauchy loss. |
| One-H200 synthetic data smoke | Passed: native multi-step teacher wrote one valid `[16,7,40,64]` latent. |
| One-H200 DMD smoke | Passed: one fake update and one student update wrote student, archive, and resumable checkpoints. |
| Controlled one-step rollout | Passed: executed `6 x 3 x 1` student calls, preserved the anchor exactly, and emitted finite seven-unit latent output. |
| Native teacher timing | A true `20/10` teacher sample took approximately `9.9s` after model warm-up on one local H200. |

The local scheduler temporarily refused a two-GPU allocation with
`QOSMaxGRESPerUser`, so this environment could not perform the final NCCL DDP
smoke itself. The supplied two-GPU script is the exact end-to-end test to run
on Berzelius before the 8-GPU paper-scale jobs.

At the measured warm throughput, 350k teacher samples on eight equivalent
GPUs are approximately five days of pure generation. The 72-hour job is
therefore intentionally resumable; use one or more resubmissions rather than
changing the target set mid-run.
