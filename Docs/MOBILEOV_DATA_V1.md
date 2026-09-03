# MobileOV-Data-v1

## Why This Release Exists

Earlier Mobile-OV runs changed data, losses, teacher targets, and first-frame behavior at the same time. A failed run therefore could not tell us whether the model recipe was wrong or whether the supervision itself contradicted deployment. MobileOV-Data-v1 freezes one traceable data contract before the next full training run.

The release does not pretend that all examples are interchangeable. It preserves six separate pools and records the source, prompt variants, verification evidence, capability tags, split family, and physical artifact for every row.

## The Six Pools

| Pool | Purpose | Required evidence |
|---|---|---|
| `image_broad` | Broad language coverage for DreamLite condition distillation | Clean caption variants |
| `image_compositional` | Counting, binding, color, relations, scenes, actions, and styles | Capability tags and clean caption variants |
| `image_grounded` | Keep the condition tied to visible content | Decodable image plus the frozen V12 verification record |
| `video_real` | Real temporal dynamics for monolithic continued training | Decodable OpenVid clip, NeoDragon latent, multi-frame SigLIP2 score, and motion checks |
| `video_teacher_t2v` | Preserve released native NeoDragon behavior and support DMD | Native monolithic teacher trajectory generated from text |
| `video_anchor_teacher` | Train exactly the deployment boundary that was previously missing | DreamLite V11 anchor and native monolithic continuation generated from that same image |

The existing T2V trajectories are retained because they preserve native generation behavior. They are not mislabeled as external-anchor data. The new anchor-conditioned pool is separate, so a training recipe can control how much deployment adaptation is applied without erasing native motion.

## Quality Cascade

### Image data

The image pools reuse the immutable ImageBridge-Data-v1/V12 release. It already removes VBench-near prompts, verifies image readability, ranks grounded pairs with SigLIP2, and uses Qwen3.6 only to adjudicate difficult candidates. MobileOV-Data-v1 does not rewrite those captions or silently change the prior release.

### Real video data

`score_mobileov_video_data.py` samples six frames from each 49-frame clip. A pair must satisfy all of the following:

1. Every sampled frame is readable.
2. A multi-frame contact sheet has acceptable SigLIP2 agreement with the caption.
3. RGB frame change and optical flow are above the bottom motion decile, removing near-static clips.
4. The largest adjacent-frame change is below the top transition percentile, removing likely hard cuts or corruption.

The thresholds are computed from the scored source distribution and saved in `video_high_precision.summary.json`; they are not hidden constants. The default job scores up to 500K existing OpenVid latent records on eight GPUs.

### Deployment-matched teacher data

The prompt bank is deterministic and benchmark-clean. By default it draws 80% from verified dynamic video captions and 20% from image sources that explicitly contain action, with compositional prompts prioritized inside each source. If an image quota cannot be filled, the builder backfills from the globally verified prompt pool rather than inventing synthetic grammar.

For every selected prompt:

1. DreamLite-mobile with the V11-balanced SmolVLM2 image bridge generates an anchor.
2. The released native NeoDragon monolithic teacher receives that exact image and the same prompt.
3. The saved tensor contains the anchor unit and all six autoregressive teacher video units.
4. The payload stores the anchor SHA-256, generation seed, teacher identity, bridge checkpoint, and the explicit contract `same_dreamlite_anchor_native_teacher`.

This avoids the invalid pairing of a DreamLite image with an unrelated OpenVid future.

## Canonical Schema

Every frozen manifest uses the same columns, including `record_id`, `split`, `pool`, `task`, `prompt`, three caption lengths, `capabilities`, source provenance, artifact paths, generation settings, verification evidence, and `family_id`. Paths are resolved when the release is frozen so that training cannot accidentally reinterpret a relative latent path from another directory.

The train/validation split is derived from normalized prompt identity. The same concept therefore cannot enter both splits through different source manifests. Exact and high-overlap VBench prompts are removed before export.

## Build The Full Release

On Berzelius, from the repository root:

```bash
git pull --ff-only
bash scripts/submit_mobileov_data_v1.sh
```

This submits three dependent jobs:

1. `build_mobileov_video_cascade_v1_1node8gpu.sbatch` scores and filters real videos, then freezes the 100K prompt bank.
2. `prepare_mobileov_anchor_teacher_v1_1node8gpu.sbatch` generates exact DreamLite-anchor/native-teacher trajectories and resumes safely after interruption.
3. `freeze_mobileov_data_v1_1node1gpu.sbatch` writes the canonical manifests and runs the release preflight.

The final gate is:

```text
data/mobileov_data_v1/releases/v1/stats/preflight.json
```

Training may use the release only when `passed` is `true`. The preflight verifies source and output hashes, row counts, schema, global record uniqueness, split-family isolation, sampled image decoding, sampled latent shapes, and sampled anchor-to-trajectory hashes.

## Reproducibility Boundary

`configs/mobileov_data_v1.yaml` is the paper-facing data specification. It fixes source roles, paths, quotas, caption priorities, benchmark filters, and the prompt-bank seed. `release.json` and `release_summary.json` record the exact resulting artifacts and hashes. Sampling ratios for a later model recipe must be versioned separately; changing a training ratio must not silently create a new dataset.

This release removes data construction as an uncontrolled variable. It does not guarantee that a particular loss is correct. If a model still collapses after the preflight passes, the remaining investigation can focus on optimization, teacher objectives, and architecture rather than guessing whether OpenVid clips, captions, or first-frame pairs were malformed.
