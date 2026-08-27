# ImageBridge-Data-v1

## Objective

ImageBridge-Data-v1 is the reproducible data layer for DreamLite Image Bridge
V12. It addresses the semantic failures identified by the V8/V9 ablations
without turning the training set into an undocumented collection of generated
prompts.

The core rule is simple: **preserve source captions and use Qwen3.6 only as a
visual verifier, not as a free-form caption generator.** This avoids grammar
artifacts, invented facts, and an untraceable prompt distribution.

## Three Disjoint Pools

| Pool | V12 probability | Purpose |
|---|---:|---|
| D1 broad | 70% | Preserve general image quality, coverage, and the stable V8 behavior. |
| D2 compositional | 20% | Increase count, color binding, multiple objects, spatial relations, scenes, actions, and rendered text. |
| D2 grounded | 10% | Use real image-caption pairs selected by the auditable SigLIP2 + Qwen3.6 cascade. |

The builder assigns each deduplicated caption to exactly one pool. A grounded
record takes precedence over compositional and broad records, while
`source_names` retains every source in which the caption appeared.

The 70/20/10 values are **sampling probabilities**, not target file sizes. The
trainer reads broad and compositional manifests with normalized weights 7/9 and
2/9, gives the grounded source zero normal-loader weight, and explicitly samples
a grounded batch with probability 0.10.

## Inputs

The default Berzilius job uses existing project assets:

- JourneyDB train-ready captions;
- Short-Caption source captions;
- the previously path-verified Short-Caption image manifest;
- optional additional compositional manifests supplied explicitly.

No LAION/COYO download is required for this first controlled V12 experiment.
The catalog stores source path, source key, source role, source file metadata,
and optionally the SHA-256 of every input manifest.

## Eight-Hour Build And Verify

Run from the `Neo_Mobile-OV` repository on Berzilius.

1. Create the isolated Qwen3.6 environment once. This keeps Transformers 5.x
   away from the pinned DreamLite/NeoDragon training environment.

   ```bash
   sbatch scripts/setup_image_bridge_data_env_1node1gpu.sbatch
   ```

2. Submit the complete two-GPU cascade. It builds the immutable catalog when
   needed, scores at most 160K grounded candidates with SigLIP2, sends a balanced
   12K hard subset to Qwen3.6, and freezes the final manifests.

   ```bash
   sbatch scripts/build_image_bridge_grounding_cascade_8h_1node2gpu.sbatch
   ```

   The job requests two GPUs because Qwen3.6-35B does not fit safely on one
   40GB A100 without CPU offload. It prints the allocated GPU model at startup,
   uses GPU 0 for batched SigLIP2 scoring, then lets Qwen use both GPUs. Qwen is
   bounded to three hours. SigLIP scoring is also bounded to three hours and
   must produce at least 110K valid output rows. This reserves up to two hours
   for selection, model loading, merging, validation, and final writes before
   the eight-hour SLURM limit.

3. Existing pilot work is not discarded. The cascade bootstraps valid rows from
   `qwen36_pilot.jsonl`, reuses accepted records, and retries records whose latest
   Qwen result is an error. There is no second 20K pilot.

4. Qwen uses a compact factual-verification schema:

   ```json
   {"caption_supported": true, "confidence": 0.0, "failed_claims": []}
   ```

   It verifies only balanced hard/ambiguous records and previous errors. It does
   not rewrite captions. SigLIP2 ranks the remaining records but is never treated
   as a factual annotation model.

5. Create a deterministic 100-accepted/100-rejected human audit sheet when the
   Qwen run finishes.

   ```bash
     /proj/cvl/users/x_fahkh2/envs/neo_mobileov/bin/python tools/sample_image_bridge_qwen_audit.py \
     --input-jsonl data/image_bridge_v1/annotations/qwen36_cascade.jsonl \
     --output-csv data/image_bridge_v1/annotations/qwen36_cascade_human_audit.csv
   ```

6. Fill `human_supported` with `true` or `false`, then require at least 95%
   agreement before launching the full gate.

   ```bash
     /proj/cvl/users/x_fahkh2/envs/neo_mobileov/bin/python tools/sample_image_bridge_qwen_audit.py \
     --input-jsonl data/image_bridge_v1/annotations/qwen36_cascade.jsonl \
     --output-csv data/image_bridge_v1/annotations/qwen36_cascade_human_audit.csv \
     --score-filled-sheet
   ```

The scorer and verifier are resumable. Rerunning the same command skips completed
SigLIP IDs and valid Qwen decisions. Both automatically split a batch after CUDA
OOM. `MODEL_ID`/revision overrides are recorded in summaries, while output
manifests include model scores and verification provenance.

## Outputs

The builder writes to `data/image_bridge_v1/` atomically:

- `catalog.sqlite3`: canonical deduplicated catalog;
- `manifests/d1_broad_train.csv`;
- `manifests/d2_compositional_train.csv`;
- `manifests/d2_grounded_candidates.csv`;
- `manifests/d2_grounded_siglip2_scored.csv`;
- `manifests/d2_grounded_qwen_adjudication.csv`;
- `manifests/d2_grounded_candidate_100k.csv`;
- `manifests/d2_grounded_high_precision_50k.csv`;
- `manifests/validation.csv` and `hard_validation.csv`;
- `mixtures/v12_70_20_10.json`: exact trainer contract;
- `source_registry.json` and `stats/summary.json`.

`d2_grounded_candidate_100k.csv` is an expansion/audit pool and must not be used
as the default training source. `d2_grounded_high_precision_50k.csv` is the V12
grounded training manifest. It excludes unreadable pairs, benchmark leakage,
valid Qwen rejections, and Qwen errors. Its exact hashes and capability counts
are stored in `stats/grounding_cascade_summary.json`.

## Reproducibility Contract

- Captions are only whitespace-normalized; they are not rewritten.
- Deduplication uses the SHA-256 of normalized caption text.
- Dataset split and subset ordering use a seeded SHA-256 key.
- The three train pools are mutually exclusive.
- Capability mining is transparent lexical routing, not a hidden label model.
- SigLIP2 is used only for high-throughput ranking; hard claims are routed to Qwen.
- Qwen labels, failed claims, errors, and raw responses remain auditable in JSONL.
- Exact and high lexical-overlap VBench prompts are removed before selection.
- Selection uses fixed capability quotas and deterministic score/hash tie-breaking.
- Existing output directories are never overwritten unless `FORCE=1` is set.
- A failed build remains in `.partial` form and cannot be mistaken for a final dataset.

This design deliberately separates **data construction**, **visual
verification**, and **V12 optimization**. That separation makes the eventual
paper ablation interpretable: improvements can be attributed to the data
mixture rather than a simultaneous, undocumented change in captions or loss.

## V12 Training

After the machine preflight passes, launch the controlled from-scratch V12 run:

```bash
sbatch scripts/train_dreamlite_compact_v12_grounding_cascade_1node8gpu.sbatch
```

V12 preserves the V11-balanced content-aware and frozen-DreamLite functional
distillation recipe. Its only intended experimental change is the frozen
benchmark-clean 70/20/10 release. Synthetic semantic prompt injection is
disabled because the explicit compositional pool now supplies that curriculum.
The default run uses a global batch of 32 for 160K steps, updates the resumable
latest checkpoint every 5K steps, and archives a checkpoint every 20K steps.
The default is a true from-scratch run (`RESUME=none`). If SLURM interrupts the
job, reuse its original `OUT` and submit with `RESUME=auto` to continue safely.
