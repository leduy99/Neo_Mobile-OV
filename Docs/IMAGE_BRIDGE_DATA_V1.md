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
| D2 grounded | 10% | Use real image-caption pairs whose central claims pass Qwen3.6 visual verification. |

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

## Build And Verify

Run from the `Neo_Mobile-OV` repository on Berzilius.

1. Create the isolated Qwen3.6 environment once. This keeps Transformers 5.x
   away from the pinned DreamLite/NeoDragon training environment.

   ```bash
   sbatch scripts/setup_image_bridge_data_env_1node1gpu.sbatch
   ```

2. Build the immutable catalog and candidate views.

   ```bash
   sbatch scripts/build_image_bridge_data_v1_1node1gpu.sbatch
   ```

3. Run the 20k Qwen3.6 pilot.

   ```bash
   RUN_MODE=pilot sbatch scripts/verify_image_bridge_data_v1_qwen36_1node1gpu.sbatch
   ```

4. Create a deterministic 100-accepted/100-rejected human audit sheet.

   ```bash
   /proj/cvl/users/x_fahkh2/envs/neo_mobileov/bin/python tools/sample_image_bridge_qwen_audit.py \
     --input-jsonl data/image_bridge_v1/annotations/qwen36_pilot.jsonl \
     --output-csv data/image_bridge_v1/annotations/qwen36_pilot_human_audit.csv
   ```

5. Fill `human_supported` with `true` or `false`, then require at least 95%
   agreement before launching the full gate.

   ```bash
   /proj/cvl/users/x_fahkh2/envs/neo_mobileov/bin/python tools/sample_image_bridge_qwen_audit.py \
     --input-jsonl data/image_bridge_v1/annotations/qwen36_pilot.jsonl \
     --output-csv data/image_bridge_v1/annotations/qwen36_pilot_human_audit.csv \
     --score-filled-sheet
   ```

6. If the audit passes, verify the full grounded candidate pool.

   ```bash
   RUN_MODE=full sbatch scripts/verify_image_bridge_data_v1_qwen36_1node1gpu.sbatch
   ```

All SLURM scripts retain one active GPU heartbeat and write logs under `logs/`.
`MODEL_ID` and `MODEL_REVISION` can point to a local checkpoint or a pinned
Hugging Face revision. The verifier records the resolved model revision and
runtime package versions in its summary.

## Outputs

The builder writes to `data/image_bridge_v1/` atomically:

- `catalog.sqlite3`: canonical deduplicated catalog;
- `manifests/d1_broad_train.csv`;
- `manifests/d2_compositional_train.csv`;
- `manifests/d2_grounded_candidates.csv`;
- `manifests/validation.csv` and `hard_validation.csv`;
- `mixtures/v12_70_20_10.json`: exact trainer contract;
- `source_registry.json` and `stats/summary.json`.

The full verifier creates
`manifests/d2_grounded_qwen_verified.csv`. This is the only grounded manifest
allowed in a V12 paper run. The unverified candidate file must never be used as
grounded supervision.

## Reproducibility Contract

- Captions are only whitespace-normalized; they are not rewritten.
- Deduplication uses the SHA-256 of normalized caption text.
- Dataset split and subset ordering use a seeded SHA-256 key.
- The three train pools are mutually exclusive.
- Capability mining is transparent lexical routing, not a hidden label model.
- Qwen labels and raw responses remain auditable in JSONL.
- Existing output directories are never overwritten unless `FORCE=1` is set.
- A failed build remains in `.partial` form and cannot be mistaken for a final dataset.

This design deliberately separates **data construction**, **visual
verification**, and **V12 optimization**. That separation makes the eventual
paper ablation interpretable: improvements can be attributed to the data
mixture rather than a simultaneous, undocumented change in captions or loss.
