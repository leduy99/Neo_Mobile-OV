# Appendix - Evidence and Reproduction Index

## 1. Purpose

This appendix maps the consolidated conclusions to implementation and saved evidence. Paths are relative to the `New-Mobile-OV` repository unless an absolute remote path or Hugging Face path is shown.

The report intentionally does not duplicate full command-line documentation. Use the referenced scripts for current defaults and the corresponding source documents for design details.

---

## 2. Data and infrastructure

| Work item | Main implementation | Primary artifact |
|---|---|---|
| OpenVid raw download | `scripts/download_openvid_raw_to_repo_berzelius.sbatch` | `download_data/data/openvid/raw/parts/` |
| Caption import | `scripts/copy_openvid_captions_to_repo.sh` | recaptioned OpenVid manifest |
| Two-second clip manifest | `tools/data_prepare/prepare_openvid_neodragon.py` | `data/openvid_neodragon_2s/manifest.csv` |
| Offline VAE encoding | `scripts/encode_neodragon_vae_latents.sbatch` | `data/openvid_neodragon_2s_latents/` |
| WanVAE anchor study | anchor-similarity evaluation tools | `output/wanvae_anchor_similarity_100_910/summary.json` |
| WanVAE negative control | independent-call/frame-10 sanity test | `output/wanvae_anchor_sanity_100_911/summary.json` |
| H200 rollout memory study | `scripts/benchmark_neodragon_rollout_bptt_h200.sbatch` | rollout-memory benchmark output |

Expected offline latent layout:

```text
data/openvid_neodragon_2s_latents/
  latents/
  shards/
  failed/
```

The presence of these three directories alone does not prove completion. Completion should be checked from merged shard counts, failure counts, and the expected manifest cardinality.

---

## 3. Video experiment matrix

| Experiment | Training script | Core trainer | Default output family |
|---|---|---|---|
| Old bridge 200K | `scripts/train_neodragon_text_bridge_recaption_1node8gpu_200k.sbatch` | `tools/train_neodragon_text_bridge.py` | `output/neo_bridge_8gpu_200k/` |
| Exp1 functional bridge | `scripts/exp1_train_neodragon_bridge_functional_distill_1node8gpu.sbatch` | `tools/train_neodragon_text_bridge.py` | `output/neo_exp1_bridge_functional/` |
| Exp1 two-epoch continuation | `scripts/exp1_continue_64k_2epochs_1node8gpu.sbatch` | `tools/train_neodragon_text_bridge.py` | `output/neo_exp1_continue_64k_2epochs/` |
| Exp1 nominal 200K continuation | `scripts/exp1_continue_to_200k_1node8gpu.sbatch` | `tools/train_neodragon_text_bridge.py` | `output/neo_exp1_continue_to200k/` |
| Exp1 full rollout | `scripts/exp1_rollout_distill_64k_to100k_1node8gpu.sbatch` | `tools/train_neodragon_bridge_rollout_distill.py` | `output/neo_exp1_rollout_64k_to100k/` |
| Legacy Exp2 | `scripts/exp2_train_neodragon_joint_flow_distill_1node8gpu.sbatch` | `tools/train_neodragon_dit_bridge.py` | `output/neo_exp2_joint_flow_distill/` |
| Corrected Exp2 | `scripts/exp2_corrected_train_neodragon_joint_distill_1node8gpu.sbatch` | `tools/train_neodragon_dit_bridge.py` | `output/neo_exp2_corrected_joint_distill/` |
| Exp3 | `scripts/exp3_train_neodragon_joint_from_scratch_1node8gpu.sbatch` | `tools/train_neodragon_dit_bridge.py` | `output/neo_exp3_joint_from_scratch/` |
| Exp4 | `scripts/exp4_train_neodragon_flow_only_from_scratch_1node8gpu.sbatch` | `tools/train_neodragon_dit_bridge.py` | `output/neo_exp4_flow_only/` |
| Exp5 | `scripts/exp5_train_neodragon_staged_1node8gpu.sbatch` | `tools/train_neodragon_exp5.py` | `output/neo_exp5_staged/` |
| Exp6 | `scripts/exp6_train_neodragon_hybrid_recovery_1node8gpu.sbatch` | `tools/train_neodragon_hybrid_recovery.py` | `output/neo_exp6_hybrid_recovery/` |

### Published checkpoint families used in evaluation

```text
Amshaker/Mobile-OV/
  neo_exp1_bridge_functional/17108893/neodragon_text_bridge_latest.pt
  neo_exp1_continue_to200k/17155429/neodragon_text_bridge_step200000.pt
  neo_exp1_rollout_64k_to100k/neodragon_rollout_bridge_step100000.pt
  neo_exp2_joint_flow_distill/17108894/neodragon_dit_bridge_latest.pt
  neo_exp3_joint_from_scratch/17104365/neodragon_dit_bridge_latest.pt
  neo_exp4_flow_only/17104367/neodragon_dit_bridge_latest.pt
  neo_exp5_staged/17112635/neodragon_exp5_latest.pt
  neo_exp6_hybrid_recovery/exp6_v1_pilot/neodragon_exp6_step002000.pt
```

The exact local filename should always be recorded in an evaluation output. A `latest` filename can point to a different training step after a remote run resumes.

---

## 4. Video evaluation evidence

| Research question | Primary saved evidence |
|---|---|
| Did nominal 200K improve Exp1 64K? | `output/neo_exp1_64k_vs_200k_20260728/summary.json` |
| Did rollout 70K differ from earlier checkpoints? | `output/neo_exp1_rollout70k_only_20260729/summary.json` |
| Did rollout 80K/100K close the native motion gap? | `output/native_anchor_exp1_rollout100k_controlled_20260730/metrics.json` |
| Did Exp2/3/4 improve with longer training? | `output/neo_exp2345_old_vs_latest_20260723/summary.json` |
| Which Exp5 component caused degradation? | `output/neo_exp5_step70k_test_20260722/` and component-swap outputs |
| How do stage/unit responses differ? | `output/exp5_vs_native_stage_unit_6prompt_20260723/summary.json` |
| What does native NeoDragon generate on the six prompts? | `output/neodragon_native_6prompt_20260723/` |
| How fast are native Hybrid and Monolithic modes? | `output/neodragon_native_mode_benchmark_20260723/summary.tsv` |
| Does Exp6 improve end-to-end behavior? | `output/exp6_2k_image_bridge_v2_100k_20260730/metrics.json` |

Useful ablation launchers:

```text
scripts/ablate_neodragon_exp2.sbatch
scripts/ablate_neodragon_exp3_exp4.sbatch
scripts/test_neodragon_exp3_exp4.sbatch
scripts/test_neodragon_checkpoint.sbatch
scripts/benchmark_neodragon_inference.sbatch
```

---

## 5. Image Bridge evidence

| Work item | Main implementation | Primary evidence |
|---|---|---|
| Image Bridge V1 training | `scripts/train_ssd1b_image_bridge_distill_1node8gpu.sbatch` | `ssd1b_image_bridge_distill/ssd1b_image_bridge_step100000.pt` |
| Comprehensive V1 evaluation | `scripts/evaluate_ssd1b_image_bridge_comprehensive.sbatch` | `output/ssd1b_image_bridge_comprehensive_eval/metrics.json` |
| Prompt-modifier ablation | comprehensive evaluator without modifier | `output/ssd1b_image_bridge_eval_no_modifier/metrics.json` |
| Image Bridge V2 training | `scripts/train_ssd1b_image_bridge_v2_1node8gpu.sbatch` | `ssd1b_image_bridge_v2/ssd1b_image_bridge_step100000.pt` |
| V2 evaluation | `tools/evaluate_ssd1b_image_bridge.py` | `output/ssd1b_image_bridge_v2_step100k_eval/metrics.json` |
| Red-panda diagnosis | `tools/analyze_ssd1b_red_panda.py` | `output/ssd1b_red_panda_diagnostic/metrics.json` |
| Native/image-bridge I2V control | `scripts/infer_image_bridge_to_exp1_rollout.sbatch` | `output/image_bridge_to_exp1_rollout_controlled_modifier/metrics.json` |

Published checkpoints:

```text
Amshaker/Mobile-OV/
  ssd1b_image_bridge_distill/ssd1b_image_bridge_step100000.pt
  ssd1b_image_bridge_v2/ssd1b_image_bridge_step100000.pt
```

---

## 6. Report visual artifacts

The copied images in `Docs/consolidated_research_report/assets/` are stable report assets. Their source evaluations remain under `output/`.

| Asset | Purpose |
|---|---|
| `exp1_64k_vs_200k.jpg` | Exp1 nominal continuation comparison |
| `rollout_80k_100k_native.jpg` | Full-rollout checkpoints against native |
| `exp234_old_vs_final.png` | Exp2/3/4 progression overview |
| `exp1_vs_exp5_red_panda.jpg` | Exp5 degradation relative to Exp1 |
| `exp5_component_ablation.jpg` | Bridge-versus-DiT attribution |
| `exp5_stage_unit_heatmaps.png` | Stage/unit response structure |
| `image_bridge_metrics.png` | V1 aggregate condition diagnostics |
| `image_bridge_v1.jpg` | V1 native/student image examples |
| `image_bridge_v2.jpg` | V2 native/student image examples |
| `image_bridge_v2_red_panda.jpg` | Fine-grained semantic failure |
| `exp6_native_condition.jpg` | Exp6 under native text condition |
| `exp6_exp1_condition.jpg` | Exp6 under Exp1 video condition |
| `exp6_end_to_end.jpg` | Image Bridge plus Exp1 plus Exp6 |

---

## 7. Minimum metadata for every future result

Every new result should record:

```text
repository commit
checkpoint path and global step
model-only or resume checkpoint
prompt text and prompt modifier
seed
first-frame source
generation noise source
Hybrid/Monolithic schedule
frame count, FPS, and resolution
precision
GPU type
warm or cold timing protocol
all metric definitions
output directory
```

Without this metadata, a result may still be useful for debugging, but it should not be promoted to a model-selection conclusion.
