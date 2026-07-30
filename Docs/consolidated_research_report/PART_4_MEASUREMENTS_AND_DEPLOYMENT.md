# Part 4 - Measurements, Complexity, and Deployment

## 1. How to read these measurements

This part consolidates the measurements collected across the project. The numbers come from several protocols, so they must not be mixed without context.

| Evidence type | Meaning |
|---|---|
| Local measured | Collected from the current code and checkpoints on the local H200 system |
| Historical local | Collected earlier for Mobile-OV/SANA-Video or On-device Sora under different output settings |
| Derived | Calculated from measured parameter counts or checkpoint contents |
| Paper reported | Taken from the NeoDragon paper and mobile deployment report |

The most comparable speed result in this repository is the local native NeoDragon Hybrid versus Monolithic benchmark because both modes used the same machine, prompt, output shape, precision, and warm-pipeline protocol. Historical Mobile-OV and On-device Sora measurements are useful engineering references, but they are not an apples-to-apples quality or speed comparison.

---

## 2. Native NeoDragon component inventory

The following parameter inventory was measured from the native NeoDragon Hybrid pipeline.

| Component | Parameters | Dense BF16 size |
|---|---:|---:|
| Native text-encoder bundle | 947,861,632 | 1.766 GiB |
| NeoDragon ContextAdapter | 130,041,344 | 0.242 GiB |
| NeoDragon DiT | 1,512,155,200 | 2.817 GiB |
| Video VAE | 119,543,507 | 0.223 GiB |
| SSD1B first-frame pipeline | 2,232,708,587 | 4.159 GiB |
| **Native Hybrid total** | **4,942,310,270** | **9.206 GiB** |

The SSD1B first-frame pipeline contains the image generator, image VAE, and its two CLIP text encoders. Its local FP16 files are approximately:

| SSD1B component | File size |
|---|---:|
| SSD1B UNet | 2.480 GiB |
| SSD1B VAE | 0.156 GiB |
| SSD1B dual CLIP encoders | 1.523 GiB |
| **Total** | **4.159 GiB** |

The native NeoDragon text bundle and the SSD1B dual CLIPs are separate condition systems. This duplication is one motivation for the shared Mobile-OV understanding backbone, but removing either system is only justified after its replacement passes output-level quality tests.

### SSD1B dual CLIP breakdown

| Encoder | Parameters | FP16 file size | Approximate compute at 77 tokens |
|---|---:|---:|---:|
| CLIP-L | 123,060,480 | 234.7 MiB | 6.649 GMAC / 13.299 GFLOPs |
| CLIP-bigG | 694,659,840 | 1.294 GiB | 48.930 GMAC / 97.860 GFLOPs |
| **Combined** | **817,720,320** | **1.523 GiB** | **55.579 GMAC / 111.158 GFLOPs** |

The theoretical dense storage of both encoders is approximately 3.27 GB in FP32, 1.64 GB in FP16, 0.82 GB in INT8, or 0.41 GB in INT4 before quantization metadata and runtime buffers.

---

## 3. Mobile-OV deployment variants

The following totals are logical unique-weight estimates. They are not measured runtime peaks and do not include every framework buffer, tokenizer file, allocator cache, or quantization scale.

### Variant A: native NeoDragon Hybrid

```text
Native video text bundle + ContextAdapter
+ NeoDragon DiT
+ video VAE
+ native SSD1B first-frame pipeline
= 4.942B parameters
```

This is the strongest released generation reference and the safest current deployment baseline.

### Variant B: Mobile-OV video bridge, native SSD1B CLIPs

This variant replaces the native video text bundle and ContextAdapter with the shared SmolVLM2 backbone and the trained video bridge, while keeping native SSD1B text conditioning:

```text
4.942B
- 1.078B native video text bundle and ContextAdapter
+ 0.513B SmolVLM2 and video bridge
= 4.377B unique parameters
```

Theoretical dense BF16 weight size: approximately **8.15 GiB**.

This is the most practical current Mobile-OV configuration because Exp1 demonstrated that the frozen NeoDragon DiT can remain usable with a distilled video bridge, while the native SSD1B CLIPs still preserve first-frame quality.

### Variant C: fully shared Mobile-OV text backbone

This variant also replaces the SSD1B dual CLIPs with the 11.15M-parameter image bridge:

```text
Variant B
- 817.72M SSD1B CLIP parameters
+ 11.15M image bridge parameters
= 3.571B unique parameters
```

Theoretical dense BF16 weight size: approximately **6.65 GiB**.

This is attractive for deployment, but it is not yet quality-ready. Image Bridge V1 and V2 showed semantic and distribution gaps, especially on prompts such as the red-panda example. The size benefit is real, but current evidence does not justify removing the native SSD1B CLIPs.

---

## 4. Checkpoint size is not deployment size

Several checkpoints appeared unexpectedly large or small because they package different states.

| Checkpoint family | Observed size | What it contains |
|---|---:|---|
| Exp1 bridge checkpoints | about 0.98-1.08 GB | Shared SmolVLM2 weights, bridge state, and training metadata |
| Image Bridge V1 | about 135 MB | Image-bridge state and optimizer/training metadata; frozen teachers are not embedded |
| Image Bridge V2 | about 145 MB | Same principle as V1 with additional training state |
| Exp2/3/4 joint checkpoints | about 4.17 GB | DiT, bridge, and associated state |
| Exp5 staged checkpoint | about 10.24 GB | Model weights plus large Adam optimizer states for DiT and bridge |
| Exp6 checkpoint | about 6.16 GB | Approximately 1.57B trainable/model parameters stored in FP32-like training form |

Exp5 is a useful example: its file included roughly 2.92 GiB of DiT weights, 0.96 GiB of bridge/shared weights, 5.63 GiB of DiT Adam state, and additional bridge optimizer state. A resume checkpoint is therefore not equivalent to an inference package.

For mobile reporting, use:

1. Unique deployed parameter count.
2. Actual converted package size at the target precision.
3. Runtime peak memory.
4. Temporary activation memory.

Do not infer any of these directly from a training checkpoint file.

---

## 5. Native NeoDragon speed and complexity

### Local H200 warm-pipeline benchmark

Protocol:

- H200 NVL.
- BF16.
- 320 x 512 output.
- 49 frames at 24 fps.
- Warm full pipeline after model loading.
- Same prompt and measurement harness for both modes.

| Metric | Hybrid 1-1-1 | Monolithic | Interpretation |
|---|---:|---:|---|
| Seconds per video | **2.152** | 9.412 | Hybrid is 4.37x faster |
| Generated frames per compute second | **22.77** | 5.21 | Hybrid is close to real time |
| Compute time / video duration | **1.054x** | 4.610x | Lower is better |
| Peak allocated VRAM | 14.00 GiB | **9.84 GiB** | Hybrid trades memory for speed |
| Peak reserved VRAM | 14.84 GiB | lower | Hybrid includes the first-frame path |
| Resident parameters | 4.942B | **2.710B** | Hybrid loads more components |
| Dense BF16 parameter size | 9.206 GiB | **5.047 GiB** | Monolithic is smaller in memory |
| Estimated executed work | **37.76 TFLOPs** | 395.39 TFLOPs | Lower-bound estimate |
| DiT calls | **18** | 240 | Main source of speed difference |
| DiT batch items | **18** | 480 | Monolithic CFG expands executed work |
| First-frame UNet calls | 4 | 0 | Hybrid pays an image-anchor cost |

![Native mode and stage analysis](assets/exp5_stage_unit_heatmaps.png)

The key lesson is that parameter count does not predict latency by itself. Hybrid is larger because it includes SSD1B and additional condition components, but its distilled 1-1-1 schedule executes dramatically fewer DiT calls. It is therefore much faster despite having more resident weights.

### Paper-reported mobile measurements

The NeoDragon mobile report measured the two SSD1B CLIPs at approximately:

| Device | CLIP-L | CLIP-bigG | Combined |
|---|---:|---:|---:|
| Snapdragon 8 Elite Gen 4 | 14.0 ms | 76.5 ms | 90.5 ms |
| Snapdragon X Elite | 5.9 ms | 43.6 ms | 49.5 ms |

The same report gives approximately 6.7 seconds end-to-end for 49 frames at 640 x 1024 and a 3.5 GB peak for the full NeoDragon mobile pipeline. These are paper-reported deployment measurements, not reproduced local numbers.

Relative to 6.7 seconds, the dual-CLIP forward adds only about 1.4% latency on Snapdragon 8 Elite. Its larger cost is storage and model residency. Therefore, keeping both CLIPs temporarily is a reasonable quality-first decision; eliminating them should be treated as a footprint optimization, not an urgent speed optimization.

---

## 6. Image-condition latency and memory

Image Bridge V1 was benchmarked against native SSD1B dual CLIP conditioning on H200.

### Parameters

| Condition path | Total parameters | Dense BF16 parameter memory |
|---|---:|---:|
| SmolVLM2 + Image Bridge | 518,630,597 | 989.21 MiB |
| Native SSD1B dual CLIPs | 817,720,320 | 1,559.68 MiB |
| Image Bridge trainable head only | 11,148,293 | 21.26 MiB |

The shared path reduces unique weights only if SmolVLM2 is already resident for understanding or video generation. If loaded only for image generation, its latency and activations must still be counted.

### Prompt-forward latency

| Batch size | SmolVLM2 + bridge | Native dual CLIPs |
|---:|---:|---:|
| 1 | 18.86 ms/prompt | **15.59 ms/prompt** |
| 4 | 5.38 ms/prompt | **5.09 ms/prompt** |
| 16 | 2.59 ms/prompt | **1.60 ms/prompt** |

### Temporary peak allocation

| Batch size | SmolVLM2 + bridge | Native dual CLIPs |
|---:|---:|---:|
| 1 | 79.07 MiB | **10.23 MiB** |
| 4 | 316.28 MiB | **39.13 MiB** |
| 16 | 1,265.13 MiB | **156.47 MiB** |

On H200, the current image bridge is not faster and uses more temporary activation memory. Its potential value is architectural sharing and reduced unique static weights, not raw text-forward speed.

The first-frame generation itself measured approximately 0.239 seconds with native conditioning and 0.233 seconds with Image Bridge V1 in the initial 30-prompt harness. Later controlled examples were around 0.22-0.26 seconds. These small timing differences are not meaningful enough to override the visible quality gap.

![Image Bridge measurement overview](assets/image_bridge_metrics.png)

---

## 7. Historical generation baselines

The earlier project measured On-device Sora and Mobile-OV 135K with a sampling-only protocol:

| Model | Seconds/video | Frames and resolution | Denoising | Peak VRAM |
|---|---:|---|---|---:|
| On-device Sora | 7.648 | 51 frames, 240p | 30 steps, max LPL 2 | 22.67 GiB |
| Mobile-OV 135K | 27.600 | 81 frames, 480 x 832 | 24 steps, CFG 6 | 16.49 GiB |
| Native NeoDragon Hybrid | 2.152 | 49 frames, 320 x 512 | Hybrid 1-1-1 | 14.00 GiB |

This table is a system-history reference only. Resolution, frame count, schedulers, guidance, model paths, and hardware protocols differ. It shows why NeoDragon became attractive as an efficient generation baseline, but it does not establish a quality-adjusted speed ranking.

---

## 8. Understanding-backbone measurements

An earlier prompt-forward comparison evaluated SmolVLM2 and MiniCPM-V-4.6 1.3B:

| Model | Batch 1 | Batch 4 | Batch 20 |
|---|---:|---:|---:|
| SmolVLM2 | **14.73 ms/prompt** | **4.53 ms/prompt** | **0.98 ms/prompt** |
| MiniCPM-V-4.6 | 88.59 ms/prompt | 22.52 ms/prompt | 4.71 ms/prompt |

SmolVLM2 was approximately 4.8-6.0x faster in this text-forward benchmark.

The historical last-hidden-state prompt-pair cosine summary was:

| Model | Mean cosine | Std | Min | Max |
|---|---:|---:|---:|---:|
| SmolVLM2 | 0.7975 | 0.0607 | 0.6473 | 0.9768 |
| MiniCPM-V-4.6 | 0.8205 | 0.0497 | 0.7011 | 0.9774 |

This result suggested limited separation in the selected final hidden state, but it is not a complete semantic-collapse diagnosis. A later layer audit showed that early SmolVLM2 layers retained higher diversity and that the original first/last naming could be misread. The practical conclusion was to select and validate the exact bridge source layers rather than judging the backbone from one final-layer average.

---

## 9. Historical VBench context

The earlier Mobile-OV/SANA-Video branch reported the following aggregate VBench trend:

| Run | Quality | Semantic | Total |
|---|---:|---:|---:|
| Mobile-OV 60K, short captions | 0.8473 | 0.4021 | 0.7581 |
| Mobile-OV 135K, extended captions | 0.8486 | 0.4879 | 0.7764 |

The main gain came from semantic alignment after extending very short prompts into one concise sentence. This supported the hypothesis that prompt distribution and bridge conditioning matter substantially.

These numbers should not be compared directly with:

- The six-prompt NeoDragon diagnostics in this report.
- Native NeoDragon paper results.
- A standard four-video-per-prompt VBench run.

The historical Mobile-OV evaluation used one generated video per prompt and duplicated samples where the evaluator required four entries. It is useful for within-model comparison, not for a definitive leaderboard.

---

## 10. Deployment implications

1. **Keep the Hybrid schedule.** It delivers the largest speed benefit by reducing executed DiT work.
2. **Keep native SSD1B CLIPs for now.** Their mobile latency is modest relative to the full pipeline, while current image bridges do not preserve quality reliably.
3. **Use the Mobile-OV video bridge only with a frozen or strongly protected DiT.** Exp1 is the only consistently useful trained path.
4. **Report unique deployed weights, not training checkpoint size.**
5. **Cache prompt embeddings when prompts repeat.** This makes the CLIP latency nearly irrelevant for repeated generation.
6. **Load condition encoders sequentially if memory is constrained.** SSD1B conditioning can be released before the video stage, although the implementation must account for reload cost.
7. **Quantize after quality is stable.** INT8 or INT4 dual CLIPs could substantially reduce storage without forcing a weak replacement bridge into the pipeline.

The current engineering bottleneck is no longer raw generation latency. It is preserving native quality while sharing representations across understanding, image generation, and video generation.
