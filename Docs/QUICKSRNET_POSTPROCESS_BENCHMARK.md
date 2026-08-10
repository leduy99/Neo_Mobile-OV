# QuickSRNet Post-Process Benchmark

## Purpose

NeoDragon's final reported mobile pipeline includes a 2x QuickSRNet output
stage after video generation.  Our prior DreamLite + Exp1 VBench runs did not
apply a super-resolution model, so their image-quality scores are not fully
comparable with NeoDragon's final pipeline.  This utility applies the **same
post-process to already-generated MP4 files**.  It does not rerun DreamLite,
SmolVLM2, the bridge, NeoDragon, or VAE decoding.

The default is Qualcomm's released public **QuickSRNet Medium 2x** float
checkpoint.  It is a self-contained PyTorch reproduction of Qualcomm's public
AIMET Model Zoo architecture and is downloaded to `checkpoints/quicksrnet/` on
first use.  The exact optimized QuickSR export used in NeoDragon's final
mobile package is not part of the public NeoDragon checkout, so results must
be labelled **public QuickSRNet 2x reproduction**, not the official deployed
NeoDragon binary.

## Berzelius Commands

Apply 2x QuickSRNet to a directory of existing videos:

```bash
INPUT_DIR=output/dreamlite_v7_exp1_64k_vbench5x/videos \
  sbatch scripts/apply_quicksrnet_existing_videos_1gpu.sbatch
```

This writes `output/dreamlite_v7_exp1_64k_vbench5x/videos_quicksrnet_medium_2x`
by default and saves per-video timings to `quicksrnet_apply_report.json`.
Use the same command with a native NeoDragon video directory to establish the
fair native-with-QSR control.

Measure GPU-only QSR latency without writing a video:

```bash
MODE=benchmark \
INPUT_VIDEO=output/dreamlite_v7_exp1_64k_vbench5x/videos/example.mp4 \
  sbatch scripts/apply_quicksrnet_existing_videos_1gpu.sbatch
```

The job performs 10 warmup passes and 30 measured passes by default.  Its JSON
reports clip latency, milliseconds per frame, and PyTorch peak memory.  It
separates QSR forward time from MP4 decoding/writing, which are system and
codec dependent.

## Fair Latency Protocol

For the final comparison, use one GPU and identical seed, frame count,
resolution, and sampling schedule for both complete pipelines:

| Pipeline | Required stages |
| --- | --- |
| Native NeoDragon | Native text stack, SSD1B first frame, VAE anchor encoding, Hybrid DiT, VAE decode, QuickSRNet 2x |
| Mobile-OV | SmolVLM2, DreamLite compact bridge/first frame, NeoDragon video bridge, same Hybrid DiT, same VAE decode, same QuickSRNet 2x |

Report cold-load and warm runs separately.  The existing video tool only
isolates the shared final stage and confirms its cost on the same clips; it is
not a replacement for the full pipeline benchmark.

## Mobile Interpretation

The H200/A100 benchmark is a reproducibility and relative-cost measurement,
not an iPhone latency prediction.  Quantizing a PyTorch checkpoint does not
produce an iPhone or Android deployment by itself: the runtime, operator
fusion, tiling, memory bandwidth, and NPU delegate dominate the result.

Useful public Android reference: the NeoDragon paper reports roughly **6.7 s
end-to-end** for the final 640x1024, 49-frame pipeline on Snapdragon 8 Elite,
and reports a QuickSR contribution of only about **5--7 ms** on Qualcomm target
hardware.  Therefore QSR should be measured and included, but it is not
expected to be the dominant mobile latency.  There is no defensible numerical
iPhone 17 latency estimate without compiling the same W8A16/QNN/Core ML model
and measuring on-device.  Any cross-device conversion from H200 timing would
be misleading and should be reported only as a labelled engineering estimate,
never as a benchmark result.

## Local GPU Sanity Measurement

The implementation was exercised on an H200 NVL with the public QuickSRNet
Medium 2x float checkpoint in FP16, using one existing 49-frame `512x320` MP4.
After 10 warmups and 30 measured runs, the GPU-forward-only path measured
**14.99 ms per clip** or **0.306 ms per frame**, with **0.082 GiB** PyTorch peak
allocated memory.  A
single file-to-file smoke run was **76.61 ms per frame**, which includes CPU
decode, host/device copies, and MP4 re-encoding; it should not be presented as
the neural-network latency.  The Berzelius job uses 10 warmup and 30 measured
GPU-forward repetitions by default for a more stable machine-specific number.

## Provenance

- Qualcomm QuickSRNet public weights and architecture: [AIMET Model Zoo](https://github.com/quic/aimet-model-zoo), BSD-3-Clause; the local notice is [`third_party/licenses/QUICKSRNET_BSD_3_CLAUSE.md`](../third_party/licenses/QUICKSRNET_BSD_3_CLAUSE.md).
- NeoDragon pipeline and on-device measurements: [NeoDragon technical report](https://arxiv.org/html/2511.06055v1).
