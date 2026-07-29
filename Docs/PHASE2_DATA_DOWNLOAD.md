# Phase 2 Dataset Downloads

The Phase 2 understanding and planning data comes from the official Hugging
Face repositories:

- `lmms-lab/LLaVA-OneVision-Data`: approximately 282 GiB across 650 files.
- `lmms-lab/LLaVA-Video-178K`: approximately 1.16 TiB across 302 files.

Both Berzelius jobs request one GPU and keep it visibly active while a process
pool downloads independent repository files. The downloader pins the source
commit, retries failed files, verifies file sizes, and resumes completed files
when the same job is submitted again.

## Submit

```bash
sbatch scripts/download_llava_onevision_berzelius.sbatch
sbatch scripts/download_llava_video_178k_berzelius.sbatch
```

The default is eight concurrent download processes. It can be overridden at
submission time:

```bash
DOWNLOAD_WORKERS=12 sbatch scripts/download_llava_video_178k_berzelius.sbatch
```

Set `HF_TOKEN` before submission when authenticated Hugging Face access is
available. The token is inherited by workers and is never stored in the repo.

## Outputs

```text
download_data/data/llava_onevision/
download_data/data/llava_video_178k/
```

Each output contains:

```text
.source_revision.json
.download_manifest.json
download_summary.json
.download_complete
```

`.download_complete` is created only after every selected repository file
matches its expected size. If a job times out or a transfer fails, submit the
same command again; the pinned revision and completed files are reused.

The jobs download the original tarballs, JSON, and parquet files. They do not
extract archives, which avoids temporarily doubling storage use.

## Monitor

```bash
squeue --me
tail -n 20 logs/mov-dl-llava-ov-<JOBID>.out
tail -n 20 logs/mov-dl-llava-video-<JOBID>.out
```
