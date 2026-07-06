#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import signal
import time

import torch


def parse_devices(value: str) -> list[int]:
    if value == "all":
        return list(range(torch.cuda.device_count()))
    devices: list[int] = []
    for item in value.split(","):
        item = item.strip()
        if item:
            devices.append(int(item))
    return devices


def main() -> None:
    parser = argparse.ArgumentParser(description="Keep allocated Slurm GPUs visibly active during CPU/network stages.")
    parser.add_argument("--devices", default="all", help="'all' visible GPUs, or a comma list such as 0,1.")
    parser.add_argument("--interval", type=float, default=15.0)
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--label", default=os.environ.get("SLURM_JOB_NAME", "gpu-heartbeat"))
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available; cannot run GPU heartbeat.")
    devices = parse_devices(args.devices)
    if not devices:
        raise RuntimeError("No CUDA devices selected for GPU heartbeat.")

    stop = False

    def _stop(*_: object) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)

    tensors: list[tuple[int, torch.Tensor]] = []
    for device in devices:
        with torch.cuda.device(device):
            tensor = torch.randn((args.size, args.size), device=f"cuda:{device}", dtype=torch.float16)
            tensors.append((device, tensor))
    print(
        f"[{args.label}] GPU heartbeat active on devices={devices}, "
        f"tensor_shape={args.size}x{args.size}, interval={args.interval}s",
        flush=True,
    )

    tick = 0
    while not stop:
        for device, tensor in tensors:
            with torch.cuda.device(device):
                tensor.mul_(1.0001).add_(0.0001)
                if tick % 4 == 0:
                    torch.cuda.synchronize(device)
        tick += 1
        time.sleep(max(0.1, args.interval))

    print(f"[{args.label}] GPU heartbeat stopped.", flush=True)


if __name__ == "__main__":
    main()
