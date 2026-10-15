#!/usr/bin/env python
"""Report CUDA liveness without manufacturing GPU utilization."""
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import signal
import threading

import torch


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--interval", type=float, default=30)
    parser.add_argument("--ready-file", type=Path, required=True)
    parser.add_argument("--label", default="job-b-data")
    args = parser.parse_args(argv)
    if not math.isfinite(args.interval) or args.interval < 1:
        parser.error("Heartbeat interval must be finite and at least one second")
    return args


def run(args, *, cuda=None, stop=None):
    cuda = torch.cuda if cuda is None else cuda
    stop = threading.Event() if stop is None else stop
    if not cuda.is_available() or cuda.device_count() != 1:
        raise RuntimeError("Expected exactly one visible, allocated CUDA GPU")
    cuda.init()
    cuda.synchronize(0)
    first = True
    try:
        while not stop.is_set():
            free, total = cuda.mem_get_info(0)
            print(json.dumps(dict(label=args.label, type="cuda_status_heartbeat",
                                  utc=datetime.now(timezone.utc).isoformat(),
                                  device=cuda.get_device_name(0), free_gib=free / 1024**3,
                                  total_gib=total / 1024**3, artificial_gpu_work=False)), flush=True)
            if first:
                temporary = args.ready_file.with_suffix(".tmp")
                temporary.write_text("ready\n", encoding="ascii")
                temporary.replace(args.ready_file)
                first = False
            if stop.wait(args.interval):
                break
    finally:
        args.ready_file.unlink(missing_ok=True)


def main(argv=None):
    args = parse_args(argv)
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    run(args, stop=stop)


if __name__ == "__main__":
    main()
