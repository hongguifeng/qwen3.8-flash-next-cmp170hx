#!/usr/bin/env python3
"""Verify safetensors files by checking the header against the file size."""
import json
import os
import struct
import sys

DEST = os.environ.get(
    "MODEL_DIR", os.path.expanduser("~/models/Qwen3.8-Flash-Next-AutoRound-3bpw-MTP")
)


def check(path):
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        if n <= 0 or n > 100_000_000:
            return f"bad header length {n}"
        header = json.loads(f.read(n))
    tensors = {k: v for k, v in header.items() if k != "__metadata__"}
    end = max(v["data_offsets"][1] for v in tensors.values())
    if 8 + n + end != size:
        return f"size mismatch: file {size} != 8+{n}+{end}={8 + n + end}"
    return f"ok {len(tensors)} tensors, {size / 2**30:.2f} GiB"


rc = 0
for name in sorted(os.listdir(DEST)):
    if not name.endswith(".safetensors"):
        continue
    path = os.path.join(DEST, name)
    parts = os.path.exists(path + ".part")
    if parts:
        print(f"SKIP (pending) {name}")
        continue
    status = check(path)
    if not status.startswith("ok"):
        rc = 1
    print(f"{'FAIL' if rc else 'PASS'} {name}: {status}")
sys.exit(rc)
