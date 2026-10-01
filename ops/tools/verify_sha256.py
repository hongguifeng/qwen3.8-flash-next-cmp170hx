#!/usr/bin/env python3
"""Verify every checkpoint file against the sha256 published by the Hub."""
import hashlib
import os
import sys

import requests

REPO = "klee100/Qwen3.8-Flash-Next-AutoRound-3bpw-MTP"
REVISION = "ce0e0b94083895bd836b916f29bf105c40a8162a"
DEST = os.environ.get(
    "MODEL_DIR", os.path.expanduser("~/models/Qwen3.8-Flash-Next-AutoRound-3bpw-MTP")
)


def expected():
    url = f"https://huggingface.co/api/models/{REPO}/revision/{REVISION}?blobs=true"
    out = {}
    for s in requests.get(url, timeout=60).json().get("siblings", []):
        lfs = s.get("lfs") or {}
        if lfs.get("sha256") or lfs.get("oid"):
            out[s["rfilename"]] = lfs.get("sha256") or lfs["oid"].replace("sha256:", "")
    return out


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(8 * 1024 * 1024)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


want = expected()
bad = []
for name, ref in sorted(want.items()):
    path = os.path.join(DEST, name)
    if not os.path.exists(path):
        print(f"MISSING {name}")
        bad.append(name)
        continue
    got = sha256(path)
    ok = got == ref
    print(f"{'OK  ' if ok else 'BAD '} {name} ({os.path.getsize(path) / 2**30:.2f} GiB)")
    if not ok:
        bad.append(name)
print("all files verified" if not bad else f"FAILED: {bad}")
sys.exit(1 if bad else 0)
