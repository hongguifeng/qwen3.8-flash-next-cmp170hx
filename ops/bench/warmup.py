#!/usr/bin/env python3
"""Warm up Triton (and measure) prefill for the served model.

The GDN / linear-attention Triton kernels are compiled (and autotuned) lazily, the
first time an engine step uses them, *inside* that step -- so the first long prompt
after a container start is 2-6x slower than steady state.  Run this once after
`run_container.sh` comes up (it also doubles as a fresh-id prefill measurement,
because every request uses brand-new token ids and therefore can never be served
by the prefix cache).

  warmup.py                      # warm 2048/8192/32768/131072
  warmup.py 8192 32768           # specific lengths
  warmup.py --reps 2 32768 131072

Keep every length < --max-model-len (262144): prompt_tokens + max_tokens must fit, so
262144 is rejected with HTTP 400.
"""
import argparse
import json
import random
import time
import urllib.request

import _cfg

ap = argparse.ArgumentParser()
ap.add_argument("lengths", nargs="*", type=int, default=[2048, 8192, 32768, 131072])
ap.add_argument("--reps", type=int, default=1, help="requests per length (1st pays compile)")
ap.add_argument("--url", default=None, help="默认取 config/engine.env 的 QWEN_PORT")
ap.add_argument("--model", default=None)
ap.add_argument("--wait", type=float, default=0.0, help="seconds to sleep between requests")
args = ap.parse_args()
args.url = args.url or (_cfg.api_url() + "/v1/completions")
args.model = args.model or _cfg.model_name()


def fresh_ids(n: int) -> list[int]:
    r = random.Random(time.time_ns() ^ n)
    return [r.randrange(1, 150000) for _ in range(n)]


def send(prompt, timeout=3600) -> float:
    body = json.dumps({"model": args.model, "prompt": prompt, "max_tokens": 1,
                       "temperature": 0, "ignore_eos": True}).encode()
    req = urllib.request.Request(args.url, data=body,
                                headers={"Content-Type": "application/json"})
    t = time.perf_counter()
    body = json.loads(urllib.request.urlopen(req, timeout=timeout).read())
    assert body["usage"]["prompt_tokens"] == len(prompt), body["usage"]
    return time.perf_counter() - t


print(f"# {args.url}  lengths={args.lengths} reps={args.reps}")
for n in args.lengths:
    lat = []
    for rep in range(args.reps):
        dt = send(fresh_ids(n))
        lat.append(dt)
        print(f"fresh prefill {n:7d} tok  rep={rep}  {dt:8.3f} s  {n/dt:7.0f} tok/s",
              flush=True)
        if args.wait:
            time.sleep(args.wait)
    if len(lat) > 1:
        print(f"           {n:7d} tok  best of {len(lat)}  {min(lat):8.3f} s  "
              f"{n/min(lat):7.0f} tok/s  (steady state excludes the first/compile hit)",
              flush=True)
