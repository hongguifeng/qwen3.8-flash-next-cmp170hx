#!/usr/bin/env python3
"""Reusable prefill A/B probe.

Measures the *real* prefill rate (TTFT of a max_tokens=1 request, i.e. true
prompt-processing wall time -- NOT the engine's `Avg prompt throughput` log line,
which is only scheduled tokens / log interval and is meaningless here).

  cold   : fresh random token ids every run -> no prefix-cache hit, cold PLE rows
  warm   : the same ids again              -> prefix-cache hit (no compute at all)
  rows   : same ids but a different first token -> prefix cache MISS, PLE rows
           ~warm.  Separates "PLE row cache" from "prefix cache".

Usage: prefill_ab.py [label]   (writes prefill_ab_<label>.json)
"""
import json, random, sys, time, urllib.request

import _cfg

URL = _cfg.api_url() + "/v1/completions"
LABEL = sys.argv[1] if len(sys.argv) > 1 else time.strftime("%H%M%S")
OUT = f"/home/hong/code/qwen3.8-flash-next-cmp170hx/ops/measurements/prefill_ab_{LABEL}.json"
rng = random.Random(20260928)


def ids(n):
    return [rng.randrange(1, 150000) for _ in range(n)]


def send(prompt, max_tokens=1):
    body = json.dumps({"model": "Qwen3.8-Flash-Next", "prompt": prompt,
                       "max_tokens": max_tokens, "temperature": 0,
                       "ignore_eos": True}).encode()
    req = urllib.request.Request(URL, data=body,
                                 headers={"Content-Type": "application/json"})
    t = time.perf_counter()
    urllib.request.urlopen(req, timeout=1800).read()
    return time.perf_counter() - t


res = {}
for n in (8192, 32768):
    a = ids(n)                                   # fresh -> cold everything
    dt = send(a); res[f"cold_{n}"] = dt
    print(f"cold  {n:6d}  TTFT {dt:7.2f}s  {n/dt:7.0f} tok/s", flush=True)
    b = [a[0] ^ 7] + a[1:]                       # prefix miss, PLE rows warm
    dt = send(b); res[f"rowswarm_{n}"] = dt
    print(f"rows  {n:6d}  TTFT {dt:7.2f}s  {n/dt:7.0f} tok/s", flush=True)
    dt = send(a); res[f"prefixwarm_{n}"] = dt
    print(f"warm  {n:6d}  TTFT {dt:7.2f}s  {n/dt:7.0f} tok/s", flush=True)

# decode sanity: 65 tokens at an 8k context
a = ids(8192)
send(a)
t1 = send(a, 1); t65 = send(a, 65)
res["decode_ms_per_step_8k"] = (t65 - t1) / 63 * 1e3
print(f"decode 8k ctx: {(t65-t1)/63*1e3:.1f} ms/step -> {63/(t65-t1):.1f} tok/s", flush=True)

json.dump(res, open(OUT, "w"), indent=2)
print(f"wrote {OUT}")
