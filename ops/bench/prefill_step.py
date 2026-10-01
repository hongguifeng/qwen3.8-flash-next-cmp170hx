#!/usr/bin/env python3
"""Prefill probes that are actually cold.

Every run generates brand-new random token ids (seed from the clock), so the
prefix cache can never serve the prompt, and the PLE rows are cold too.

  single(N)      one fresh N-token prompt -> TTFT is the real prompt-processing time
  conc(N, k)     k fresh N-token prompts at once -> aggregate prompt tok/s, i.e.
                 throughput as a function of tokens/step (the scheduler splits
                 max_num_batched_tokens across the k sequences)

Usage: prefill_step.py <label> [N ...]
"""
import json, random, sys, threading, time, urllib.request

import _cfg

URL = _cfg.api_url() + "/v1/completions"
LABEL = sys.argv[1] if len(sys.argv) > 1 else "x"
SIZES = [int(x) for x in sys.argv[2:]] or [8192, 32768]
OUT = f"/home/hong/code/qwen3.8-flash-next-cmp170hx/ops/measurements/prefill_step_{LABEL}.json"


def fresh(n):
    r = random.Random(time.time_ns() ^ n)
    return [r.randrange(1, 150000) for _ in range(n)]


def send(prompt, max_tokens=1):
    body = json.dumps({"model": "Qwen3.8-Flash-Next", "prompt": prompt,
                       "max_tokens": max_tokens, "temperature": 0,
                       "ignore_eos": True}).encode()
    req = urllib.request.Request(URL, data=body,
                                 headers={"Content-Type": "application/json"})
    t = time.perf_counter()
    urllib.request.urlopen(req, timeout=3600).read()
    return time.perf_counter() - t


def conc(n, k):
    lat = [None] * k
    prompts = [fresh(n) for _ in range(k)]

    def work(i):
        lat[i] = send(prompts[i])

    th = [threading.Thread(target=work, args=(i,)) for i in range(k)]
    t = time.perf_counter()
    for x in th: x.start()
    for x in th: x.join()
    wall = time.perf_counter() - t
    return wall, lat


res = {}
for n in SIZES:
    dt = send(fresh(n))
    res[f"cold1_{n}"] = dt
    print(f"1 seq  {n:6d} tok  TTFT {dt:7.2f}s   {n/dt:8.0f} tok/s", flush=True)
for n in SIZES:
    for k in (2, 4):
        wall, lat = conc(n, k)
        res[f"cold{k}_{n}"] = wall
        print(f"{k} seq  {n:6d} tok  wall {wall:7.2f}s   agg {k*n/wall:8.0f} tok/s "
              f"(slowest {max(lat):.1f}s)", flush=True)

json.dump(res, open(OUT, "w"), indent=2)
print(f"wrote {OUT}")
