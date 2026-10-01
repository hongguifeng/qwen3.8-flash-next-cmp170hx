#!/usr/bin/env python3
"""Probe prefill throughput at a given prompt length/content style.

usage: prefill_probe.py <target_tokens> {repeat|unique} [--port 9393]

Sends one completion request (max_tokens=1) and reports the *prefill* rate
= prompt_tokens / (time to first token).  While it runs it polls
/metrics + /proc/diskstats so we can see whether the rate is constant
(IO bound) or degrades with position (quadratic attention).
"""
import json
import os
import random
import string
import sys
import threading
import time
import urllib.request

PORT = int(os.environ.get("PORT", "9393"))
BASE = f"http://127.0.0.1:{PORT}"
MODEL = os.environ.get("SERVED_NAME", "Qwen3.8-Flash-Next")


def post(path, payload, timeout=3600):
    req = urllib.request.Request(
        BASE + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def get(path, timeout=10):
    with urllib.request.urlopen(BASE + path, timeout=timeout) as r:
        return r.read().decode()


def n_tokens(text):
    return len(post("/tokenize", {"model": MODEL, "prompt": text})["tokens"])


def build(mode, target, seed=0):
    rnd = random.Random(seed)
    tag = "".join(rnd.choices(string.ascii_lowercase, k=8)) + " "
    if mode == "unique":
        # random words from a huge space -> nearly every token trigram is novel
        unit = " ".join(
            "".join(rnd.choices(string.ascii_lowercase + string.digits, k=6))
            for _ in range(40)
        )
        unit += "\n"
    else:
        # fixed prose repeated many times: internally repetitive (same token
        # ngrams over and over) but the random tag makes it novel across runs,
        # so vLLM's prefix cache can't serve it.
        unit = (
            tag
            + "The quick brown fox jumps over the lazy dog. "
            "Pack my box with five dozen liquor jugs. "
            "How vexingly quick daft zebras jump! "
        )
    assert n_tokens(unit) > 0
    reps = max(1, target // n_tokens(unit))
    text = unit * reps
    # trim to ~target by tokenizing the tail once
    while reps > 1 and n_tokens(text) > target * 1.03:
        reps -= max(1, reps // 20)
        text = unit * reps
    return text


def disk_read_bytes():
    tot = 0
    try:
        with open("/proc/diskstats") as f:
            for line in f:
                p = line.split()
                tot += int(p[5]) * 512  # sectors read
    except OSError:
        pass
    return tot


def metric(name):
    try:
        for line in get("/metrics").splitlines():
            if line.startswith(name + "{"):
                return float(line.rsplit(" ", 1)[1])
    except Exception:
        pass
    return float("nan")


class Monitor(threading.Thread):
    daemon = True

    def __init__(self, total):
        super().__init__()
        self.total = total
        self.stop = threading.Event()
        self.t0 = time.time()

    def run(self):
        last_p = last_d = last_t = None
        while not self.stop.wait(2.0):
            now = time.time()
            p = metric("vllm:prompt_tokens_total")
            d = disk_read_bytes()
            if last_p is not None and not (p != p):
                rate = (p - last_p) / (now - last_t)
                dr = (d - last_d) / (now - last_t) / 1e9
                print(
                    f"    t={now - self.t0:6.1f}s  prompt={p:8.0f}/{self.total}"
                    f"  ({100 * p / max(self.total, 1):5.1f}%)"
                    f"  prefill={rate:7.1f} tok/s  disk_read={dr:5.2f} GB/s",
                    flush=True,
                )
            last_p, last_d, last_t = p, d, now


def main():
    target = int(sys.argv[1])
    mode = sys.argv[2] if len(sys.argv) > 2 else "repeat"
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    text = build(mode, target, seed)
    n = n_tokens(text)
    print(f"[probe] mode={mode} target={target} actual_prompt_tokens={n}"
          f" chars={len(text)}")
    mon = Monitor(n)
    mon.start()
    t0 = time.time()
    try:
        post("/v1/completions", {
            "model": MODEL,
            "prompt": text,
            "max_tokens": 1,
            "temperature": 0.0,
        })
    except Exception as e:
        print(f"[probe] request failed: {e}")
    dt = time.time() - t0
    mon.stop.set()
    time.sleep(0.1)
    print(f"[probe] prefill {n} tokens in {dt:.1f}s -> {n / dt:.1f} tok/s"
          f" (TTFT ~= {dt:.1f}s)")


if __name__ == "__main__":
    main()
