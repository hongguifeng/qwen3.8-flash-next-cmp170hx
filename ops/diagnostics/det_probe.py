#!/usr/bin/env python3
# det_probe.py — 引擎输出确定性探针（2026-09-30 加入）
#
# 用途：给"同一请求跑 N 次"测噪声底，用于判断
#   (a) 引擎当前是否 bit 可复现（本项目实测：不）；
#   (b) 任何"改动是否影响模型输出"的 A/B 必须先有噪声底，否则结论无效。
#
# 用法: python3 ops/diagnostics/det_probe.py 6      # 只读，不打搅引擎（发几个 12 token 的小请求）
# 关键观察点：第 2..N 次请求引擎状态完全相同（命中前缀缓存、batch=1、同一图大小），
#            它们之间的差异只能来自内核级不确定性。
#
"""Cheap determinism probe for the live engine (read-only, tiny requests).

Sends N identical greedy requests one at a time and compares generated tokens,
their logprobs and the top-k candidate logprobs. Repeats 2..N run under
identical engine conditions (prefix-cache hit, batch size 1, same CUDA-graph
size), so any difference between them is kernel-level nondeterminism.
"""
import json
import sys
import time
import urllib.request

URL = "http://127.0.0.1:9393/v1/completions"

LONG_PROMPT = (
    "The quarterly logistics review covers warehouse throughput, fleet "
    "utilisation, cold-chain compliance, driver rosters, peak-season surge "
    "planning, supplier contracts, and the new telemetry rollout."
)
SHORT_PROMPT = "Summarise the note."


def ask(prompt, max_tokens=12, nlogprobs=5):
    body = {
        "model": "Qwen3.8-Flash-Next",
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "logprobs": nlogprobs,
        "ignore_eos": True,
    }
    req = urllib.request.Request(
        URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=300) as r:
        out = json.loads(r.read())
    dt = time.time() - t0
    ch = out["choices"][0]
    lp = ch["logprobs"]
    rec = {
        "tokens": lp["tokens"],
        "tok_lp": lp["token_logprobs"],
        "top": lp["top_logprobs"],
        "ptok": out["usage"]["prompt_tokens"],
        "dt": dt,
    }
    return rec


def cmp(a, b):
    ids_same = a["tokens"] == b["tokens"]
    d_sample = max(abs(x - y) for x, y in zip(a["tok_lp"], b["tok_lp"]))
    d_top = 0.0
    for ta, tb in zip(a["top"], b["top"]):
        keys = set(ta) | set(tb)
        for k in keys:
            d_top = max(d_top, abs(ta.get(k, -99.0) - tb.get(k, -99.0)))
    return ids_same, d_sample, d_top


def run(label, prompt, reps, max_tokens=12):
    print(f"### {label}  (prompt_tokens~{len(prompt.split())} words, reps={reps})")
    runs = []
    for i in range(reps):
        rec = ask(prompt, max_tokens=max_tokens)
        runs.append(rec)
        print(
            f"  run {i}: ptok={rec['ptok']} wall={rec['dt']:.2f}s "
            f"first_tok={rec['tokens'][0]!r} lp0={rec['tok_lp'][0]:.9f}",
            flush=True,
        )
    for i in range(1, reps):
        ids, ds, dt = cmp(runs[0], runs[i])
        print(f"  run0 vs run{i}: ids_same={ids} max|dlogprob|={ds:.3g} max|d_top5|={dt:.3g}")
    print("  -- repeats that share identical engine state (2..N) --")
    for i in range(2, reps):
        ids, ds, dt = cmp(runs[1], runs[i])
        print(f"  run1 vs run{i}: ids_same={ids} max|dlogprob|={ds:.3g} max|d_top5|={dt:.3g}")
    return runs


if __name__ == "__main__":
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    run("LONG prompt (prefill > 16 tokens)", LONG_PROMPT, reps)
    print()
    run("SHORT prompt control", SHORT_PROMPT, 3, max_tokens=8)
