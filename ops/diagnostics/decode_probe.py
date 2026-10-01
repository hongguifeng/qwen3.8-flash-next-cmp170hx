"""Precise single-request decode timing: TTFT + per-token gaps from SSE."""
import json, statistics, sys, time
import requests

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:9393"
TOKENS = int(sys.argv[2]) if len(sys.argv) > 2 else 512
PROMPT = "Write a detailed technical explanation of how a write-ahead log works in a database."

stamps = []
t0 = time.perf_counter()
with requests.post(
    URL + "/v1/chat/completions",
    json={
        "model": "Qwen3.8-Flash-Next",
        "messages": [{"role": "user", "content": PROMPT}],
        "temperature": 0,
        "max_tokens": TOKENS,
        "ignore_eos": True,
        "stream": True,
        "chat_template_kwargs": {"enable_thinking": False},
        "stream_options": {"include_usage": True},
    },
    stream=True,
    timeout=1800,
) as r:
    r.raise_for_status()
    usage = None
    for line in r.iter_lines():
        if not line.startswith(b"data: "):
            continue
        body = line[6:]
        if body == b"[DONE]":
            continue
        d = json.loads(body)
        if d.get("usage"):
            usage = d["usage"]
        for ch in d.get("choices") or []:
            delta = ch.get("delta") or {}
            if delta.get("content"):
                stamps.append(time.perf_counter() - t0)

n = usage["completion_tokens"] if usage else len(stamps)
ttft = stamps[0] if stamps else 0
gaps = [b - a for a, b in zip(stamps, stamps[1:])]
body = stamps[-1] - ttft
print(f"prompt_tokens={usage['prompt_tokens'] if usage else '?'} completion_tokens={n}")
print(f"TTFT {ttft:.3f}s  total {stamps[-1]:.3f}s")
print(f"decode (after first token): {len(gaps)} gaps, {len(gaps)/body:.2f} tok/s")
if gaps:
    q = statistics.quantiles(gaps, n=100)
    print(f"per-token gap: mean {statistics.mean(gaps)*1000:.1f} ms  "
          f"median {statistics.median(gaps)*1000:.1f} ms  "
          f"p10 {q[9]*1000:.1f}  p90 {q[89]*1000:.1f}  max {max(gaps)*1000:.1f} ms")
