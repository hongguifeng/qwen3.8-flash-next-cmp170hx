#!/usr/bin/env python3
"""GPU memory-path probe: streaming vs page-scattered gather vs compute.

Runs inside the engine container (no vLLM imports, its own CUDA context).
Footprint is kept small (~300 MiB) so it fits next to the 96%-full KV cache.

  docker exec hong-pc /opt/vllm/.venv/bin/python /vmm_probe.py <label>

The point: a poisoned engine does 7x more GPU-timeline work per prefill
segment and 8-15x more inside the QSA gather kernels, while pure compute and
D2D streaming are unaffected.  If the *gather* rate here is degraded but
streaming/compute are not, the poison is a GPU virtual-memory / page-locality
effect rather than anything in the engine's kernels or its Python.
"""
import sys
import time

import torch

LABEL = sys.argv[1] if len(sys.argv) > 1 else "x"
DEV = "cuda"


def timed(fn, iters=10):
    for _ in range(2):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def report(name, sec, byts):
    print(f"{LABEL:8s} {name:34s} {sec*1e3:9.3f} ms  {byts/sec/1e9:7.1f} GB/s", flush=True)


print(f"--- {LABEL}: device={torch.cuda.get_device_name(0)} "
      f"free={torch.cuda.mem_get_info()[0]/2**20:.0f} MiB", flush=True)

# ---------------------------------------------------------------- compute
a = torch.randn(4096, 4096, device=DEV, dtype=torch.bfloat16)
b = torch.randn(4096, 4096, device=DEV, dtype=torch.bfloat16)
sec = timed(lambda: a @ b, 20)
report("bf16 4096^3 matmul (2*FLOP)", sec, 2 * 4096**3)

# ---------------------------------------------------------------- streaming
MB = 2**20
src = torch.randn(256 * MB // 4, device=DEV, dtype=torch.float32)
dst = torch.empty_like(src)
sec = timed(lambda: dst.copy_(src), 20)
report("streaming copy 256 MiB", sec, 2 * src.numel() * 4)

sec = timed(lambda: src.sum(), 20)
report("streaming reduce 256 MiB", sec, src.numel() * 4)

# ---------------------------------------------------------------- gathers
n = src.numel()
for page_kb in (4, 16, 64):
    step = page_kb * 1024 // 4            # fp32 elems per "page"
    pages = max(1, n // step)
    pick = torch.randint(0, pages - 1, (pages,), device=DEV).long() * step
    off = torch.arange(step, device=DEV)
    idx = (pick[:, None] + off[None, :]).reshape(-1)[: n // 2]
    out = torch.empty(idx.numel(), device=DEV, dtype=torch.float32)
    sec = timed(lambda: torch.index_select(src, 0, idx, out=out), 10)
    report(f"page-scattered gather {page_kb:2d} KiB", sec, out.numel() * 4)

# random element-wise gather (worst case, one 4B load per TLB entry)
idx = torch.randint(0, n - 1, (64 * MB // 4,), device=DEV).long()
out = torch.empty(idx.numel(), device=DEV, dtype=torch.float32)
sec = timed(lambda: torch.index_select(src, 0, idx, out=out), 10)
report("random element gather 64 MiB", sec, out.numel() * 4)

# ---------------------------------------------------------------- small/L2
small = torch.randn(8 * MB // 4, device=DEV, dtype=torch.float32)
sid = torch.randint(0, small.numel() - 1, (8 * MB // 4,), device=DEV).long()
sout = torch.empty(sid.numel(), device=DEV, dtype=torch.float32)
sec = timed(lambda: torch.index_select(small, 0, sid, out=sout), 10)
report("random gather 8 MiB (L2-ish)", sec, sout.numel() * 4)
