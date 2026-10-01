Follow-up on the same system (single 170HX, 64 GiB, WSL2 Ubuntu 22.04 guest, Windows KMD 616.92,
vLLM 0.29 serving a Qwen3-Next-style hybrid model: GDN linear attention + QSA sparse paged
attention + MoE + PLE SSD offload, MTP=1, cudagraph FULL_AND_PIECEWISE,
VLLM_WSL2_ENABLE_PIN_MEMORY=1, --max-model-len 262144, KV pool ~9.2 GiB, gpu-mem-util 0.96,
--max-num-batched-tokens 2048, PLE SSD offload 512 MiB row cache).

## What we now know (measured today; this closes the carrier question)

The failure: the FIRST prefill request with >=96K tokens in a fresh container permanently
degrades *prefill* throughput (fresh 2048-token prefill 0.95 s -> 2.1 s, worsening with further
activity, measured 2.29 -> 2.82 -> 4.00 s). Decode is unaffected. Only context/process
destruction recovers it (100%).

Just-measured decisive battery, run INSIDE the poisoned engine process (torch-only, so it is
safe inside a graph replay), 4096 rows x 2 KiB random-row gather = the failing shape:
  8 freshly allocated 64 MiB buffers at 8 DISTINCT virtual addresses spanning 48 GiB:
     poisoned 39.68 / 39.68 / 39.68 / 39.68 / 39.68 / 39.68 / 39.68 / 39.68 ms
     clean     0.174-0.176 ms                                              => 225x, uniform
  dense contiguous copy of THE SAME 8 buffers, same instant:
     poisoned 0.090-0.092 ms, clean 0.092 ms                                => 1.00x
  A fresh Python process in the same container at the same poisoned instant doing the exact
     same torch gather: 25-45 GB/s, 60 consecutive reps max 0.183 ms, zero outliers => healthy.
  Grid sweep in the poisoned state: grid 16..65536 CTAs, 46-145 GB/s, every configuration fine.
  Footprint sweep in a fresh process: 16 MiB..3 GiB, flat 22-28 GB/s.
  Per-access completion latency thermometer (inline PTX %clock64 gated on the loaded value,
  MLP 1..1024 per SM): poisoned/clean 1.05x-2.7x, NO sample >10 us, max 5.5 us.
  Per-CTA %globaltimer trace of the production kernel: concurrency constant 222/222, no gaps,
  median CTA 1.9x slower but p95 32 ms and max 67 ms.
  Kernel log is clean (no Xid/TDR); Windows NVML shows no ECC (unsupported), no remapped rows,
  no thermal/power throttle, P0.

Conclusion we drew: the carrier is a CUDA-context-scoped translation/replay resource that
charges per *scattered short request*; it is uniform across the entire address space and across
all allocations, dense streaming never triggers it, a new process/context is immune, and one
>=96K-token prefill turns it on irreversibly. Chunked prefill (2048-token chunks, i.e. the 96K
request arrives as 48 chunks) does NOT prevent it, so we read the trigger as the *accumulated
mapped/touched working set* rather than a burst.

## Question: practical remedies

The WSL2 GPU-PV layer is closed; we cannot patch it. We need the best available *engineering*
answer, and we want to test only the changes with the highest chance of actually working.

1) Is "destroy the context" really the only reset, or is there any arrangement of allocation /
   mapping order that avoids the threshold in the first place? Concretely, which allocation-
   topology knobs could plausibly move the threshold, and what is the mechanism you would
   expect for each? Specifically evaluate:
   - reducing the total mapped/touched device bytes (smaller KV pool: lower --max-model-len,
     lower gpu-memory-utilization, fp8 KV cache, more PLE offload) so the >=96K request never
     crosses the threshold;
   - PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True vs False;
   - PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync (VMM-based suballocation, different
     mapping granularity and VA topology);
   - preallocating or pre-touching the whole KV/activation working set at startup (vLLM already
     does a memory-profiling pass at init) so later allocations are not new mappings;
   - disabling prefix caching, disabling MTP, changing cudagraph capture sizes, or disabling
     cudagraphs entirely (does the captured graph's memory pool create the particular mapping
     topology?);
   - splitting one 96K request into two 48K requests (we have not tested it);
   - isolating long-context requests into a separate process/instance;
   - anything on the Windows/dxgkrnl/driver side that a user can legitimately set (WDDM
     residency policy, hardware-accelerated GPU scheduling, page size, TCC vs WDDM, driver
     version, Windows build, driver "prefer maximum performance", disabling GPU scheduling).
2) Rank the candidate interventions by (probability of actually working) x (cost to test), and
   for each give: the exact change, the cheapest test, and the predicted observable that would
   falsify it. Assume each test costs one ~6-minute engine restart plus a ~1-minute measurement,
   so we can afford maybe 4-6 tests total.
3) If none of them works: is "restart on demand" the correct engineering answer, and what is the
   most reliable way to *detect* the poisoned state cheaply from outside the process (we can
   poll /metrics and run a fresh 2048-token prefill, but we would like something faster and
   less invasive)?
4) How do we confirm a candidate fix is real rather than coincidence? Our plan: fresh container,
   apply exactly one change, run the same trigger (one 98304-token prefill) and the same fresh
   2048-token prefill measurement, and require the clean number (<=1.0 s) both after the trigger
   and after 20 more minutes of activity. Is there a stronger or more economical protocol?

Constraints: read-only inspection only. Do NOT restart, stop or reconfigure the container
`hong-pc`, and do NOT run GPU-heavy benchmarks. Answer in English, concrete, quantitative, and
ordered by expected value.
