You are advising on a vLLM prefill slowdown mystery on a single-GPU WSL2 box (NVIDIA 170HX 64GB, sm_80, 74 SMs, driver 610.43.03, kernel 6.x WSL2). Read-only role for you: do NOT suggest restarting/stopping/reconfiguring the engine container, and do NOT run GPU benchmarks. Just analyze the evidence and design the next measurement.

# The bug
The engine serves a Qwen3.8-Flash-Next MoE (48 layers) with MTP spec-decode. After ONE >=96K-token prefill request, the engine enters a "poisoned" state that persists for many minutes:
- fresh 2048-token prefill: 0.93 s -> 1.98 s (2.1x)
- a *particular* attention op (QSA sparse attention, reads the paged KV cache via a block table) goes 2.13 ms -> 22.3 ms (10.6x)
- decode is barely affected (21.1 vs 20.5 ms/token), MoE is 1.5x slower, dense GEMMs unaffected.
- The state is REVERSIBLE without restart: running my in-engine probe repeatedly (private stream, extra kernel launches) drains it back to clean within seconds. Severity varies over time (a request can take 3.2 s or 34 s). The FIRST >=96K request after container start reproduces it; later ones often don't.

# What is experimentally REFUTED (all measured)
1. Same compiled cubin, byte-identical launch parameters (grid/blocks/per-CTA work counts/selection widths identical to clean), same block-table ids, same data values (zero-filled K/V equally slow), eager vs CUDA-graph replay, host submission cost (0.1-0.4 ms) => not operands/placement/launch/host.
2. Clocks/power: 1477 MHz avg SM clock, no throttle flags, 105 W while slow.
3. Device throughput: 4096^3 bf16 GEMM = 0.69 ms (190 TFLOPS) at the SAME instant the attention op is 10x slow, on the SAME stream/window.
4. No idle gaps: a per-CTA %globaltimer trace of the attention kernel shows continuous coverage (union of CTA intervals == full envelope, env-minus-union = 0.000 ms), constant concurrency (~220 of 4096 CTAs in flight in BOTH states), all 74 SMs used. So it is not scheduling starvation or reduced residency.
5. Not the number of CTAs or their distribution.
6. Profile of per-CTA duration: clean median 112 us / p95 202 us; poisoned median 1217 us / p95 2292 us (about 11x uniformly) in one moment, but in a *later, more severe* moment: median 296 us, p95 22699 us, max 57648 us (79x tail), while the envelope grew 3.66 -> 88 ms.

# NEW decisive data (my in-engine probe, all arms launched at the SAME instant, same stream, interleaved)
Probe arms. "KSC": 16384 rows x 2048 B gathered by a random row index from the real KV-cache allocation (43 MB pool). "FSC": identical pattern from a FRESH 64 MB torch buffer. "STR": 16384 rows at fixed 2-row stride (4 KB apart) from that fresh buffer. "KRD": dense read of 16 blocks (52 MB) of the real KV pool. "BW": contiguous 32 MiB uint8 copy. "LAT": 1000-iteration dependent-load chain, 296 CTAs x 32 lanes = 9472 concurrent chains. "G4": 4096^3 bf16 matmul.

CLEAN state: G4 0.71 ms | BW 1365 GB/s | LAT 250 ns | KRD 618 GB/s | KSC 0.15 ms (223 GB/s) | attention op OE 2.13 ms
POISONED state: G4 0.69 ms | BW 1394 GB/s | LAT 400 ns | KRD 626 GB/s | KSC 10.4 ms (1.6 GB/s) | FSC 19.9 ms (0.8 GB/s) | STR 19.9 ms (0.8 GB/s) | attention op OE 22.3 ms
Also: KSC's SECOND repetition is ~2x slower than the first (10.4 -> 20.8 ms), i.e. the state degrades while the scattered pattern runs.
And a variant of the attention kernel that forced K/V data arrival every iteration (full-tile tl.sum) took 70 ms instead of 2.1 ms even in the CLEAN state, i.e. latency-bound variants are catastrophically sensitive; the production kernel (Triton software-pipelined, num_stages=2) hides most of it.

# My reading
The poisoned state looks like: a small fraction of memory accesses/pages acquire ENORMOUS latency (microseconds to ~ms), which destroys latency-sensitive scattered/short-row kernels (2 KB row gathers: 100-800x) while deep-MLP streaming and dense bandwidth-bound work is unaffected (BW 1.4 TB/s, dense GEMM full speed, 9472-way random 8B loads only 1.6x worse). Consistent with: the attention op is a scattered 256 B-per-token gather over a page-table-selected 43 MB pool, low MLP per warp -> it pays the long-tail latency; MoE dense GEMMs do not.

# Questions
1. Do you agree with "rare, enormous-latency accesses" as the mechanism, or is there a better single explanation consistent with ALL of the above (especially: fresh-buffer scatter FSC/STR equally slow, second rep worse than first, and a full stop/restart not being needed to clear)?
2. What exactly should I measure next to pin the tail latency itself? I can write CUDA/Triton kernels and run them inside the engine process, in both states, one execution per state. I have no CUPTI/ncu/nsys (CUPTI_ERROR_CMP_DEVICE_NOT_SUPPORTED; perf_event_open denied), but I have inline-asm %globaltimer, %smid, %clock64 and CUDA events.
   Specifically: how would you design (a) a per-access latency histogram over ~10^6 independent random accesses at controlled MLP (1, 4, 64, 1024 outstanding), (b) a test that distinguishes "a few pages are slow" from "all accesses are slow with a heavy tail", (c) a test for whether the long-latency fraction depends on access size (8 B, 128 B, 2 KB, 1 MB), and (d) whether out-of-order/deep-MLP hides it (a bandwidth-vs-MLP curve)?
3. Is there a plausible WSL2/driver-level explanation (dxgkrnl, GPU page residency/migration, memory carve-out, ECC scrub, compression) that fits: reversible without restart, severity drifting, first-trigger-only, and dense traffic unaffected while scattered collapses?
4. Which single next experiment has the highest information per unit risk, and what result would falsify my mechanism?

Be concrete and quantitative. English.
