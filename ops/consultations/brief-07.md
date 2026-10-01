Context: single 170HX (GA100-class, 64 GiB, 74 SMs, cc 8.0) under WSL2 (Ubuntu 22.04 guest,
Windows KMD 616.92), vLLM 0.29 serving a Qwen3-Next-style hybrid model (gated-delta-net linear
attention + QSA sparse paged attention + MoE + PLE SSD offload), MTP=1, cudagraph
FULL_AND_PIECEWISE, env VLLM_WSL2_ENABLE_PIN_MEMORY=1 (worth 30-35% if removed).

Phenomenon: in a fresh container, the FIRST prefill request with >=96K tokens permanently
degrades *prefill* throughput. Fresh 2048-token prefill goes 0.90-1.0 s -> 2.0 s, and with
further GPU activity it worsens further (measured today: 2.29 -> 2.82 -> 4.00 s across ~20 min
of probing). Decode (bs=1) is unaffected. A container restart clears it completely (100%).
No kernel/driver error appears: the WSL dmesg ring buffer is now clean and stays at zero new
messages while poisoned -- no Xid, no TDR, no reset, no paging/migration message.
No thermal/power throttle: SM 1477 MHz, 105 W, all clocks_event_reasons = 0.
Hardware health in the poisoned state: bf16 8192^3 GEMM = 206.7 TFLOPS, D2D memcpy = 1.37 TB/s.

ALREADY ELIMINATED (measured, with numbers):
- PLE/SSD/Python bookkeeping, KV-cache scan width, illegal indexer budgets, spec-decode params.
- QSA kernel launch parameters are BYTE-IDENTICAL between the clean and poisoned 2048-token
  request; a 4-cell crossover probe (real/zero K/V x real/table-shifted) gave 4 identical slow
  cells, so placement/table values/data values are irrelevant.
- Per-CTA %globaltimer/%smid trace of the production QSA kernel (4096 CTAs, 74 SMs):
  clean vs poisoned envelope 2.19 ms vs 89.9 ms; D[med/p95/max] = 190/347/389 us clean vs
  675/31752/66722 us poisoned; CTA concurrency constant at 222/222 with no scheduling gap;
  phase medians (prologue/index/KV-load-wait/math+rest/epilogue) 1/1/43/145/0 us clean vs
  1/1/103/544/1 us poisoned. So: no residency loss, no gaps -- it is intra-CTA, heavy tail.
- Per-access *completion* latency thermometer (inline PTX: %clock64 read, then ld.global, then a
  second %clock64 gated on a predicate derived from the loaded value; MLP 1/4/32/512/1024 per
  SM; both a fresh incompressible 64 MiB buffer and the real KV pool; ~1M samples per state):
  poisoned/clean ratio 1.05x (MLP 1024) to 2.7x (MLP 1); NO sample above 10 us, max 5.5 us;
  the latency-vs-MLP curve shape is identical in both states. (Caveat: that thermometer's address
  permutation footprint was only ~1 MiB, so it could not have exercised page-table coverage.)
- Grid-size sweep, today, IN THE POISONED STATE, into a fresh incompressible 268 MiB buffer:
  256 B rows at 4 KB stride, total bytes fixed, grid = 16/64/256/1024/4096/16384/65536 CTAs
  -> 46-145 GB/s in every configuration. So CTA count / grid size is NOT the discriminator.

NEW, TODAY (this is the sharp part):
Same-instant arms inside the engine process while poisoned:
  G4  4096^3 bf16 GEMM                    0.69 ms        1.0x
  BW  32 MiB contiguous copy              1394 GB/s      1.0x
  KRD dense read of the real KV pool      626 GB/s       1.0x
  LAT 8 B random loads, per-warp MLP 1    698 ns/load    2.8x
  KSC 16384 random 2 KB rows, real KV pool  1.6 GB/s    73x worse
  FSC 4 KB-stride 256 B rows, fresh 64 MB buffer 0.8 GB/s 200x worse
  STR same pattern, different permutation  0.8 GB/s     200x worse
  OE  production QSA kernel               22.3 ms        10.6x worse
FSC/KSC's SECOND repetition is worse than the first (10.4 -> 20.8 ms): the pattern itself
worsens the state.
Now, the decisive new control: the EXACT failing torch pattern (out.copy_(src[random_index]),
256 B rows) replicated *outside the engine*, in a fresh Python process inside the same
container, at the same poisoned instant: 0.10-0.20 ms per 4 MiB pass = 25-45 GB/s, and 60
consecutive repetitions had max 0.183 ms with ZERO outliers beyond 4x the minimum.
A footprint sweep in that fresh process (16 MiB / 64 MiB / 256 MiB / 1 GiB / 2 GiB / 3 GiB,
i.e. up to 786k 4 KB pages) stayed flat at 22-28 GB/s for the scatter and 34-56 GB/s for the
dense control.
Also: the WSL2 dmesg ring buffer had been saturated for 3.7 h by a runaway monitor loop I had
accidentally left running (a shell loop invoking `nvidia-smi --query-gpu=clocks...` that hung in
D state, issuing ~850 *failed* dxgkio_escape ioctls per second, 186235 samples written). I killed
it: 0 new messages. The poison reproduces identically with the storm gone (so it was a confound,
not the cause) -- and it means the log was masking any real driver error all along.
Windows-side NVML (real nvidia-smi.exe on the host): ECC is N/A everywhere (unsupported on this
board), Remapped Rows/Banks N/A, HW Thermal / HW Power Brake / SW Thermal slowdown Not Active,
P0, memory 61657/65536 MiB used.

So, as of now: the pathology is *process-local* (engine process devastated, fresh process in the
same container at the same instant completely healthy, same driver, same device), *pattern
selective* (scattered short rows catastrophic, dense streaming and single contiguous copies full
speed, 8 B random single-outstanding loads nearly full speed), worsens with activity, and is
cleared only by restarting the process/context.

Questions:
1) Which process-local, pattern-selective device state can produce this? Enumerate the candidates
   you consider live, given that a fresh context created *after* the poisoning is healthy.
   In particular: per-context GPU page tables / UVMM mappings for a ~61 GiB context vs a ~3 GiB
   one; page-table *level* occupancy; ranges that have been evicted/reclaimed/remapped under
   dxgkrnl residency bookkeeping; sector/L2 partition aliasing computed from *virtual* addresses;
   replay/in-flight capacity that is per-context; anything in the WSL2 GPU-PV layer bound to a
   process's allocation history.
2) My proposed next discriminator: using CUDA VMM, reserve a huge VA (e.g. 32-48 GiB), map ONE
   small physical allocation (64 MiB) repeatedly (aliasing) across that VA, then run the same
   scatter pattern over the whole VA range but always landing in mapped pages, comparing with the
   same 64 MiB physical allocation not aliased. If the aliased case collapses, page-table/TLB
   coverage is the carrier; if not, page tables are excluded. Are the confounds acceptable
   (aliasing must not be coalesced; the walker must not detect repetition)? What would you change?
3) If (2) is negative, what single experiment has the best chance of discriminating the remaining
   candidates, given that CUPTI/nsys/ncu/perf are all unavailable, and that the only workable
   instruments are: CUDA events, inline PTX (%clock64, %globaltimer with 1024 ns granularity,
   %smid, %lanemask), Triton kernels with arbitrary grids, torch allocation patterns, and
   read-only Windows/WSL-side observability?
4) Why would a single >=96K-token prefill (which allocates a few GiB more KV and touches new
   pages) permanently and increasingly damage a process's scattered-access throughput, while
   leaving dense access intact -- and why does activity make it worse rather than drain it?
5) If this is a WSL2 GPU-PV/dxgkrnl defect, is there any *legitimate* mitigation short of
   restarting the engine (e.g. allocation placement, pre-touching, page size selection, avoiding
   a particular allocator, disabling pin-memory, a driver-level setting), and what would you
   measure to confirm it?

Constraints: read-only inspection only. Do NOT restart, stop or reconfigure the container
`hong-pc`, and do NOT run GPU-heavy benchmarks. Answer in English, concrete and quantitative.
