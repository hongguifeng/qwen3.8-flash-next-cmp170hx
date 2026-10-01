# Consulting request (round 2): root-cause a persistent PREFILL slowdown; no CUPTI available

You are helping debug a live local inference service. Be concrete and quantitative. You may inspect the
machine read-only (`docker exec hong-pc ...`, `cat /proc/...`, `nvidia-smi`, `iostat`, `py-spy dump/record`,
`curl -s localhost:9393/metrics`, `docker cp`), and you may read the patched vLLM source inside the container
(e.g. `/opt/vllm/src/vllm/models/qwen4_exp/nvidia/*.py`). You must NOT restart/stop/rm/reconfigure the container
`hong-pc`, must not change any file inside it, and must not run GPU-heavy benchmarks. Container python is
`/opt/vllm/.venv/bin/python` (there is no `python3` in PATH).

## 0. Environment
- WSL2 (kernel 6.18.33.2-microsoft-standard-WSL2), NVIDIA driver 610.43.03 inside guest, Docker container `hong-pc`.
- GPU: one "170HX" (custom sm_80, 64 GiB, 74 SMs), SM clock pinned/stable 1485 MHz (max 1695), mem 1728 MHz.
  Measured raw: bf16 8192^3 matmul = 5.32 ms best / median 5.67 ms = 194-207 TFLOPS; D2D copy 1.37-1.67 TB/s.
  Idle 35 W, saturated ~249 W. PCIe gen2 x8 (~3.2 GB/s each way).
- Stack: patched vLLM `0.29.1rc1.dev402+ga5a30471f.ple1`, torch `2.13.0+cu130`, Triton JIT (cache persisted on a
  host bind mount). Engine runs with `VLLM_USE_BREAKABLE_CUDAGRAPH=1`, `cudagraph_mode=FULL_AND_PIECEWISE`,
  capture sizes `[1..16,18,20,...,32,64,128,256,512,1024,2048]`, chunked prefill `max_num_batched_tokens=2048`,
  `max_num_seqs=4`, `max_model_len=262144`, MTP speculative decoding with `num_speculative_tokens=1`,
  prefix caching ON, `--mamba-cache-mode align`, `gpu_memory_utilization 0.96` (64.8 GiB reserved; ~0.7 GiB free).
- Model: 48 layers = 36 GDN linear-attention (gated delta net, chunked-prefill Triton kernels) + 12 full attention
  (`full_attention_interval=4`), MoE 512 experts top-10, hidden 2560. Plus one special "PLE" layer (`layers.1`)
  whose per-token embeddings are read from a 95.4 GiB SSD-backed row table (16 rows/token, 320 B/row) through a
  native AIO reader with a 512 MiB / 1,198,372-row LRU row cache, 16 workers, AIO queue depth 256, and an
  optional per-request read-ahead of 16384 tokens. Chunk = 2048 tokens = 32768 PLE rows = 10.5 MiB useful
  (device traffic 13.8x that: 4427 B read per 320 B row).

## 1. Symptom
(a) A single request with >=~64-100K prompt tokens is slow (143K real English text = 270.97 s = 528 tok/s), and
(b) **afterwards the engine stays in a "poisoned" state** in which every subsequent *prefill* is 4-9x slower and
this persists across unrelated, short, fresh requests until the container is restarted:
| request | clean container | poisoned container |
|---|---|---|
| fresh 8K ids | 2.22-2.31 s (3548-3695 tok/s) | 9.29-13.05 s (628-882 tok/s) |
| fresh 32K ids | 9.58-9.88 s (3318-3420 tok/s) | 41.5-54.9 s (572-789 tok/s) |
Decode is NOT affected: in the poisoned state a `max_tokens=64` request measured 15.3 ms/token (clean-level;
clean conc=1 = 110-120 tok/s). Chunk size has no effect (2048 in both states; 8192 chunks are slower in both
because they miss the captured CUDA-graph sizes and run eager). Prompt content/ids has no effect.

## 2. What I measured this round (all reproducible; current container is in the poisoned state)
### 2.1 A/B: the PLE read-ahead is NOT the trigger
Restarted with the PLE prefetch disabled (`ple_ssd_prefetch_tokens=0`, verified no prefetcher thread):
clean 8K 2.97 s (2756 tok/s), clean 32K 11.25 s (2912), then 143K real text 270.97 s (528), then fresh 8K
10.28 s (797) and fresh 32K 44.73 s (733) -> **still poisoned**. (Prefetch off *hurts* the clean state ~24%, so it
stays on in production.)
### 2.2 Per-chunk CUDA-event phase timers inside the PLE layer (I mount an instrumented copy of the PLE module)
Per chunk (2048 tokens / 32768 PLE rows), milliseconds:
| phase | clean 32K | poisoned 32K |
|---|---|---|
| chunk wall time | 546 | 2305 |
| `io` (real native-AIO device read) | 191 | 209 |
| `W_ids_sync` (worker waits for the compute-stream event that must precede the pre-ids work) | 257 | **1826** |
| `W_table_read` (LRU lookup/evict + H2D enqueue, worker) | 221 | 267 |
| `M_wait_pending` (engine thread blocked on the PLE future = 91% of the chunk) | 477 | **2092** |
| python lookup/fill/join | ~25 | ~51 |
The SSD path is at full speed in the poisoned state (~157k rows/s), identical to a standalone harness using the
same `.so` (162 us for one 4 KiB read at depth 1; 140k rows/s at depth 256). So the po*isoned cost is that the GPU
takes ~1.8 s instead of ~0.26 s to drain the work queued before the PLE id handoff.*
PLE is a depth-1 pipeline: `start_prefetch()` copies ngram ids device->device, records an event, submits a future;
the worker (`_read_and_copy`) first does `self._ids_ready.synchronize()` (waits for the compute stream to reach
that event), then reads the SSD, then H2D; the engine thread later calls `_pending.result()` in `_finalize_prefetch`.
### 2.3 py-spy (all threads, 150 Hz, 14 s) during a poisoned 8K prefill (15.58 s, 526 tok/s)
- Engine MainThread: **73% of samples in `concurrent/futures/_base.py:451 Future.result()` <- `_finalize_prefetch`
  <- `<lambda>` <- `torch/cuda/graphs.py:219 replay()`**; 18% idle waiting for the next request.
  => the engine thread is blocked *inside a CUDA-graph replay* at the PLE break; graphs ARE replayed (not eager),
  and the depth-1 PLE handshake exposes the raw GPU time on the engine thread.
- PLE worker thread: 58.4% `Event.synchronize` (the `_ids_ready` wait), 27% idle, 10.4% in the native read.
### 2.4 Concurrent-GEMM GPU-availability probe (separate process in the same container, continuous 8192^3 bf16,
5.3 ms/call = 206-211 TFLOPS when uninterrupted, sampling loop)
While the poisoned engine served a prefill, the probe GEMM was *starved of the GPU for blocks of ~1.9-2.7 s*
every ~3.4 s (13 blocks in 2060 samples), running at full speed in between. During those blocks `nvidia-smi`
showed SM util 100% but only ~86-152 W (vs 249 W saturated), clocks 1485 MHz. Same probe, decode-only request:
0.98 s / 64 tokens = 15.3 ms/token (clean level). => the engine monopolizes the whole device in ~2 s low-power
blocks, and only *prefill* does this.
### 2.5 Ruled out, each by measurement
GPU clock/power limits (1485 MHz in every phase; only a `0x4` SW power cap during the long request, peak 268 W);
GPU raw capability in the poisoned state (194-207 TFLOPS, 1.37 TB/s D2D); Triton JIT/autotune recompiles (0 files
added to the persisted Triton cache during poisoning; only 3 one-shot spec-decode kernel warnings at startup);
token recompute / preemption (`request_prefill_kv_computed_tokens_sum == prompt_tokens_total == 224903`;
`num_preemptions_total` 0); KV/prefix residency (`kv_cache_usage_perc` 0 and `prefix_cache_hits_total` 0 after the
long run); host RAM/swap (RSS 4.4 GB, VmSwap 0); thread leak/spin (55 threads, 0.5% CPU idle, disk 0 r/s idle);
AIO-context decay (standalone reader full speed while poisoned); prompt content; chunk size; `ple_ssd_cache_mb`
512 vs 4096.
### 2.6 Kernel-level profiling is NOT available in this environment (verified just now, free)
- `torch.profiler` with CUDA activity: `CUPTI_ERROR_CMP_DEVICE_NOT_SUPPORTED (42)`, 0 CUDA events.
- `nsys profile` (2026.2.1, present at /usr/local/bin/nsys): "Importer error status: An unknown error occurred.
  Unable to retrieve the importer version: skipping importation of the QDSTRM file" -> no `.nsys-rep` produced.
- `ncu`: `ERR_NVGPUCTRPERM` (needs `NVreg_RestrictProfilingToAdminUsers=0` on the Windows host driver - not
  settable from inside WSL2).
- `nsys status -e`: root enabled, but `perf_event_open syscall available: Fail` / CPU profiling env: Fail.
- CUDA *events* do work (my PLE timers are event-based and consistent to a few ms).

## 3. Proposed next step (ONE restart, ~7 min) and my questions
**Plan A - event-based per-module GPU attribution without CUPTI.** I have a working "patch one module file into the
container" mechanism (an env var in my launcher bind-mounts a host file over
`/opt/vllm/src/vllm/models/qwen4_exp/nvidia/ple_ssd.py`). At import time I would install a global forward hook
(`torch.nn.modules.module.register_module_forward_hook`) that, for whitelisted class names, records a
`torch.cuda.Event` pair and accumulates elapsed device time per class per engine step, logging every N steps.
Target classes (from `models/qwen4_exp/nvidia/model.py`): `Qwen4ExpDecoderLayer` (:173, forward :276),
`Qwen4ExpSparseMoeBlock` (:159, MoE incl. the third-party "humming" indexed-GEMM experts), the PLE layer
(`Qwen4ExpPLELayer`), the GDN linear-attention module and the full-attention module (in the transformers-side
modeling file), and the MTP/lm_head module. Then run the same sequence (clean 8K, clean 32K, 143K text, fresh 8K,
fresh 32K) and compare per-class GPU time per chunk between clean and poisoned.
Q1: Is this sound given *breakable CUDA graphs*? Python forward code runs every step and the graph pieces are
replayed in place, so hooks should fire every step and an end-event on the current stream should measure the
replayed piece. Pitfalls I should pre-empt: the PLE layer uses its *own* CUDA stream (event ordering!), async
ops not on the current stream, event-pool exhaustion, hook overhead, inclusive-vs-self attribution with nested
hooks, and anything that would force a sync/`.item()` and thereby *break* the graph-replay path or change the
behaviour being measured. Also: is `register_module_forward_hook` (global) compatible with this patched vLLM
(its MoE/quantization layers may bypass `nn.Module.__call__` in places)? Any better hook point?
Q2: In a no-CUPTI environment, is there a *better* way to attribute the time to a module/kernel that I have
missed? (NVTX+nsys is dead for the same reason; `torch.cuda.nvtx`; `cudaProfilerStart/Stop`;
`torch.cuda.memory._record_memory_history`; `nvidia-smi dmon`/PCIe counters; `/proc/driver/nvidia/...`;
anything in the `nvidia` python packages installed in the venv?)
**Plan B - config bisect (one restart each).** Available knobs: `QWEN_EAGER` (`--enforce-eager`),
`QWEN_GRAPH_MODE`, `QWEN_MTP`, `QWEN_CAPTURE_SIZES`, `QWEN_SEQS`, plus `VLLM_HUMMING_MOE_GEMM_TYPE=auto|indexed|
grouped|grouped_contiguous`, `--no-enable-prefix-caching`, `--mamba-cache-mode {align,...}`.
Q3: Which single restart is most informative? E.g. does `--enforce-eager` (a) *remove* the poisoning (=> CUDA-graph
machinery/pool implicated) or (b) keep it but uniformly ~2x slower (=> graphs innocent, the slow path is
data/state-dependent)? Is it better to combine eager + Plan A instrumentation in one restart (risk of confound: the
clean baseline also becomes 2x slower, but the *ratio* clean:poisoned should be preserved)?
**Plan C - hypothesis-driven micro-experiments.** Use `/metrics` (`vllm:num_requests_running`, `num_waiting`,
`iteration_tokens_total`, `spec_decode_num_*`, `engine_sleep_state`, `cache_config_info`) and cheap API calls to
discriminate: is the poisoned state simply *more sequences in flight* (max_num_seqs=4; maybe after a long request
the scheduler keeps more work in flight so each PLE chunk waits behind other requests' chunks and the per-chunk
wall time looks worse)? Counter-evidence: the concurrent-GEMM probe shows the GPU genuinely busy in ~2 s blocks,
and my chunk timer is measured inside the request's own PLE call. Would `QWEN_SEQS=1` be a decisive test?
## 4. Mechanism candidates I would like you to rank and attack
C1. **Spin-wait/semaphore kernel** left in a bad state or mis-tuned: stream-K/split-K style kernels (the MoE
"humming" backend logs tuning configs like `{'block_shape':[16,64,128],'use_stream_k':True,'num_write_splits':1,
'num_sms':74,'num_stages':5}`; my notes say `max_k_block` was capped 128) - such kernels occupy SMs while waiting
on flags, which would explain 100% SM utilisation at only ~100 W, starving the other process, and *persisting* if
some counter/flag/queue is left wrong.
C2. **Data-shaped slowdown**: some metadata tensor whose *content* the long request changed and which persists in
the engine (KV block table / mamba state index table / GDN chunk metadata / a "slot" table that grew or got
fragmented), making subsequent prefill kernels read from far-apart addresses (poor locality) even though shapes
are identical. Note `kv cache group sizes [1584,1584,1584,1584,8,1584]` (one group of 8!) with `mamba_cache_mode
align`, and `Current kv cache memory in use is 9.22 GiB`.
C3. **Allocator / CUDA-graph-pool state**: after a huge request the caching allocator may hold a very different
set of blocks, and the engine may now take a different code path (extra copies, different padding) with the same
shapes.
C4. **WSL2 guest-driver scheduling degradation after a huge burst** - partially refuted: a *separate process*
inside the same container still got full GEMM speed in the poisoned state, and a fresh container is instantly fast,
so it is per-process/per-context rather than device-wide.
C5. Something in the *chunked-prefill scheduler*: e.g. after the long request the engine schedules *prefill-only*
steps in a way that gives each step a much larger `num_tokens` (or batches several requests' chunks), so the
per-chunk time grows - I measured `M_wait_pending` per chunk, but I never verified `num_tokens` actually processed
per engine step in the poisoned state. Is there a cheap way to see the tokens per step (`iteration_tokens_total`
histogram? engine log with `--enable-logging`?). 
C6. Something in the *GDN/mamba state* path: `mamba_cache_mode=align`, 36 GDN layers, 12 full attention. The GDN
chunked-prefill kernels (`chunk_scaled_dot_kkt_fwd`, `recompute_w_u_fwd`, `l2norm_fwd`, ...) were the ones that
triggered Triton JIT compiles at startup. Could a state/chunk-size index (e.g. `chunk_indices` sized by the longest
sequence ever seen) make them slower for *all* subsequent requests?
## 5. Deliverable I want from you
1. Your ranked judgement of the mechanism candidates (with reasoning about the exact evidence), especially which
one explains "same shapes/config, 7x slower, prefill-only, ~100 W at 100% SM util, starves other contexts,
cleared by process restart".
2. ONE recommended next restart: either Plan A (with the precise hook/aggregation design + pitfalls), or Plan B
(which single knob), or a combination - plus the exact env/args and the interpretation matrix for its outcomes.
3. The cheap, no-restart checks I should run *right now* in the poisoned container to discriminate further
(e.g. specific `/metrics` queries, specific `docker exec` one-liners, specific HTTP probes such as "2 concurrent
prefills", "one 2048-token request repeated 16 times", "same request with `max_tokens=1` vs 64").
4. Anything in the numbers above that contradicts my conclusion "the PLE/SSD path is exonerated; the cost is
GPU-side prefill work".
Keep it under ~1500 words if you can.
