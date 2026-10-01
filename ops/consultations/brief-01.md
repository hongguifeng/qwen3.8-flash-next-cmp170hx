# Consultation: why does the PLE SSD-offload read path collapse persistently?

You are debugging a vLLM deployment on a single-machine WSL2 setup. I have done extensive
measurement; I need you to reason over this evidence and tell me (a) the most likely mechanism and
(b) the exact next measurements/experiments that would discriminate between remaining hypotheses.
Feel free to read the files listed at the end. Do NOT restart, stop or reconfigure the running
container `hong-pc` — I need it alive. Read-only inspection (`docker exec hong-pc ...`, iostat,
py-spy, /proc) is welcome.

## Environment
- Windows 11 + WSL2, single GPU **RTX A170HX** (GA100-based mining card, 64 GiB HBM2e, 74 SMs,
  SM clock pinned 1485 MHz / 1695 max, mem 1728 MHz, PCIe **gen2 x8 => ~3.2 GB/s** host<->device).
- The Linux rootfs and the model files live on **/dev/sdd: 1 TB, WSL2 "Virtual Disk", and the kernel
  reports `rotational=1`** (`/sys/block/sdd/queue/rotational` = 1, scheduler=none, nr_requests=1267,
  logical_block_size=512, physical_block_size=4096).
- Model: `Qwen3.8-Flash-Next-AutoRound-3bpw-MTP` (142.5 GiB): MoE 512 experts top-10, 48 layers,
  hidden 2560. PLE ("position learning enhancement" / n-gram embedding) is a **320,001,536-row x
  160-dim BF16 table = 95.37 GiB (320 bytes/row)**, stored in 11 safetensors shards, and it is
  **offloaded to SSD** because it does not fit in VRAM.
- Container `hong-pc` runs vLLM 0.29.1rc1.dev402 with
  `--max-model-len 262144 --max-num-seqs 4 --max-num-batched-tokens 2048 --gpu-memory-utilization 0.96
   --enable-prefix-caching --enable-chunked-prefill --mamba-cache-mode align`
  `--additional-config {ple_ssd_offload:true, ple_ssd_workers:16, ple_ssd_cache_mb:512,
   ple_ssd_native_library:/opt/vllm/optimization/ple_ssd_io.so, ple_ssd_io_depth:256,
   ple_ssd_prefetch_tokens:16384}` and `VLLM_WSL2_ENABLE_PIN_MEMORY=1`. MTP spec-decode (1 token).
  API on :9393. There is exactly **one** PLE layer (layers.1), so 16 rows/token = 5 KiB/token of
  random SSD reads on the critical path.

## How the PLE SSD path works (source at /tmp/plesrc/ple_ssd.py, /tmp/plesrc/ple_ssd_io.c)
Per engine step (chunk of <=2048 tokens) the PLE layer at layer 1 does:
1. `compute_ngram_ids(...)` on GPU -> 2048*16 = 32768 int64 row ids (ids are a hash => uniformly
   random over the whole 320M-row space).
2. `start_prefetch()`: `self._copy_ready.synchronize()` (CPU waits for the previous chunk's rows H2D)
   -> `self._device_ids[:tokens].copy_(ngram_ids)` (**synchronous D2H** on the compute stream) ->
   `_ids` H2D on a side stream + `_ids_ready.record()` -> submit a job to a 1-thread
   `ThreadPoolExecutor` named `ple-ssd-prefetch`.
3. that worker: `_ids_ready.synchronize()` -> `PLESSDTable.read(ids, out)` -> `_copy_to_gpu()` (H2D on
   the side stream, 10.5 MB/chunk).
4. `_finalize_prefetch()`: `self._pending.result()` (**CPU blocks on the worker; the whole pipeline is
   depth-1, no cross-chunk pipelining**) -> `current_stream().wait_event(_copy_ready)` ->
   `output.copy_()`.
`PLESSDTable.read()` serializes ALL readers through one `threading.Condition` (`self._reading` flag;
demand reads get priority over read-ahead). `_read()` = `np.unique(32768 ids)` -> Python loop over
unique rows against an LRU `OrderedDict[int, bytes]` (`cache_limit = 512MiB/(320+128) = 1,198,372
rows`) -> `PLESSDNativeReader.read(missing)` -> Python loop building `bytes` per row ->
`b"".join(...)` -> numpy assign.
`PLESSDNativeReader.read()` builds offset/fd arrays and calls the native `rows_read()`
(`/tmp/plesrc/ple_ssd_io.c`): for each iocb it sets
`aio_offset = off & ~4095`, `aio_nbytes = ((off & 4095) + 320 + 4095) & ~4095`
(**=> every 320-byte row becomes a sector-aligned >=4 KiB O_DIRECT read**), submits in batches of
`depth=256` and **drains each batch completely (all 256 completions) before submitting the next**.
A separate `PLEPromptPrefetcher` (1 thread, `ple_ssd_prefetch_tokens=16384`) is submitted **once per
request** in `add_request` and reads the first 16384 tokens of the prompt (262144 rows = 84 MB of
useful data) through the same table lock in 256-row `prefetch=True` batches.

## Measured device capability (my own harness re-using the container's ple_ssd_io.so, on the real
95 GiB shard, offsets uniformly random over the whole file, 32768-row calls)
```
depth=1   6154 rows/s  (=> 162 us per single 4 KiB read, pure latency)
depth=2  11759 rows/s
depth=8  34509 rows/s
depth=32 66849 rows/s
depth=64 87862 rows/s
depth=128 114749 rows/s
depth=256 140329 rows/s     <-- 0.234 s per 32768 rows
```
`/proc/diskstats` for sdd shows **4427 bytes read per 320-byte row** (confirming the 13.8x
amplification; 138 MiB of device traffic per 10.5 MB of useful rows). So the device CAN do 140k
rows/s = ~620 MB/s of 4 KiB random reads at QD 256, and the delay scales almost linearly with depth
(the AIO *is* asynchronous despite rotational=1).

## Measured Python cost of the same path (micro-benchmark inside the container)
```
full 1.2M-row LRU: 32768 get+move_to_end   22.1 ms
full 1.2M-row LRU: 32768 insert+popitem    14.1 ms
native path: out[i].tobytes() x32768        9.3 ms
b"".join(...) 2.3 ms | np.unique 2.2 ms | 10.5 MB numpy assign 0.4 ms
```
=> ~36 ms worst case per 2048-token chunk = 6% of a fast step. Python bookkeeping is NOT the cause.

## The symptom (measured over many fresh container restarts, all with brand-new random token ids or
## fresh text so the prefix cache cannot hide anything)
| | fresh container (clean) | after one ~100K-190K-token request ("poisoned") |
|---|---:|---:|
| prefill 8192 tok | 2.25 s = 3643 tok/s | 9.29-13.05 s = 628-882 tok/s |
| prefill 32768 tok | 9.88 s = 3318 tok/s | 41.5-54.9 s = 572-789 tok/s |
| single-stream decode | ~110 tok/s (62-67 chunk/s) | 62-73 tok/s (34.5-40.8 chunk/s), +13 ms/step |
| prefix-cached re-hit of a 32K prompt | 0.95 s | 3.63 s |
| per-token prefill cost | ~0.28 ms | ~1.12 ms (+0.85 ms/token, i.e. ~56 us per PLE row) |

- The degradation is **superlinear in the long prompt's length and happens even as the very first
  request of a fresh container**: 100K tok = 51.8 s (1928 tok/s), 152K = 316 s (481), 186K = 563-573 s
  (326-331), i.e. ~1.3M rows/s -> ~5.3k rows/s.
- **After that, SHORT prompts stay 4-5x slow and decode 1.7x slow; only `docker restart` of the
  container restores full speed** (verified 4 times, e.g. immediately after restart 8K = 2.25 s /
  3643 tok/s, 32K = 9.88 s / 3318 tok/s). Restarting cannot be a GPU-clock effect: SM clock was
  sampled at 1 Hz through a clean run, a 186K-token run and the poisoned runs that followed, and it
  was **1485 MHz in every phase** (throttle reasons 0x1 = GpuIdle; 0x4 = SW power cap only during
  the long run, peak 268 W). Poisoned short runs drew ~100 W average vs 44 W for the clean 8K run.
- `iostat /dev/sdd`: in the poisoned state random reads go from **r/s 1.5k -> 52k, r_await 0.26 ->
  1.1-1.6 ms, %util up to 39%** (normally ~1.5%).
- py-spy in the poisoned state: MainThread inside `_finalize_prefetch` -> `self._pending.result()`
  (the `Future`), and the `ple-ssd-prefetch_0` thread inside `torch/cuda/streams.py:254`
  (`Event.synchronize`, i.e. `_ids_ready.synchronize()`), EngineCore CPU ~94% (~1 core burned).
  GPU utilization 90-100% but only 86-152 W.
- Ruled out already: (a) PLE row-cache size — `ple_ssd_cache_mb=4096` was no better (117K prompt =
  870 tok/s); (b) host memory/pinned-memory pressure (MemAvailable 53 GB, VmSwap 0, VmPin 0);
  (c) CPU Python cost (measured above); (d) GPU clock throttling (measured above); (e) content of the
  prompt (random ids vs real English/CJK text give the same rates); (f) a standalone run of the
  container's own native reader reaches 140k rows/s on the same file *after* the poisoned state,
  i.e. the device itself is not left slow.

## What I need from you
1. Given the above, what is the most likely mechanism for the persistent 4-5x collapse of a
   depth-1, single-threaded, demand-driven read pipeline whose device can do 140k rows/s?
   Specifically consider: the single `self._condition` serialization; the 256-row "drain before next
   batch" loop in `rows_read`; the one-window-per-request `PLEPromptPrefetcher` (16384 tokens = 84 MB
   per request) racing the demand reads and filling/evicting the 1.2M-row LRU; the
   `_copy_ready.synchronize()` / `_ids_ready.synchronize()` handshakes; whether an ever-growing
   `self._ids`-style buffer or a kernel/AIO-context state (io_setup depth 256) could persist and
   degrade; and whether the WSL2 virtual disk's queue/scheduler could end up throttled after a burst
   (the poisoned iostat numbers: 52k r/s, r_await 1.1-1.6 ms, 39% util).
2. Which single experiment would discriminate best, and what exact commands/counters would you use?
   (Please keep it to experiments that take minutes, not hours, and remember each container restart
   costs ~6 minutes.)
3. If the mechanism is the read-ahead prefetcher or the LRU thrash, what config/serving-flag change
   would you try first? (`ple_ssd_prefetch_tokens=0` is available, as are `ple_ssd_cache_mb`,
   `ple_ssd_workers`, `ple_ssd_io_depth`, `VLLM_HUMMING_*`, and the patch source is at
   /opt/vllm/src/vllm/models/qwen4_exp/nvidia/ple_ssd.py which I can edit before a restart.)

## Files you can read
- /tmp/plesrc/ple_ssd.py, /tmp/plesrc/ple_ssd_io.c, /tmp/plesrc/ple_layer.py,
  /tmp/plesrc/ngram_embedding.py, /tmp/plesrc/model_state.py  (the extracted PLE source)
- /tmp/ioh.py (the device harness), /tmp/micro.py (the Python-cost microbenchmark),
  /tmp/mon.log (1 Hz nvidia-smi + iostat during clean/long/poisoned phases), /tmp/seq.log
  (the request timings), /home/hong/vllm/OPS.md (my running notes)
