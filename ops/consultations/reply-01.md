The measurements point to a **depth-one latency amplifier**. The SSD is involved, but the persistent state is probably created above the raw AIO layer.

The strongest clue is the poisoned-state stack:

- `ple-ssd-prefetch_0` spends almost all its time in `_ids_ready.synchronize()`.
- The engine thread waits in `_pending.result()`.
- The GPU reports high utilization at only 86–152 W.
- The standalone native reader still reaches about 140k rows/s.
- The SSD is at only about 39% utilization, with roughly 1.1–1.6 ms request latency.

That means the poisoned path is usually waiting for the GPU stream to reach the ID handoff, while the single worker and single outstanding lookup expose every delay to the engine step.

For a 2048-token chunk, the demand lookup is 32,768 rows:

- Raw reader time at the measured 140k rows/s: about 234 ms.
- Python bookkeeping: about 36 ms.
- PCIe transfer of 10.5 MiB at 3.2 GB/s: about 3–4 ms.
- Observed poisoned penalty: about 0.85 ms/token, or about 1.74 seconds per chunk.

The extra 1.7 seconds is therefore much larger than the measured SSD read, Python, or PCIe costs. The PLE pipeline turns any delay in the CUDA stream, request scheduling, or persistent GPU memory state into a full CPU-visible stall.

The relevant synchronization chain is in [ple_ssd.py](/tmp/plesrc/ple_ssd.py:430):

```text
main stream produces ngram IDs
    -> synchronous device_ids.copy_
    -> side stream copies IDs to pinned host memory
    -> ids_ready event
    -> worker waits in Event.synchronize()
    -> native SSD read
    -> side-stream H2D copy
    -> main thread waits on Future.result()
```

Because there is only one `_pending` future, the next chunk cannot be prepared while the current chunk is being finalized. This makes the path extremely sensitive to latency variance.

**Likely causes ranked**

| Mechanism | Assessment |
|---|---|
| Long-lived CUDA/vLLM state after a long prefill | Most plausible root cause. Candidates include KV/prefix-cache residency, allocator or graph-state changes, and a changed execution schedule. |
| PLE depth-one synchronization chain | Definite amplifier. It explains why the main thread and one CPU core stall so visibly. |
| Prompt read-ahead | Plausible trigger or amplifier. It adds 262,144 random rows per request and competes for the same lock and LRU. |
| LRU pollution | Possible secondary effect. The current evidence does not support cache capacity as the main cause. |
| WSL2 disk burst throttling | Possible transient contributor, unlikely to explain the persistent state by itself. |
| AIO context degradation | Low probability. The C code drains every batch, and the context has no obvious accumulating state. |
| Growing `_ids` or host buffers | Essentially ruled out by the source. The buffers are allocated once at fixed shape. |
| `ple_ssd_workers` | Irrelevant while the native reader is enabled. Native mode uses one `rows_read()` call, regardless of the worker count. |

The read-ahead implementation in [ple_ssd.py](/tmp/plesrc/ple_ssd.py:300) has two real costs:

1. It reads 262,144 random rows for the first 16,384 prompt tokens.
2. It inserts those rows into the same demand LRU.

The demand-priority condition helps once a demand reader is already waiting, but it does not prevent a read-ahead batch from acquiring the lock during gaps between demand requests. The prefetcher can therefore add I/O and cache churn.

Its scale is still limited. For a 186k-token request, demand needs roughly 3 million rows. The read-ahead window adds only 262k rows, about 9% more row traffic. At the measured raw rate, those extra rows cost roughly 1.9 seconds. They can trigger an unfavorable WSL/VHDX burst or pollute the cache, but they cannot directly explain a 563-second request or a persistent 4–5x slowdown after the request finishes.

The LRU capacity result is also informative. With 1.2 million rows over a 320-million-row uniform key space, the expected random hit rate is only about:

```text
1.2M / 320M = 0.375%
```

Increasing the cache to 4 GiB should materially help if capacity misses were the cause. It did not. The cache can still suffer from churn or internal hash-table growth after millions of insert/delete operations, so a zero-cache test remains useful, but simple cache capacity is not the explanation.

The WSL2 disk measurements look more like a workload symptom than a saturated device. At 52k 4 KiB reads/s and 39% utilization, the disk still has substantial headroom relative to the 140k-row/s test. The elevated `r_await` can result from the production access pattern, VHDX scheduling, or a burst of small read-ahead batches. The fact that a fresh native reader reaches full speed after poisoning strongly argues against a permanently throttled underlying disk.

One additional hypothesis deserves an explicit check: `--enable-prefix-caching` may leave a large long-prompt KV prefix resident after the request. That state would survive subsequent requests and disappear on container restart. It could change available KV blocks, allocator behavior, or execution scheduling. Fresh prompts avoid prefix-cache hits, but they do not necessarily avoid the memory occupancy caused by cached prefixes.

**The single best A/B experiment**

Disable only prompt read-ahead:

```bash
QWEN_SSD_PREFETCH=0 bash /home/hong/vllm/run_container.sh --now
```

Keep these unchanged:

```text
QWEN_BATCH_TOKENS=2048
QWEN_SSD_CACHE_MB=512
QWEN_SSD_DEPTH=256
QWEN_SSD_WORKERS=16
VLLM_WSL2_ENABLE_PIN_MEMORY=1
```

`QWEN_SSD_PREFETCH=0` prevents `PLEPromptPrefetcher` from being constructed in [model_state.py](/tmp/plesrc/model_state.py:29). It therefore removes the read-ahead thread, its shared-lock traffic, and its LRU pollution while leaving the demand path unchanged.

After the container is ready, run this sequence:

```bash
python3 /home/hong/vllm/prefill_step.py prefetch0_clean 8192 32768

python3 /home/hong/vllm/prefill_step.py prefetch0_long 100000

python3 /home/hong/vllm/prefill_step.py prefetch0_after 8192 32768
```

The 100k-token request should reproduce the state transition in about a minute or two. The important comparison is:

```text
prefetch0_clean 8K / 32K
prefetch0_after 8K / 32K
```

Interpret the result this way:

- **Short prompts remain fast after the long request:** read-ahead is a causal trigger or major amplifier. Keep it disabled initially. A smaller value such as 2048 or 4096 could then be tested.
- **The long request improves, but short prompts still become slow:** read-ahead contributes extra work, but another persistent state remains.
- **The same collapse occurs with nearly identical timings:** read-ahead is not the root cause. Focus on CUDA/vLLM state, prefix/KV residency, or demand-path timing.
- **The collapse remains and the demand read phase itself becomes slow:** test cache churn and same-process AIO reuse.
- **The collapse remains while demand reads stay near the standalone rate and `ids_ready` wait grows:** the cause is above the SSD reader, most likely GPU execution state or scheduling.

Run these monitors during the three requests:

```bash
iostat -dxm 1 | tee /tmp/iostat-prefetch0.log
```

```bash
nvidia-smi \
  --query-gpu=timestamp,utilization.gpu,power.draw,clocks.sm,clocks.mem,memory.used,pstate,pcie.link.gen.current,pcie.link.width.current \
  --format=csv -l 1 | tee /tmp/nvidia-prefetch0.log
```

```bash
curl -s http://127.0.0.1:9393/metrics |
  rg 'kv_cache_usage_perc|prefix_cache_(queries|hits)_total|prompt_tokens_by_source'
```

Capture the metrics immediately before the long request, immediately after it, and after the short poisoned request. In particular, check whether `kv_cache_usage_perc` remains elevated after the long request. If it does, run a separate A/B with prefix caching disabled. That is the cleanest test for a persistent GPU-side cache state.

**The most valuable phase timing**

The next code patch should measure four durations independently in `_read_and_copy()` and `start_prefetch()`:

```python
# _read_and_copy
t0 = time.perf_counter()
self._ids_ready.synchronize()
t1 = time.perf_counter()

self._table.read(...)
t2 = time.perf_counter()

self._copy_to_gpu(tokens)
t3 = time.perf_counter()

logger.info(
    "PLE chunk tokens=%d ids_wait_ms=%.3f table_ms=%.3f copy_enqueue_ms=%.3f",
    tokens,
    (t1 - t0) * 1000,
    (t2 - t1) * 1000,
    (t3 - t2) * 1000,
)
```

Also time this call in `start_prefetch()`:

```python
t0 = time.perf_counter()
self._copy_ready.synchronize()
t1 = time.perf_counter()
```

Log aggregate means and p95 values every 32 or 64 chunks instead of logging every chunk.

The key signatures are:

| Measurement | Meaning |
|---|---|
| `ids_wait_ms` rises from roughly normal to seconds | GPU stream or execution scheduling is the primary delay. |
| `table_ms` rises while `ids_wait_ms` stays stable | Demand I/O, lock contention, LRU churn, or VHDX behavior is involved. |
| `copy_ready` wait rises | The previous PLE H2D or its consumer is extending the serialization chain. |
| All phase times stay normal but request latency rises | Look at vLLM scheduler, KV cache, graph mode, or another CPU-side engine stall. |

The `rows_read()` loop in [ple_ssd_io.c](/tmp/plesrc/ple_ssd_io.c:54) drains each batch completely before submitting the next. That is inefficient for latency hiding, but the fixed-context reuse experiment can test whether it accumulates state:

- Run the native reader in the poisoned container.
- Create one `Reader(depth=256)`.
- Execute 8–16 identical 32,768-row calls on that same object.
- Record throughput for every call.
- Repeat with a new `Reader` per call.

If the same context stays near 100k–140k rows/s, AIO-context degradation is eliminated. If only reused contexts decay, the native context or its interaction with WSL2 becomes a serious suspect.

For that experiment, modify `/tmp/ioh.py` only in a temporary copy so that one `Reader` is reused, then run it inside the container. The important counters are:

```text
rows/s per call
/proc/diskstats sectors read
/proc/diskstats read milliseconds
/proc/diskstats I/O milliseconds
iostat r/s, r_await, aqu-sz, %util
```

A persistent raw disk problem should reproduce in the same-process reader and should show rising device latency or utilization. A production pipeline problem will leave the standalone reader fast.

**Configuration change I would keep first**

Use:

```text
ple_ssd_prefetch_tokens = 0
```

through:

```bash
QWEN_SSD_PREFETCH=0
```

The current evidence gives little reason to increase `ple_ssd_cache_mb`; 4096 MiB already failed to restore performance. I would test `ple_ssd_cache_mb=0` only after the prefetch A/B, because it distinguishes cache churn from cache capacity:

```bash
QWEN_SSD_PREFETCH=0 \
QWEN_SSD_CACHE_MB=0 \
bash /home/hong/vllm/run_container.sh --now
```

The native path also makes `ple_ssd_workers` ineffective, and depth 256 is already close to the measured device optimum. Those settings are lower-priority experiments.

If disabling read-ahead fixes the state transition, the durable patch should give read-ahead its own cache or reader and keep it from inserting random rows into the demand LRU. The larger structural fix is to remove the depth-one join: retain multiple host/device staging slots, allow the next ID batch to be copied while the current result is consumed, and make the demand reader independent of the prompt read-ahead lock.
