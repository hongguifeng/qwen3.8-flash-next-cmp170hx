## Bottom line

**Choose Plan A, but change its implementation: ordinary forward hooks do not execute on every breakable-graph replay.** I inspected the running container’s source. There are also two important measurement corrections:

- The mounted `ple_ssd.py`’s `_PT` timers use **`time.perf_counter()`**, not CUDA events.
- `iteration_tokens_total` is **not reliable evidence of actual chunk size** in this build: its prefill accounting comes from request-level prefill statistics attached to engine outputs.

I made no changes and submitted no inference requests.

## 1. Ranked mechanisms

| Rank | Candidate | Judgment |
|---|---|---|
| **1** | **C1: Humming synchronization/persistent-kernel pathology** | Best fit for low-power GPU occupancy, context starvation and prefill-specific slowdown. There is concrete code supporting investigation—not proof. |
| **2** | **C3: graph-owned state/workspace lifetime or aliasing** | Stronger than generic allocator fragmentation. Could be the trigger for C1 or corrupted metadata rather than a competing explanation. |
| **3** | **C6/C2: GDN or attention metadata/state** | Prefill-only behavior fits. The specific “chunk indices retain longest-ever length” theory is weakened by the implementation. |
| **4** | **C4: context-specific driver scheduling/residency** | Separate-process GEMM excludes device-wide degradation, not an engine-context problem. Keep this if module timings show broadly distributed inflation. |
| **5** | **C5: additional scheduler work** | Cheap to check, but hard to reconcile with the fixed token budget and single-request reproductions. |

**Concrete C1 evidence:** `HummingExpertsBase` allocates a persistent `int32[1024]` lock tensor and passes it to GEMMs. Humming’s `utils/ptx/barrier.cuh` contains actual global-memory polling loops. `epilogue/pipeline.cuh` invokes them when `slice_count > 1`. Thus **`num_write_splits=1` does not exclude Stream-K inter-CTA synchronization**.

However, an irrecoverably wrong lock commonly causes a hang, not repeatable finite 2-second completion. Investigate delayed progress, interference, aliasing and scheduling—not simply “a semaphore stayed nonzero.”

**Concrete C3 evidence:** `breakable_cudagraph.py` shares graph pools and deliberately weak-references captured arguments. This makes workspace lifetime/reuse worth examining; it does not demonstrate a bug. A static replay ordinarily does not become 7× slower merely because the caching allocator holds different free blocks.

**Concrete C6 evidence:** `v1/attention/backends/gdn_attn.py` rebuilds `chunk_indices` and `chunk_offsets` from the **current prefill query-start locations**. A longest-ever high-water mark is not apparent there. Stale captured pointers or wrong current metadata remain possible.

Finally, NVIDIA “100% utilization” means kernels were active throughout the sampling interval—not that all SMs were productively occupied. The GEMM starvation adds useful evidence, but does not identify a spin kernel.

## 2. ONE recommended restart: graph-preserving event attribution

Keep the existing configuration and reproduce poisoning. **Do not combine eager mode with this first attribution run.**

For the currently running, prefetch-disabled baseline, retain:

```bash
QWEN_EAGER=0
QWEN_GRAPH_MODE=FULL_AND_PIECEWISE
QWEN_MTP=1
QWEN_SEQS=4
QWEN_BATCH_TOKENS=2048
QWEN_SSD_PREFETCH=0
VLLM_USE_BREAKABLE_CUDAGRAPH=1
```

Keep capture sizes, Humming selection, prefix caching and `mamba-cache-mode=align` unchanged. Add only your host-mounted instrumentation. Using prefetch=16384 is reasonable operationally, but would change this baseline.

### The crucial hook correction

The actual implementation is:

```python
def replay(self):
    for r in self.segments:
        r()
```

Those callables are **captured graph replays and saved eager-break functions**, not a new execution of the entire model’s Python forward. A stack inside `CUDAGraph.replay()` confirms graph execution, not recurrent module-hook execution.

Use two complementary levels:

1. **Time existing graph segments and eager callables** by wrapping `BreakableCUDAGraphCapture.replay`’s segment invocations with current-stream event pairs and host timestamps. Do not introduce new breaks.
2. **Embed module-boundary timing events during capture**, using both pre- and post-hooks and:
   ```python
   torch.cuda.Event(enable_timing=True, external=True)
   ```
   Keep these events alive with the captured entry. Their record nodes execute on replay even though the hooks do not. Validate that elapsed times update across two replays before the long test.

Instrument the production prefill descriptor(s), initially around **2048 padded tokens**, rather than every decode capture size.

### Attribution and collection

- Key records by **graph entry, module instance/layer index, invocation index**, not class alone.
- Include decoder layers, MoE, `QwenGatedDeltaNetAttention`, the actual full-attention class, PLE, and target-versus-MTP identity.
- Explicitly wrap `HummingExpertsBase.humming_forward` to separate **w13/w2** if feasible; it is not an `nn.Module` hook point.
- Decoder `self.mlp(...)` and attention calls do use module dispatch. Hyperconnection `.combine_and_mix(...)` and direct `.forward(...)` calls bypass normal hooks.
- Retain per-layer measurements; report **class sum plus worst layers**. Do not sum inclusive decoder and child times together.
- Preallocate/initialize events outside capture. Collect only after completion, using `query()`, **before the same event records are overwritten by another replay**. Drop an unavailable sample rather than adding synchronization.
- No `.item()`, tensor printing, per-module synchronization or new stream waits.
- Time PLE’s copy stream separately. A compute-stream interval spanning its host break includes SSD-induced idle time; it is not pure kernel time.
- Log CPU-side actual scheduled tokens, padded descriptor, request count, prefill/decode counts, and graph-entry identity.

### Interpretation

| Result after the long prompt | Next target |
|---|---|
| MoE w13/w2 supplies most of the extra ~1.6–1.8 s | Humming Stream-K/locks/workspaces; then backend bisect |
| GDN supplies the increase | Chunk metadata, state indices, captured pointer lifetime |
| Full attention alone grows | Context lengths/block tables/backend dispatch |
| Kernels stable; eager gaps/stream dependencies grow | PLE coordination or driver waits, not SSD throughput |
| Different descriptor/work amount appears | Dispatch/scheduler explanation |
| Poisoning disappears with instrumentation | Instrumentation perturbed timing/lifetime; no exoneration |

An eager bisect is useful later, but **“eager fixes it” does not uniquely implicate graph pools**, and **“eager retains it” does not prove graphs innocent**. Eager changes kernel dispatch, allocation lifetimes and overlap; the clean/poisoned ratio need not be preserved.

## 3. Cheap checks now

### A. Establish genuinely isolated execution

```bash
curl -s localhost:9393/metrics | grep -E \
'^vllm:(num_requests_running|num_requests_waiting|kv_cache_usage_perc|num_preemptions_total|engine_sleep_state)'
```

At my snapshot: **running=1, waiting=0, KV usage=24.3%**. That is not an idle baseline. Wait for existing work to finish; do not cancel it.

`QWEN_SEQS=1` is not decisive: it also changes capture/batching configuration. Demonstrating one running request with no other work is the better first check.

### B. Do not infer chunk size from this histogram

Although its HELP says “per engine_step,” `loggers.py:1137` observes:

```python
iteration_stats.prompt_token_stats.computed \
    + iteration_stats.num_generation_tokens
```

Those prompt statistics describe prefill accounting associated with outputs. **Buckets above 2048 do not prove oversized GPU steps.**

Use existing PLE `start_tok/start_n` totals over complete isolated requests for padded PLE-call size. For actual scheduled tokens, instrument `scheduler_output.num_scheduled_tokens` at the runner boundary on the planned restart.

### C. Small API probes, only after isolation

Use `/v1/completions`, fresh token-ID arrays, `temperature=0`, `ignore_eos=true`:

- **Fresh 512 versus 2048 tokens**, `max_tokens=1`: does the slowdown track capture/work size?
- **Two or three fresh 2048-token requests sequentially**: are first chunks already slow? This tests persistence without another long prompt.
- **Fresh 8192 tokens, max_tokens=1 versus 64**: measure streaming **TTFT separately from decode intervals**, not total latency.
- Two concurrent 2048-token prefills are optional; they mainly test scheduling/throughput and are less diagnostic than isolation.

Avoid repeated identical prompts unless intentionally testing prefix caching. Sixteen repetitions are unnecessary initially.

## 4. Is PLE exonerated?

**The SSD service-time increase cannot explain the slowdown.** Chunk wall time grows by **1759 ms**, while native I/O grows only **18 ms**; `W_ids_sync` grows **1569 ms**, about **89%** of the wall-time increase.

But the conclusion should be narrower:

- `_ids_ready` is actually recorded on the **PLE stream after D2H**, following waits on the compute stream and `_previous_use`. Its host wait is not exclusively arithmetic kernel time.
- It can expose **previous-chunk tail work**, not merely current work before `layers.1`.
- Current `_PT` statistics use rolling two-second host buckets; start/completion counts can cross bucket boundaries.
- **15.3 ms/token = 65.4 tok/s**, not 110–120 tok/s. Decode is less affected, but the quoted numbers alone do not establish unchanged performance.

No NVTX, allocator history or `cudaProfilerStart/Stop` substitute provides kernel timing without a functioning tracing backend. CUDA graph debug dumps can identify topology/kernel names, not durations. The nsys importer error is not itself proof capture failed, but **CUDA events remain the most direct next instrument here**.
