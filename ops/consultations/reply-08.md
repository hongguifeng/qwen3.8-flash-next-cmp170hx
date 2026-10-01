## Recommendation

**Treat the engine’s CUDA context as the failure domain, and process replacement as the only *demonstrated* recovery. Spend the next tests on prevention—not speculative “flush/reset” APIs.**

My order would be:

1. **Explicitly reduce the KV pool to 7.5 GiB.**
2. **Try native allocator + expandable segments.**
3. **Try FP8 QSA KV storage.**
4. **Disable CUDA graphs.**
5. **Try `cudaMallocAsync`, if no earlier candidate succeeds.**
6. **Use the remaining restart to reproduce the best candidate**, rather than testing another weak hypothesis.

If request segmentation is acceptable for your application, substitute the **two sequential 48K requests experiment** for test 5.

I performed only read-only inspection of configuration, source, package metadata, and logs. **No container changes, restarts, or GPU benchmarks were run.**

---

## 1. What the evidence establishes—and what remains an inference

Your battery strongly establishes a **persistent, engine-context-associated scattered-access pathology**, not a globally degraded GPU or simply bad allocations.

One qualification matters for choosing remedies:

> A translation/replay-resource failure and an accumulated-working-set trigger are plausible engineering models, but neither the specific resource nor “lifetime accumulated bytes” is established.

A 98,304-token request still reaches a **98,304-token key/history extent**, despite being processed as 48 query chunks. It can exercise different indexer workspaces, access patterns, or execution paths from two independent 49,152-token requests.

Thus:

- Chunking disproves “a single 96K-query burst is required.”
- It **does not distinguish** cumulative mapping/touch history from maximum live context length or a long-context-specific execution path.

I would not spend your budget trying to name the internal driver resource. But I would preserve these distinctions when interpreting a successful workaround.

### Important findings from this installation

The inspected build is `vllm 0.29.1rc1.dev402+ga5a30471f.ple1`, with PyTorch `2.13.0+cu130`.

Startup logs report:

| Item | Reported value |
|---|---:|
| Model-loading GPU memory | 47.32 GiB |
| Allocated KV pool | 9.22 GiB |
| Reported KV token capacity | 332,439 |
| Actual CUDA-graph pool | 1.57 GiB |
| Estimated CUDA-graph memory | 2.26 GiB |
| Post-capture recommendation to stay within configured budget | **7.51 GiB KV** |
| Post-capture recommendation to use available GPU memory, with its safety allowance | **8.61 GiB KV** |

**These are vLLM accounting figures, not proof of WDDM eviction.** Nevertheless, a 9.22 GiB pool exceeding both later recommendations makes memory headroom your strongest first intervention.

Two other findings:

- Your custom QSA backend explicitly supports `fp8` / `fp8_e4m3` KV storage.
- The PLE 512 MiB row cache is an `OrderedDict` of CPU-side byte strings—not a 512 MiB GPU cache.

---

## 2. Ranked test plan

I interpret expected value as **chance of a usable remedy per test cost**, including compatibility and performance costs. There is no defensible dataset for assigning numerical success probabilities here; the ranking is an engineering judgment.

### Common screening test

For every candidate:

1. Start a fresh engine with that candidate only.
2. Establish its own clean 2048-token prefill baseline.
3. Run the exact **98,304-token uncached trigger**, with minimal output.
4. Immediately repeat the uncached 2048-token prefill.
5. If available, also run the already-validated **same-context gather sentinel**.

Use the stronger confirmation protocol in §6 only for survivors.

| Rank | Exact change | Why it could work | Cheapest falsifier |
|---|---|---|---|
| **1** | Add `--kv-cache-memory-bytes 8053063680`—**7.5 GiB**. Keep `--max-model-len 262144`. | Removes about **1.72 GiB**, or **18.7% of the KV allocation**, and restores meaningful residency/mapping headroom. | Pool is demonstrably smaller, the trigger completes, but the gather returns to ~40 ms or short-prefill latency again approximately doubles. |
| **2** | Set `PYTORCH_CUDA_ALLOC_CONF=backend:native,expandable_segments:True` before engine launch. | Changes allocation/VA topology, block reuse, and graph-pool fragmentation without changing model arithmetic. | Option is supported and effective, but the same trigger still poisons the context. |
| **3** | Add `--kv-cache-dtype fp8_e4m3`. | Roughly halves **main QSA K/V bytes touched for a given context**, potentially avoiding a touched-working-set threshold. Your backend explicitly implements this path. | FP8 is actually active, but the same-context gather still poisons. Faster prefill alone does **not** establish a fix. |
| **4** | In the existing compilation configuration, change only `"cudagraph_mode":"FULL_AND_PIECEWISE"` to `"cudagraph_mode":"NONE"`. | Removes captured-graph pools and graph-related lifetime/topology effects. | No graphs are captured, yet the same trigger produces the same persistent degradation relative to this configuration’s clean baseline. |
| **5** | Set `PYTORCH_CUDA_ALLOC_CONF=backend:cudaMallocAsync`, without expandable-segment settings. | Uses CUDA’s stream-ordered pool allocator, changing reuse, backing allocation, and mapping behavior. | It starts successfully with that backend but still poisons. Startup incompatibility makes it unusable, not a falsification of the mapping hypothesis. |

**Budget:** four screens are approximately **28 minutes** under your estimate. Reserve two additional startup slots for replication/control. If all four fail, spend one on async allocation and stop.

### Why 7.5 GiB first?

It is a meaningful **one-flag** test that may preserve the advertised context length.

A rough capacity projection is:

\[
332{,}439 \times \frac{7.5}{9.22} \approx 270{,}400\text{ tokens}.
\]

That is only about **3% above 262,144**, so check actual hybrid-cache capacity at startup; padding/grouping can invalidate proportional estimates.

If it cannot initialize at 262,144, that is **not a poisoning result**. A more aggressive, explicitly two-setting fallback is:

```text
--kv-cache-memory-bytes 6442450944
--max-model-len 131072
```

That is **6 GiB KV**, freeing approximately **3.22 GiB** while still admitting the 98,304-token trigger. Treat it as a combined reduced-footprint configuration, not evidence identifying which flag mattered.

---

## 3. Evaluation of the allocation and execution knobs

### A. Smaller KV pool, lower maximum length, FP8, additional offload

These affect different quantities.

**Explicit KV bytes / lower `gpu-memory-utilization`:**

- Reduce allocated pool bytes.
- May reduce residency pressure and mapping metadata.
- **Do not necessarily reduce bytes touched by the same single 98K request** if it uses the same number of BF16 cache blocks.

The explicit byte limit is preferable here because it avoids another profiling estimate. In this build it overrides automatic KV sizing from `gpu-memory-utilization`.

At 64 GiB, reducing utilization by **0.01** changes the nominal budget by **0.64 GiB**. A large reduction can make the declared 262K maximum fail initialization.

**Lower `--max-model-len`:**

- Is an admission limit and may reduce length-dependent buffers/metadata.
- **Does not generally shrink the automatically budgeted KV pool proportionally.**
- A limit below 98K avoids this particular request, but that is an admission workaround—not evidence that poisoning is impossible after many shorter requests.

**FP8:**

- Reduces the main QSA cache footprint for the **same token history**.
- Does not automatically halve the total allocated KV pool: automatic sizing can spend the savings on more blocks.
- Does not halve GDN state, all indexer caches, padding, or graph memory.

For your main model’s 12 QSA layers, two KV heads, and head dimension 256, the unpadded main K/V payload at 98,304 tokens is:

\[
98304 \times 12 \times 2_{\mathrm{K,V}} \times 2_{\mathrm{heads}}
\times256\times2_{\mathrm{bytes}}
=2.25\ \mathrm{GiB}.
\]

FP8 reduces that component to **1.125 GiB**, before accounting for MTP, side caches, and padding. That makes it a useful **touched-bytes** experiment distinct from shrinking unused pool capacity.

However, FP8 also changes kernels and numerical behavior. Require an accuracy check and valid scaling; backend support alone is not an accuracy guarantee.

**More PLE offload:**

Your PLE table is already SSD-backed. Reducing `ple_ssd_cache_mb` from 512 to 128 saves roughly **384 MiB of CPU cache**, not VRAM.

Offloading other currently GPU-resident parameters could free VRAM, but introduces transfer and pinned-memory costs. It is lower priority than the explicit KV limit.

### B. `expandable_segments:True` versus `False`

No allocator configuration override was visible in the inspected container environment. Thus explicit `False` is likely a repeat of the native default, unless application code changes it.

With expandable segments, PyTorch reserves larger VA ranges and maps backing storage incrementally, allowing adjacent free blocks to merge within a segment. This can improve pool reuse and reduce fragmentation. [1]

**Plausible benefit:** fewer disconnected allocator segments, different allocation ordering and backing-map lifecycle.

**Not guaranteed:**

- Larger GPU hardware pages.
- Fewer GPU PTEs.
- Fewer translation misses.
- Physically contiguous storage.
- A single arena covering every library allocation.

It could also make the problematic topology worse. Treat `True` as one worthwhile experiment, not a known WSL fix.

### C. `backend:cudaMallocAsync`

This is a different allocator implementation, not merely another native-allocator tuning knob.

It can change pooling, reuse, synchronization, and physical backing decisions. But **“VMM-based” does not imply a guaranteed larger translation page size**, nor does it bypass GPU-PV.

Check that the active engine backend really changed; do not rely solely on an environment variable in a launcher. Also expect some native-allocator statistics or tuning options to be inapplicable.

Do not combine async allocation, expandable segments, and graph changes in the first test.

### D. Preallocating or pre-touching everything

**Low priority.**

vLLM already allocates its KV storage at initialization. Assigning more KV blocks to a request does not normally mean creating a new CUDA allocation for each block.

Preallocation helps only if the trigger requires **additional backing allocations, registration churn, or an unlucky late allocation order**.

Pre-touching helps only if benign first-touch/residency establishment avoids a later problematic transition. It may instead:

- Poison the context at startup.
- Merely move the threshold earlier.
- Do nothing because dense touching does not exercise the failing scattered-access path.

A profiling pass also does not prove that the full long-history execution path or every cache page was exercised.

If you eventually test it, touch the **actual retained production allocations**, not disposable dummy buffers. If the sentinel is slow before the first request, pre-touching has simply moved the failure.

### E. Prefix caching, MTP, graph capture sizes

**Prefix caching off:** replace `--enable-prefix-caching` with `--no-enable-prefix-caching`.

This can reduce retained logical cache content and alter GDN checkpoint/cache reuse. It does **not** normally unmap the preallocated pool. Because the first long request triggers the problem, cross-request prefix retention is not the leading explanation.

**MTP off:** remove the `--speculative-config` argument.

This can remove draft-model work, state, and graph allocations. But automatic KV sizing may reclaim the freed space, leaving total GPU allocation similar. Lower priority unless the long-prefill path specifically enters problematic draft-model code.

**Fewer capture sizes:** potentially reduces graph-pool footprint or topology complexity, but graph pools may share memory. Memory savings are not proportional to the number of deleted sizes.

**No graphs:** the cleanest first graph experiment. Prefer mode `NONE` over `--enforce-eager` for isolation: `--enforce-eager` also changes compilation behavior more broadly.

For all three, record actual KV and graph-pool sizes. **Automatic KV expansion can hide the memory savings you intended to test.**

### F. Two independent 48K requests

This is useful, but it is not equivalent to chunked prefill.

Test:

1. Fresh context.
2. An uncached **49,152-token** request; wait for completion.
3. Check the sentinel.
4. A second independent, uncached **49,152-token** request.
5. Check again.

Interpretation:

- **Both healthy:** supports a peak-live-history/access-pattern explanation over a simple total-token counter.
- **Second poisons:** compatible with cumulative state, retained cache content, or touching additional pool blocks.
- Neither outcome alone proves a mapping-resource model.

With prefix caching enabled, the requests may retain different blocks, so a positive result is less clean than the “both healthy” result.

Crucially, splitting the application input into two independent requests **changes model semantics**. Sending the previous 48K again as history, or retaining its KV state, restores the long context and may restore the trigger.

### G. Pin-memory switch

I would not spend one of the first four tests here.

NVIDIA documents WSL pinned-memory limitations, so host mappings remain relevant. [2] But your PLE implementation explicitly creates staging tensors with `pin_memory=True` in:

```text
/opt/vllm/src/vllm/models/qwen4_exp/nvidia/ple_ssd.py
```

Therefore:

```text
VLLM_WSL2_ENABLE_PIN_MEMORY=0
```

is **not an all-pinning-off experiment**. A negative result would not rule out PLE’s pinned mappings.

---

## 4. Recovery, isolation, and Windows-side options

### Is context destruction the only reset?

**It is the only reset demonstrated by your measurements. I would build operations around that fact.**

There is no documented CUDA API that means “reset this context’s WSL translation/replay machinery while preserving all allocations and graphs.”

I would not allocate test slots to:

- `torch.cuda.empty_cache()`
- Synchronization or idle waits
- CUDA memory-pool trimming
- L2-persistence reset APIs
- vLLM sleep/wake
- Destroying selected graphs and hoping the context recovers

These can change allocator/cache state but do not promise to clear the observed condition.

`cudaDeviceReset()` is destructive to context resources and is **not a safe in-place recovery procedure for a live PyTorch/vLLM engine**. Replacing the CUDA-owning worker process is safer. The entire container need not be the conceptual reset unit, although restarting the service/container may be the simplest reliable implementation.

### Separate long-context instance

**High-confidence containment principle; significant capacity cost on this GPU.**

Your loaded model consumes about **47.32 GiB**. Two independently loaded copies require approximately **94.64 GiB before KV and activations**. Two normal hot instances therefore do not fit on 64 GiB.

Practical choices:

1. Route long contexts to another GPU/host.
2. Time-multiplex a disposable long-context worker on this GPU.
3. Batch long-context jobs, then replace the worker before returning to short-prefill service.

Do not assume vLLM sleep/wake destroys the context. Shared-weight/IPC multi-process designs are possible engineering projects, not a cheap first remedy.

### Windows / driver interventions

| Option | Assessment |
|---|---|
| **Another compatible NVIDIA Windows driver branch/version** | Legitimate and potentially decisive. Best host-side experiment because the suspected implementation lives below vLLM. No evidence here identifies a specifically good replacement for 616.92. Keep CUDA 13.0 compatibility and supported device requirements in view. |
| **Windows build / WSL update** | Legitimate, but changes a broad stack and costs a host-level maintenance window. Test independently from the driver if attribution matters. |
| **HAGS toggle** | Legitimate if exposed for this adapter. Lower priority: changes scheduling implementation, not an advertised translation-resource limit. Your no-gap CTA trace and healthy decode weaken a generic scheduling explanation. |
| **WDDM residency policy** | No supported general user switch guarantees that this WSL CUDA context’s allocations remain resident or enlarges its translation resources. Native graphics residency APIs are not a drop-in policy control for vLLM in WSL. |
| **GPU page size** | No supported general WSL user setting forces the desired CUDA GPU page size. CPU huge pages and Windows page-file sizing are not substitutes. |
| **TCC** | Not a drop-in WSL remedy. NVIDIA’s WSL support documentation specifies WDDM, not TCC. [2] |
| **Prefer maximum performance** | Very low expected value with P0, unchanged streaming bandwidth, and unaffected decode. It does not reset mapping state. |
| **“Disable GPU scheduling”** | Disabling HAGS selects the other scheduling path; it does not remove WDDM scheduling. |
| **TDR registry changes** | Not a remedy for this case: no TDR is occurring. |
| **Native Linux** | The strongest architectural way to remove GPU-PV from the path, but a migration project—not a seven-minute tuning test. Still requires validation. |

For a driver/HAGS/Windows candidate, the falsifier is the same: after a genuinely fresh engine, the identical trigger still produces the same-context scattered-access slowdown.

---

## 5. Cheap, reliable detection from outside

### Without engine instrumentation

**There is no demonstrated reliable passive external detector.**

Your fresh-process gather is specifically unsuitable: it remains healthy while the engine is poisoned. Likewise, NVML, GPU utilization, decode throughput, and ordinary health endpoints need not reveal the condition.

Existing metrics can provide suspicion:

- `vllm:request_prefill_time_seconds`
- `vllm:request_queue_time_seconds`
- `vllm:time_to_first_token_seconds`

I verified these metric names in this build.

But aggregate prefill histograms mix prompt lengths, cache hits, concurrency, and I/O conditions. A raw average or p95 is not a poison detector.

For an isolated known request, histogram `_sum`/`_count` deltas can be useful. With mixed traffic, use per-request timing and workload information.

**Without a hook, your uncached 2048-token, one-output-token request remains the best validated external active check.** A 256/512-token canary might be cheaper, but do not assume it exercises the same failing shape.

### Recommended instrumentation for a future build

Expose a metric from a probe **executed inside the actual engine context**, requested or polled externally.

Reuse the exact validated failing gather:

- 64 MiB source
- 4096 rows × 2 KiB output = 8 MiB
- Fixed preallocated indices/output/events
- Approximately **72 MiB** persistent device footprint
- Execution at a safe scheduler boundary, outside graph capture
- CUDA-event timing, not HTTP latency

Suggested initial alarm:

```text
gather > 5 ms AND > 10 × its clean baseline,
confirmed by a second measurement
```

Your known separation is enormous:

- Clean: approximately **0.175 ms**
- Poisoned: approximately **39.68 ms**

Three measurements cost approximately:

- **0.525 ms clean**
- **119 ms poisoned**

Once per minute, even the poisoned measurement consumes only about **0.2% of GPU time**, excluding orchestration overhead.

Expose both latency and sample timestamp; a stale metric must not be interpreted as healthy. Calibrate the sentinel before/after poisoning and keep its allocation present in both control and candidate runs.

**Operationally:** stop admitting new short-prefill work after confirmation, drain appropriately, replace the CUDA-owning process, and require a clean readiness measurement. For known long-context jobs, proactively schedule replacement afterward rather than waiting for user-visible degradation.

---

## 6. Stronger and more economical confirmation

Your proposed test is good, but **“≤1.0 s after 20 minutes” needs two refinements**.

### Separate absence of poisoning from absolute performance

Graph disabling, FP8, or another allocator may change clean performance.

Use two independent criteria:

1. **Poisoning criterion:** no persistent jump relative to that candidate’s own pre-trigger baseline.
2. **Product criterion:** the absolute short-prefill latency is acceptable—e.g. ≤1.0 s.

For screening, a reasonable provisional gate is:

- Median of three post-trigger short-prefill measurements **≤1.2×** the candidate’s clean median.
- Same-context gather remains near baseline, provisionally **<2×**, with no multi-millisecond samples.

The gather is important: a candidate can accelerate computation enough to partially hide a still-poisoned context.

### Avoid accidental cache-hit “success”

- Verify **98,304 actual input tokens**.
- Ensure the long trigger is not prefix-cached.
- Make the 2048-token canary uncached from its first cache block.
- Standardize PLE warmness as much as practical.
- Use minimal output and an idle queue for measurements.

A changed token near the end does not prevent reuse of the earlier prefix.

### Stage the validation

**Stage 1 — fail fast**

Run one canonical trigger and immediate checks. Reject failures without a 20-minute soak.

**Stage 2 — challenge the survivor**

Within the surviving context:

- Run at least two more **uncached 98K** requests.
- Exercise independent shorter prompts that cause cache reuse/churn.
- Check the sentinel after each long request.

Twenty minutes of mostly decode is a weak test because decode is unaffected.

**Stage 3 — reproduce across fresh contexts**

Prioritize at least **two independent candidate startups** over a long single-process soak. Allocator address placement and initialization order may themselves affect the result.

An economical sequence is:

```text
A: original configuration → failure
B: candidate → immediate pass + repeated triggers
B: candidate, new process → repeat pass
```

Your existing reproducible A measurements may supply the control. For a surprising driver/topology result, an **A–B–A–B** sequence is stronger.

If retaining 262K support matters, a 98K pass establishes only that **the old trigger no longer works**. Later test longer contexts; the threshold may merely have moved.

---

## Bottom line

**Start with the 7.5 GiB explicit KV limit.** Your own startup accounting gives it a concrete rationale beyond speculation.

Then test **expandable segments, FP8 QSA KV, and no CUDA graphs**, reserving startup budget for replication. Do not spend early trials on PLE row-cache size, maximum-performance mode, page-size folklore, or cache-flush APIs.

If these fail, **controlled process replacement is the correct engineering answer**, not a workaround to be embarrassed about. Pair it with long-request isolation/admission policy and a same-context gather metric. That directly uses your strongest findings: **the failure is measurable, context-associated, and reliably cleared by process/context destruction.**

### Sources and inspected implementation

1. [PyTorch: CUDA caching allocator, expandable segments, and graph-pool fragmentation](https://docs.pytorch.org/devlogs/eager/2026-06-01-cuda-caching-allocator/).
2. [NVIDIA CUDA on WSL User Guide: WDDM support and pinned-memory limitations](https://docs.nvidia.com/cuda/archive/13.1.2/wsl-user-guide/index.html).
3. Local inspected sources under `/opt/vllm/src/vllm/`: `config/cache.py`, `v1/worker/gpu_worker.py`, `models/qwen4_exp/nvidia/{qsa.py,indexer_qsa.py,ple_ssd.py}`, and `v1/metrics/loggers.py`. These support the build-specific flags, memory accounting, FP8 support, PLE allocation details, and metric names above.
