## First: three important corrections from the running container’s source

I inspected files and read-only endpoints; I did **not** change the service or submit workloads.

1. **The quoted “stable scan width” is a request scan, not a context scan.** In `common/qsa_cache.py`, `REQUEST_SCAN_SIZE = next_power_of_2(request_capacity)`, with capacity taken from `max_num_seqs`: **4 here**. The compressed-work buffer capacity is **515 rows**, but its live slice is `(num_tokens + 3*num_requests)//4`: **one work item for a two-token, one-request decode**. This particular code does not support H1.

2. **Those three rejection/resampling kernels belong to speculative token sampling**, in `v1/worker/gpu/spec_decode/rejection_sampler_utils.py`, not QSA selection. QSA calls `torch.ops._C.persistent_topk` on sm_80, selecting **512 compressed positions**, then expanding them into up to **2051 token positions**, plus a count column.

3. **The current fused pre-indexer uses reductions and stores, not atomic accumulation.** Thus the description and inspected source differ. Record source hashes and loaded-library identity before interpreting further measurements.

Also, proposed overrides are not currently valid diagnostics: `persistent_topk` accepts only **k=512,1024,2048**. Budget 256/compression 4 gives k=64; budget 2048/compression 8 gives k=256. Both would fail without changing the implementation.

## Ranked judgement

| Rank | Hypothesis | Assessment and discriminator |
|---|---|---|
| **1** | **H6: stale/corrupted device metadata or scratch ownership/lifetime** | Best fit for cross-request persistence, fixed short-decode cost and shared target/draft involvement. Check visible lengths, padded rows, block tables, scratch aliasing, and captured buffer ownership—not just graph replay machinery. |
| **2** | **H1, narrowed to wrong live bounds or captured capacity** | Decode logits width comes from `page_table.shape[1] * page_size`; prefill width comes from current `max_seq_len`. Decode launch capacity can be large from startup. A capacity that was already large cannot explain the flip without a changed pointer, live bound, branch, or buffer contents. |
| **3** | **H2: locality/residency** | Plausible for gathers, but “fragmentation causes 10× latency” is not established. KV blocks are logical allocations within preallocated tensors, not independent CUDA allocations. Include WSL/driver memory residency and graph-pool pressure, not just scattered block IDs. |
| **4** | **H4: MTP state interaction** | Could trigger incorrect ownership/reuse, but target and draft both slowing is also consistent with a common primitive. Inspect step-0 production and later reuse before disabling MTP. |
| **5** | **H5: GDN state** | Not the first target: QSA explains **75/98 ≈ 77%** of the clean decode regression. Missing GDN intervals are not proof of zero GDN cost. |
| **6** | **H3 as stated** | The named sampling kernels are unrelated. **QSA top-k degeneration remains testable**, but it is histogram/radix selection, not that rejection loop. |

An important H6 subcase: `persistent_topk.cuh` contains **inter-CTA spin waits**. However, its cooperative radix path starts above **32,768 compressed candidates = 131,072 logical tokens**. Correct metadata for a 98,304-token trigger gives only **24,576 candidates**. The inspected source already includes workspace memset and length clamping; verify that the loaded binary contains those fixes rather than proposing them again blindly.

# One ordered plan

### 1. Freeze the evidence and run the smallest discriminator — no restart

Save source hashes, loaded `_C_stable_libtorch` identity, existing event logs and metrics. Label every measurement with graph/capture ID and target versus draft.

Using the existing request harness, run **three fresh 256-token prompts**, each generating **16 tokens**, one at a time. Compare steady decode steps—not TTFT—with the existing poisoned 2K/8K measurements. Keep MTP and every service flag unchanged.

At this length, compressed visibility is approximately **64–68**, below k=512. Correct top-k takes its trivial branch: it emits sequential indices and does **not read logits or enter the radix barrier**.

**Decision:**  
- If QSA still costs approximately 6 ms extra per layer, ordinary candidate scanning and selection-distribution degeneration become weak explanations, **provided metadata is correct**. Prioritize metadata, pre-indexer, attention core and residency.  
- If tiny decode recovers, prioritize length-dependent selection/gathers and boundary transitions.

This does not validate device metadata. Do **not** run 200K yet: it crosses the actual radix threshold and introduces a second regime while destroying useful evidence.

### 2. Establish what can actually be reset — still no restart

The current `/openapi.json` advertises **neither `/reset_prefix_cache` nor sleep/wake routes**. There is no exposed reset workaround to test now.

The inspected prefix-reset implementation clears **hash mappings**, not QSA storage, top-k workspace or graph pools. If an authorized deployment later exposes it, the controlled test is:

```bash
curl -X POST \
 'http://localhost:9393/reset_prefix_cache?reset_running_requests=false'
```

Run only while idle and require `{"success":true}`.

Recovery would implicate prefix-associated lifecycle/reuse; failure would **not** exclude stale tensor contents or locality.

Do not use sleep/wake here: sleep mode was not enabled in the displayed launch arguments; level 1 discards KV and offloads weights, while level 2 discards all GPU memory. Neither is a surgical QSA reset. Allocating fresh requests can rotate logical blocks but cannot guarantee fresh physical storage or cleared scratch.

**Proceed to one instrumentation restart, performed by the operator.**

### 3. Spend the first restart on decisive instrumentation, not a knob change

**Flags/env: unchanged.** Retain MTP, prefix caching, budget 2048, compression 4, batch size 2048 and current graph configuration.

Mount patches adding external baked event pairs around:

- `qsa.py`: projections, indexer call, sparse attention, output projection.
- `indexer_qsa.py`: fused pre-indexer and index expansion.
- `ops/qsa_indexer.py`: paged logits kernel and `_topk` **separately**.
- `common/qsa_cache.py`: metadata launch, using ordinary stream events outside capture where appropriate.

The fused norm/RoPE/cache update is **one kernel**; boundary events cannot subdivide it without changing the kernel. Do not instrument speculative rejection kernels as QSA.

For prefill, keep each measured interval inside one segment. Use **exclusive stage times**, not sums of nested module intervals.

Also install a bounded snapshot callback now, so another restart is unnecessary. For **one main QSA layer and the draft layer**, preserve selected intermediate tensors at their production point into preallocated buffers; harvest after completion. Reading arbitrary intermediates after replay is unsafe because their storage may already have been reused.

Run:

1. Clean 256-token and 2K probes.
2. The known 98,304-token trigger, recording every step.
3. Identical fresh short probes.

Define the flip as **three comparable steps exceeding 3× their clean QSA baseline**. Locate the stage accounting for most of the approximately **6.2 ms/layer** increase.

**Decision:** follow that stage into step 4. If stage timings fail to reconcile with the parent interval, fix instrumentation/stream coverage before changing model behaviour.

### 4. In the same instrumented process, run a same-input isolation test

This is the **cheapest decisive experiment missing from the list**.

Preserve a clean, two-row input snapshot for the implicated primitive. After poisoning, replay that **unchanged input** through an isolated invocation, with independent outputs/workspace. Use only a few repetitions, outside serving traffic—not a throughput benchmark.

Compare:

- Clean snapshot before versus after poisoning.
- Poisoned snapshot versus clean snapshot after poisoning.
- Original captured invocation versus isolated invocation.

For top-k, preserve logits, visible lengths, shape and stride. A two-row, 65,536-column FP32 logits snapshot is only **512 KiB**. Check results against a CPU reference, allowing arbitrary ordering/tie choices.

**Decision:**

- **Poisoned data slow; clean data fast:** inspect numerical values, bounds and index distribution. Not generic allocator fragmentation.
- **Original graph slow; isolated identical input fast:** inspect captured pointers, shared scratch, lifetime and dependencies. This is not evidence that “CUDA graphs are slow.”
- **Unchanged clean input also slows in isolation:** process/device state or residency becomes stronger. A few equivalent calls in an independent process distinguish process-local from device-wide effects, although changed addresses prevent a pure placement comparison.
- **Invalid metadata or indices:** treat as a correctness bug immediately; performance knobs are inappropriate.

### 5. Use a second restart only for the identified cause

Apply **one targeted fix**, preserving the original configuration.

Examples of decision-directed fixes—not exploratory knobs:

- Bounds/ownership violation: repair the producer and explicitly initialize padded/reused rows.
- Top-k-only regression with valid inputs: replace `_topk` temporarily with an exact, non-cooperative masked selection implementation retaining **k=512** and `-1` padding. Recovery isolates the original primitive, not necessarily its barrier.
- Residency evidence with unchanged-input slowdown: lower `--gpu-memory-utilization` from **0.96 to 0.90**, provided the resulting cache still accommodates the trigger without preemption.

Only if snapshots specifically implicate MTP reuse should this restart remove `--speculative-config`; do not assume `QWEN_MTP=0` controls this build.

Validate with the same clean→98K→short sequence twice. Require short-decode QSA time within **20% of clean**, valid outputs and no new preemption.

## Log these invariants

Every step: graph ID, segment, target/draft/MTP step, actual/padded token counts, request count, query starts, sequence lengths, `skip_topk`, stage times and request/cache/preemption metrics.

At capture and whenever changed: tensor **shape, stride, storage offset, pointer**, logits width, launch grid, work-buffer live/capacity sizes and top-k implementation.

For sampled device snapshots:

- `visible_blocks = floor((logical_position+1)/4)` for valid rows; **zero for padding**.
- Physical block/slot bounds and active K-work entries.
- Packed count:  
  **`4*min(visible_blocks,512) + ((position+1)%4)`**, hence **0–2051**.
- Selected indices: causal, in range, expected padding.
- Logits/Q/K: finite values, range, zero/tie prevalence.
- MTP reused rows tied to the correct producer and request.

Shapes and CPU mirrors need no kernels. Device values require an explicit snapshot/readback; `/metrics` and py-spy cannot provide them.
