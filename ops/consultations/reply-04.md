Two source details affect the next test. Prefix-cache reset clears hashes but does not reset the free-block queue, so it will not reliably recover low IDs. Also, `align` gathers state IDs from each group’s block table; it does not map QSA IDs modulo the eight-block group. I’m favoring a small crossover test that holds one QSA invocation’s inputs fixed and varies KV placement and launch path independently.
**My next experiment would be a one-layer, same-context QSA crossover: original versus relocated K/V, each launched eagerly and through a single-kernel CUDA graph.** That separates address-sensitive behavior from submission-sensitive behavior while requiring only a small per-layer scratch buffer.

Your evidence localizes the slowdown well, but I would retain one distinction: **30 ms between CUDA events is not yet proof that the kernel spent 30 ms executing on SMs.** The interval can include submission gaps, descheduling, and waits. Full clocks plus low power is consistent with memory stalls, but also with intermittent execution.

I inspected the local source and confirmed that the container’s QSA kernel, Mamba block-table helper, and block-pool implementation match it. I made no changes and ran no GPU workloads.

**1. Ranked mechanisms**

| Rank | Mechanism | Assessment |
|---|---|---|
| **1** | **H1, broadened to allocation-specific memory behavior:** residency/backing, translation, or cache behavior affecting engine allocations | Fits the healthy external allocations and sensitivity to request history. But “the long request leaves the TLB dirty” is insufficient: a repeatedly accessed short working set should replace old translations. Persistence requires a continuing cause—different mappings, placement, working set, interference, or residency decisions. |
| **2** | **H4, narrowed to an engine-context or prefill-path execution/submission problem**, including the relevant part of H2 | Still compatible with event timing. Healthy decode and external probes argue against a uniform device-wide or context-wide slowdown, but not against particular submissions, streams, graph executions, or allocations. |
| **3** | **Retained software state or buffer aliasing/lifetime problems outside the recorded launch fingerprint** | Worth keeping above pure L2 speculation. Equal shapes and count sums do not establish equal pointers, strides, complete selection arrays, graph-buffer relationships, or dependency edges. There is no positive evidence of corruption, however. |
| **4** | **H3: address-dependent L2/TLB conflicts caused by allocation order** | Physically plausible, but the ID summaries do not establish it. A uniform 13.6× QSA penalty across layers needs a repeatable address-layout explanation. |
| **5** | **H2 as a fixed completion/event latency per layer** | Strongly disfavored by the token-dependent excess and unaffected long-context chunks. A fixed 28 ms penalty does not naturally become approximately proportional to token count. |
| **6** | **H5 as the cause of poison** | Excluded as a necessary cause by your uninstrumented reproduction. Instrumentation can still affect attribution and absolute timing. |

There is an important constraint on **any KV-only explanation**:

- Total measured GPU excess: **2613.0 − 679.6 = 1933.4 ms**.
- Direct QSA-attention excess: **12 × (30.0 − 2.2) = 333.6 ms**, approximately **17%**.
- GDN-module excess: **1588.5 − 422.0 = 1166.5 ms**, approximately **60%**.

If `GDN_FUSED` covers the same prefill core invocations and remains unchanged, most of the GDN excess lies outside that core. The local wrapper includes input/output projections outside the measured core. **A bad QSA KV address cannot directly explain those projections becoming slower.** A common mechanism must also affect their allocations or execution, or the inclusive module interval must contain additional gaps.

What you can reasonably discard:

- Sustained clock/power throttling as the main cause.
- A global loss of compute throughput or HBM bandwidth.
- PLE SSD service time as the primary cause.
- The previously suspected larger logical QSA workload on poisoned short requests.
- A spin loop **inside the measured sparse-attention kernel**: the inspected kernel has no inter-CTA polling loop. TOPK is separate and did not inflate.

The unchanged 96K throughput is especially unfavorable to a universal slow mode. It does not prove that every trigger chunk is unaffected unless those chunks were individually timed; an aggregate 50–64 seconds can hide some localized penalties.

**2. ONE experiment: frozen-input QSA placement × launch crossover**

Perform this at **one QSA layer**, on one fresh 2048-token request in each state. Use the engine’s own CUDA context. The external context cannot answer the address-specific question using its own allocations.

The four measurements are:

| | Original K/V allocation | Relocated copy of the referenced K/V blocks |
|---|---:|---:|
| Isolated eager launch | \(T_{OE}\) | \(T_{RE}\) |
| Single-kernel graph replay | \(T_{OG}\) | \(T_{RG}\) |

Also retain the normal production invocation’s approximately **2.2/30 ms** measurement as the reference.

**Preparation**

At the next instrumented startup, reserve enough scratch for **only the K/V blocks referenced by one 2048-token request at one layer**, plus a block-table copy. Reserve it before KV sizing rather than assuming a runtime allocation will succeed.

For an unpadded BF16 layout with head dimension 256:

\[
B_{\mathrm{KV}}=2\times2048\times H_{\mathrm{KV}}\times256\times2
             =2\ \mathrm{MiB}\times H_{\mathrm{KV}}.
\]

Round up to full storage blocks and use the actual strides to calculate the allocation. Four KV heads would require approximately **8 MiB before padding**. This does not require duplicating the engine-scale KV pool.

Use the existing Q, selected indices, token-to-request mapping, gate, and output buffers. The inspected `NUM_SPLITS=1` kernel only reads its inputs and writes its output, so repeating it before downstream consumption does not update KV or recurrent state.

**Procedure**

1. At the chosen eager break, hold the consumer back and ensure the input producers have completed. Keep all operands alive and unchanged during the diagnostic.
2. **Measure the original-address case before copying or reading back K/V.** A diagnostic copy could warm caches or change residency.
3. Copy the referenced **whole storage blocks** into scratch. Preserve the K/V layout and strides. Build a private block table that maps the same logical blocks to their copied locations.
4. Launch the **same compiled kernel** with the same grid, specialization, strides, counts, scales, Q, selections, gate, and output. Change only K/V pointers and the private table for the relocated case. Avoid going through a wrapper that might choose a new specialization.
5. Measure both address cases on the same diagnostic stream, first eagerly and then through graphs containing only **start event → kernel → end event**, with no unresolved incoming dependencies. Use external timing-event nodes for capture.
6. Use three repetitions per cell, reverse the order on alternating repetitions, and retain first-use timings separately. Keep compilation and graph construction outside the timed intervals.

Do this once clean and once poisoned. Twelve poisoned invocations at 30 ms each would add roughly **0.36 seconds of kernel time per state**, plus preparation. No pool-sized benchmark is needed.

**Interpretation**

| Poisoned result | Conclusion supported |
|---|---|
| Original ≈30 ms; relocated ≈2–4 ms **under both launch modes** | Strong evidence that K/V placement/backing/access locality is causal. A placement ratio above **4×**, reproducible in both orders, would be decisive enough to pursue H1/H3. |
| Both placements ≈30 ms eagerly but ≈2–4 ms in the single-kernel graphs | Strong evidence for an eager submission/timing-gap mechanism. The memory addresses alone cannot account for the difference. |
| Both isolated modes are fast, although the production invocation remains ≈30 ms | The surrounding execution sequence matters. Incoming dependencies, interference, or first-touch/residency effects remain candidates. Use the first original-address measurement to distinguish this from an effect introduced by copying. |
| Original improves strongly on immediate repetition, before relocation | Temporal warming or residency matters. Do not attribute subsequent relocated-buffer speed to address placement alone. |
| All four remain ≈30 ms | Neither relocating K/V nor removing the surrounding launch chain is sufficient. Other operands, context scheduling during execution, and internal kernel behavior remain open. |

That last outcome is deliberately not labeled “TLB proven.” Broad H1 and H4 can overlap: driver residency management is both allocation-specific and context-mediated. No timing-only experiment guarantees a unique hardware diagnosis.

**3. The block-ID shift can matter—but these are not hardware physical-page IDs**

In the QSA kernel, the relevant address calculation is essentially:

```text
selected logical token
    → block_table[logical token / PAGE_SIZE]
    → K/V base + block_id × block_stride + within-block offset
```

The source uses a 64-bit block-offset calculation. See the [QSA kernel](/home/hong/vllm/src/vllm/models/qwen4_exp/nvidia/ops/qsa.py:105).

Thus, an ID shift changes the **virtual addresses accessed inside a CUDA allocation**. It does not tell you which HBM pages, L2 sets, memory partitions, or GPU translation entries back those addresses.

Three mechanisms are plausible:

- **Residency/backing differences:** different pool offsets can access differently treated memory regions. Healthy fresh external allocations do not exclude this.
- **Cache conflicts:** repeated block strides and relative K/V/Q/output placement can produce unfavorable cache behavior. Contiguous blocks can still conflict with other streams of addresses.
- **Translation effects:** different addresses can change translation working sets or conflicts. But for the same number of densely accessed pages, moving to higher IDs does not inherently increase TLB demand.

The inspected [`align` helper](/home/hong/vllm/src/vllm/v1/attention/backends/utils.py:1134) gathers entries beginning at:

```text
(seq_len - 1) // mamba_block_size
```

from the **corresponding Mamba block table**. It does not take the QSA block ID modulo eight. The eight-block group is therefore a reason to identify the group and inspect its actual state indices, not evidence of an eight-slot collision mechanism by itself.

For cheaply testing allocation order:

- **Prefix-cache reset will not reliably recover low IDs.** The inspected [reset implementation](/home/hong/vllm/src/vllm/v1/core/block_pool.py:821) clears hashes without rebuilding the free-block queue.
- Filling/freeing requests can move through that queue, but also changes cache history and residency. It is a confounded way to test H3.
- The relocated-KV arm above tests address sensitivity directly. For a later allocation-policy intervention, select known-free blocks through the allocator and log the resulting per-group tables; changing table IDs alone without moving their contents is invalid.

**4. What to dump—and what a dump cannot establish**

**No static buffer dump can distinguish a memory stall from a fence wait.** It can expose the software condition that causes one. For one matched clean/poisoned invocation per relevant layer, record:

| State | Why it matters |
|---|---|
| Actual Q/K/V/output/gate/selection pointers; storage bases, offsets, shapes, byte strides and dtypes | Tests whether “identical kernel” also means the same storage relationships. |
| Complete referenced block-table entries, per group | A min/max/jump summary can conceal different mappings. |
| Per-row valid counts and selected-index arrays, or exact comparisons where expected equal | Equal maxima and sums do not establish equal access sequences or tile-count distributions. |
| GDN state indices, `has_initial_state`, sequence offsets, prefill/decode counts | Tests wrong-slot selection or an unintended recurrent-state path. |
| Graph/segment identity, stream identity, and producer/wait/record relationships | Tests whether an interval includes a dependency that changed. |
| Host duration of the exact raw kernel-launch call | A launch call expanding toward 30 ms changes how to interpret the surrounding CUDA-event interval. |

Read these back **after timing**, because `.cpu()`/`.item()` can drain queues and perturb the state you want to measure.

If you add kernel-side telemetry to this experiment, the most useful small trace is **per-CTA entry/exit `%globaltimer` timestamps and SM IDs**, stored without atomics in uniquely indexed slots. For \(N\) CTAs, two 64-bit timestamps and one 32-bit SM ID cost about **20N bytes**.

Compare:

\[
T_{\text{CTA envelope}}=\max(t_{\text{exit}})-\min(t_{\text{entry}})
\]

with the surrounding event interval.

- Event time ≈30 ms, CTA envelope ≈2 ms: most of the excess lies outside the observed CTA execution envelope.
- Both ≈30 ms, with longer matched per-CTA lifetimes: the delay occurs while CTAs are in flight.
- Similar per-CTA lifetimes but much wider launch-wave spacing: investigate scheduling, admission, or reduced concurrent progress.

**Even per-CTA elapsed time includes preemption; neither `%globaltimer` nor `clock64()` is an active-cycle counter.** That trace locates the missing time. The placement × launch crossover supplies the causal intervention needed to interpret it.
