The strongest current hypothesis is a process or CUDA-context scoped virtual-memory failure in the WSL2 GPU-PV path. The evidence is much less consistent with damaged DRAM, a global L2 problem, SM throttling, or ordinary kernel scheduling.

The fresh-process result is especially valuable: the same scatter kernel is healthy on the same device, at the same time, with the same driver. That makes a global hardware condition very unlikely. It does not yet distinguish a CUDA-context resource from a dxgkrnl per-process allocation or residency ledger, because a fresh process creates both a new process record and a new CUDA context.

## 1. Live candidates

| Candidate | Why it remains live | Expected signature |
|---|---|---|
| Context-scoped GPU page tables, TLBs, and page-walk caches | A large prefill can increase the number of mapped and touched pages by millions. A new context can have clean translation state while the old one retains pathological mappings or occupancy. | Strong dependence on distinct virtual pages and VA span; high penalty at low MLP; sensitivity to VA relocation or VMM aliases. |
| dxgkrnl/WDDM GPU-PV residency and mapping bookkeeping | WSL2 does not expose all residency, remapping, and replay activity through guest `dmesg`. A process can retain a damaged or highly fragmented allocation history while a new process gets a clean host-side record. | Performance follows allocation history, mapping history, or process identity. Releasing/remapping allocations may help without resetting the GPU. |
| Page-table level occupancy | The relevant limit may be an intermediate page-table level, a page-table cache, or a set-associative TLB/walk cache rather than total bytes. | Sharp threshold when VA mappings cross a particular range, alignment, or number of populated table entries. |
| Per-context replay, page-walk, or fault capacity | Your CTA traces show no scheduling gap, but a warp can remain inside the CTA while requests are replayed or wait for translation. A replay resource can be saturated without generating an OS-visible fault. | Heavy-tailed CTA completion times, increasing degradation after repeated scatter passes, sensitivity to MLP and number of unique pages. |
| Physical allocation placement or partition color | The engine process may receive a different physical-page distribution after holding approximately 61 GiB. This can be process-history dependent under WDDM residency. | Performance follows the backing allocation after remapping, rather than the virtual address. |
| Virtual-address-dependent cache, request, or translation partitioning | Some pretranslation structures can use VA bits. This could produce aliasing or set imbalance even if DRAM channel selection itself is based on physical addresses. | Performance follows VA bits, alignment, or high-level page-table indices. |
| CUDA allocator and mapping topology | Caching allocator fragmentation, expandable segments, CUDA graph mappings, and separate allocation classes can create different VA gaps, page sizes, and mapping boundaries. | New VMM allocations and legacy allocator allocations behave differently in the same process. |
| Process-scoped WSL2 host state | The important state may live above the CUDA context: dxgkrnl handles, host allocation objects, residency lists, or process budgets. | A second CUDA context in the same process remains bad, but a new process is healthy. |
| Pinned-host/BAR or staging-path state | Still technically live, but lower probability. The device-only fresh-buffer result and full dense-copy bandwidth make it a poor fit. | Changing pinned-memory configuration changes the scatter pathology specifically, not just the overall baseline. |

The numbers support a translation or replay mechanism:

- At 4 KiB pages, 61 GiB represents about 15.99 million pages; 3 GiB represents about 786,432 pages.
- At 64 KiB pages, those figures are about 999,424 and 49,152 pages.
- At 2 MiB pages, they are about 31,232 and 1,536 pages.

If the GPU uses 512-entry page-table pages, 4 KiB mappings for 61 GiB require roughly 31,232 leaf PTE pages, versus about 1,536 for 3 GiB. The exact GPU page-table format and effective page size are implementation details, but the possible occupancy ratio is large enough to explain a threshold.

A CUDA allocation size alone does not tell you the effective GPU page size. WSL2, WDDM, the CUDA allocator, and VMM allocation granularity can all affect it.

The sector or DRAM partition hypothesis needs more caution. Conventional DRAM channel selection is generally based on physical-address bits after translation. A virtual address can still affect TLB sets, page-walk structures, cache indexing before translation, or request partitioning, but “virtual addresses select DRAM partitions” should be treated as an experiment-specific hypothesis rather than an assumed architectural fact.

The 8-byte random-load result is also consistent with translation pressure being conditional on request shape. A single outstanding load has little ability to create a large replay backlog. The 256-byte rows create more sectors, more page-boundary interactions, and more concurrent outstanding work per CTA. Dense streaming can amortize translation and sustain high memory-level parallelism even while scattered requests are trapped in a translation or replay bottleneck.

## 2. The VMM alias experiment

The experiment is useful, but a negative result would not exclude page tables by itself.

A 32–48 GiB reservation with no mappings does not consume equivalent leaf-page-table state. The mappings must actually be installed across the range. Repeatedly mapping a 64 MiB physical allocation should create distinct virtual mappings, and the GPU page walker normally works from the VA and PTEs; it does not normally inspect whether identical PTE contents came from the same CUDA allocation handle. However, the driver may batch mappings, use larger mapping granularity, or maintain allocation-level metadata that makes the alias case different from a normal 32–48 GiB allocation.

The main confounds are:

1. **Cache reuse.** Every alias refers to the same 64 MiB of physical data. If the cache is physically tagged, repeated aliases can reuse the same lines even though the virtual addresses differ. That can make the alias case artificially fast.

2. **Allocation granularity.** The mapping may use a recommended or large granularity. A 32 GiB VA span mapped at 2 MiB granularity produces far fewer leaf entries than the same span mapped at 4 KiB or 64 KiB granularity.

3. **Mapping regularity.** A fully regular repeated mapping may be handled more efficiently by the driver or page-walk caches than the irregular allocation topology produced by the engine.

4. **Different residency semantics.** One 64 MiB physical object may generate one residency record even when aliased thousands of times. A real 32 GiB allocation may generate millions of page-level residency records.

5. **Repeated physical pages.** The experiment tests virtual translation pressure and alias behavior, but it does not reproduce physical-page diversity, physical placement, or per-page residency metadata.

I would change the experiment into a small factorial design:

| Arm | Virtual span | Physical backing | Purpose |
|---|---:|---:|---|
| A | 64 MiB | 64 MiB | Baseline |
| B | 32–48 GiB mapped | 64 MiB aliased | Large VA/PTE footprint with repeated backing |
| C | 32–48 GiB mapped | As much unique backing as practical | Large VA and physical footprint |
| D | Same large VA span | Repeated backing with deliberately irregular alias placement | Tests whether regular repetition is being optimized |

Because the device is already using about 61.6 GiB, arm C may be impractical in the current state. A smaller span is acceptable if you first establish the threshold curve. The important comparison is between VA footprint and unique physical-page count.

Use the smallest supported VMM granularity, or at least record the granularity returned by `cuMemGetAllocationGranularity`. Put aliases at nonuniform VA offsets, include guard gaps, and ensure that every mapping is actually touched. Reserve-only space should be treated as a control, not as equivalent to mapped space.

For the kernel:

- Keep the number of load instructions and rows constant across arms.
- Use an incompressible buffer and a streaming cache operator where possible.
- Randomize alias order and physical offsets.
- Make the output observable so loads cannot be removed.
- Measure a low-MLP version and a higher-MLP version separately.
- Record event time, `%clock64` samples, and CTA p50/p95/p99/max.
- Do not compare total useful bytes as the primary metric; compare the same number of row operations.

An aliased case that becomes pathological would strongly implicate VA translation, page-walk coverage, VA-indexed structures, or alias handling. It would not uniquely prove ordinary page-table exhaustion.

An aliased case that remains healthy would only say that repeated aliases do not reproduce the complete failure. It would leave open:

- unique physical-page residency metadata,
- page-level remapping state,
- physical placement,
- allocation-level dxgkrnl bookkeeping,
- replay state associated with the original allocation history,
- or a mapping granularity different from the one used by the real KV allocation.

## 3. Best single discriminator after a negative alias result

The highest-value experiment is a **same-process VA-by-backing crossover**. Perform it inside the poisoned engine process, using CUDA VMM and two equal-size physical allocations mapped into two deliberately different virtual ranges.

Create:

- `P0`, `P1`: two equal-size, incompressible physical allocations;
- `V0`, `V1`: two reserved VA ranges with the same mapping granularity but different high-level VA indices and alignment;
- the same scatter kernel and the same logical row sequence.

Run these four mappings:

| Mapping | Interpretation |
|---|---|
| `P0 -> V0` |  |
| `P0 -> V1` | Same physical backing, different VA |
| `P1 -> V0` | Different physical backing, same VA |
| `P1 -> V1` |  |

Synchronize before and after every remap. Pre-touch the backing allocations so first-touch behavior is not mixed into the main measurement. A 64–256 MiB test object is probably enough to start; sample only a fixed subset if the larger object would violate the no-heavy-benchmark constraint.

Interpretation:

- **Slow follows `V0` or `V1`:** VA-indexed translation, page-table level, TLB, or VA-related partitioning.
- **Slow follows `P0` or `P1`:** physical placement, residency state, or allocation-level backing state.
- **All four are slow:** context-wide replay/translation pressure or process-scoped dxgkrnl state.
- **Only one VA/backing combination is slow:** interaction between mapping history and backing residency.
- **The first pass is faster and later passes degrade:** replay, residency churn, or a self-reinforcing cache/page-walk state.

This experiment is more discriminating than simply comparing an aliased 64 MiB object with a unique 64 MiB object because it swaps one variable at a time. It also directly tests whether the pathology follows the virtual address or the physical allocation.

Use a VMM allocation rather than a PyTorch caching-allocator allocation for at least one arm. If both VMM and legacy allocations are available, add one legacy allocation as a third backing class. That can reveal whether the CUDA allocator is producing a particular mapping or residency topology.

One limitation is that CUDA does not provide a supported API to inspect the GPU’s actual PTE hierarchy or physical DRAM color. The experiment infers the carrier from what performance follows.

## 4. Why the 96K prefill can cause permanent and increasing damage

A large prefill can cross several independent thresholds at once:

1. It adds several GiB of KV storage.
2. It maps and touches hundreds of thousands to millions of additional pages.
3. It creates more intermediate page-table nodes.
4. It increases the number of live WDDM/dxgkrnl allocation and residency records.
5. It changes allocator fragmentation and future VA placement.
6. It exercises sparse, page-diverse accesses immediately after creating those mappings.

The trigger may therefore be a threshold in a finite structure rather than a linear amount of memory. For example, the extra KV may populate enough leaf mappings to overflow a page-walk cache set, consume a context-scoped replay budget, force a less favorable mapping mode, or push dxgkrnl into a different residency regime.

The “permanent” part does not require a permanent physical fault. It can mean that the process retains:

- the large mapping topology,
- fragmented allocation state,
- page-level residency metadata,
- a bad VA-to-backing arrangement,
- or a context-scoped queue/cache state.

The “worsens with activity” result is compatible with a self-reinforcing mechanism:

- each scattered pass touches many new or weakly resident pages;
- translation misses consume finite walk and replay resources;
- stalled requests remain associated with the same CTA;
- new requests continue to compete for those resources;
- page-table, TLB, and residency structures are continually churned;
- a subset of CTAs accumulates a very long tail.

That explains why your CTA concurrency remains 222/222 while the p95 and maximum durations become tens of milliseconds. The CTAs are resident, but some warps inside them are waiting or replaying. It also explains why a per-access thermometer can show only microsecond-scale completions: the bad time may be concentrated in a small population of requests or in repeated request/replay cycles that a one-megabyte footprint never triggers.

Dense reads remain fast because they:

- amortize translation over many adjacent sectors;
- allow coalescing and prefetch-like behavior;
- sustain high memory-level parallelism;
- do not repeatedly jump across unrelated page mappings;
- hide individual translation delays behind other useful work.

A single 8-byte random load has the opposite problem from the 256-byte row test: it has little work per request and low MLP, but it also creates less sector and replay pressure. Its 2.8× slowdown is therefore compatible with the 73–200× collapse of the row-scatter cases.

## 5. Possible mitigations

The practical mitigations depend on which arm of the crossover experiment wins.

The lowest-risk workload mitigations are:

- Keep the process below the token or KV threshold that triggers the state transition.
- Use chunked prefill so a large request does not create and touch its entire new working set in one burst.
- Preallocate the KV layout early and reuse it, avoiding allocator churn and late VA fragmentation.
- Avoid repeated allocation/free cycles around large KV and temporary buffers.
- If supported by the model/runtime, isolate very large-context requests in a separate worker process.

Chunked prefill deserves a careful qualification: it may prevent a burst-induced replay or residency failure, but it may still fail if the final mapped working set itself crosses the problematic threshold. Test both the largest individual chunk and the accumulated mapped-page count.

Pre-touching can help if the problem is lazy first-touch residency. It will not help if the problem is page-table occupancy or a bad process-level mapping ledger; it may simply move the failure earlier. The useful test is to separate:

1. allocation,
2. mapping,
3. first touch,
4. scatter access.

Measure the scatter kernel after each stage.

A memory-pool change is worth testing in a controlled process:

- legacy PyTorch allocator versus `cudaMallocAsync`, where supported;
- fixed-size persistent allocations versus expandable segments;
- VMM allocations with the reported recommended granularity;
- fewer large contiguous allocations versus many fragmented temporary allocations.

There is no supported `nvidia-smi` or CUDA switch that tells WSL2 GPU-PV to disable this class of replay, force a particular WDDM residency policy, or force large GPU pages for ordinary `cudaMalloc` allocations. CUDA VMM lets you control VA reservation and mapping granularity, but it does not expose the complete GPU PTE policy.

Disabling pinned host memory is a valid A/B test because it changes the host allocation and staging path. Given that the failure reproduces on device-side scatter into a fresh device buffer, I would expect it to change the baseline more than the poisoned/clean ratio. A result where disabling pin memory prevents the degradation would make the WSL2 host-memory/PV path much more likely.

A potentially useful recovery test, if the engine can safely release temporary allocations, is:

- release the extra KV or test mappings;
- trim the CUDA memory pool;
- synchronize;
- repeat the scatter probe.

If performance recovers while the CUDA context remains alive, allocation or residency state is implicated. If it does not recover until the process/context is destroyed, context-scoped state or an unrecoverable dxgkrnl bookkeeping defect becomes more likely.

For confirmation, record the following at each step:

- total mapped and allocated device bytes;
- virtual addresses and alignment of the relevant buffers;
- VMM allocation granularity;
- number of unique pages touched by the probe;
- CUDA event time for a fixed operation count;
- `%clock64` p50/p95/p99/max for sampled loads;
- CTA duration p50/p95/p99/max;
- whether the second and tenth repetitions are slower than the first;
- Windows-side per-process GPU memory and process lifetime;
- whether freeing/remapping changes the result.

The most convincing WSL2 GPU-PV signature would be a repeatable threshold in mapped-page or allocation history, with the slowdown following the process or mapping/backing crossover, no corresponding clock or thermal change, and recovery after freeing the relevant mappings or destroying only the affected CUDA context. A clean new process plus a bad old process is already strong evidence for that class of defect; the crossover experiment should identify whether the carrier is VA state, backing/residency state, or context-wide replay state.
