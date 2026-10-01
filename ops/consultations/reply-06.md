1. I agree with the mechanism at the phenomenological level, with one refinement: I would call it an **address-sensitive, heavy-tailed completion/replay regime** rather than asserting that DRAM accesses themselves become slow.

A timed load in a CUDA kernel measures more than DRAM latency. It can include:

- L1/L2 miss handling
- TLB lookup and page-table walks
- replay after a failed or delayed memory transaction
- memory-fabric queueing
- warp scoreboard delay
- residency or migration handling
- scheduler delay before the load is issued or retired

Your data strongly supports that this combined path has developed a long tail.

The key facts fit together:

- The attention CTA remains resident while its warps wait on memory-related dependencies. Continuous CTA coverage and constant active CTA count do not imply useful progress.
- Dense GEMM and streaming copies remain fast, so aggregate arithmetic and bandwidth are healthy.
- Sparse 2 KB accesses collapse, while large streaming work survives. That is the normal signature of poor latency hiding.
- The fresh-buffer FSC and STR results show that the problem is not specific to KV contents or block-table-selected KV pages.
- The second KSC pass getting slower is especially significant. A normal cache miss would tend to improve on the second pass. That result suggests an active subsystem state: replay pressure, translation/residency work, queue buildup, or some other state that the probe itself worsens.
- The 9,472 concurrent dependent chains are already a high-global-MLP experiment. Their 1.6x degradation does not contradict a severe low-MLP tail; many chains can hide the individual waits.

The strongest single explanation is therefore:

> The large prefill leaves the GPU or driver in a state where some address-sensitive memory operations have a very long completion tail. The production attention kernel exposes that tail because each CTA has little independent work and relatively few stages of software pipelining. Streaming and dense kernels hide it.

This explanation includes VM translation, page residency, replay, and memory-fabric queueing as possible implementations. The present data does not distinguish those implementations.

“Few pages become slow” remains possible. FSC and STR being slow only show that the effect is not restricted to the original KV allocation. A small set of bad pages or mappings in the fresh allocation could still explain both. KRD can also hide a small number of pathological pages through high concurrency.

2. I would measure **effective per-load completion latency** first. Use `%globaltimer` around a single explicit load, record every sample, and correlate each sample with address, virtual page, SM, time, and test arm.

A conceptual per-sample sequence is:

```cuda
t0 = globaltimer();

asm volatile(
    "ld.global.u64 %0, [%1];"
    : "=l"(value)
    : "l"(address));

t1 = globaltimer();

sink ^= value;
record(dt = t1 - t0, address, page_id, smid, t0);
```

Use inline assembly or another mechanism that prevents load elimination. Keep the result live through a reduction or sink. Do not put a CUDA event around every access; use CUDA events only around the whole kernel. Calibrate the timer bracket with an empty bracket and retain the raw timings as well as timer-subtracted values.

Use one active lane per warp for the 8-byte test. This prevents 32 lanes from merging into a single coalesced transaction. Ensure simultaneously active workers use addresses separated by at least 128 bytes. For a 128-byte logical access, use a fixed sequence of smaller loads inside the timed region and define the sample as the whole 128-byte request.

Record at least:

- `dt` from `%globaltimer`
- `%clock64` delta from the same region
- `%smid`
- virtual address or a compact address tag
- candidate page number at 4 KB, 64 KB, and 2 MB granularity
- arm and MLP level
- sample timestamp or tile number

A 16-byte record for one million samples is only about 16 MB. Avoid atomics in the measurement path. Write records into preallocated output storage, preferably in short per-thread batches so the output store is less interleaved with the timed load.

Use logarithmic bins after the kernel completes. For example:

- below 0.5 µs
- 0.5–1 µs
- 1–2 µs
- 2–4 µs
- 4–8 µs
- 8–16 µs
- 16–32 µs
- 32–64 µs
- 64–128 µs
- 128–256 µs
- 256–512 µs
- 0.5–1 ms
- 1–2 ms
- 2–4 ms
- above 4 ms

Also report p50, p90, p99, p99.9, p99.99, maximum, and survival probabilities at fixed thresholds. With \(10^6\) samples, probabilities around \(10^{-4}\) are measured reasonably well; a single event at \(10^{-6}\) is not enough to characterize a distribution.

The important qualification is that the probe is itself state-changing. A single million-sample kernel cannot be assumed stationary because your second KSC repetition already gets worse. Divide the kernel into time-ordered tiles, for example 64 tiles of 16,384 samples, and retain the tile number. The result should show both the aggregate distribution and whether the distribution changes during the shot.

A good one-launch layout is:

- 8-byte random loads at MLP 1
- 8-byte random loads at MLP 4
- 8-byte random loads at MLP 64
- 8-byte random loads at MLP 1024
- repeat those arms in several short tiles
- randomize the arm order across tile groups

This gives you a time-resolved factorial experiment in one kernel launch. “One launch per state” is compatible with this, but a stationary histogram is not compatible with a probe that changes state. The tile history is part of the result.

For MLP, define it explicitly as **outstanding logical loads per SM**. A useful implementation is one persistent CTA per SM, with one active lane per warp:

- MLP 1: one active worker per SM
- MLP 4: four active workers per SM
- MLP 64: 64 active workers per SM
- MLP 1024: 1,024 active workers per SM

The 1024 case uses 32 warps with one active lane each. Record `%smid` and the number of workers observed per SM; discard or separately classify samples from SMs that did not receive the intended worker count. If controlling one CTA per SM is difficult, use smaller levels and report the actual observed active-worker count rather than calling the nominal launch configuration MLP.

A barrier at the beginning of each round makes the loads approximately concurrent within a CTA. It does not make the issue timestamps identical, so record the per-load timer values and inspect the issue spread. The actual MLP is the number of loads issued before earlier loads complete, not merely the number of resident threads.

3. To distinguish a few bad pages from a global heavy-tailed distribution, reuse the same address set many times and attach every timing sample to a page tag.

Use approximately:

- 1,024 candidate 64 KB pages in the existing 64 MB fresh buffer
- 16–64 sampled offsets per page
- a randomized page/offset order
- at least 16 passes over the same tagged address set

The exact hardware page size is unknown, so compute tags at 4 KB, 64 KB, and 2 MB boundaries. Also include offsets within each candidate page so that a page-level effect can be separated from a particular cache-line or L2-set effect.

For each candidate page \(i\), compute:

\[
q_i(T) =
\frac{\text{number of samples from page }i\text{ with }dt>T}
{\text{number of samples from page }i}
\]

Use several thresholds \(T\), such as:

- 10 times the clean median
- 1 µs
- 10 µs
- 100 µs
- 1 ms

Then examine:

- the distribution of \(q_i\)
- whether the same pages remain slow across passes
- the variance of \(q_i\) beyond binomial sampling noise
- correlation with SM
- correlation with sample time
- whether pass 2 is slower for every page or only for a subset

Expected signatures:

| Observation | Likely implication |
|---|---|
| A small fraction of page IDs have very high \(q_i\) across every pass | A few pages, mappings, or physical regions are pathological |
| \(q_i\) is nearly uniform across pages, but the overall tail is much heavier in the poisoned state | A global address-sensitive latency distribution |
| The same page alternates between fast and slow depending on time | Dynamic residency, replay, queueing, or migration |
| Slow samples cluster by `%smid` | SM-local cache, partition, or scheduler state |
| Slow samples occur across all SMs in the same time windows | Global GPU or driver state |
| Pass 2 becomes slower for nearly every page | Probe-induced global degradation |
| Only one allocation is slow | Allocation or mapping-specific issue |
| Fresh and KV pages have the same per-page statistics | Allocation-independent memory-system state |

A stable page subset would refine your hypothesis from “rare accesses globally become slow” to “a small subset of pages or mappings becomes pathological.” Uniform page behavior with a long tail would support your current interpretation.

For the page test, use an address permutation that is identical between passes. A fresh random permutation on every pass would prevent you from determining whether the same page is responsible. Keep the permutation fixed but change the order, so you can distinguish page identity from temporal position.

4. For access-size dependence, measure two different quantities:

- **request-level completion time** for an 8 B, 128 B, 2 KB, or 1 MB region
- **subrequest-level timing** for the 128-byte chunks inside that region

A 1 MB access is not one hardware load, so its total duration cannot be compared directly with an 8-byte load without decomposing it.

For each size, choose random aligned bases and time:

```text
t0
read the complete region
t1
```

For 2 KB and larger regions, use the same number of threads and the same memory instruction sequence in every state. For 1 MB, use a fixed cooperative reader, such as a warp or CTA, and divide the region into 128-byte or 256-byte chunks.

Use fewer samples for large requests. For example:

- 8 B: \(10^6\) samples
- 128 B: \(10^5\) samples
- 2 KB: \(10^4\) samples
- 1 MB: \(10^3\) samples

Keep either total bytes or total regions fixed and report which normalization you used. A useful fixed-byte design is 256 MB per arm, though the large requests may be more informative with a fixed number of regions.

Interpret the results as follows:

- If per-128-byte chunk tail probability is approximately constant across request sizes, while the probability that a whole request is slow follows

  \[
  P(\text{at least one tail}) \approx 1-(1-q)^N
  \]

  where \(N\) is the number of chunks, then a rare per-transaction tail is likely.

- If latency jumps at a particular page-sized working set or stride, translation or residency is implicated.

- If 1 MB reads have a large total-time tail but their individual chunks are normal, the problem may be request-level scheduling, cooperative synchronization, or an issue with the large-reader kernel.

- If 8 B and 128 B accesses already show the poisoned tail, the issue is probably below the production kernel’s 2 KB row construction.

- If only 2 KB or larger accesses show the problem, suspect transaction aggregation, warp-level replay, coalescing, or the production access pattern.

Use multiple strides for this test: 4 KB, 8 KB, 16 KB, 32 KB, 64 KB, and 2 MB. A sharp change at one stride is more informative than the absolute result from one 4 KB stride.

5. For the bandwidth-versus-MLP curve, use a fixed large working set and vary only the number of independent operations that can be in flight.

Use:

- 8 B random loads
- 128 B random loads
- a working set larger than the effective L2 capacity
- MLP levels of 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, and 1,024 per SM
- identical total bytes at each level

For the throughput version, issue a window of independent loads, retain their results, and consume them only after the window has been issued. Do not make it a dependent chain. Time the whole window and calculate:

\[
\text{bandwidth} =
\frac{\text{bytes transferred}}
{\text{kernel elapsed time}}
\]

Also retain the per-load timing data for a smaller subset of samples.

The curves distinguish several cases:

- **Latency-only poisoning:** poisoned bandwidth is poor at MLP 1 or 4, then approaches clean bandwidth at high MLP. The per-load tail remains, but out-of-order work hides it.
- **Replay or transaction-capacity poisoning:** bandwidth remains depressed even at MLP 1024, and the number of completed bytes per unit time stays low.
- **Translation/residency poisoning:** the curve depends strongly on page working set and stride, with a large transition when the number of pages exceeds some capacity.
- **SM-local pathology:** different SMs produce materially different curves.

Your existing BW and KRD arms already show that high-MLP streaming bandwidth is healthy. This new curve tells you where the latency is hidden and where it stops being hideable.

6. The WSL2/driver-level explanation is plausible, but I would rank the candidate mechanisms this way:

1. GPU VM, TLB, page-walk, residency, or replay state.
2. A GPU memory-fabric or partition scheduler state that preferentially hurts sparse transactions.
3. WSL2/WDDM/dxgkrnl memory-management activity causing the first large working set to enter an abnormal residency or mapping state.
4. A cache or address-partition conflict.
5. ECC handling, compression, or a static memory carve-out.

The first three fit the lifecycle:

- the first large prefill crosses a lazy-allocation or mapping threshold
- the state persists after the request
- later requests often reuse already-created mappings or pools
- the severity changes as queues, mappings, or resident pages evolve
- the probe changes the state without a restart
- streaming traffic can hide or amortize the affected path

A WSL2-specific memory-management issue is possible because CUDA inside WSL2 ultimately crosses a paravirtualized WDDM path. That does not mean `dxgkrnl` is the cause; it is one possible location for residency, mapping, or submission state.

A page-fault or residency explanation needs careful testing. If a small number of pages are nonresident, random 2 KB access will expose the faults, while a 52 MB sequential read can prefetch, overlap, and hide them. That would explain KRD remaining near 626 GB/s. Strong evidence would be:

- a first-touch penalty
- stable page identity for the slow samples
- sensitivity to page working-set size
- a sharp stride or page-size transition
- the same effect in both KV and fresh allocations

ECC scrubbing and compression fit less well. ECC problems usually leave more stable physical-address signatures and can affect streaming throughput or generate persistent hardware reports. Compression may be access-pattern dependent, but it does not naturally explain repeated millisecond-scale tails with recovery driven by an unrelated probe. A static memory carve-out cannot explain the state drifting over minutes.

One additional control has high value: compare passive elapsed time with active probe traffic. After poisoning, allow the same amount of wall-clock time to pass with no GPU work, then run a very small sentinel. Separately, run the active probe for the same duration and compare recovery. If only traffic recovers the state, “background drain” is probably the wrong description; the probe is changing residency, translation state, cache state, or the queue itself.

7. The highest-information next experiment is a **single tagged, per-access latency thermometer on the existing fresh buffer and KV buffer**, with repeated page IDs, MLP 1 and MLP 64 or 1024, and time-resolved tiles.

I would use roughly:

- 256,000 to 1,000,000 total 8-byte samples
- fixed tagged addresses from both allocations
- 16 or more passes over the same page-tagged set
- MLP 1 and MLP 64 in randomized tile order
- `%globaltimer`, `%clock64`, `%smid`, page tag, address tag, and tile number
- no allocation, no engine restart, no CUDA graph changes, and no large output atomics

This one measurement answers the most valuable questions simultaneously:

- Does a poisoned per-load tail actually exist?
- Does it occur in the fresh buffer?
- Is it attached to a small set of pages?
- Does high MLP hide it?
- Does the probe worsen the distribution over time?
- Is the effect global or SM-local?

The strongest falsifier for your mechanism would be:

> In the poisoned state, the fresh-buffer per-load latency CDF remains close to the clean CDF at MLP 1, including the high percentiles and maximum, while the 2 KB production-shaped kernel remains 10x slower.

For example, if p50, p99.9, p99.99, and the millisecond tail are all within roughly 20–30% of clean on individually timed fresh-buffer loads, then rare enormous memory-access latency is not sufficient. The next suspects would be QSA-specific warp replay, instruction issue, address-generation serialization, synchronization, or a software-pipeline interaction.

A second falsifier for the **global** version of the mechanism would be a result where nearly all poison-state tail events are attached to a small, repeatable set of page IDs. That would still support a memory-latency explanation, but the correct model would be “a few pathological pages or mappings,” rather than “the whole random-access path has acquired a heavy tail.”
