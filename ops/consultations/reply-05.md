The evidence places the fault at the GPU execution and scheduling boundary around the attention workload. The QSA data, physical placement, values, launch mode, host submission, SM clock, and tensor-core throughput have all been substantially isolated.

The most decisive next measurement is a **single execution of the diagnostic kernel with per-CTA `%globaltimer` start/end timestamps, plus `%smid` if possible**. That directly separates time spent waiting for CTAs to run from time spent executing inside CTAs.

Subjective prior ranking, conditioned on your observations:

1. **Residual GPU work, stream scheduling, or a queued dependency backlog** — highest probability.
2. **Shared memory-system pressure**: global-memory request queues, TLB/page-walk/replay state, L2/DRAM partition pressure, or interconnect backpressure.
3. **Reduced effective SM residency caused by concurrent work or scheduler interference**.
4. **CUDA-graph or driver scheduling state amplifying the condition**.
5. **WSL2/NVIDIA driver context-level scheduling defect**.
6. **A changed Triton binary or static occupancy calculation** — low probability under the stated facts.
7. **Clock, thermal, or general hardware failure** — very low probability given the GEMM result and clock telemetry.

These categories overlap. For example, residual graph work can create reduced effective occupancy and memory-system pressure simultaneously.

The first explanation fits the strongest facts: the first large prefill creates the condition, later large prefills do not reproduce it, draining progressively clears it, and an external GEMM is healthy in short bursts while being starved for seconds. Blocking the engine thread only prevents new submissions; it does not necessarily remove already queued graph work, outstanding memory operations, stream dependencies, or resident/interleaved GPU work.

The 4096³ GEMM proves that tensor-core compute throughput is available. It does not prove that attention gets normal memory request service or normal CTA residency. A compute-heavy GEMM can complete at 190 TFLOPS during a period in which a memory-heavy attention kernel suffers long gaps between CTA waves.

The graph results should be interpreted as evidence of a changing state rather than as a stable graph-versus-eager speed difference. The fact that `OE` falls from 21.8 ms to 7 ms while `OG` and `RG` are already near 2 ms shows that the state is changing during the probe sequence. The 14.1 ms `RG` result during the fully bad measurement may also reflect a graph-specific scheduling path, but it is not enough to establish that graph replay intrinsically makes the kernel 1.5× faster.

For the next measurement, compile one minimal diagnostic variant with the same grid, specialization, `num_warps`, `num_stages`, operands, and launch dimensions. Add:

- One 64-bit start timestamp per CTA.
- One 64-bit end timestamp per CTA.
- One `%smid` value per CTA, if inline PTX is practical.
- No per-thread trace and no extra barriers.

Read `%globaltimer` once at the earliest useful point before the first K/V load, and once after the existing computation immediately before the CTA returns. Store only from one lane per CTA. The trace buffer is tiny compared with the 198 MiB scratch buffer, so the trace traffic itself should be negligible.

Use the eager private-stream invocation first. Run exactly one diagnostic execution in each condition:

1. A clean-state execution using the diagnostic binary.
2. The first execution after the normal large-prefill trigger, with the probe disabled during triggering and armed only after the bad state is established.

Do not use three diagnostic repetitions for this measurement. Your own results show that repeated probing changes the state.

Record the CUDA-event duration around the diagnostic kernel as `T_evt`. For CTA `i`, record:

```text
D_i = end_i - start_i
T_first = min(start_i)
T_last  = max(end_i)
T_env   = T_last - T_first
```

Also form the union of all CTA intervals:

```text
U = length of union of [start_i, end_i] for every CTA
```

This gives useful quantities:

```text
head_tail = T_evt - T_env
internal_no_CTA_time = T_env - U
total_no_CTA_time = T_evt - U
median_D = median(D_i)
p95_D = 95th percentile(D_i)
```

`T_evt` and the `%globaltimer` values should be compared in nanoseconds after checking their scale on a clean run. Fixed event overhead is insignificant at 2.2–21.8 ms, but use ratios and fractions rather than claiming nanosecond accuracy.

For the requested envelope view, normalize every CTA interval to the first and last CTA timestamps:

```text
x_start_i = 100 * (start_i - T_first) / T_env
x_end_i   = 100 * (end_i   - T_first) / T_env
```

Plot or tabulate each CTA as an interval from `x_start_i` to `x_end_i`, once for clean and once for bad. Sort once by CTA ID and once by start time. The start-time ordering matters because bad-state scheduling may change CTA launch order.

The key comparison matrix is:

| Bad-state observation relative to clean | Interpretation |
|---|---|
| `T_evt` grows by about 10×, but `median_D` and `p95_D` stay within roughly 1.0–1.5× | Individual CTAs still execute normally. The lost time is outside CTA execution. |
| `T_evt` grows by about 10×, `T_env` remains close to clean, and `head_tail` accounts for most of the difference | Delay before the first CTA or after the last CTA. This points to stream scheduling, a queued dependency, launch admission, or a tail synchronization. |
| `T_evt` grows by about 10×, `T_env` also grows, `D_i` remains near clean, and `T_env-U` is large | CTAs execute at normal speed, but there are long gaps with no CTA active. This is reduced effective residency, scheduler starvation, or interleaving with other GPU work. |
| `T_evt` grows by about 10×, `median_D` and `p95_D` grow by a similar factor, while the interval pattern and active-CTA overlap remain similar | The CTAs themselves are slowed in flight. Investigate memory-system pressure, translation/replay behavior, or a workload-specific SM-side stall. |
| `median_D` grows, but only for CTAs on particular `%smid` values or in a subset of launch waves | Per-SM, memory-partition, or concurrent-work interference. This is stronger evidence for scheduling or partition-local resource contention than for a global kernel slowdown. |
| `D_i` has two distinct populations, with one population near clean and one much slower | Intermittent interference, wave-level overlap, or SM-local contention. Compare the slow population against `%smid` and start time. |
| `T_evt` and `T_env` are both large, but `U` is close to `T_env` | CTAs are continuously in flight. The dominant problem is inside CTA execution, not a no-CTA scheduling gap. |
| The trace binary’s cubin/hash or compile metadata differs between runs | A code-object or JIT-cache issue remains possible. Compare registers per thread, static shared memory, local memory, warps, stages, and the cubin hash. |
| The diagnostic binary and compile metadata are identical, but active CTA overlap is lower in bad state | Static occupancy did not change. Effective occupancy or co-residency changed at runtime. |

For your current numbers, the two clean hypotheses make very different predictions:

- **Outside-CTA delay:** production-like `T_evt` changes from approximately 2.2 ms to 21.8 ms, but `D_i` remains around 2–3 ms. You would see long blank portions in the normalized envelope or a much smaller number of overlapping intervals.
- **In-flight CTA slowdown:** `D_i` itself moves toward roughly 15–22 ms, while CTAs remain continuously active over most of the event interval.

That distinction is much stronger than another relocated-buffer or graph/eager arm because those arms already establish that the slowdown follows the GPU’s current state rather than the selected address or data.

The `%smid` result is useful for the occupancy question. If possible, derive the number of simultaneously active traced CTAs per SM from the intervals. A changed theoretical occupancy cannot occur for the same cubin and the same launch attributes. A changed **effective** occupancy can occur if other streams retain resident work, if the driver temporarily restricts admission, or if the scheduler interleaves the kernel differently.

If the first trace shows that CTA execution itself is slow, add four phase markers in a subsequent clean/bad pair around the major QSA stages: index/table processing, K/V load and selection, reduction/attention arithmetic, and output/store. Compare:

```text
phase_ratio_k =
    bad_phase_duration_k / clean_phase_duration_k
```

A large ratio confined to the load/selection phase supports memory-system or translation/replay pressure. A roughly uniform ratio across all phases supports a broader in-flight scheduling/resource problem. A ratio concentrated around barriers or final stores supports synchronization or completion behavior.

A changed binary is the least likely outcome. If the same cubin and static attributes are used, the binary’s theoretical occupancy is fixed. The behavior you are seeing can still come from dynamic residency, queued work, or memory-system state, all of which can clear progressively without recompilation.
