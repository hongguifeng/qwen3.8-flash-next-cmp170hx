The new result changes the weighting substantially. It weakens the generic “Windows WDDM memory management” hypothesis, but it strengthens a narrower hypothesis involving the host’s MCDM compute path together with WSL2 GPU paravirtualization, or an engine operation that crosses those paths.

Your result does not weaken the context scoped resource model. It only shows that resident allocation size plus one isolated gather kernel is insufficient.

**(a) MCDM versus WDDM**

There are three relevant environments:

| Environment | CUDA execution path | Host NVIDIA model |
|---|---|---|
| Native Linux reference | Native Linux kernel driver | Linux NVIDIA driver |
| WSL2 engine | Linux CUDA user space → `/dev/dxg` → dxgkrnl/GPU-PV → Windows host | MCDM |
| Native Windows torch | Windows CUDA user space → Windows KMD | MCDM |

Native Windows torch and WSL2 share the host MCDM driver, but they do not share the same CUDA-to-driver path. WSL2 adds dxgkrnl, VMBus/GPU-PV, Linux-side CUDA stubs, and a different context and memory-management boundary.

That means the current native Windows result excludes neither:

- a WSL2 GPU-PV/dxgkrnl problem;
- an MCDM problem exercised only through the GPU-PV path;
- a WSL-specific pinned-memory path;
- an engine-specific sequence involving graphs, host staging, and long-lived allocations.

It does make a purely WDDM-specific explanation less likely. WDDM’s graphics scheduling and residency behavior are no longer the best generic explanation because the affected host adapter is running MCDM. MCDM is intended for compute-only devices and removes much of the graphics-oriented scheduling path.

MCDM is still a credible suspect for a memory-management defect. It is a less mature and less commonly exercised path for massive single-context CUDA workloads, and your cross-context `cuMemGetInfo` observation is consistent with context-scoped accounting. That observation is not necessarily an API bug: CUDA memory-information APIs are not a reliable global physical-residency oracle under Windows memory management. Treat the result as evidence that the API cannot be used to infer total physical residency in this setup.

My assessment is:

| Suspect | Current weight | Reason |
|---|---:|---|
| Generic WDDM overhead | Low | The adapter is MCDM, and the failure is not ordinary host-to-device bandwidth loss. |
| MCDM memory-management defect | Medium | Context-scoped accounting and compute-only memory paths remain relevant. |
| WSL2 GPU-PV/dxgkrnl interaction | High | WSL is the only environment running the engine, and it adds a separate memory/context boundary. |
| Windows build or WSL kernel implementation | Medium | A dxgkrnl/GPU-PV protocol or accounting regression could be build-specific. |
| HAGS | Low | HAGS is primarily a graphics scheduling policy. It is a poor match for a persistent CUDA translation/replay resource failure on MCDM. |
| NVIDIA driver branch | Medium to high, conditional | Valuable if a portable synthetic workload reproduces the failure on native Windows or if WSL engine A/Bs implicate the driver path. |

The Windows-side A/B order should therefore be:

1. **NVIDIA driver branch**, using a supported driver that still supports the CMP 170HX and the same CUDA stack.
2. **WSL package/kernel version**, because this directly changes the GPU-PV/dxgkrnl side without changing the native Windows CUDA path.
3. **Windows build**, especially if the WSL package and driver are held constant.
4. **HAGS**, as a low-probability control rather than a leading experiment.
5. `.wslconfig` memory or swap settings only if host pinned-memory pressure or system-memory residency becomes visible.

HAGS should not be your expensive first A/B. The fact that `HwSchMode` is disabled is useful metadata, but changing it is unlikely to alter a device-only scattered-load throughput collapse.

The relevant architecture references are [Microsoft’s MCDM architecture](https://learn.microsoft.com/en-us/windows-hardware/drivers/display/mcdm-architecture), [GPU paravirtualization](https://learn.microsoft.com/en-us/windows-hardware/drivers/display/gpu-paravirtualization), and NVIDIA’s [CUDA on WSL guide](https://docs.nvidia.com/cuda/wsl-user-guide/).

**(b) Missing ingredients**

The most useful distinction is between an ingredient that creates more mappings or outstanding translation work and an ingredient that merely consumes memory. Your ballast test primarily tested the latter.

My ranking by expected value per unit cost is:

| Rank | Candidate | Value per cost |
|---:|---|---:|
| 1 | Allocation/mapping churn and deferred frees | Very high |
| 2 | Long logical sequence plus long-context execution path | Very high |
| 3 | Concurrent side-stream activity combined with pinned staging | High |
| 4 | Prefix/cache/state retention and allocator lifetime | High |
| 5 | CUDA graph pools and graph recapture | Medium |
| 6 | An otherwise unspecified driver-side context resource | High diagnostic value, high experiment cost |

The sixth item is better treated as the resource class being investigated than as a workload ingredient. The other items are possible ways the engine could fill or fragment that resource.

**1. Allocation churn and number of mappings — highest priority**

Ten large ballast allocations are a weak approximation of the engine’s allocation history. The engine may create many more allocation descriptors, segments, VA ranges, allocator blocks, graph-pool allocations, and transient workspaces even when the final resident byte count is similar.

A driver may maintain:

- per-context page-table or mapping metadata;
- VA range descriptors;
- translation-cache invalidation state;
- residency or eviction bookkeeping;
- replay queues associated with mappings;
- allocator segment metadata.

A high mapping count can therefore poison scattered accesses while leaving a dense contiguous copy nearly unaffected. This matches your observed pattern better than a simple bandwidth or capacity problem.

The cheapest portable test is a same-process allocation-count ladder:

1. Allocate the 64 MiB scatter source and destination.
2. Measure scatter and dense copies.
3. Allocate and free varied-size device tensors, for example 4 KiB through 16 MiB, while keeping total allocated bytes bounded.
4. Repeat at 1,000, 10,000, and 50,000 allocation/free operations.
5. Run `empty_cache()` at selected points, then measure again.
6. Separately retain the allocations instead of freeing them.
7. Report scatter/dense ratio against allocation count, not only against bytes.

Use the same CUDA C++ source on WSL2 and native Windows if possible. A raw CUDA Runtime or Driver API harness will remove PyTorch allocator differences. Keep the total device allocation small enough that this remains an allocation stress test rather than a memory-capacity test.

A particularly useful variant is:

- many small allocations;
- one large allocation split into tensor views;
- PyTorch caching allocator reuse;
- explicit free and cache release;
- CUDA virtual-memory API mappings.

If only the first or last case triggers the failure, you have localized the missing operation.

**2. Long-context execution path and one logical sequence**

A 96K request does more than allocate 96K worth of bytes. It causes the engine to maintain one sequence with a long history and to execute operations whose metadata, index ranges, offsets, recurrent state, and KV references depend on the accumulated context.

The local source confirms several relevant paths:

- chunked prefill at 2,048 tokens;
- a maximum logical context set independently of the chunk size;
- PLE prompt read-ahead over up to 16,384 remaining tokens;
- Mamba/GDN state handling;
- CUDA graph execution around host breaks;
- PLE IDs and staging buffers sized by `max_total_tokens`.

The relevant launcher settings are in [scripts/serve.sh](/home/hong/code/qwen3.8-flash-next-cmp170hx/scripts/serve.sh:20) and [docs/GUIDE.md](/home/hong/code/qwen3.8-flash-next-cmp170hx/docs/GUIDE.md:81).

The key point is that chunking does not reset the logical sequence. A 96K request processed as 2K chunks can still exercise a long-context path. Forty-eight independent 2K requests may never create the same metadata lifetime, history-dependent indexing, or persistent allocator state.

The cheapest portable test is a toy long-context harness with the same logical distinction:

- one sequence with lengths 2K, 16K, 32K, 64K, and 96K;
- process every sequence in 2K chunks;
- maintain a persistent history buffer and index metadata;
- perform the same scattered device reads against the accumulated history;
- compare it with 48 independent 2K sequences using the same total number of tokens and approximately the same total bytes;
- measure after every chunk and after the sequence completes.

Do not make this a large attention benchmark. The purpose is to reproduce the lifetime and indexing pattern, not model throughput.

A stronger engine-side version is to disable one long-context feature at a time:

- disable PLE prompt read-ahead with `QWEN_SSD_PREFETCH=0`;
- disable the complete PLE SSD path;
- disable prefix caching;
- disable MTP;
- use eager execution;
- retain the normal 2K chunk size.

The first two are especially informative because `QWEN_SSD_PREFETCH=0` removes only prompt read-ahead, while disabling PLE removes the demand-read and pinned-staging path as well.

**3. Concurrent side-stream activity and pinned staging**

Pinned memory is a plausible contributor, but the mechanism needs to be stated carefully.

The local PLE implementation creates:

- pinned CPU ID storage;
- a device ID buffer;
- pinned CPU BF16 staging storage;
- a device BF16 staging buffer;
- a separate CUDA stream;
- multiple CUDA events;
- asynchronous device-to-host and host-to-device copies.

The relevant code is in [patches/qwen38-ple-ssd.patch](/home/hong/code/qwen3.8-flash-next-cmp170hx/patches/qwen38-ple-ssd.patch:1117). The sequence copies IDs from the graph-produced device buffer to pinned host memory, lets the host path fill pinned BF16 storage, and then copies that storage back to a device buffer on a separate stream. The implementation explicitly synchronizes those buffers with events at lines 1186–1207.

Pinned host pages normally participate in DMA mapping and GPU-visible address translation. Under WSL2, that path also crosses GPU-PV. A large or frequently registered pinned working set could consume per-context mapping metadata or create invalidation pressure.

That could affect device-only accesses if the driver shares translation bookkeeping, invalidation queues, or replay resources between host-mapped and device-local mappings. It is less likely if the only effect is ordinary copy-engine DMA. A contiguous H2D transfer should not by itself make an SM load from ordinary device memory 225 times slower.

The cheapest portable test has two variants:

1. Allocate persistent pinned buffers at 0, 64, 256, and 1,024 MiB.
2. Have a CPU worker fill them while a side CUDA stream performs nonblocking H2D copies.
3. Run the device-only random-row gather on the main stream during and after that activity.
4. Compare against pageable host buffers, no side stream, and pinned buffers that are freed before the gather.
5. Repeat with several copy sizes and with the copies continuously in flight.

Use `torch.empty(..., pin_memory=True)` only for an initial test. A raw CUDA test using `cuMemHostAlloc` and, where supported, `cuMemHostRegister` is better because it tests the actual CUDA registration APIs.

The exact Linux native AIO implementation cannot run identically on native Windows; it uses Linux `io_submit` and Linux ABI structures. Therefore, native AIO depth 256 should not be treated as the portable ingredient. The portable ingredient is:

> CPU producer plus persistent pinned host buffers plus asynchronous H2D traffic plus a device-only gather in the same context.

A second diagnostic test can use `cudaHostAllocMapped` and let a kernel read host-mapped memory directly. That does not exactly match the engine, but it determines whether the host-mapped address class itself has unusual behavior on either platform. If WSL rejects or materially changes that API, that is itself useful evidence.

**4. Prefix cache, deferred frees, and state lifetime**

The launcher enables prefix caching and aligned Mamba state caching. A long request can leave behind:

- cached KV blocks;
- sequence metadata;
- recurrent state allocations;
- allocator segments that are technically reusable but still mapped;
- graph-pool references held by Python or graph objects;
- delayed stream-ordered frees.

This candidate is closely related to allocation churn, but it is cheap to test separately.

Run the same 96K workload with prefix caching disabled and compare:

- the first slow gather;
- the number of allocations;
- the number of retained bytes after the request;
- whether a second request gets slower;
- whether `empty_cache()` changes anything.

A useful control is to insert a full device synchronization and allocator cleanup after each 2K chunk in a test build. If that changes the threshold, lifetime and deferred-release behavior are implicated even if the final resident byte count is unchanged.

**5. CUDA graph pools and recapture**

The launcher captures many graph sizes through 2,048 tokens and uses breakable graphs around the PLE host breaks. The source creates a global graph pool and uses `BreakableCUDAGraphCapture` when PLE offload is active; see [patches/qwen38-ple-ssd.patch](/home/hong/code/qwen3.8-flash-next-cmp170hx/patches/qwen38-ple-ssd.patch:656).

Graph pools can retain allocations and VA mappings for the lifetime of the graph executable. Recapture can add more pools or more address ranges. This is a plausible amplifier, especially if the graph capture and PLE host-break code cause allocations to be created on multiple streams.

It is a lower-ranked primary cause because:

- graph capture sizes are configured at startup;
- the default sizes stop at 2,048 tokens;
- your trigger is tied to long accumulated context;
- graph replay itself usually reuses addresses rather than mapping new memory on every replay.

The cheapest portable test is a graph-count ladder:

- 0 graphs;
- 1 graph;
- 8 graphs;
- 32 graphs;
- repeated capture and destruction;
- one shared graph pool versus separate pools.

Each graph should contain only small allocations, a copy, and a lightweight kernel. Measure the device-only gather after capture, after replay, and after graph destruction.

The corresponding engine A/B is `QWEN_EAGER=1`. If eager mode prevents the collapse, then graph pools or graph/host-break interaction move near the top of the list. If eager mode does not change it, graph capture becomes less likely.

**6. A driver-side context-scoped resource**

This remains fully compatible with the negative reproducer.

A resource can be:

- per CUDA context;
- owned by the process’s primary context;
- invisible to a fresh process;
- reset only when the context is destroyed;
- exhausted by a sequence of operations rather than by final resident bytes.

Your fresh-process result is therefore expected under this model. A fresh process is testing a new resource namespace.

The cheapest same-platform control is a single-process reset test:

1. Create one CUDA context.
2. Run the workload or synthetic stress.
3. Measure gather degradation.
4. Destroy the context.
5. Create a fresh context in the same process.
6. Repeat the gather.
7. Compare with a second process.

A raw Driver API test is preferable because it can explicitly retain and release primary contexts. PyTorch’s CUDA primary-context behavior can otherwise obscure the boundary.

This test identifies context scoping, but it will not identify the exact resource. For that, use the staged synthetic tests above and a virtual-memory mapping stress.

**A useful added test: CUDA virtual-memory mapping churn**

The closest portable synthetic stress to a translation-resource hypothesis is the CUDA virtual-memory API:

- query allocation granularity with `cuMemGetAllocationGranularity`;
- reserve a large virtual address range;
- create many small physical allocations;
- map and unmap them at distinct virtual addresses;
- run the same sparse gather before and after;
- repeat with persistent mappings and with map/unmap churn;
- destroy the mappings and measure recovery.

This crosses a more direct VA-management path than ten ordinary `cudaMalloc` calls. It should be implemented in a small C++ Driver API program and run with the same source and architecture-targeted kernel on WSL2 and native Windows.

Keep the first run bounded, such as 4,096 to 16,384 mappings, and avoid a large arithmetic workload. This is an address-management experiment, not a throughput benchmark. If the API is unsupported or has a different allocation granularity on one platform, record that result rather than trying to force equivalence.

**(c) Cheapest decisive platform experiment**

Use a three-stage decision tree.

The first stage is a raw CUDA C++ harness that runs byte-for-byte the same source on WSL2 and native Windows. Avoid PyTorch for this test. It should contain these independent modes:

| Mode | Operation |
|---|---|
| Baseline | Device-only dense copy and random-row gather |
| Allocation | Vary allocation count while holding total bytes approximately constant |
| Pinned | Persistent pinned buffers and side-stream H2D copies |
| Graph | Capture and replay a controlled number of graph pools |
| VMM | Reserve/map/unmap many device virtual-memory ranges |
| Long sequence | Repeat 2K chunks against one growing logical history |
| Combined | Allocation churn plus pinned side-stream activity plus sparse gather |

Record:

- median and p99 gather time;
- dense-copy time;
- scatter/dense ratio;
- time and allocation count at the point of degradation;
- whether the effect survives stress teardown;
- whether a new context resets it;
- host and device memory usage if available.

Interpretation:

| Result | Strongest interpretation |
|---|---|
| WSL2 fails, native Windows remains flat | WSL2 GPU-PV/dxgkrnl or WSL CUDA path |
| Both fail at similar mapping/copy thresholds | Host MCDM/NVIDIA driver or hardware-level resource |
| Neither fails, but the engine does | Engine sequencing or a combined operation still missing from the harness |
| Pinned mode alone fails only under WSL2 | GPU-PV host-memory mapping path |
| VMM mode fails on both | Translation/VA resource hypothesis becomes strong |
| Allocation count fails while byte ladder remains flat | Mapping metadata or allocator churn |
| Graph mode fails only with pinned side-stream mode | Graph/host-break/stream interaction |

The second stage is a small set of WSL engine ablations. These are more decisive than further ballast experiments because they run inside the context that actually becomes poisoned.

Use a fresh engine process for each trial and change one feature at a time:

1. Full configuration.
2. `QWEN_EAGER=1`.
3. `QWEN_SSD_PREFETCH=0`.
4. PLE SSD disabled entirely.
5. `VLLM_WSL2_ENABLE_PIN_MEMORY=0`.
6. Prefix caching disabled.
7. MTP disabled.
8. Allocation/pinning combinations that isolate the first positive result.

The most informative first three are eager mode, PLE read-ahead disabled, and PLE entirely disabled. `QWEN_SSD_PREFETCH=0` does not remove the demand-read pinned staging path, so it distinguishes prompt read-ahead from the basic PLE transfer path.

The third stage is the Windows-side host observation during one healthy and one poisoned run.

Useful read-only sources of evidence include:

- GPUView/WPR traces from the `Microsoft-Windows-DxgKrnl` provider;
- Windows GPU Engine and GPU Process Memory performance counters;
- dedicated/local adapter memory and shared memory counters;
- NVIDIA process memory and utilization queries;
- CUPTI or Nsight Systems activity traces for CUDA contexts, kernels, copies, and stream overlap;
- timestamps around the long prefill and the first slow gather.

GPUView and dxgkrnl ETW can show:

- process and context activity;
- DMA packet submission;
- queue stalls;
- context switches;
- residency changes;
- allocation and paging events, where exposed;
- host/device transfer timing.

They are unlikely to expose the internal NVIDIA SM TLB, replay queue, or translation-cache occupancy directly. A flat ETW residency trace therefore does not falsify a private translation-resource failure. It can still distinguish a paging/residency problem from a kernel-side throughput problem.

WMI/PerfMon GPU counters are useful for host memory and residency correlation, but they will not give you a direct “translation resource used” counter. There is no documented Windows registry knob that sizes NVIDIA translation or replay resources. Avoid undocumented driver patches, TCC emulation, or registry changes intended to disable paging; they would destroy the value of the comparison.

**(d) Interpretation of the negative reproducer**

Your interpretation is fair, with one important qualification.

It refutes:

> A fresh process with a large resident device working set and the same scattered short-row access pattern is sufficient to cause the collapse.

It does not refute:

> A particular CUDA context accumulates a resource deficit after a specific sequence of allocations, mappings, host registrations, graph captures, stream interactions, or long-context kernels.

It also does not prove that the ballast allocations were physically resident in the same way as the engine’s allocations. The cross-process `cuMemGetInfo` result is evidence that the API’s free-memory view is context-scoped or virtualized under this configuration. A 60 GiB allocation ladder in a fresh process can therefore be a poor test of global physical residency while still being a valid test of that process’s ordinary device allocation path.

The strongest negative control to run before downgrading the context-resource hypothesis is:

1. Keep the entire test in one process and one CUDA context.
2. Use the actual engine context or a debug hook inside the engine process.
3. Reproduce the allocation and host-staging sequence before running the gather.
4. Include allocation-count and VMM mapping ladders, not only byte ladders.
5. Force-touch each allocated page before timing.
6. Measure during the stress, immediately after stress, after `empty_cache()`, and after context destruction.
7. Compare one long logical sequence with independent 2K sequences.
8. Compare with pinned side-stream copies active and fully drained.

If a standalone one-context program with mapping churn, pinned side-stream traffic, graph pools, and a growing logical sequence remains healthy on both platforms, the engine-code explanation becomes the leading one. The likely missing unit would then be the exact interaction between long-context model kernels, allocator lifetime, PLE host breaks, and the engine’s CUDA graph state rather than “resident memory plus gather.”

The most efficient current plan is therefore:

1. Build the same raw CUDA harness for WSL2 and native Windows.
2. Add allocation-count, pinned-side-stream, graph-pool, and VMM modes.
3. Run WSL engine A/Bs for eager mode, PLE read-ahead, PLE offload, and prefix caching.
4. Only if those results implicate the platform layer, spend the cost of a driver-branch or WSL/Windows-build A/B.
5. Use ETW/GPUView to classify residency and scheduling behavior, while treating the internal translation/replay resource as observable only indirectly.
