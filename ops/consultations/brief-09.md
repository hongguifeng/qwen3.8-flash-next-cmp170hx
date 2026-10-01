# Consultation #9 — after the first portable minimal reproducer FAILED to reproduce (both WSL2 and Windows-native)

You are helping on a debugging project: a vLLM engine (Qwen3.8-Flash-Next, hybrid GDN+QSA+MoE+PLE-SSD+MTP) running in Docker inside **WSL2** on Windows 11 24H2 with an NVIDIA CMP 170HX (cc 8.0, 64 GiB, 74 SMs).

## Established engine-level facts (unchanged)
A single >=96K-token prefill request puts the engine's CUDA context into a persistent "poisoned" state:
- a *scattered short-row* access pattern (e.g. `dst.copy_(src[idx])`, 16384 rows x 2 KiB, 64 MiB src footprint) goes from **0.175 ms to 39.68 ms (~225x)**;
- a *dense contiguous* copy of **the same buffers at the same instant** is **1.0x** (0.091 ms);
- 8 freshly allocated 64 MiB buffers at 8 VAs spanning 48 GiB: **all exactly equally slow (39.68 ms)**;
- the production QSA kernel is only **10.6x** slow; a per-access latency thermometer (MLP 1..1024, 5 arms x 2 buffers) shows the **latency CDF is essentially unchanged** (ratios 1.05-2.7x, **zero samples >10 us**, max 5.5 us) => "rare enormous per-access latency" is falsified; it is a *throughput/translation-resource* effect;
- grid/CTA-count sweep, VA-region/physical-block/allocation identity, data values, kernel identity, placement table values, launch parameters (byte-identical QSALOG), ECC/row-remap/throttling, driver error paths (zero Xid/TDR/reset, 0 dmesg messages) are ALL excluded;
- **a fresh process is immune while the engine is poisoned**; only ctx/process destruction resets; chunked 2048-token prefill does NOT prevent it; the trigger needs an accumulated long context; the effect gets worse with use.

## NEW result #1 — the minimal portable reproducer does NOT reproduce, on either platform
I wrote a pure-torch reproducer (`--gb` ladder of ballast, re-measuring a 16384x2 KiB random-row gather over a 64 MiB src vs a dense copy) and ran it with the engine stopped (62.5 GiB physically free) on:
- **WSL2 (Linux guest, same driver stack as the engine)**: ladder 0->60 GiB resident: scatter **0.160-0.187 ms (179-209 GB/s)** at EVERY step, dense 0.093-0.112 ms. No degradation. After free+empty_cache: 0.166 ms.
- **Windows-native (torch 2.14.0+cu126, same GPU via the native KMD path)**: ladder 0->60 GiB resident: scatter **0.163-0.182 ms (184-206 GB/s)** at every step, dense 0.092-0.112 ms. No degradation.

=> "grown resident mapped working set + scattered short rows" in a fresh process is **NOT sufficient** to reproduce the engine's collapse. The trigger requires something the engine does that my reproducer does not, and I need to find a *portable* version of that ingredient so I can dichotomize WSL2 vs Windows-native.

## NEW result #2 — `cuMemGetInfo` over-reports free memory by ~60 GiB under this platform
- Engine holding 61.5 GiB (windows nvidia-smi: used 61657 MiB, free 3446 MiB of 65536).
- A **fresh process inside the same WSL container** reports `mem_get_info() = free 62.5 GiB / total 64.0 GiB`.
- **Windows-native** torch reports the same `free 62.58 GiB / total 64.0 GiB` while WSL holds 61.6 GiB.
- A small 2 GiB allocation while physically only 3.4 GiB was free **succeeded** on both.
=> On MCDM the free-memory accounting is per-context / unaware of other contexts (or over-commit/eviction is in play). I flag this because vLLM sizes its KV pool from such an API.

## NEW result #3 — platform identity
- Windows 11 24H2 build 26100.4061, Windows KMD **616.92**, **Driver Model = MCDM (Microsoft Compute Driver Model)** for both Current and Pending (no WDDM graphics driver for this compute-only card, no TCC fields at all anywhere in nvidia-smi; TCC is impossible/unavailable and NVIDIA documents WSL CUDA as WDDM-only).
- HAGS (`HwSchMode`) = 0x1 = **disabled**. GSP N/A. Compute Mode Default. GPU Virtualization Mode None.
- The project's own reference environment (`docs/GUIDE.md`) is **native Linux x86-64 with driver 610.43.03**, PyTorch 2.13.0, Humming kernels 0.1.12 — i.e. the fast reference numbers were NOT measured under WSL2.
- vLLM has no native Windows support in this build (Linux-only Triton QSA kernels, a Linux-only `ple_ssd_io.so`, CUDA-graph capture), so "run the engine natively on Windows" is not an option.

## What I want from you (be concrete, quantitative, and rank by expected value per unit cost)
**(a)** Does the **MCDM vs WDDM** distinction change your assessment from consultation #8? For a per-context translation/replay-resource failure, is MCDM a stronger or weaker suspect than WDDM, and does it change which Windows-side knob (driver branch, HAGS, WSL settings, Windows build) is worth a costly A/B?

**(b)** Given the negative reproducer, enumerate the candidate **missing ingredients** that a >=96K-token prefill in this engine exercises and my ballast+gather reproducer did not, and rank them. For each: the mechanistic reason it could produce a *context-scoped, scattered-access-only, translation-resource-like* collapse, plus the **cheapest minimal test** that runs identically on **WSL2 and Windows-native**. Please explicitly evaluate at least these candidates and any you add:
  1. **pinned host memory / GPU-PV host-mapped staging** (this engine has a WSL2-specific env switch `VLLM_WSL2_ENABLE_PIN_MEMORY=1` worth 30-35% performance; the PLE-SSD path allocates `pin_memory=True` staging tensors and issues native AIO at depth 256; NVIDIA documents WSL pinned-memory limitations). Could GPU access to *host-mapped* pages be the translation resource, and why would that then slow access to *device-only* buffers?
  2. **allocation churn / number of new mappings** (KV block allocator churning ~48 chunks x many blocks, vs my 10 big ballast allocations).
  3. **cudagraph pool re-capture / graph memory pools**.
  4. **a long-context-specific execution path** (indexer workspaces sized by context length, PLE prefetch_tokens=16384, FLA/GDN chunked kernels at 98K history).
  5. a **driver-side per-context resource that only *that* process's context accumulates** (which my fresh-process reproducer structurally cannot see, and which a *fresh* context is immune to by definition).
  6. Anything a >=96K single request does that 48 x 2048-token requests do not.

**(c)** If candidate (b)-1 or (b)-5 is right, my reproducer approach may be fundamentally unable to discriminate. Propose the **cheapest genuinely decisive experiment** to separate "WSL2/GPU-PV layer" from "Windows KMD/MCDM layer" from "engine code" — including any *synthetic* stress (e.g. a standalone CUDA/pytorch program that deliberately exhausts a translation-class resource) that could cross the same resource on both platforms, and any *read-only* Windows-side observability (dxgkrnl counters, ETW/GPUView, WMI, driver registry knobs) that could show per-context translation/replay resource usage for comparison.

**(d)** Sanity-check my interpretation of the negative reproducer: is it fair to say it *only* refutes "resident size + gather pattern is sufficient", and NOT the context-scoped-resource model? Is there a *stronger* negative control I should run before concluding?

## Constraints
READ-ONLY: do not restart, stop, or reconfigure the container `hong-pc`; do not run GPU-heavy benchmarks; inspect configuration/source/logs read-only. Answer in English, concrete and quantitative, with a ranked plan.
