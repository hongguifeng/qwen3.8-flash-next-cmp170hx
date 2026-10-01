# Consultation #4 — "poison" root cause: I have localized it to SM stalls in the engine's own prefill kernels; I need ONE more discriminating experiment

You advised me three times before on this vLLM "poison". Thanks to your corrections I now have a much tighter picture. I am stuck on the *mechanism* (why identical kernels stall ~7x) and on choosing the next measurement. Please answer with (1) a ranked mechanism list, (2) ONE concrete experiment feasible with the tools I list, (3) whether the block-id shift can have a real mechanism.

## Environment (fixed, cannot change)
- WSL2 (Windows host), single CMP 170HX (GA100, cc 8.0, 74 SM, 64 GiB HBM2e, PCIe gen2 x8). GPU is SHARED with the Windows desktop.
- Docker image with vLLM 0.29.1rc1 (custom qwen4_exp build), torch 2.13.0+cu130. Qwen3.8-Flash-Next AutoRound 3bpw + MTP=1, PLE-SSD offload (95 GiB PLE tables on SSD).
- Serve config: max_model_len 262144, max_num_batched_tokens 2048 (chunk), max_num_seqs 4, gpu_memory_utilization 0.96, prefix caching, chunked prefill, FULL_AND_PIECEWISE cudagraphs, mamba_cache_mode=align.
- VRAM: weights 47.32 GiB, cudagraph pool 2.23 GiB, misc ~2.7 GiB, **KV pool only ~9.2 GiB**, total 61.4/64 GiB (measured: a 96K request needs ~2.9 GiB KV; max_model_len 262144 needs 7.25 GiB KV). `gpu_memory_utilization=0.92` and below fails at startup (not enough KV), so the usable window is 0.93–0.96. **No free VRAM while the engine runs**, so I cannot allocate a big buffer from an external process to reproduce.
- Profilers: torch.profiler -> CUPTI_ERROR_CMP_DEVICE_NOT_SUPPORTED(42); nsys -> import error; ncu -> ERR_NVGPUCTRPERM; perf_event_open -> Fail. **CUDA events work. External CUDA context (separate process) works. Container restarts cost ~6 min.**

## The phenomenon (all numbers I re-measured this session)
Trigger: ONE prompt of >=64K–96K tokens (chunked). Then:
| fresh prompt (new random ids) | clean | poisoned |
|---|---|---|
| 2048 | 0.646–0.681 s (3172 tok/s) | 1.969–2.128 s (1040) |
| 512 | ~0.22 s | 0.714 s |
| 8192 | ~2.39 s | 10.068 s |
| 32768 | ~10 s | 44.152 s |
| 98304 (the trigger itself) | 49.969 s (1967 tok/s) | (2nd run also ~56–64 s, i.e. NOT slowed) |
| decode (28-token prompt, 64 tokens) | ~19.9–21.3 ms/tok | 21.1–21.2 ms/tok (unaffected) |
Cleared ONLY by restarting the container (fresh CUDA context). Persists >=10 min idle. Not prefetch-related: `ple_ssd_prefetch_tokens=0` A/B did not fix it (and costs 24% clean).

## What is measured inside the engine (CUPTI-free: `external=True` CUDA events baked into captured graphs + host timers)
For the SAME 2048-token fresh request, same container, before/after the trigger:
- **QSA launch parameters are IDENTICAL (byte-for-byte) clean vs poisoned**: `_qsa_sparse_paged` prefill: nq=2048, sel_w=2051, n_tiles=65, n_splits=1, warps=1, block_n=32, cnt_max=2048, cnt_sum=2.098e6, ti_max=2047; indexer PLOG_PREFILL nq=2048, w(=logits width)=512, grid g1=32, g2=1, vis_max=512, vis_sum=5.238e5; TOPK w=512; METADATA tok=2048 seq_max=2048 vis_sum=1.311e6. (The big values w~2.46e4, vis_sum~4.09e7, ti_max~9.8e4, cnt_max=2050 belong to the 96K trigger's long-context chunks, not to poisoned short requests.)
- **The only state difference I can find**: physical KV block ids of the fresh request. clean `bt0_max=9.33[2..13] bt0_min=3.33 bt0_jumps=31.3`; poisoned `bt0_max=42.5..50[17..83] bt0_min=1..8.5 bt0_jumps=1.5..2` (i.e. poisoned blocks are CONTIGUOUS but sit at higher pool indices).
- Cost distribution, same-shape 2048-token target prefill, per-layer in-graph events (ms): sum_gpu 679.6 -> 2613.0; **QSA ATTENTION per layer 2.2 -> 30.0 (x13.6, uniform over all 12 QSA layers)**; ATNQ total 85.8 -> 784.0; **GDN (gated-delta-net chunked linear attn) 422.0 -> 1588.5 (x3.8)**; MoE 134.5 -> 197.2 (x1.5); PAGED_LOGITS / TOPK / PRE_INDEXER ~0.0–0.5 unchanged; sum_host (host launch path inside the replay loop) 647.7 -> 2488.0. MTP draft prefill (nl=1, nseg=3): 13.1 -> 94.2 ms, its single attention kernel 2.2 -> 30.0 ms.
- All 103 pipeline segments inflate uniformly ~7x; GDN_FUSED (the fused core) is UNCHANGED while the whole GDN module is 3.8x.
- **Clocks/power, 10 Hz sampling inside the request window (new this session)**: trigger (clean): SM 1461 MHz busy-avg, 136 W busy-avg, 301 W peak, `0x4 SW Power Cap` active in 97/602 samples. **Poisoned fresh 2048: SM 1477 MHz avg (peak 1485), 105 W avg, 158.8 W peak, ZERO throttle flags.** => full clock + low power + slow = SM stalls, not throttling, not a capability drop.
- Independent probes (separate CUDA context, run in both states, ~300 MiB footprint): bf16 8192^3 156–200 TFLOPS, streaming copy 1541–1568 GB/s, 4/16/64 KiB page-scattered gather 316–329 GB/s, random element gather 47–58 GB/s — **identical clean vs poisoned**. So the raw device memory path is healthy for freshly-allocated buffers.
- PLE/SSD (per-layer host-side prefetch chain: ids D2H -> SSD read -> H2D -> graph segment): device AIO time `io` unchanged (191 ms vs 209 ms per 2048-token chunk). Worker `W_ids_sync` and main-thread `M_wait_pending` per layer equal exactly one segment GPU time (clean 5.4 ms/layer vs segment 5.9; poisoned 38 ms/layer vs segment 30–42) => the PLE waits are a consequence, not a cause. Row cache LRU saturated (1.2M rows) in both states.
- Per-token excess: ~+20–26 us per token per layer for 512/2048/8192/32768 fresh requests; a 32-token prefill is unaffected; the 96K trigger's own chunks are unaffected.

## Structural facts you may need
- 48 decoder layers = 36 GDN (linear attn) + 12 QSA; QSA prefill path runs through `@eager_break_during_capture` (so the prefill graph is a 103-segment "breakable cudagraph" with eager PLE breaks between segments), decode runs inline in 5-segment graphs.
- QSA: `build_qsa_metadata` (Triton), `_qsa_pre_indexer` (reduce/store, not atomic), `persistent_topk` (accepts only k=512/1024/2048; per your earlier note it has inter-CTA spin waits; its cooperative-radix path needs compressed candidates > 32768 = 131072 logical tokens, not reached at 96K), `expand_qsa_block_indices`, `qsa_sparse_paged_attention`.
- KV cache: 6 groups with sizes [1584,1584,1584,1584, **8**, 1584] blocks, `kv lcm block sizes 1584`. `mamba_cache_mode=align`, number_of_conv_states=3, mamba_ssm_dtype=float32, head_dim 256, indexer_budget 2048, indexer_compress_ratio 4, indexer_n_heads 4, indexer_kv_heads 1, indexer_head_dim 128, full_attention_interval 4.
- Allocation: `PYTORCH_CUDA_ALLOC_CONF` unset (so no expandable_segments/VMM), no CuMem allocator (sleep mode off).

## My own candidates (rank/refute these, add what I'm missing)
H1 TLB/page-mapping locality degradation of the engine's own long-lived regions (KV/QSA caches) after the long request's churn, at 96% VRAM with WSL2/dxgkrnl sharing the device with the Windows desktop. Against: external probes healthy; but they cannot allocate at engine scale (no free VRAM).
H2 Cross-stream / completion-event latency growth (the pipeline is strictly serial per layer: segment -> ids -> SSD -> H2D -> segment). Against: the trigger's chunks use the same per-layer chain and are NOT slowed.
H3 L2 set-conflict / address-aliasing caused by the block-id shift (bt0 moved from 2..13 to 42..50) — the same work with the same indices, but landing on worse L2/TLB sets. Note `mamba_cache_mode=align` means GDN state slots are derived from the block table, and one KV group has only 8 blocks.
H4 A sticky device/driver state in WSL2 GPU-PV (e.g. the WDDM/dxgkrnl submission path latching a slow mode, or the fence/writeback path) that only a new context clears.
H5 My instrumentation perturbing: I believe not — the poison reproduces identically in fully un-instrumented containers (2.0 s vs 0.65 s this session, no patches mounted).

## Questions
1. Rank the mechanisms; which can be excluded on the evidence above?
2. Give me ONE concrete experiment that discriminates the top two, achievable with: CUDA events in-graph, an external CUDA context in the container (small footprint only), host-side WSL2 shells, ~6-min container restarts, and config knobs (chunk size, ple_ssd_cache_mb, max_model_len, gpu_memory_utilization floor 0.93, mamba_cache_mode, prefix caching, seq count). No CUPTI/ncu/perf.
3. Does the physical block-id shift have a plausible hardware/driver mechanism (L2 aliasing, TLB, state-cache mapping under `mamba_cache_mode=align`), and can I test it cheaply (e.g. force a different allocation order, or pre-fill/free the KV pool so a short request gets LOW block ids after poisoning)?
4. Is there any *state* I should dump once, per layer, in the poisoned state that would separate "stall on memory" from "stall on a dependency/fence" without CUPTI? (I can bake arbitrary CUDA events into the graphs, and I can read any buffer from the host at replay time.)

Be concrete and quantitative; answer in English. Do not restart, stop or reconfigure the container; do not run GPU-heavy benchmarks; read-only inspection of /home/hong/vllm and the container is fine.
