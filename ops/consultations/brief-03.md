# Consulting request (round 3): we localised a persistent per-step slowdown to the QSA sparse-attention path. Design the *best* plan.

You are helping debug a live local inference service (container `hong-pc`, port 9393). You may inspect the machine
read-only (`docker exec hong-pc ...`, `curl localhost:9393/metrics`, `/proc`, `nvidia-smi`, `iostat`, `py-spy`,
and read any file in the container, e.g. `/opt/vllm/src/vllm/models/qwen4_exp/**`). You must NOT restart/stop/rm the
container or change files in it, and must not run GPU-heavy benchmarks. Container python: `/opt/vllm/.venv/bin/python`.
Answer in English, concrete and quantitative, and give me ONE ordered plan (not a menu of dice rolls).
Aim for <= 1800 words.

## 0. Environment / model (all measured earlier)
WSL2, driver 610.43.03, one custom sm_80 "170HX" (64 GiB, 74 SMs, 1485 MHz fixed, 194-207 TFLOPS bf16 dense measured,
1.4-1.7 TB/s D2D, idle 35 W, saturated 249 W). Patched vLLM `0.29.1rc1.dev402+ga5a30471f.ple1`, torch 2.13+cu130,
Triton, `VLLM_USE_BREAKABLE_CUDAGRAPH=1`, `cudagraph_mode=FULL_AND_PIECEWISE`, capture sizes <=2048,
`max_num_batched_tokens=2048`, `max_num_seqs=4`, `max_model_len=262144`, MTP spec-decode (1 token), prefix caching ON,
`--mamba-cache-mode align`, gpu_mem_util 0.96 (64.8 GiB reserved, ~0.7 GiB free).
Model `Qwen3.8-Flash-Next` (3 bpw AutoRound, 142 GiB): 48 layers = 36 GDN gated-delta-net linear attention + 12 full
attention (`full_attention_interval=4`), MoE 512 experts top-10, hidden 2560, plus one PLE layer (layer 1) whose
embeddings come from a 95 GiB SSD table. Full attention is a **custom sparse backend "QSA"**
(`models/qwen4_exp/nvidia/qsa.py`: `Qwen4ExpQSAAttention`, backend `QWEN4_EXP_QSA_TRITON`, `is_sparse()=True`).

## 1. Symptom (reproducible ~10x)
One request with >= ~64-96K prompt tokens flips the engine into a state where **every subsequent step is 6-9x slower**
until the container is restarted. Fresh random-id prompts, same container, two clean runs:
| prompt tokens | clean | poisoned |
|---|---:|---:|
| 512 | 0.447 s (1145 tok/s) | 0.964 s (531) |
| 2048 | 0.662 s (3093) | 2.364 s (866) |
| 8192 | 2.006 s (4083) | 11.947 s (686) |
| 32768 | 9.828 s (3334) | 52.697 s (622) |
The poisoning request itself (98304 fresh ids) takes 49.6-59.8 s = ~2000 tok/s while a clean 32K request runs at 3334.

## 2. Method this round (CUPTI is unusable here, so I used baked CUDA events)
* `torch.profiler` (CUDA activity) -> `CUPTI_ERROR_CMP_DEVICE_NOT_SUPPORTED (42)`, 0 CUDA events.
  `nsys profile` (2026.2.1) -> "Importer error ... skipping importation of the QDSTRM file", no `.nsys-rep`.
  `ncu` -> `ERR_NVGPUCTRPERM` (needs `NVreg_RestrictProfilingToAdminUsers=0` on the Windows host driver).
* Working substitute: during CUDA-graph **capture**, global forward hooks record
  `torch.cuda.Event(enable_timing=True, external=True)` pairs at module boundaries; those record nodes are baked
  into the graph and **execute on every replay**; I harvest `elapsed_time` one step later (no sync in the hot path).
  Validated standalone first: a graph with 1x2048^3 matmul = 1.665 ms, with 20x = 33.14 ms, values update per replay.
* Also wrapped `BreakableCUDAGraphCapture.replay()` to time each segment (graph replay / eager break) with normal
  events + host clock. Segment intervals cannot straddle a graph break; module intervals can (artefact noted below).
* Structure measured: prefill uses **103-segment piecewise captures**, decode/MTP use **3-segment FULL captures**;
  each capture contains 144 module intervals (48 layers x {decoder-layer, attention, MoE}).
* Instrumentation lives in `~/vllm/cg_instr.py` (host) mounted as `vllm/compilation/cg_instr.py`, installed at import
  time by an instrumented copy of `ple_ssd.py` (which also carries per-chunk PLE phase timers, `PLESTAT` lines).

## 3. Result: the extra cost is in attention, not in MoE
Main model, one **prefill chunk (2048 tokens)**, GPU time summed over all module intervals in that step (ms):
| step | total | ATNQ (12 QSA full-attn layers) | GDN (36 linear-attn layers) | MOE (48 layers) |
|---|---:|---:|---:|---:|
| clean | **389 / 475** | 66 | 217 | 112 |
| mid-trigger | 6674 | 2483 | 912 | 140 |
| poisoned | **2518 / 2617** | 1085 | 1300 | 190 |
**Decode** steps (FULL 3-segment captures, `sum_host` = 0.6 ms so it is pure GPU time inside the graph):
| per decode step | clean | poisoned |
|---|---:|---:|
| total | **12.4 ms** | **110.5 ms** (+98) |
| ATNQ (12 layers) | 5.6 ms | **80.6 ms** (+75 => +6.2 ms per layer per step!) |
| MOE (48 layers) | 3.6 ms | 4.8 ms |
The 1-layer MTP draft model behaves the same (21.5 -> 107.4 ms; its QSA attention 13.4 -> 77.2 ms).
Caveat: module intervals for modules whose forward spans a graph break contain the break gap (the 12 full-attention
modules sit *at* the piecewise splits, so their prefill numbers are inflated by that artefact; the *decode* FULL
captures have no splits, so the decode +6.2 ms/layer is clean). Phases ruled out (all measured): PLE/SSD path
(device reads full speed, disabling the SSD prefetch does not help), GPU clocks/power, GPU raw compute, Triton
recompiles, token recompute/preemption, KV/prefix-cache metrics (usage 0 afterwards), host RAM/threads, chunk size,
prompt content, the CUDA-graph machinery itself. Earlier concurrent-GEMM probe: while a poisoned step runs, an
external GEMM is starved of the GPU in ~2 s blocks; `nvidia-smi` shows SM util 100% but only ~100 W (vs 249 W
saturated) and clocks pinned at 1485 MHz. So: SMs are busy but not doing useful FLOPs - latency/scan/spin-like.

## 4. What the QSA path is (from the source in the container)
`nvidia/indexer_qsa.py::QSAIndexer`: `token_topk = config.indexer_budget`, `compress_ratio = config.indexer_compress_ratio`,
`output_width = token_topk + compress_ratio - 1`, `packed_output_width = output_width + 1`; `skip_topk` (False by default;
"MTP step 0 selects the target-aligned rows; later steps reuse them while continuing to update the QSA side cache").
Model config (`/model/config.json` -> `text_config`): **`indexer_budget: 2048`**, **`indexer_compress_ratio: 4`**,
`indexer_n_heads: 4`, `indexer_kv_heads: 1`, `indexer_head_dim: 128`, `full_attention_interval: 4`, `ngram_size: 3`,
`split_ngram_parts: 128`, `number_of_conv_states: 3`, `mamba_ssm_dtype: float32`, `head_dim: 256`.
i.e. **per query token the indexer selects up to 2048 positions out of the visible candidates** and writes ~2052
ints/token. `nvidia/ops/qsa_pre_indexer.py::_qsa_pre_indexer_kernel` fuses RMSNorm+RoPE and compresses K with
**atomic accumulation into a "circular" state/compressed cache**; grid = `num_k_work + num_q_work`,
`num_q_work = cdiv(num_tokens,2)*cdiv(num_q_heads,2)`, `num_k_work = k_work_metadata.shape[0]`.
`common/qsa_cache.py` builds per-token `visible_blocks` (a Triton kernel, lines ~212-275) and the metadata
(`visible_blocks_buffer[:num_tokens]`, `token_to_req`, `logical_positions`, `slot_mapping`, lines ~390-440).
Comments there: "Let the dependent grid start launch setup while these CTAs finish" and
"**Pad for tl.arange while keeping the scan width stable across live batches**".
The startup Triton-compile log shows these sampling kernels compile on first use:
`_qsa_pre_indexer_kernel`, `_compute_local_logits_stats_kernel`, `_rejection_kernel`, `_resample_kernel`
(i.e. the top-k selection appears to be *sampling-based*: local logits stats + rejection + resample).

## 5. Candidate mechanisms (rank/triage them)
H1 **High-water-mark "scan width"**: some buffer/scan width is sized (or kept "stable") by the longest batch/context
ever seen, so after a 96K request every later step - including a 2-token decode step - scans a 96K-scale buffer.
Evidence: `visible_blocks` and the "scan width" comment above; decode cost is ~constant (110 ms/step) for current
contexts of 2K/8K/32K, i.e. NOT proportional to the *current* context; +6.2 ms/layer for a 2-token decode is far more
than any real attention work.
H2 **Memory locality / allocator fragmentation**: after the 96K request the cache/state blocks of subsequent short
requests are physically scattered, so the latency-bound gathers (compressed KV rows, GDN state) go from L2 to HBM
(10x latency), while the bandwidth-bound MoE weight streaming is unaffected. (Would also explain 100% SM util at 100 W.)
H3 **Sampling-loop degeneration** (rejection/resample runs to its max iteration count because the distribution/logits
or the candidate pool is degenerate after the long request).
H4 **MTP/`skip_topk` state interaction** (frozen rows from step 0 + side cache updates) - the 1-layer draft model is
hit just as hard.
H5 **GDN/mamba state cache** (`--mamba-cache-mode align`, odd `kv cache group sizes [1584,1584,1584,1584,8,1584]`) -
explains the GDN side of the prefill inflation, but GDN does not appear in the poisoned decode steps.
H6 something else you can justify.

## 6. Candidate experiments/knobs I have available (rank them; I want the minimum number of 5-minute restarts)
Free (no restart): scale the trigger (96K vs a bigger 200K request) and re-probe; concurrent requests; prefix-cached
request; `/metrics` (`vllm:num_requests_running`, `kv_cache_usage_perc`, `prefix_cache_*`, `iteration_tokens_total`,
`spec_decode_num_*`, `engine_sleep_state`); py-spy; `iostat`; `nvidia-smi dmon`.
Per restart (5-7 min each), I can add any of:
(a) baked events inside the QSA path, bracketing these groups: q/k projection, norm+RoPE, compressed-K/state update,
local-logits-stats+rejection+resample sampling, top-k selection, attention core, metadata build
(the files are `qsa.py`, `indexer_qsa.py`, `ops/qsa_pre_indexer.py`, `common/qsa_cache.py`; I mount patched copies);
(b) `--no-enable-prefix-caching`;
(c) an hf-override of `indexer_budget` (e.g. 256 instead of 2048) and/or `indexer_compress_ratio` (8 instead of 4)
- changes model behaviour, but as a *diagnostic* it separates "cost ∝ candidates selected" from "fixed scan cost";
(d) `QWEN_MTP=0` (no speculative decoding);
(e) `--mamba-cache-mode` variants; (f) lower `--gpu-memory-utilization` (fewer KV blocks => less fragmentation?);
(g) smaller `max_num_batched_tokens` (e.g. 512) to shrink per-step token work;
(h) `--kv-cache-dtype`/block-size changes.
Also: is there any way to *clear* the poisoned state without restarting (vLLM APIs like `/reset_prefix_cache`,
`sleep`/`wake_up`, or forcing a new KV-block allocation), both as a workaround and as a discriminator?

## 7. Deliverable
1. Your ranked judgement of H1-H6, with the specific evidence that discriminates them, and any hypothesis I missed.
2. ONE ordered plan (step 1, 2, 3...) that keeps the number of restarts minimal, with the exact env/flags/patch for
each step, what I should measure, and the decision rule after each step (what outcome sends me where). Include the
free steps first. State explicitly what each step can and cannot prove.
3. The invariants/quantities to log every step to catch the flip (e.g. buffer shapes, scan widths, candidate counts,
`visible_blocks` stats) - ideally something I can read from the running engine without new kernels.
4. If you think the cheapest *decisive* experiment is something I have not listed, say so and give it.
