# 5th consult: the isolated kernel ITSELF runs 10x slower in the "bad" state, but the device is healthy

You are advising on a WSL2 + Docker + GPU0 (NVIDIA cc8.0, 74 SMs, 64 GiB, driver 610.43.03) vLLM deployment
(Qwen3.8-Flash-Next, custom QSA sparse prefill attention). Read-only advice only: do NOT restart/stop/reconfigure
any container and do NOT run GPU-heavy benchmarks. Answer in English, quantitative and concrete.

## Phenomenon
After the FIRST >=96K-token prefill request in a container, subsequent fresh small prefill requests are 2-4x slower
(fresh 2048 tokens: 0.84 s -> 2.03 s; fresh 8192: 2.32 s -> 9.72 s; in-chunk rate 0.33 -> 1.0 ms/token).
Decode is NOT affected (21 ms/token both states). 512-token requests scale about proportionally.
The state is NOT sticky: it clears progressively if the GPU pipeline is drained/quiesced (see below).

## Instrument (validated)
Inside the production QSA attention op, right after the production launch (which itself replays inside captured
CUDA-graph segments), with the engine thread blocked, I re-run the SAME compiled Triton kernel on a private stream
using THE SAME operands (a snapshot of the current call's tensors? no - the current call's actual tensors), 3 reps,
CUDA-event timing, one sync per rep (GPU drained, isolated). Arms:
OE = original K/V + production selection/block table, eager launch.
RE = the referenced K/V *blocks* copied (whole blocks, layout/strides preserved) into a fresh 198 MiB scratch +
     private compacted block table, eager.
SH = same K/V storage, block table shifted (+7 mod nblocks) -> different physical addresses, eager.
OG/RG = original / relocated, replayed as a SINGLE-KERNEL CUDA graph.
OE2 = OE repeated LAST (tests whether probing itself clears the state).
RK = scratch K/V filled with ZEROS (data-value test), eager.
G4 = 4096^3 bf16 matmul on the same private stream (137 GFLOP) - device-health control.
Arming is done with a file, so the probe is COMPLETELY OFF while I trigger the bad state (observer effect is real:
a container with the probe always on never enters the bad state). Host launch-enqueue cost is also recorded.

## Clean state (same shapes, nq=2048, sel_w=2051, cnt_max=2048, cnt_sum=2098176, ti_max=2047)
OE 2.17-2.67 ms, RE 2.10-2.61, SH 2.14-2.73, OG 2.10-2.69, RG 2.06-2.60  => all four equal.
(OE 2.2 ms matches the earlier in-graph CUDA-event value for the production invocation, so the method is trusted.)

## Bad state
Fully bad moment, nq=2048, IDENTICAL work profile to clean (cnt_max=2048, cnt_sum=2098176, ti_max=2047):
  OE 21.8 ms, RE 22.1, SH 22.6, OG 21.8, RG 14.1   (=> 10x; ALL ARMS SLOW)
  host launch-enqueue cost 0.03-0.4 ms (one RE outlier 2.63 ms) => host submission cannot explain 20 ms.
Partially recovered moment, SAME probe call: OE 7.0 while OG 2.11 / RG 1.99 (stable across the 3 reps of each arm)
  => at that instant graph replay is 3.4x faster than eager, i.e. the state was being cleared as the probe ran.
Bad state, chunk shapes nq=1584/1856 (cnt_sum 3.25e6/3.80e6, ti_max 6335/8191):
  G4 = 0.70-0.74 ms (=190 TFLOPS, FULL HEALTH) while OE 3.34-3.93, RE 3.27-3.77, SH 3.33-3.93,
  OG 3.34-3.82, RG 3.28-3.73, OE2 3.35-3.78, RK (zero-filled K/V) 3.12-3.69 ms
  => placement, values, and launch mode all irrelevant; device fully available in the same instant.

## Other established facts
- SM clock in the bad state: 1477 MHz (full), 105 W (low), zero throttle flags; a 4096^3 GEMM from an external
  container ran 206-211 TFLOPS but its blocks were starved 1.9-5.0 s at a time while the engine worked.
- Earlier in-graph event accounting (bad/clean): attention 30.0/2.2 ms per layer (all 12 layers exactly 30.0),
  GDN module 1588/422 ms while the narrow GDN fused kernel bracket is unchanged, MoE 197/134 ms,
  sum_host inside the replay loop 2488/648 ms per chunk.
- Reversibility: while probing, request latency went 2.805 -> 1.404 s and afterwards a fresh 2048 measured
  0.89 s (fully clean, no restart). Reproducibility is stateful: the FIRST 96K trigger after container start
  poisons (3/3), a second 96K trigger in the same container does not (0.684 s).
- No profiler available: CUPTI returns CUPTI_ERROR_CMP_DEVICE_NOT_SUPPORTED, nsys/ncu/perf all fail.
  Available: CUDA events, host timers, py-spy, Triton kernel source patching (constexprs, extra output buffers),
  a second container for external control kernels, nvidia-smi.

## Question
Given that (a) the same compiled kernel with the same grid/specialization, the same operand contents, the same
address relations, the same data values and the same launch mode can be 10x slower in one moment and 1x in another,
(b) the device is demonstrably healthy in the bad moment (190 TFLOPS GEMM on the same stream), and (c) the state is
progressively cleared by draining the pipeline, what are the most likely remaining mechanisms, and what is the single
most decisive next measurement I can make with only the tools listed above?

Please rank mechanisms by prior probability and for the top ones give a concrete, executable experiment (what to
instrument, what numbers to compare, and the exact interpretation matrix), including the per-CTA %globaltimer
envelope idea: what exactly should I compare, and what would distinguish "time lost outside the CTAs" from
"CTA in-flight slowed", from "the kernel binary/occupancy changed"?
