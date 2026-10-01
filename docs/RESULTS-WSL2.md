# WSL2 native deployment — measured results (pass3)

Measured 2026-09-29 on this machine (Windows 11 host, WSL2, **no Docker container**).
This file is the performance record for the native deployment; the reference-machine
numbers live in [RESULTS.md](RESULTS.md) (pass2) and [PERFORMANCE.md](PERFORMANCE.md).
Raw artifacts: [`results/pass3-wsl2-native/`](../results/pass3-wsl2-native) ·
settings snapshot: [manifest.json](../results/pass3-wsl2-native/manifest.json).

## Environment and configuration

| Item | Value |
| --- | --- |
| Host | Windows 11 24H2 (26100.4061), 88 GB RAM, 16 cores, WSL 2.7.11.0 (kernel 6.18.33.2), `networkingMode=mirrored` |
| Compute GPU | NVIDIA CMP 170HX, 65 536 MiB, compute capability 8.0, PCIe gen2 ×8 (display output is a separate Radeon RX550) |
| GPU clocks / power | HBM 1 728 MHz fixed (= hardware max), SM idle 210 MHz / load peak 1 485 MHz (cap 1 695); **power limit 220 W** (default 250, max 300) since 2026-10-01 — see the re-measure section below |
| Runtime | native vLLM 0.29.1rc1.dev402+ga5a30471f.ple1, torch 2.13.0+cu130, triton 3.7.1, Python 3.12 |
| Patches | `patches/qwen38-ple-ssd.patch` (upstream, 1345 lines) + `ops/patches/qsa-alloc-heal.patch` (local allocator healer, 65 lines) |
| Model | `Qwen3.8-Flash-Next-AutoRound-3bpw-MTP` (143 GB), 95.4 GiB BF16 PLE table read from SSD via `O_DIRECT` + AIO |
| Serving config | **MTP=2**, 2048-token prefill chunks, `max-num-seqs 4`, `gpu-memory-utilization 0.94`, context 262 144, bf16, FULL_AND_PIECEWISE graphs, PLE SSD offload (16 workers, 512 MiB row cache, AIO depth 256, 16 384-token read-ahead) |
| Env knobs | `VLLM_WSL2_ENABLE_PIN_MEMORY=1` (without it decode drops 30–35 %), `VLLM_USE_BREAKABLE_CUDAGRAPH=1`, `OMP_NUM_THREADS=1`, `QSA_ALLOC_HEAL=1` |

Startup: `/health` returned 200 after **251–321 s** (weights + CUDA-graph capture);
VRAM in use 60 163–60 401 MiB of 65 536 MiB.

## Generation (128 output tokens, temperature 0, thinking off, EOS ignored)

Three-round medians, same protocol as pass2. Aggregate includes prefill and HTTP
overhead; per-request decode excludes first-token latency. Raw:
[`final-wsl2-bench.json`](../results/pass3-wsl2-native/final-wsl2-bench.json).

| Concurrent requests | Aggregate tok/s | Per-request decode tok/s | TTFT | (pass2 reference) |
| ---: | ---: | ---: | ---: | --- |
| 1 | **122.55** | **131.19** | 0.071 s | 103.30 / 110.99 / 0.105 s |
| 4 | **305.42** | **90.87** | 0.145 s | 277.22 / 77.95 / 0.195 s |
| 8 | 304.07 | 86.63 | 0.817 s | 428.97 / 60.63 / 0.183 s |

## Generation (512 output tokens)

Two/three-round medians. Raw:
[`final-wsl2-long-bench.json`](../results/pass3-wsl2-native/final-wsl2-long-bench.json).

| Concurrent requests | Aggregate tok/s | Per-request decode tok/s | TTFT |
| ---: | ---: | ---: | ---: |
| 1 | **130.45** | **132.33** | 0.072 s |
| 4 | 291.37 | 85.23 | 0.137 s |
| 8 | 328.22 | 88.71 | 2.907 s |

pass2 measured 116.88 single-user decode and 435.28 aggregate tok/s at concurrency
eight. Single- and four-stream numbers are 10–18 % **higher** here; the
concurrency-eight aggregate is lower because this deployment keeps
`max-num-seqs 4` (pass2 used 16 sequences), so eight clients are served in two
waves — that also explains the 2.9 s TTFT at concurrency eight. Raise `QWEN_SEQS`
if multi-client aggregate matters more than single-stream latency.

## Prefill (fresh deterministic token IDs, prefix cache avoided)

Three rounds with a **fresh `--seed`** (`random.Random(seed + length*7 + repeat)`, so
every round is an uncached prompt). Raw:
[`final-wsl2-prefill.json`](../results/pass3-wsl2-native/final-wsl2-prefill.json).

| Prompt tokens | Round latencies | Median | Prompt tok/s | (pass2 reference) |
| ---: | --- | ---: | ---: | --- |
| 512 | 0.370 / 0.195 / 0.266 s | 0.266 s | 1 925 | 0.317 s / 1 616 |
| 2 048 | 0.714 / 0.687 / 0.693 s | 0.693 s | 2 956 | 1.069 s / 1 915 |
| 8 192 | 2.184 / 2.400 / 2.158 s | 2.184 s | 3 750 | 3.108 s / 2 636 |
| 32 768 | 10.262 / 10.622 / 10.404 s | 10.404 s | 3 149 | 15.372 s / 2 132 |

Prefill is 1.2–1.5 × faster than pass2 across the board (NVMe + 88 GB RAM versus
the reference's SATA + 15 GB RAM; the PLE SSD path is the bottleneck in both).
Constant in-flight throughput (≈2.4–3.1 K tok/s at 128 K tokens) means the SSD path
is not degrading with context length.

⚠️ **Repeating the same prompt is served from the prefix cache** and is *not* a
prefill measurement: re-running the same seed took 8 192 tokens from 2.184 s to
0.418 s and 32 768 tokens from 10.404 s to 0.677 s. Always pass a fresh `--seed`.

## MTP depth A/B (why MTP=2 is the default)

Same prompts, both configs fully warmed (≥2 500 decode tokens) and measured while
the host had ≥33 GB free RAM. `tok/s = (1 + accepted drafts) / step time`.

| Depth | Tokens per step | Step p50 | Steady-state decode | Runs |
| ---: | ---: | ---: | --- | --- |
| 1 | 1.70 | 15.2–15.7 ms | 106.3 – 116.0 tok/s (≈111) | 6 |
| **2** | **2.20** | 17.2–17.6 ms | **118.4 – 130.9 tok/s (≈126)** | 9 |

The second draft pass costs only ≈2 ms but returns ≈0.5 token/step ⇒ +13.5 %.
Prefill is unaffected (2048: 0.64 s, 8192: 2.11 s at both depths).

⚠️ **Decode step time is sensitive to host memory pressure**: the same MTP=2 binary
measured p50 22.2 ms / ≈100 tok/s while the WSL memory balloon held 57.8 GB and
Windows had 6.9 GB free, and p50 17.4 ms / ≈126 tok/s after the balloon deflated
(51 GB free). Any decode comparison needs ≥2 500 warm-up tokens **and**
`vmmemWSL < 32 GB` (`bin/drop_host_cache.sh`).

## Long context and allocator healing

| Prompt | Deployment | Prefill | Post-check (fresh 2 048) |
| ---: | --- | ---: | ---: |
| 131 072 | native | 54.63 s (2 399 tok/s) | 0.680 / 0.687 s ✔ |
| 131 072 | native, 2026-10-01 re-measure | **48.66 / 49.00 s (2 694 tok/s)** | 0.653 / 0.653 s ✔ |
| 131 072 | Docker (+ healer) | 53.54 s (2 450 tok/s) | 0.676 / 0.668 / 0.688 s ✔ |
| 196 608 | Docker (+ healer) | 80.06 s (2 455 tok/s) | 0.668 s ✔ |
| 262 143 | Docker (+ healer) | 104.70 s (2 504 tok/s) | 0.688 s ✔ (QXHEAL fired 8×) |

Without the healer, any prompt ≥96 K drove the torch caching allocator to
`free = 0 MiB` and the engine into a state where prefill was 4.3–4.5 × slower and
decode 1.7 × slower until restart. `ops/patches/qsa-alloc-heal.patch` fixes it: the
engine now serves the full 262 144-token context at a constant ≈2.5 K tok/s.
The native deployment keeps the same patch (`QSA_ALLOC_HEAL=1`) and fired it once
during a 131 K-token prefill.

## Re-measure 2026-10-01 (new GPU power/clocks + driver KMD 610.88)

After the operator changed the card's power limit (250 W → **220 W**) and memory-clock
settings, the whole stack was re-measured on the running native deployment
(full log: `ops/OPS.md §9.33`, raw rows in `ops/measurements/perf-history.csv`).
Startup took 294 s, VRAM 59 953 MiB.

| Metric | 2026-10-01 | 2026-09-29 record |
| --- | --- | --- |
| Prefill 2 048 | **0.653 s** (3 132 tok/s) | 0.693 s median |
| Prefill 8 192 | **2.225 s** (3 682 tok/s) | 2.184 s median |
| Prefill 131 072 | **48.66 / 49.00 s** (2 694 tok/s) | 54.63 s (2 399) |
| Post-131 K fresh 2 048 | 0.653 s ✔ | 0.680 / 0.687 s |
| Decode step p50 (MTP=2) | **17.3–17.7 ms** | 17.2–17.6 ms |
| Decode tok/s (1 024 out, 68.7 % accept) | 129.0 | 118.4–130.9 |
| Concurrency 1 aggregate / decode | 115.9 – 144.3 / 127.2 – 156.7 | 122.55 / 131.19 |
| Concurrency 4 aggregate / decode | 277.6 – 319.8 / 83.4 – 90.7 | 305.42 / 90.87 |

Power/clock sampling during the run (`nvidia-smi -lms 250`):

| Phase | Power | SM clock | Throttle reason |
| --- | --- | --- | --- |
| decode (busy) | mean 177 W / peak 204 W | mean 1 478 / peak 1 485 MHz | none |
| 131 072 prefill | peak **320 W** | peak 1 485 MHz | `0x4 SW Power Cap` in 264/936 samples (28 %) |

So the 220 W cap never binds during decode and only clips transient peaks during long
prefill — where throughput came out **10.9 % higher** than the previous record anyway.
Note NVML reports the HBM clock unchanged at 1 728 MHz (its hard maximum) both before
and after, and the Windows driver is now KMD 610.88 (was 616.92 in September).

⚠️ Measurement trap observed here: the first 3–4 decode rounds after startup showed
p50 18.4–19.1 ms and converged to 17.3–17.7 ms only after the 2 500-token warm-up *and*
with the host balloon down (vmmemWSL ≤ 30 GB). Judging the GPU change on the first few
rounds would have looked like a 7 % regression that does not exist.

## Validation smoke checks

[`final-wsl2-smoke.json`](../results/pass3-wsl2-native/final-wsl2-smoke.json): 6/6
passed — factual ("Paris"), arithmetic, Python, Korean, tool call, and retrieval of
an 8-digit code from a 16 841-token prompt (3.70 s). The known upstream caveat still
applies: smoke checks are not a quality benchmark, and an arithmetic prompt can
return a wrong answer.

## Known differences from the Docker deployment

- **Windows clients must use `127.0.0.1`, never `localhost`** — mirrored networking
  does not forward the IPv6 loopback, and Windows resolves `localhost` to `::1` first
  (measured: `127.0.0.1:9393` → 200, `localhost:9393` → timeout). Docker used to work
  with `localhost` because the port was published by a Windows-side proxy.
- `http://127.0.0.1:8000/v1` forwards to `:9393` and is kept for older clients.
- `.wslconfig` currently asks for `memory=48GB`, but it is **not active yet** (guest
  still reports 62.8 GiB); it needs one `wsl --shutdown`.
- After a `wsl --shutdown` or reboot, the engine must be started again
  (`bin/start.sh`); there is no autostart by design.

## Reproducing

```bash
cd /home/hong/code/qwen3.8-flash-next-cmp170hx
bin/start.sh                    # waits for /health, then frees host RAM
PY=vllm-native/opt/vllm/.venv/bin/python
$PY benchmarks/bench_server.py --url http://127.0.0.1:9393 --label wsl2-native-128 \
    --tokens 128 --rounds 3 --output results/pass3-wsl2-native/final-wsl2-bench.json
$PY benchmarks/bench_server.py --url http://127.0.0.1:9393 --label wsl2-native-512 \
    --tokens 512 --rounds 3 --output results/pass3-wsl2-native/final-wsl2-long-bench.json
$PY benchmarks/bench_prefill.py --label wsl2-native-prefill --seed $RANDOM \
    --lengths 512 2048 8192 32768 --repeats 3 \
    --output results/pass3-wsl2-native/final-wsl2-prefill.json   # fresh seed each time!
$PY benchmarks/verify_server.py --url http://127.0.0.1:9393 --long --tools \
    --output results/pass3-wsl2-native/final-wsl2-smoke.json
bin/bench.sh                    # quick check; appends a row to ops/measurements/perf-history.csv
```
