# pass3 — WSL2 native deployment (2026-09-29)

This is the **native (no Docker)** re-measurement of the same checkpoint on the same
GPU, run from `vllm-native/` inside WSL2. Read
[`docs/RESULTS-WSL2.md`](../../docs/RESULTS-WSL2.md) for the tables, comparisons
against pass2, and caveats.

| File | What it contains |
| --- | --- |
| `manifest.json` | hardware, runtime versions, patches, serve arguments, env knobs |
| `final-wsl2-bench.json` | three-round 128-token chat sweep, concurrency 1 / 4 / 8 |
| `final-wsl2-long-bench.json` | three-round 512-token chat sweep, concurrency 1 / 4 / 8 |
| `final-wsl2-prefill.json` | fresh 512 / 2K / 8K / 32K prefill sweeps (fresh seed, prefix cache avoided) |
| `final-wsl2-smoke.json` | six serving smoke checks (includes Korean, tool call, 16.8K retrieval) |

Headline (median of rounds): single-stream decode **131 tok/s** (128 tokens) /
**132 tok/s** (512 tokens), aggregate **305 tok/s** at concurrency four, prefill
**0.693 s @ 2K** and **10.40 s @ 32K**, all with **MTP=2** and a 262 144-token context.

Produced with the project's own benchmark tools:

```bash
PY=vllm-native/opt/vllm/.venv/bin/python
$PY benchmarks/bench_server.py --url http://127.0.0.1:9393 --label wsl2-native-128 \
    --tokens 128 --rounds 3 --output final-wsl2-bench.json
$PY benchmarks/bench_server.py --url http://127.0.0.1:9393 --label wsl2-native-512 \
    --tokens 512 --rounds 3 --output final-wsl2-long-bench.json
$PY benchmarks/bench_prefill.py --label wsl2-native-prefill --seed $RANDOM \
    --lengths 512 2048 8192 32768 --repeats 3 --output final-wsl2-prefill.json
$PY benchmarks/verify_server.py --url http://127.0.0.1:9393 --long --tools \
    --output final-wsl2-smoke.json
```

Note: `bench_prefill.py` uses a fixed per-(length, round) seed, so a second
invocation with the default seed hits the **prefix cache** and looks 3–15 × faster.
Pass a fresh `--seed` for every measurement run. Ad-hoc runs of `bin/bench.sh` are
appended to `ops/measurements/perf-history.csv`.
