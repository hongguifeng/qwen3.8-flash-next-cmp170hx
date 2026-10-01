#!/bin/bash
# Report recorded checkpoint progress and current rate from the downloader state.
D=~/models/Qwen3.8-Flash-Next-AutoRound-3bpw-MTP
rec() {
python3 - "$D" <<'PY'
import json, os, sys
d = sys.argv[1]
CH = 16 * 1024 * 1024
try:
    s = json.load(open(os.path.join(d, ".fdl-state.json")))
except Exception:
    print(0)
    raise SystemExit
tot = 0
for n, idxs in s.items():
    part = os.path.join(d, n + ".part")
    size = os.path.getsize(part) if os.path.exists(part) else 0
    if not size:
        final = os.path.join(d, n)
        size = os.path.getsize(final) if os.path.exists(final) else 0
    tot += sum(min(CH, size - i * CH) for i in set(idxs))
print(int(tot))
PY
}
a=$(rec)
sleep "$1"
b=$(rec)
python3 -c "
a, b, dt = $a, $b, $1
tot = 142.56 * 2**30
rate = (b - a) / 2**20 / dt
eta = (tot - b) / 2**20 / max(rate, 0.01) / 3600
print(f'progress {b/2**30:6.2f}/{tot/2**30:.2f} GiB ({b/tot*100:5.1f}%)  rate {rate:5.2f} MiB/s  ETA {eta:5.2f} h')"
