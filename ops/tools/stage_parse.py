#!/usr/bin/env python3
"""Group CGSTEP lines by capture shape (nl= layers, nseg= segments, desc=) and
average every `cls=` entry, so clean and poisoned phases can be compared."""
import collections, re, sys

path = sys.argv[1]
lo = int(sys.argv[2]) if len(sys.argv) > 2 else 0
hi = int(sys.argv[3]) if len(sys.argv) > 3 else 10**9

shape = collections.defaultdict(lambda: collections.defaultdict(float))
cnt = collections.Counter()
sums = collections.defaultdict(lambda: [0.0, 0.0, 0.0])
desc = {}
for i, line in enumerate(open(path, errors="replace"), 1):
    if i <= lo or i > hi or "CGSTEP" not in line:
        continue
    m = re.search(r"CGSTEP nl=(-?\d+) nseg=(\d+) desc=(.*?) sum_gpu=([\d.]+) sum_host=([\d.]+)", line)
    if not m:
        continue
    key = (int(m.group(1)), int(m.group(2)), m.group(3)[:40])
    cnt[key] += 1
    desc[key] = m.group(3)
    sums[key][0] += float(m.group(4))
    sums[key][1] += float(m.group(5))
    cm = re.search(r"cls=(.*?) top5=", line)
    if cm:
        for tok in cm.group(1).split():
            if "=" in tok:
                k, v = tok.rsplit("=", 1)
                try:
                    shape[key][k] += float(v)
                except ValueError:
                    pass

for key in sorted(cnt, key=lambda k: -cnt[k]):
    nl, nseg, _ = key
    n = cnt[key]
    print(f"\n### nl={nl} nseg={nseg} desc={desc[key]}  samples={n}  avg sum_gpu={sums[key][0]/n:.2f} avg sum_host={sums[key][1]/n:.2f} ms")
    items = sorted(shape[key].items(), key=lambda kv: -kv[1])
    line = []
    for k, v in items:
        av = v / n
        if av < 0.005:
            continue
        line.append(f"{k}={av:.2f}")
    for i in range(0, len(line), 8):
        print("   " + "  ".join(line[i:i+8]))
