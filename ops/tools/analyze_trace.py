"""Summarize a kineto/chrome trace: kernel time, launch count, stream gaps."""
import gzip, json, sys, collections

path = sys.argv[1]
op = gzip.open if path.endswith(".gz") else open
d = json.load(op(path, "rt"))
ev = d["traceEvents"] if isinstance(d, dict) else d
kern = [e for e in ev if e.get("cat") == "kernel"]
kern.sort(key=lambda e: e["ts"])
total = sum(e["dur"] for e in kern)
print(f"events={len(ev)} kernels={len(kern)} kernel_total={total/1e6:.3f}s")
span = (kern[-1]["ts"] + kern[-1]["dur"] - kern[0]["ts"]) / 1e6 if kern else 0
print(f"kernel span={span:.3f}s  busy={100*total/1e6/span:.1f}%")
by = collections.Counter()
for e in kern:
    by[e["name"]] += e["dur"]
print("\n-- top kernels by total time --")
for n, t in by.most_common(20):
    c = sum(1 for e in kern if e["name"] == n)
    print(f"  {t/1e6:7.3f}s {100*t/total:5.1f}%  n={c:5d}  {t/c:9.1f}us  {n[:70]}")
# gaps during the steady-state window
gaps = [(kern[i+1]["ts"] - (kern[i]["ts"] + kern[i]["dur"]), i) for i in range(len(kern)-1)]
big = sorted([g for g in gaps if g[0] > 200], reverse=True)[:10]
print(f"\n-- gaps >200us: {len(big)} of {len(gaps)} --")
for g, i in big[:10]:
    nxt = kern[i+1]["name"]
    pr = kern[i]["name"]
    print(f"  gap {g:7.0f}us  after {pr[:38]:40s} before {nxt[:38]}")
# metadata ops (memcpy/memset)
for cat in ("gpu_memcpy", "gpu_memset", "cuda_runtime", "python_function"):
    c = [e for e in ev if e.get("cat") == cat]
    if c:
        print(f"\n{cat}: {len(c)} events, total {sum(e['dur'] for e in c)/1e6:.3f}s")
