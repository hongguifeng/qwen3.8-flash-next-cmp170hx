"""Portable minimal reproducer #4 -- the missing COMBINATION: high occupancy AND allocation churn.

Previous negative results (both platforms, fresh process):
  * 0->60 GiB resident ballast, few allocations        -> no degradation   (minrepro.py)
  * 50000 alloc/free ops, device nearly EMPTY (~1 GiB) -> no degradation   (minrepro.py #3)
Never tested: churn while the device is ~95% occupied, forcing the memory manager to create new
mappings / split page tables at the very end of physical memory -- which is what a >=96K-token
prefill does (47 GiB weights + 9 GiB KV + graph pools + transient workspaces at peak occupancy).

  scatter : dst.copy_(src[idx])     16384 random 2 KiB rows out of a 64 MiB src  (engine's shape)
  dense   : dst.copy_(src[:16384])  same buffers, contiguous (control)

Phases: baseline -> ballast to --ballast-gb (mixed-size chunks, each touched) -> churn batches
(small alloc/touch/free, every --flush ops a real cache release so genuine mappings are created).
"""
import argparse, gc, os, platform, statistics as st, time
import torch

MiB, GiB = 2 ** 20, 2 ** 30


def mem():
    f, t = torch.cuda.mem_get_info()
    return f, t, torch.cuda.memory_allocated(), torch.cuda.memory_reserved()


def bench(fn, reps=5, inner=10):
    fn()
    torch.cuda.synchronize()
    ev = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(reps)]
    for a, b in ev:
        b.record()
        for _ in range(inner):
            fn()
        a.record()
    torch.cuda.synchronize()
    return sorted(b.elapsed_time(a) / inner for a, b in ev)[len(ev) // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ballast-gb', type=float, default=58.0)
    ap.add_argument('--ballast-one', action='store_true', help='one single huge allocation (low fragmentation)')
    ap.add_argument('--churn', type=str, default='0,2000,20000')
    ap.add_argument('--flush', type=int, default=256, help='empty_cache() every N churn ops')
    ap.add_argument('--rows', type=int, default=16384)
    ap.add_argument('--row-bytes', type=int, default=2048)
    args = ap.parse_args()

    w = args.row_bytes // 8
    src = torch.randint(-(2 ** 62), 2 ** 62, (args.rows * 2, w), dtype=torch.int64, device='cuda')
    dst = torch.empty((args.rows, w), dtype=torch.int64, device='cuda')
    idx = torch.randint(0, args.rows * 2, (args.rows,), device='cuda')
    lab = [torch.empty(1 * MiB, dtype=torch.uint8, device='cuda')]    # tiny allocator sentinel
    sc = lambda: dst.copy_(src[idx])
    dn = lambda: dst.copy_(src[:args.rows])
    ref = [None, None]

    def report(tag):
        s = bench(sc)
        d = bench(dn)
        lab[0].fill_(1)
        if ref[0] is None:
            ref[0], ref[1] = s, d
        f, t, al, rs = mem()
        print(f"{tag:34s} scatter {s:8.3f} ms x{s/ref[0]:6.2f}  dense {d:7.3f} ms x{d/ref[1]:5.2f}  "
              f"| free {f/GiB:5.2f} alloc {al/GiB:5.2f} reserved {rs/GiB:5.2f} GiB", flush=True)
        return s

    print(f"platform={platform.platform()} python={platform.python_version()} torch={torch.__version__}", flush=True)
    print(f"device={torch.cuda.get_device_name(0)} free={torch.cuda.mem_get_info()[0]/GiB:.2f} GiB", flush=True)
    report("BASELINE")

    # ---- ballast
    done = 0.0
    sizes = [32 * MiB, 64 * MiB, 128 * MiB, 256 * MiB]
    ballast = []
    if args.ballast_one:
        n = int(args.ballast_gb * GiB)
        t = torch.empty(n, dtype=torch.uint8, device='cuda')
        t.fill_(2)
        ballast.append(t)
        done = n
        report(f"ballast one-shot {done/GiB:.1f} GiB")
    while done < args.ballast_gb * GiB:
        n = int(min(sizes[(len(ballast)) % 4], args.ballast_gb * GiB - done))
        if n < MiB:
            break
        try:
            t = torch.empty(n, dtype=torch.uint8, device='cuda')
            t.fill_(2)
        except RuntimeError as e:
            print(f"ballast stopped at {done/GiB:.1f} GiB: {e}", flush=True)
            break
        ballast.append(t)
        done += n
    report(f"ballast {done/GiB:.1f} GiB ({len(ballast)} chunks)")

    # ---- churn at high occupancy
    csizes = [4 * 1024, 64 * 1024, 512 * 1024, 2 * MiB, 8 * MiB]
    target = [int(x) for x in args.churn.split(',') if x]
    ops = 0
    ring = []
    for tgt in target:
        while ops < tgt:
            try:
                t = torch.empty(csizes[ops % 5], dtype=torch.uint8, device='cuda')
                t.fill_(3)
                ring.append(t)
                del ring[:-16]                                  # keep <= ~48 MiB alive
                del t
            except RuntimeError:
                torch.cuda.empty_cache()
                print(f"  churn op {ops}: alloc failed, freed cache and continued", flush=True)
            ops += 1
            if ops % args.flush == 0:
                torch.cuda.synchronize()
                torch.cuda.empty_cache()                        # force real cudaFree/cudaMalloc
        report(f"churn {ops} ops @ high occ.")
    del ring
    gc.collect()
    torch.cuda.empty_cache()
    report("after churn teardown")
    del ballast
    gc.collect()
    torch.cuda.empty_cache()
    report("after ballast free")


if __name__ == '__main__':
    main()
