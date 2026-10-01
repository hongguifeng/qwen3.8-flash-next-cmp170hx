"""Portable minimal reproducer #5 -- SCATTERED-ACCESS VOLUME.

Every earlier reproducer issued only ~1e5 individual short-row scattered accesses (16k rows x ~10
reps).  A single >=96K-token prefill in this engine issues on the order of 1e7-1e8 of them (many
layers x 2 kv heads x 16 chunks x 98k tokens, 256 B-2 KiB per access, scattered over a 9 GiB pool).
This tests whether *volume* of scattered translation events is the missing ingredient.

  phase: BASELINE -> run N passes of dst.copy_(src[idx]) (each pass = `rows` independent short-row
  gathers) in batches, measuring scatter+dense after every batch.

Footprint stays tiny (src 64 MiB + dst/idx), so it can run with the engine alive.
"""
import argparse, platform, statistics as st
import torch

MiB, GiB = 2 ** 20, 2 ** 30


def bench(fn, reps=7, inner=10):
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
    ap.add_argument('--passes', type=int, default=60000)
    ap.add_argument('--batch', type=int, default=500)
    ap.add_argument('--rows', type=int, default=16384, help='short rows gathered per pass')
    ap.add_argument('--row-bytes', type=int, default=2048)
    ap.add_argument('--src-mib', type=int, default=64)
    args = ap.parse_args()

    w = args.row_bytes // 8
    src_rows = args.src_mib * MiB // args.row_bytes
    src = torch.randint(-(2 ** 62), 2 ** 62, (src_rows, w), dtype=torch.int64, device='cuda')
    dst = torch.empty((args.rows, w), dtype=torch.int64, device='cuda')
    idx = torch.randint(0, src_rows, (args.rows,), device='cuda')
    sc = lambda: dst.copy_(src[idx])
    dn = lambda: dst.copy_(src[:args.rows])
    ref = [None, None]
    total = 0

    def report(tag):
        s = bench(sc)
        d = bench(dn)
        if ref[0] is None:
            ref[0], ref[1] = s, d
        print(f"{tag:40s} scatter {s:8.3f} ms x{s/ref[0]:6.2f}  dense {d:7.3f} ms x{d/ref[1]:5.2f}"
              f"  scattered_rows_so_far={total/1e6:.2f}M", flush=True)

    print(f"platform={platform.platform()} torch={torch.__version__} device={torch.cuda.get_device_name(0)}"
          f" rows={args.rows} roww={args.row_bytes} src={args.src_mib}MiB free={torch.cuda.mem_get_info()[0]/GiB:.2f}GiB",
          flush=True)
    report("BASELINE")
    while total < args.passes * args.rows:
        for _ in range(args.batch):
            sc()
        torch.cuda.synchronize()
        total += args.batch * args.rows
        report(f"after {total/1e6:.2f}M scattered rows")
    report("FINAL")
    if ref[0] is not None:
        print(f"VERDICT (WSL/native): scatter final/baseline = {bench(sc)/ref[0]:.2f}x "
              f"(>=3x => volume reproduced the collapse)", flush=True)


if __name__ == '__main__':
    main()
