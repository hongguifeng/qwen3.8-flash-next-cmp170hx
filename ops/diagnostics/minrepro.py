#!/usr/bin/env python3
"""Minimal reproducer: does scattered short-row GPU access collapse after the
resident (mapped+touched) device memory grows?

Deliberately uses ONLY torch ops (no Triton, no vLLM) so that the identical file
runs natively on Windows and inside WSL -- that is the point of the test: if the
collapse reproduces on native Windows too, the carrier is in the Windows
driver/MCDM/dxgkrnl layer; if only WSL reproduces it, the WSL GPU-PV guest layer
is implicated.

Arms (exactly the shapes measured in the engine probe):
  scatter : dst.copy_(src[idx])     16384 random 2 KiB rows (64 MiB footprint)
  dense   : dst.copy_(src[:16384])  same 32 MiB, contiguous
Reference: engine probe measured scatter 0.19 ms clean / 20-40 ms "poisoned",
dense 0.09 ms in both states.

Ladder: start with ~1 GiB resident, grow by --step GiB, re-measure after every
step, then free everything and re-measure once more (recovery check).
"""
import argparse
import platform
import sys
import time

import torch

ROWS = 16384
ROWW = 2048
SRC_ROWS = 32768


def bench(fn, reps=3):
    fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    return min(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gb", type=float, default=56.0, help="target resident GiB")
    ap.add_argument("--step", type=float, default=8.0, help="GiB per ladder step")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--no-ballast", action="store_true",
                    help="only measure the baseline and the recovery")
    a = ap.parse_args()

    print(f"platform={platform.platform()} python={sys.version.split()[0]} "
          f"torch={torch.__version__}", flush=True)
    dev = "cuda"
    torch.cuda.init()
    try:
        free, total = torch.cuda.mem_get_info()
        print(f"device={torch.cuda.get_device_name(0)} "
              f"free={free / 2**30:.2f} GiB total={total / 2**30:.2f} GiB", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"mem_get_info failed: {e!r}", flush=True)

    # incompressible payload so L2 compression cannot fake a fast result
    src = torch.randint(0, 255, (SRC_ROWS, ROWW), dtype=torch.uint8, device=dev)
    dst = torch.empty(ROWS, ROWW, dtype=torch.uint8, device=dev)
    idx = torch.randint(0, SRC_ROWS - 1, (ROWS,), dtype=torch.int64, device=dev)
    nbytes = ROWS * ROWW
    print(f"src=0x{src.data_ptr():x} ({SRC_ROWS * ROWW / 2**20:.0f} MiB) "
          f"pass={nbytes / 2**20:.0f} MiB", flush=True)

    sc = lambda: dst.copy_(src[idx])          # noqa: E731
    dn = lambda: dst.copy_(src[:ROWS])        # noqa: E731

    def show(tag, t_sc, t_dn):
        print(f"{tag:34s} scatter {t_sc:9.3f} ms {nbytes / (t_sc / 1e3) / 1e9:6.1f} GB/s"
              f"   dense {t_dn:7.3f} ms {nbytes / (t_dn / 1e3) / 1e9:6.1f} GB/s",
              flush=True)

    show("baseline", bench(sc, a.reps), bench(dn, a.reps))

    ballast = []
    gb = 0.0
    if not a.no_ballast:
        while gb < a.gb:
            take = min(a.step, a.gb - gb)
            try:
                b = torch.empty(int(take * 2**30), dtype=torch.uint8, device=dev)
                b.zero_()                       # touch every page
                ballast.append(b)
            except Exception as e:  # noqa: BLE001
                print(f"ballast +{take:.0f} GiB FAILED: {e!r}", flush=True)
                break
            gb += take
            torch.cuda.synchronize()
            try:
                freen = torch.cuda.mem_get_info()[0] / 2**30
            except Exception:  # noqa: BLE001
                freen = float("nan")
            show(f"resident {gb:5.0f} GiB (+{take:.0f}) free {freen:5.1f}",
                 bench(sc, a.reps), bench(dn, a.reps))

    if ballast:
        print(f"ballast VA range 0x{min(b.data_ptr() for b in ballast):x} .. "
              f"0x{max(b.data_ptr() for b in ballast):x}", flush=True)
    ballast.clear()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    show("after free + empty_cache", bench(sc, a.reps), bench(dn, a.reps))
    print(f"torch allocated={torch.cuda.memory_allocated() / 2**30:.2f} GiB "
          f"reserved={torch.cuda.memory_reserved() / 2**30:.2f} GiB", flush=True)


main()
