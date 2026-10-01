"""Footprint sweep: same random-row gather pattern, growing address footprint.

Engine (poisoned): random 256 B rows over its own ~64 MiB scratch -> 0.8 GB/s.
Fresh process, poisoned state, 256 MiB source             -> 42 GB/s.
Hypothesis: what collapses is scattered access over a *large mapped footprint*
(GPU TLB / page-walk coverage), which the engine process has (~61 GiB VA) and a
fresh small process does not.  Payload is high-entropy so L2 compression cannot
fake a fast result.
"""
import time
import torch

DEV = "cuda"
ROWS = 16384
ROW_W = 32                      # 256 B rows
MULT = 0x9E3779B97F4A7C15


def fill_hi_entropy(n):
    x = torch.arange(n, dtype=torch.int64, device=DEV)
    x.mul_(MULT)
    return x


def timed(fn, reps=12):
    fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    return sorted(ts)


def main():
    out = torch.empty(ROWS, ROW_W, dtype=torch.int64, device=DEV)
    nbytes = ROWS * ROW_W * 8
    print(f"free={torch.cuda.mem_get_info()[0]/2**30:.2f} GiB  pass={nbytes/2**20:.0f} MiB")
    for mib in (16, 64, 256, 1024, 2048, 3072):
        n_rows = mib * 2 ** 20 // (ROW_W * 8)
        try:
            src = fill_hi_entropy(n_rows * ROW_W)
            idx = torch.randint(0, n_rows, (ROWS, ROW_W), dtype=torch.int64, device=DEV)
            idx.mul_(ROW_W).add_(torch.arange(ROW_W, dtype=torch.int64, device=DEV))
            ts = timed(lambda: out.copy_(src[idx]))
            # dense control on the same buffer
            tdc = timed(lambda: out.view(-1).copy_(src[: ROWS * ROW_W]))
            print(f"footprint={mib:5d} MiB  pages~{n_rows*256//4096:9d} | "
                  f"scatter med={ts[len(ts)//2]:8.3f} max={ts[-1]:8.3f} ms "
                  f"{nbytes/(ts[len(ts)//2]/1e3)/1e9:7.1f} GB/s | "
                  f"dense med={tdc[len(tdc)//2]:7.3f} ms "
                  f"{nbytes/(tdc[len(tdc)//2]/1e3)/1e9:7.1f} GB/s")
            del src, idx
            torch.cuda.empty_cache()
        except Exception as e:  # noqa: BLE001
            print(f"footprint={mib} MiB FAILED {e!r}")
            torch.cuda.empty_cache()


main()
