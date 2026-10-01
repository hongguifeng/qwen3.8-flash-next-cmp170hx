"""Grid-size sweep of a short-row (256 B @ 4 KB stride) read pattern vs a dense control.

Same instrument that produced the earlier clean references (STR/FSC = 176 GB/s clean,
0.8 GB/s poisoned at torch-default huge grids).  Here the *only* variable is the number
of CTAs, so the answer needs no cross-state comparison: we look at the internal contrast
inside the currently poisoned engine.
"""
import time
import torch
import triton
import triton.language as tl

VEC = 32          # 32 x int64 = 256 B per row  (matches the KV 128-dim bf16 chunk)
STRIDE_W = 512    # 512 words = 4 KB stride     (matches the STR/FSC poisoning pattern)
N_ROWS = 65536    # 65536 x 256 B = 16 MiB of useful data per pass
GRIDS = [65536, 16384, 4096, 1024, 256, 64, 16]


@triton.jit
def k_rows(src, dst, ROWS_PC: tl.constexpr, STRIDE_W: tl.constexpr,
           DENSE: tl.constexpr, VEC: tl.constexpr):
    pid = tl.program_id(0)
    lanes = tl.arange(0, VEC)
    acc = tl.zeros((VEC,), dtype=tl.int64)
    if DENSE:
        base = pid * ROWS_PC * VEC
        for r in range(ROWS_PC):
            acc += tl.load(src + base + r * VEC + lanes)
    else:
        base = pid * ROWS_PC
        for r in range(ROWS_PC):
            acc += tl.load(src + (base + r) * STRIDE_W + lanes)
    tl.store(dst + pid * VEC + lanes, acc)


def bench(n_rows, dense, reps=3):
    grid = (n_rows,)
    dst = torch.empty(grid[0] * VEC, dtype=torch.int64, device="cuda")
    nbytes = n_rows * VEC * 8
    k_rows[grid](src, dst, 1, STRIDE_W, dense, VEC, num_warps=1)
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        k_rows[grid](src, dst, 1, STRIDE_W, dense, VEC, num_warps=1)
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    best = min(ts)
    del dst
    return best * 1e3, nbytes / best / 1e9


def bench_grid(rows_pc, dense, reps=3):
    """rows_pc rows per CTA -> grid = N_ROWS/rows_pc CTAs, same total bytes."""
    grid = (N_ROWS // rows_pc,)
    dst = torch.empty(grid[0] * VEC, dtype=torch.int64, device="cuda")
    nbytes = N_ROWS * VEC * 8
    k_rows[grid](src, dst, rows_pc, STRIDE_W, dense, VEC, num_warps=1)
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        k_rows[grid](src, dst, rows_pc, STRIDE_W, dense, VEC, num_warps=1)
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    best = min(ts)
    del dst
    return best * 1e3, nbytes / best / 1e9


def main():
    global src
    free, total = torch.cuda.mem_get_info()
    print(f"device={torch.cuda.get_device_name(0)} free={free/2**30:.2f} GiB/{total/2**30:.1f}")
    # incompressible payload (all-zero buffers get L2-compressed -> fake fast)
    src = torch.randint(-(2 ** 62), 2 ** 62, (N_ROWS * STRIDE_W,), dtype=torch.int64,
                        device="cuda")
    # correctness sanity on the strided arm
    dst = torch.empty(64 * VEC, dtype=torch.int64, device="cuda")
    k_rows[(64,)](src, dst, 1024, STRIDE_W, False, VEC, num_warps=1)
    torch.cuda.synchronize()
    ref = torch.stack([src[(i * 1024 + r) * STRIDE_W + torch.arange(VEC)] for i in range(64)
                       for r in range(1024)]).sum(0)
    ok = bool(torch.equal(dst, ref))
    print(f"sanity strided grid=64 rows_pc=1024 -> {'OK' if ok else 'MISMATCH'}")

    for label, dense in (("DENSE ", True), ("STRIDED", False)):
        print(f"--- {label} (rows_pc -> grid, ms, GB/s) ---")
        for rows_pc in [1024, 256, 64, 16, 4, 1]:
            ms, gbs = bench_grid(rows_pc, dense)
            print(f"  rows_pc={rows_pc:5d} grid={N_ROWS//rows_pc:6d}  {ms:8.2f} ms  {gbs:7.1f} GB/s")
        # time-resolved: repeat the small-grid point to watch drain
        for k in range(3):
            ms, gbs = bench_grid(1024, dense, reps=1)
            print(f"  drain pass{k}: rows_pc=1024 grid=64  {ms:8.2f} ms  {gbs:7.1f} GB/s")
    print("done")


main()
