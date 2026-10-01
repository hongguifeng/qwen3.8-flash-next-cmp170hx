"""Replicate the engine-probe memory-control arms *outside* the engine process.

Engine arms (in `_xp_probe`, same poisoned state, same instant):
    KRD : copy_(k_cache[:16])          -> 626 GB/s  (dense)
    KSC : 16384 random 2 KB rows       -> 1.6 GB/s  (73x down)
    FSC : _fro.copy_(_frb[_fri])       -> 0.8 GB/s  (210x down)
    STR : _fro.copy_(_frb[_sa])        -> 0.8 GB/s  (200x down)
    thermometer (trition, perm-indirect, 8 B lanes) -> unaffected

Question: is the collapse the *process/allocation* or the *pattern*?
This reproduces the same patterns in a fresh process, and adds a duration/rep
sweep so rare multi-ms stalls show up as a heavy tail across repetitions.
"""
import time
import torch

DEV = "cuda"
ROWS = 16384          # rows per pass
ROW_W = 32            # int64 words per row  -> 256 B rows
SRC_ROWS = 1 << 20    # source has 1M rows (256 MB)


def timed(fn, reps):
    fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    return ts


def stats(ts, nbytes):
    ts = sorted(ts)
    return (f"min={ts[0]:8.3f} med={ts[len(ts)//2]:8.3f} p90={ts[int(len(ts)*0.9)]:8.3f} "
            f"max={ts[-1]:9.3f} ms   {nbytes/ (sum(ts)/len(ts)) / 1e9:7.2f} GB/s(mean)  "
            f"{nbytes/(ts[0]/1e3)/1e9:7.2f} GB/s(best)")


def main():
    torch.manual_seed(0)
    src = torch.randint(-(2 ** 62), 2 ** 62, (SRC_ROWS, ROW_W), dtype=torch.int64, device=DEV)
    out = torch.empty(ROWS, ROW_W, dtype=torch.int64, device=DEV)
    nbytes = ROWS * ROW_W * 8

    # index tensors: random rows (as in FSC/STR) vs sequential (dense control)
    rand_idx = torch.randint(0, SRC_ROWS, (ROWS, ROW_W), dtype=torch.int64, device=DEV)
    seq_idx = (torch.arange(ROWS, device=DEV)[:, None] * ROW_W
               + torch.arange(ROW_W, device=DEV)[None, :])
    # 2 GiB bf16 "KV-like" pool; gather 16384 random 256 B chunks from it
    pool16 = torch.randint(-(2 ** 14), 2 ** 14, (1 << 30,), dtype=torch.bfloat16, device=DEV)
    pstart = torch.randint(0, (1 << 30) - 256, (16384, 1), device=DEV)
    pidx = pstart + torch.arange(128, device=DEV)[None, :]
    pout = torch.empty(16384, 128, dtype=torch.bfloat16, device=DEV)

    print(f"src={src.numel()*8/2**20:.0f} MiB pool, pass={nbytes/2**20:.0f} MiB, rows={ROWS}")
    arms = [
        ("dense copy (contig)",        lambda: out.view(-1).copy_(src[:ROWS].reshape(-1))),
        ("gather seq idx (dense-ish)", lambda: out.copy_(src.view(-1)[seq_idx])),
        ("gather RAND idx (FSC/STR)",  lambda: out.copy_(src.view(-1)[rand_idx])),
        ("gather 256B rows from pool", lambda: pout.copy_(pool16[pidx])),
    ]
    for name, fn in arms:
        try:
            ts = timed(fn, 12)
            print(f"{name:28s} {stats(ts, nbytes)}")
        except Exception as e:  # noqa: BLE001
            print(f"{name:28s} FAILED {e!r}")

    # duration stress: repeat the worst arm many times to expose a rare tail
    print("--- 60 sequential reps of the random-index gather (tail hunt) ---")
    ts = timed(lambda: out.copy_(src.view(-1)[rand_idx]), 60)
    ts_sorted = sorted(ts)
    over = [round(t, 2) for t in ts_sorted if t > 4 * ts_sorted[0]]
    print(f"  {stats(ts, nbytes)}")
    print(f"  outliers(>4x min)={len(over)} -> {over[:20]}")


main()
