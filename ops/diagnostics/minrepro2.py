"""Portable minimal reproducer #2 -- adds the ingredients the engine has and minrepro.py lacked:
   * pinned (page-locked) HOST memory  -> GPU-PV / MCDM host-mapped staging path
   * allocation churn (many small fresh device mappings)
Runs identically on WSL2/Linux and native Windows (pure torch, no Triton/vLLM).

Arms measured every step:
  scatter : dst.copy_(src[idx])  -- 16384 random 2 KiB rows out of a 64 MiB src  (the engine's failing shape)
  dense   : dst.copy_(src[:16384]) -- same buffers, contiguous (control)
  xfer    : pinned<->device copy bandwidth (diagnostic for the host-mapping path)
"""
import argparse, ctypes, gc, os, platform, statistics as st, sys, time
import torch

MiB = 2 ** 20
GiB = 2 ** 30


def host_free_gb():
    if os.path.exists('/proc/meminfo'):
        d = {}
        for line in open('/proc/meminfo'):
            k, v = line.split(':')
            d[k] = int(v.split()[0])
        return d.get('MemAvailable', 0) / (MiB * 1024)
    class M(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
    m = M()
    m.dwLength = ctypes.sizeof(M)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.ullAvailPhys / GiB


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
    ms = sorted(b.elapsed_time(a) / inner for a, b in ev)
    return ms[len(ms) // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rows', type=int, default=16384)
    ap.add_argument('--row-bytes', type=int, default=2048)
    ap.add_argument('--pin', type=str, default='0.5', help='cumulative pinned GiB ladder, comma separated')
    ap.add_argument('--churn', type=str, default='0', help='cumulative churn allocations of 8 MiB, comma separated')
    args = ap.parse_args()

    dev = 'cuda'
    free, total = torch.cuda.mem_get_info()
    print(f"platform={platform.platform()} python={platform.python_version()} torch={torch.__version__}")
    print(f"device={torch.cuda.get_device_name(0)} free={free/GiB:.2f} GiB total={total/GiB:.2f} GiB host_free={host_free_gb():.1f} GiB")

    src_rows = args.rows * 2
    src = torch.randint(-(2 ** 62), 2 ** 62, (src_rows, args.row_bytes // 8), dtype=torch.int64, device=dev)
    dst = torch.empty((args.rows, args.row_bytes // 8), dtype=torch.int64, device=dev)
    idx = torch.randint(0, src_rows, (args.rows,), device=dev)
    xbuf = torch.empty(2 * MiB, dtype=torch.uint8, device=dev)
    print(f"src={hex(src.data_ptr())} ({src.numel()*8/MiB:.0f} MiB) pass={dst.numel()*8/MiB:.0f} MiB")

    sc = lambda: dst.copy_(src[idx])
    dn = lambda: dst.copy_(src[:args.rows])
    xf = lambda: xbuf.copy_(pin_host)          # device <- pinned host

    pin_host = torch.empty(2 * MiB, dtype=torch.uint8, pin_memory=True)
    pin_host.fill_(7)

    def report(tag):
        s = bench(sc)
        d = bench(dn)
        x = bench(xf)
        print(f"{tag:34s} scatter {s:7.3f} ms {dst.numel()*8/MiB/s*1000:8.1f} GB/s   "
              f"dense {d:6.3f} ms {dst.numel()*8/MiB/d*1000:8.1f} GB/s   pin->dev {2/x*1000:7.1f} GB/s", flush=True)

    report("baseline")

    held_pin, held_dev = [], []
    pin_ladder = [float(x) for x in args.pin.split(',') if x]
    churn_ladder = [int(x) for x in args.churn.split(',') if x]
    steps = max(len(pin_ladder), len(churn_ladder))
    for i in range(steps):
        note = ''
        if i < len(pin_ladder):
            want = int(pin_ladder[i] * GiB)
            have = sum(t.numel() for t in held_pin)
            while have < want:
                n = min(64 * MiB, want - have)
                t = torch.empty(n, dtype=torch.uint8, pin_memory=True)
                t.fill_(3)
                have += n
                held_pin.append(t)
            note += f"pin={sum(t.numel() for t in held_pin)/GiB:.2f}GiB "
        if i < len(churn_ladder):
            while len(held_dev) < churn_ladder[i]:
                t = torch.empty(2 * MiB, dtype=torch.uint8, device=dev)
                t.fill_(5)
                held_dev.append(t)
            note += f"churn={len(held_dev)}x2MiB "
        report(f"step {i} (+{note.strip()})")

    for t in held_pin + held_dev:
        del t
    held_pin.clear()
    held_dev.clear()
    gc.collect()
    torch.cuda.empty_cache()
    report("after free + empty_cache")
    print(f"torch device allocated={torch.cuda.memory_allocated()/GiB:.2f} GiB reserved={torch.cuda.memory_reserved()/GiB:.2f} GiB")


if __name__ == '__main__':
    main()
