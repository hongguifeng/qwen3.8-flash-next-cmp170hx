"""Portable minimal reproducer #3 -- adds the ranked "missing ingredients" from consultation #9.

Measured arms (identical on WSL2/Linux and native Windows, pure torch + ctypes driver API):
  scatter : dst.copy_(src[idx])     -- 16384 random 2 KiB rows out of a 64 MiB src (the engine's failing shape)
  dense   : dst.copy_(src[:16384])  -- same buffers, contiguous (control)

Stress ladders (each one cumulative, measured after every step):
  --churn  N...     allocation churn: N mixed-size alloc/free ops (4 KiB..2 MiB), recycling the
                    caching-allocator pool, and every 512 ops a real cache release (empty_cache)
                    so genuine new mappings are created.
  --vmm    N...     CUDA VMM map/unmap churn: reserve VA, pool of 2 MiB physical handles, map at
                    N distinct VAs in sequence, touch, unmap.  Forces VA-mapping bookkeeping churn.
  --vmm-keep N...   keep N VMM mappings resident (persistent mappings) instead of unmapping.
  --pin-side MiB... persistent pinned host buffers + CPU producer thread + nonblocking H2D on a
                    side CUDA stream, continuously in flight WHILE the gather is timed.
  --graphs N...     capture and keep N small CUDA graphs, then replay them.
"""
import argparse, ctypes, gc, os, platform, statistics as st, sys, threading, time
import torch
import torch.utils.dlpack as dlpack

MiB, GiB = 2 ** 20, 2 ** 30
LIB = 'nvcuda.dll' if os.name == 'nt' else 'libcuda.so.1'


# ---------------------------------------------------------------- VMM via driver API
class CULoc(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("id", ctypes.c_int)]


class CUAllocFlags(ctypes.Structure):
    _fields_ = [("compressionType", ctypes.c_ubyte), ("gpuDirectRDMACapable", ctypes.c_ubyte),
                ("usage", ctypes.c_ushort), ("reserved", ctypes.c_ubyte * 4)]


class CUAllocProp(ctypes.Structure):
    _fields_ = [("type", ctypes.c_int), ("requestedHandleTypes", ctypes.c_int),
                ("location", CULoc), ("win32HandleMetaData", ctypes.c_void_p),
                ("allocFlags", CUAllocFlags)]


class CUAccessDesc(ctypes.Structure):
    _fields_ = [("location", CULoc), ("flags", ctypes.c_int)]


class DLDevice(ctypes.Structure):
    _fields_ = [("device_type", ctypes.c_int), ("device_id", ctypes.c_int)]


class DLDataType(ctypes.Structure):
    _fields_ = [("code", ctypes.c_uint8), ("bits", ctypes.c_uint8), ("lanes", ctypes.c_uint16)]


class DLTensor(ctypes.Structure):
    _fields_ = [("data", ctypes.c_void_p), ("device", DLDevice), ("ndim", ctypes.c_int),
                ("dtype", DLDataType), ("shape", ctypes.POINTER(ctypes.c_int64)),
                ("strides", ctypes.POINTER(ctypes.c_int64)), ("byte_offset", ctypes.c_uint64)]


class DLManagedTensor(ctypes.Structure):
    _fields_ = [("dl_tensor", DLTensor), ("manager_ctx", ctypes.c_void_p), ("deleter", ctypes.c_void_p)]


ctypes.pythonapi.PyCapsule_New.restype = ctypes.py_object


class Vmm:
    def __init__(self):
        self.cu = ctypes.CDLL(LIB)
        self.keep = []
        p = CUAllocProp()
        p.type = 1                      # CU_MEM_ALLOCATION_TYPE_PINNED
        p.requestedHandleTypes = 0
        p.location.type = 1             # CU_MEM_LOCATION_TYPE_DEVICE
        p.location.id = 0
        g = ctypes.c_size_t()
        self._ck(self.cu.cuMemGetAllocationGranularity(ctypes.byref(g), ctypes.byref(p), 0), "granularity")
        self.gran = g.value
        self.prop = p

    def _ck(self, r, what):
        if r != 0:
            raise RuntimeError(f"vmm {what} -> CUDA error {r:#x}")

    def _al(self, n):
        return (n + self.gran - 1) // self.gran * self.gran

    def phys(self, nbytes):
        n = self._al(nbytes)
        h = ctypes.c_ulonglong()
        self._ck(self.cu.cuMemCreate(ctypes.byref(h), ctypes.c_size_t(n), ctypes.byref(self.prop),
                                     ctypes.c_ulonglong(0)), "cuMemCreate")
        return int(h.value), n

    def reserve(self, nbytes):
        n = self._al(nbytes)
        va = ctypes.c_ulonglong()
        self._ck(self.cu.cuMemAddressReserve(ctypes.byref(va), ctypes.c_size_t(n),
                                             ctypes.c_size_t(self.gran), ctypes.c_ulonglong(0),
                                             ctypes.c_ulonglong(0)), "cuMemAddressReserve")
        return int(va.value), n

    def map(self, va, handle, nbytes):
        self._ck(self.cu.cuMemMap(ctypes.c_ulonglong(va), ctypes.c_size_t(nbytes), ctypes.c_size_t(0),
                                  ctypes.c_ulonglong(handle), ctypes.c_ulonglong(0)), "cuMemMap")
        d = CUAccessDesc()
        d.location.type, d.location.id, d.flags = 1, 0, 3
        self._ck(self.cu.cuMemSetAccess(ctypes.c_ulonglong(va), ctypes.c_size_t(nbytes),
                                        ctypes.byref(d), ctypes.c_size_t(1)), "cuMemSetAccess")

    def unmap(self, va, nbytes):
        self._ck(self.cu.cuMemUnmap(ctypes.c_ulonglong(va), ctypes.c_size_t(nbytes)), "cuMemUnmap")

    def release(self, h):
        self._ck(self.cu.cuMemRelease(ctypes.c_ulonglong(h)), "cuMemRelease")

    def tensor(self, va, nbytes):
        t = DLManagedTensor()
        n64 = nbytes // 8
        shp = (ctypes.c_int64 * 1)(n64)
        t.dl_tensor.data = ctypes.c_void_p(va)
        t.dl_tensor.device = DLDevice(2, 0)
        t.dl_tensor.ndim = 1
        t.dl_tensor.dtype = DLDataType(0, 64, 1)
        t.dl_tensor.shape = shp
        t.dl_tensor.strides = None
        t.dl_tensor.byte_offset = 0
        cap = ctypes.pythonapi.PyCapsule_New(ctypes.byref(t), b"dltensor", None)
        self.keep.append((t, shp))
        return dlpack.from_dlpack(cap)


# ---------------------------------------------------------------- timing
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
    ap.add_argument('--rows', type=int, default=16384)
    ap.add_argument('--row-bytes', type=int, default=2048)
    ap.add_argument('--churn', type=str, default='')
    ap.add_argument('--vmm', type=str, default='')
    ap.add_argument('--vmm-keep', type=str, default='')
    ap.add_argument('--pin-side', type=str, default='')
    ap.add_argument('--graphs', type=str, default='')
    args = ap.parse_args()
    num = lambda s: [int(x) for x in s.split(',') if x]

    print(f"platform={platform.platform()} python={platform.python_version()} torch={torch.__version__}", flush=True)
    free, total = torch.cuda.mem_get_info()
    rss = ''
    if os.path.exists('/proc/self/statm'):
        rss = f" rss={int(open('/proc/self/statm').read().split()[1])*4096/GiB:.1f} GiB"
    print(f"device={torch.cuda.get_device_name(0)} mem_get_info free={free/GiB:.2f} tot={total/GiB:.2f} GiB{rss}", flush=True)

    src_rows = args.rows * 2
    w = args.row_bytes // 8
    src = torch.randint(-(2 ** 62), 2 ** 62, (src_rows, w), dtype=torch.int64, device='cuda')
    dst = torch.empty((args.rows, w), dtype=torch.int64, device='cuda')
    idx = torch.randint(0, src_rows, (args.rows,), device='cuda')
    sc = lambda: dst.copy_(src[idx])
    dn = lambda: dst.copy_(src[:args.rows])
    payload = dst.numel() * 8 / MiB

    ref = [None, None]

    def report(tag):
        s = bench(sc)
        d = bench(dn)
        if ref[0] is None:
            ref[0], ref[1] = s, d
        print(f"{tag:38s} scatter {s:8.3f} ms x{s/ref[0]:5.2f}   dense {d:7.3f} ms x{d/ref[1]:5.2f}", flush=True)
        return s

    report("BASELINE")

    # ---- allocation churn
    if args.churn:
        sizes = [4 * 1024, 64 * 1024, 512 * 1024, 2 * MiB]
        live, done = [], 0
        for target in num(args.churn):
            if target <= 0:
                continue
            while done < target:
                t = torch.empty(sizes[done % 4], dtype=torch.uint8, device='cuda')
                t.fill_(1)
                live.append(t)
                del live[:-64]
                done += 1
                if done % 512 == 0:
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
            report(f"churn {done} alloc/free ops")
        del live
        gc.collect()
        torch.cuda.empty_cache()
        report("after churn teardown")

    # ---- VMM map/unmap churn (and persistent mappings)
    if args.vmm or args.vmm_keep:
        v = Vmm()
        nphys = 16
        pool = [v.phys(2 * MiB) for _ in range(nphys)]
        nva = 8192                                   # 8192 * 2 MiB = 16 GiB VA range
        base_va, va_n = v.reserve(nva * 2 * MiB)
        print(f"VMM: granularity={v.gran} phys_pool={nphys}x2MiB VA_range={va_n/GiB:.1f} GiB "
              f"base={hex(base_va)}", flush=True)
        done = 0
        for target in num(args.vmm):
            if target <= 0:
                continue
            while done < target:
                off = done % nva
                va = base_va + off * 2 * MiB
                h = pool[done % nphys][0]
                v.map(va, h, 2 * MiB)
                t = v.tensor(va, 2 * MiB)
                t[:1].fill_(1)                       # touch -> page-table entry gets used
                torch.cuda.synchronize()             # cuMemUnmap is undefined while ops are pending
                del t
                v.unmap(va, 2 * MiB)
                done += 1
            torch.cuda.synchronize()
            report(f"vmm map/unmap churn {done}")

        held = []
        for target in num(args.vmm_keep):
            if target <= 0:
                continue
            while len(held) < min(target, nva):
                off = len(held)
                va = base_va + off * 2 * MiB
                v.map(va, pool[len(held) % nphys][0], 2 * MiB)
                t = v.tensor(va, 2 * MiB)
                t[:1].fill_(1)
                held.append(t)
            torch.cuda.synchronize()
            report(f"vmm persistent mappings {len(held)} ({len(held)*2} MiB)")

    # ---- pinned host staging + CPU producer + side-stream H2D, continuously in flight
    if args.pin_side:
        stop = threading.Event()
        nbytes = 16 * MiB
        pins = [torch.empty(nbytes // 4, dtype=torch.float32, pin_memory=True) for _ in range(4)]
        dev = torch.empty(nbytes // 4, dtype=torch.float32, device='cuda')
        side = torch.cuda.Stream()
        counter = [0]

        def producer():
            i = 0
            while not stop.is_set():
                p = pins[i % len(pins)]
                p.fill_(float(i % 7))
                with torch.cuda.stream(side):
                    dev.copy_(p, non_blocking=True)
                i += 1
                counter[0] = i

        for mb in num(args.pin_side):
            while len(pins) * (nbytes // MiB) < mb:
                pins.append(torch.empty(nbytes // 4, dtype=torch.float32, pin_memory=True))
            th = threading.Thread(target=producer, daemon=True)
            th.start()
            time.sleep(0.7)
            report(f"pinned staging {len(pins)*(nbytes//MiB)} MiB + side-stream H2D")
            stop.set()
            th.join(timeout=5)
            torch.cuda.synchronize()
            stop.clear()
            print(f"    ({counter[0]} H2D copies issued while gather was timed)", flush=True)

    # ---- CUDA graph pools
    if args.graphs:
        graphs = []
        for target in num(args.graphs):
            while len(graphs) < target:
                x = torch.empty(1 * MiB, dtype=torch.uint8, device='cuda')
                y = torch.empty_like(x)
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g):
                    y.copy_(x)
                graphs.append((g, x, y))
            for g, _, _ in graphs:
                g.replay()
            torch.cuda.synchronize()
            report(f"cuda graphs captured+replayed {len(graphs)}")
        graphs.clear()
        gc.collect()
        torch.cuda.empty_cache()
        report("after graph teardown")

    print(f"torch allocated={torch.cuda.memory_allocated()/GiB:.2f} GiB reserved={torch.cuda.memory_reserved()/GiB:.2f} GiB",
          flush=True)


if __name__ == '__main__':
    main()
