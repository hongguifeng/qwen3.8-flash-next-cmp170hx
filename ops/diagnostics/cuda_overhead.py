"""Per-call CUDA API overhead as seen from inside a WSL2 container."""
import time
import torch

torch.cuda.init()
dev = torch.device("cuda:0")
torch.zeros(1, device=dev)
torch.cuda.synchronize()

def bench(name, fn, n=300):
    fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    print(f"{name:34s} {(time.perf_counter()-t)/n*1e6:8.1f} us/call")

ev = torch.cuda.Event()
ev2 = torch.cuda.Event()
a = torch.zeros(8, device=dev)
b = torch.zeros(8, device=dev)
small_cpu = torch.zeros(16, dtype=torch.int64, pin_memory=True)
small_gpu = torch.zeros(16, dtype=torch.int64, device=dev)
stream = torch.cuda.Stream(device=dev)

bench("cudaDeviceSynchronize", lambda: torch.cuda.synchronize())
bench("event.record()+synchronize()", lambda: (ev.record(), ev.synchronize()))
bench("small D2D copy (8 floats)", lambda: b.copy_(a))
bench("small H2D copy (16 int64 pinned)", lambda: small_gpu.copy_(small_cpu, non_blocking=True))
bench("small D2H copy (16 int64 pinned)", lambda: small_cpu.copy_(small_gpu, non_blocking=True))
bench("scale_(1) tiny kernel", lambda: a.mul_(1.0))
bench("event query()", lambda: ev.query())

def wait_stream_pair():
    stream.wait_stream(torch.cuda.current_stream())
    torch.cuda.current_stream().wait_event(ev)
bench("stream/event waits (2)", wait_stream_pair)
bench("alloc-free small tensor", lambda: torch.empty(16, dtype=torch.int64, device=dev))

def alloc_pinned():
    torch.empty(16, dtype=torch.int64, pin_memory=True)
bench("pinned host alloc (16 int64)", alloc_pinned)
