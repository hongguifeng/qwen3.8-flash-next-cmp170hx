"""CPU submission cost of CUDA graph replay vs eager launches (WSL2 vs native)."""
import time
import torch

dev = "cuda:0"
torch.zeros(1, device=dev)
torch.cuda.synchronize()

def submit_cost(name, fn, n=200):
    fn(); torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    submit = (time.perf_counter() - t) / n * 1e6
    torch.cuda.synchronize()
    print(f"{name:42s} submit {submit:8.1f} us/call")

a = torch.zeros(1024, device=dev)
b = torch.zeros(1024, device=dev)
submit_cost("eager: 1 kernel launch", lambda: b.add_(1))
submit_cost("eager: 8 kernel launches", lambda: [b.add_(1) for _ in range(8)])

def make_graph(nodes):
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        for _ in range(3):
            for _ in range(nodes):
                b.add_(1)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(nodes):
            b.add_(1)
    torch.cuda.synchronize()
    return g

for nodes in (1, 8, 64, 512):
    g = make_graph(nodes)
    submit_cost(f"graph replay: {nodes} nodes", g.replay, n=100)

# realistic: many small kernels inside one graph, measured with sync per replay
g = make_graph(64)
n = 50
torch.cuda.synchronize()
t = time.perf_counter()
for _ in range(n):
    g.replay()
    torch.cuda.synchronize()
print(f"{'graph replay + synchronize (64 nodes)':42s} {(time.perf_counter()-t)/n*1e6:8.1f} us/call")
