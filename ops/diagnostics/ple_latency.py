"""Measure PLE row read latency with the production AIO helper and with pread."""
import ctypes, os, statistics, sys, time
import numpy as np

PATH = "/model/model-00001-of-00011.safetensors"
ROW = 320          # BF16 row bytes (per ngram head)
ROW_OFF = 8 + 393216  # past the safetensors header (approx; only latency matters)
LIB = "/opt/vllm/optimization/ple_ssd_io.so"
DEPTH = int(sys.argv[1]) if len(sys.argv) > 1 else 256
N = int(sys.argv[2]) if len(sys.argv) > 2 else 200
BATCH = int(sys.argv[3]) if len(sys.argv) > 3 else 16

size = os.path.getsize(PATH)
fd_plain = os.open(PATH, os.O_RDONLY)
fd_direct = os.open(f"/proc/self/fd/{fd_plain}", os.O_RDONLY | os.O_DIRECT)

rng = np.random.default_rng(7)
max_off = (size - 4096 - ROW_OFF) // 320
rows = rng.integers(0, max_off, size=N)

lib = ctypes.CDLL(LIB, use_errno=True)
lib.rows_open.argtypes = [ctypes.c_uint]; lib.rows_open.restype = ctypes.c_void_p
lib.rows_read.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                          ctypes.c_uint, ctypes.c_uint, ctypes.c_void_p]
lib.rows_read.restype = ctypes.c_int
lib.rows_close.argtypes = [ctypes.c_void_p]
reader = lib.rows_open(DEPTH)
if not reader:
    sys.exit(f"rows_open failed errno={ctypes.get_errno()}")

def aio_batch(batch_rows):
    fds = np.full(len(batch_rows), fd_direct, dtype=np.int32)
    offs = np.ascontiguousarray(ROW_OFF + batch_rows.astype(np.int64) * ROW, dtype=np.uint64)
    out = np.empty((len(batch_rows), ROW), dtype=np.uint8)
    t = time.perf_counter()
    rc = lib.rows_read(reader, fds.ctypes.data, offs.ctypes.data, len(batch_rows), ROW, out.ctypes.data)
    dt = time.perf_counter() - t
    if rc:
        raise OSError(-rc, "rows_read failed")
    return dt

lat = []
for i in range(N):
    lat.append(aio_batch(np.array([rows[i]])))
print(f"AIO single row (depth {DEPTH}): mean {statistics.mean(lat)*1e3:.2f} ms  "
      f"median {statistics.median(lat)*1e3:.2f}  p90 {sorted(lat)[int(len(lat)*0.9)]*1e3:.2f}  "
      f"p99 {sorted(lat)[int(len(lat)*0.99)]*1e3:.2f}  ({N} samples)")

bl = []
for i in range(0, N - BATCH, BATCH):
    bl.append(aio_batch(rows[i:i+BATCH]))
per = [b / BATCH for b in bl]
print(f"AIO batch {BATCH} consecutive (production batch shape): mean {statistics.mean(bl)*1e3:.2f} ms/batch -> "
      f"{statistics.mean(per)*1e3:.2f} ms/row  p90 {sorted(bl)[int(len(bl)*0.9)]*1e3:.2f} ms/batch")

# page cache / buffered reads (no O_DIRECT)
def pread_batch(batch_rows):
    t = time.perf_counter()
    for r in batch_rows:
        os.pread(fd_plain, ROW, ROW_OFF + int(r) * ROW)
    return time.perf_counter() - t

lat2 = [pread_batch(np.array([rows[i]])) for i in range(min(N, 100))]
print(f"pread buffered single row (cold): mean {statistics.mean(lat2)*1e3:.2f} ms")
lat3 = [pread_batch(np.array([rows[i]])) for i in range(min(N, 100))]
print(f"pread buffered single row (again): mean {statistics.mean(lat3)*1e3:.2f} ms")
lib.rows_close(ctypes.c_void_p(reader))
