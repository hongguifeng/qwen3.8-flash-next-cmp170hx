#!/usr/bin/env python3
"""Resumable parallel HF downloader over a mirror endpoint (range requests)."""
import json
import os
import signal
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import quote

import requests

REPO = "klee100/Qwen3.8-Flash-Next-AutoRound-3bpw-MTP"
REVISION = "ce0e0b94083895bd836b916f29bf105c40a8162a"
API = "https://hf-mirror.com"
API_META = "https://huggingface.co"
DEST = os.path.expanduser("~/models/Qwen3.8-Flash-Next-AutoRound-3bpw-MTP")
STATE = os.path.join(DEST, ".fdl-state.json")
CHUNK = 16 * 1024 * 1024
CONCURRENCY = 48
STALL_LIMIT = 420
MAX_RETRY = 10

lock = threading.Lock()
STATS = {"done": 0, "total": 0, "t0": time.time(), "fail": 0, "last": time.time()}
STATE_DATA = {}


def get_files():
    # The mirror's metadata endpoint omits sizes; fetch the manifest from the
    # upstream API (small JSON) and the blobs from the mirror.
    url = f"{API_META}/api/models/{REPO}/revision/{REVISION}?blobs=true"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    files = [
        (s["rfilename"], s["size"])
        for s in r.json().get("siblings", [])
        if s.get("size") is not None
    ]
    if not files:
        raise RuntimeError("no file sizes returned by metadata API")
    return sorted(files, key=lambda f: -f[1])


def file_url(name):
    return f"{API}/{REPO}/resolve/{REVISION}/{quote(name)}"


URL_CACHE: dict[str, tuple[str, float]] = {}
URL_LOCK = threading.Lock()
URL_TTL = 2700  # mirror redirects to a signed xet URL valid for one hour


def signed_url(name):
    """Resolve the mirror redirect once and reuse the signed URL for chunks."""
    with URL_LOCK:
        hit = URL_CACHE.get(name)
        if hit and time.time() < hit[1]:
            return hit[0]
    r = requests.get(
        file_url(name),
        headers={"Range": "bytes=0-0"},
        stream=True,
        timeout=(30, 60),
        allow_redirects=True,
    )
    final = r.url
    r.close()
    with URL_LOCK:
        URL_CACHE[name] = (final, time.time() + URL_TTL)
    return final


def invalidate_url(name):
    with URL_LOCK:
        URL_CACHE.pop(name, None)


def load_state():
    if os.path.exists(STATE):
        try:
            with open(STATE) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_state(state):
    with lock:
        STATE_DATA.clear()
        STATE_DATA.update(state)
    tmp = STATE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, STATE)


def _sigterm(_signum, _frame):
    print("signal received; saving state", flush=True)
    try:
        save_state(STATE_DATA)
    finally:
        os._exit(0)


signal.signal(signal.SIGTERM, _sigterm)
signal.signal(signal.SIGINT, _sigterm)


def fetch(name, size, idx):
    """Download one chunk; returns number of bytes written."""
    start, end = idx * CHUNK, min((idx + 1) * CHUNK, size) - 1
    last = None
    for attempt in range(MAX_RETRY):
        try:
            url = signed_url(name)
            headers = {"Range": f"bytes={start}-{end}"} if idx or size > CHUNK else {}
            deadline = time.monotonic() + 150
            with requests.get(
                url, headers=headers, stream=True, timeout=(30, 60)
            ) as r:
                if r.status_code == 200 and size > CHUNK:
                    raise RuntimeError("server ignored range")
                if r.status_code in (401, 403):
                    invalidate_url(name)
                    raise RuntimeError(f"HTTP {r.status_code}")
                if r.status_code not in (200, 206):
                    raise RuntimeError(f"HTTP {r.status_code}")
                buf = bytearray()
                for piece in r.iter_content(1024 * 512):
                    buf.extend(piece)
                    if time.monotonic() > deadline:
                        raise TimeoutError(
                            f"chunk deadline hit at {len(buf)}/{end - start + 1} bytes"
                        )
                if len(buf) != end - start + 1:
                    raise RuntimeError(f"short read {len(buf)} != {end - start + 1}")
            with open(os.path.join(DEST, name + ".part"), "r+b") as f:
                f.seek(start)
                f.write(buf)
            return len(buf)
        except Exception as exc:  # noqa: BLE001
            last = exc
            with lock:
                STATS["fail"] += 1
            time.sleep(min(1.5 * (attempt + 1), 20))
    raise RuntimeError(f"{name} chunk {idx}: {last}")


def watchdog():
    while True:
        time.sleep(60)
        with lock:
            idle = time.time() - STATS["last"]
        if idle > STALL_LIMIT:
            print(f"no chunk completed for {idle:.0f}s; saving state and exiting", flush=True)
            try:
                save_state(STATE_DATA)
            finally:
                os._exit(2)


def report():
    while True:
        time.sleep(30)
        with lock:
            done, total = STATS["done"], STATS["total"]
        el = time.time() - STATS["t0"]
        rate = done / el / 2**20 if el else 0
        eta = (total - done) / 2**20 / rate / 3600 if rate else float("inf")
        print(
            f"[{time.strftime('%H:%M:%S')}] {done / 2**30:8.2f}/{total / 2**30:.2f} GiB "
            f"({done / max(total, 1) * 100:5.1f}%)  {rate:5.2f} MiB/s  ETA {eta:5.2f} h  "
            f"retries={STATS['fail']}",
            flush=True,
        )


def main():
    os.makedirs(DEST, exist_ok=True)
    files = get_files()
    size_by_name = dict(files)
    total = sum(size_by_name.values())
    state = load_state()
    done = 0
    for name, idxs in list(state.items()):
        size = size_by_name.get(name, 0)
        for idx in set(idxs):
            done += min(CHUNK, size - idx * CHUNK)
    STATS.update(total=total, done=done)
    print(
        f"{len(files)} files, {total / 2**30:.2f} GiB total, {done / 2**30:.2f} GiB cached",
        flush=True,
    )
    threading.Thread(target=report, daemon=True).start()
    threading.Thread(target=watchdog, daemon=True).start()

    for name, size in files:
        final = os.path.join(DEST, name)
        if os.path.exists(final) and os.path.getsize(final) == size:
            print(f"== {name} already complete", flush=True)
            continue
        part = final + ".part"
        if not os.path.exists(part) or os.path.getsize(part) != size:
            with open(part, "wb") as f:
                f.truncate(size)
            state[name] = []
            with lock:
                STATS["done"] = sum(
                    min(CHUNK, size_by_name[n] - i * CHUNK)
                    for n, idxs in state.items()
                    for i in set(idxs)
                )
        have = set(state.get(name, []))
        nchunks = max(1, (size + CHUNK - 1) // CHUNK)
        todo = [i for i in range(nchunks) if i not in have]
        print(
            f"== {name}: {size / 2**30:.2f} GiB, {len(todo)}/{nchunks} chunks to fetch",
            flush=True,
        )
        if todo:
            with ThreadPoolExecutor(CONCURRENCY) as pool:
                futs = {pool.submit(fetch, name, size, i): i for i in todo}
                done_count = 0
                for fut in as_completed(futs):
                    idx = futs[fut]
                    got = fut.result()
                    done_count += 1
                    with lock:
                        STATS["done"] += got
                        STATS["last"] = time.time()
                        state.setdefault(name, []).append(idx)
                    if done_count % 8 == 0:
                        save_state(state)
            save_state(state)
        os.replace(part, final)
        print(f"== completed {name}", flush=True)
    save_state(state)
    open(os.path.expanduser("~/vllm/download-complete"), "w").write(
        time.strftime("%Y-%m-%d %H:%M:%S\n")
    )
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
