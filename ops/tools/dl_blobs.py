#!/usr/bin/env python3
"""Parallel registry blob downloader + docker-load archive builder.

Works around a stalling registry path: blobs are fetched with many range
requests, verified by digest, and re-packaged into an image archive that
`docker load` (or `ctr images import`) can ingest.
"""
import hashlib
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import requests

REPO = "18gogogo/170hx1-qwen38nextf"
TAG = "sm80"
REGISTRY = "https://registry-1.docker.io"
AUTH = "https://auth.docker.io/token"
DEST = os.path.expanduser("~/vllm/image-blobs")
CHUNK = 8 * 1024 * 1024
CONCURRENCY = 64
MAX_RETRY = 10

lock = threading.Lock()
STATS = {"done": 0, "total": 0, "retry": 0}


TOKEN = {"value": None, "ts": 0.0}
STATE_FILE = os.path.join(DEST, ".state.json")
STATE = {}


def token(force=False):
    """Docker Hub bearer tokens expire after ~5 minutes."""
    if force or TOKEN["value"] is None or time.time() - TOKEN["ts"] > 240:
        r = requests.get(
            AUTH,
            params={
                "service": "registry.docker.io",
                "scope": f"repository:{REPO}:pull",
            },
            timeout=60,
        )
        r.raise_for_status()
        TOKEN["value"] = r.json()["token"]
        TOKEN["ts"] = time.time()
    return TOKEN["value"]


def load_state():
    if os.path.exists(STATE_FILE):
        try:
            STATE.update(json.load(open(STATE_FILE)))
        except Exception:
            pass


def save_state():
    tmp = STATE_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(STATE, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, STATE_FILE)


def manifest(tok):
    r = requests.get(
        f"{REGISTRY}/v2/{REPO}/manifests/{TAG}",
        headers={
            "Authorization": f"Bearer {tok}",
            "Accept": ",".join(
                [
                    "application/vnd.oci.image.manifest.v1+json",
                    "application/vnd.docker.distribution.manifest.v2+json",
                ]
            ),
        },
        timeout=60,
    )
    r.raise_for_status()
    return r.json()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(16 * 1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def fetch_blob(digest, size, idx):
    start, end = idx * CHUNK, min((idx + 1) * CHUNK, size) - 1
    url = f"{REGISTRY}/v2/{REPO}/blobs/{digest}"
    path = os.path.join(DEST, digest.replace(":", "_") + ".part")
    token_used = None
    for attempt in range(MAX_RETRY):
        try:
            tok = token(force=(attempt > 0 and token_used is not None
                               and token_used == TOKEN["value"]))
            token_used = tok
            with requests.get(
                url,
                headers={
                    "Authorization": f"Bearer {tok}",
                    "Range": f"bytes={start}-{end}",
                },
                stream=True,
                timeout=(30, 180),
            ) as r:
                if r.status_code not in (200, 206):
                    raise RuntimeError(f"HTTP {r.status_code}")
                buf = bytearray()
                for piece in r.iter_content(1024 * 512):
                    buf.extend(piece)
                if len(buf) != end - start + 1:
                    raise RuntimeError(f"short read {len(buf)} != {end - start + 1}")
            with open(path, "r+b") as f:
                f.seek(start)
                f.write(buf)
            with lock:
                STATS["done"] += len(buf)
                STATE.setdefault(digest, [])
                if idx not in STATE[digest]:
                    STATE[digest].append(idx)
            return
        except Exception as exc:  # noqa: BLE001
            with lock:
                STATS["retry"] += 1
            if attempt == MAX_RETRY - 1:
                raise RuntimeError(f"{digest} chunk {idx}: {exc}") from exc
            time.sleep(min(1.5 * (attempt + 1), 20))


def report():
    while True:
        time.sleep(20)
        with lock:
            done, total = STATS["done"], STATS["total"]
        print(
            f"  blobs {done / 2**20:7.0f}/{total / 2**20:.0f} MiB "
            f"({done / max(total, 1) * 100:4.1f}%) retries={STATS['retry']}",
            flush=True,
        )


def download_blob(digest, size):
    part = os.path.join(DEST, digest.replace(":", "_") + ".part")
    final = os.path.join(DEST, digest.replace(":", "_"))
    if os.path.exists(final) and os.path.getsize(final) == size:
        print(f"  cached {digest[:19]} {size / 2**20:.0f} MiB", flush=True)
        with lock:
            STATS["done"] += size
        return final
    if not os.path.exists(part) or os.path.getsize(part) != size:
        with open(part, "wb") as f:
            f.truncate(size)
        STATE[digest] = []
    have = set(STATE.get(digest, []))
    nchunks = max(1, (size + CHUNK - 1) // CHUNK)
    todo = [i for i in range(nchunks) if i not in have]
    if have:
        with lock:
            STATS["done"] += sum(min(CHUNK, size - i * CHUNK) for i in have)
        print(f"  resuming {digest[:19]}: {len(have)}/{nchunks} chunks cached", flush=True)
    print(
        f"  fetching {digest[:19]} {size / 2**20:.0f} MiB "
        f"({len(todo)}/{nchunks} chunks)",
        flush=True,
    )
    with ThreadPoolExecutor(CONCURRENCY) as pool:
        futs = [pool.submit(fetch_blob, digest, size, i) for i in todo]
        for k, fut in enumerate(futs):
            fut.result()
            if k % 8 == 0:
                save_state()
    save_state()
    got = sha256_file(part)
    want = digest.split(":")[1]
    if got != want:
        os.remove(part)
        raise RuntimeError(f"digest mismatch for {digest}: {got}")
    os.replace(part, final)
    print(f"  verified {digest[:19]}", flush=True)
    return final


def main():
    os.makedirs(DEST, exist_ok=True)
    load_state()
    man = manifest(token())
    config = man["config"]
    layers = man["layers"]
    total = config["size"] + sum(l["size"] for l in layers)
    STATS["total"] = total
    print(
        f"image {REPO}:{TAG}: {len(layers)} layers, {total / 2**20:.0f} MiB",
        flush=True,
    )
    threading.Thread(target=report, daemon=True).start()

    cfg_path = download_blob(config["digest"], config["size"])
    layer_paths = [download_blob(l["digest"], l["size"]) for l in layers]

    # docker-save layout: config file, layer tars, and manifest.json
    names = {cfg_path: config["digest"].split(":")[1] + ".json"}
    tar_entries = []
    for path, layer in zip(layer_paths, layers):
        name = layer["digest"].split(":")[1] + ".tar"
        names[path] = name
        tar_entries.append(name)
    entries_cfg = config["digest"].split(":")[1] + ".json"
    order = [entries_cfg] + tar_entries

    layout = os.path.join(DEST, "layout")
    os.makedirs(layout, exist_ok=True)
    for path, name in names.items():
        target = os.path.join(layout, name)
        if not os.path.exists(target):
            os.link(path, target)
    with open(os.path.join(layout, "manifest.json"), "w") as f:
        json.dump(
            [
                {
                    "Config": entries_cfg,
                    "RepoTags": [f"{REPO}:{TAG}"],
                    "Layers": tar_entries,
                }
            ],
            f,
        )
    print("layout ready:", layout, "order:", len(order), flush=True)


if __name__ == "__main__":
    main()
