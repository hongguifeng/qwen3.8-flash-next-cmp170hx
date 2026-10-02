#!/usr/bin/env python3
"""cmp_model_dirs.py —— 逐张量**采样比对**两个 safetensors 模型目录（只读）

用途：仓库不提供 SHA-256 时，这是唯一能给出"内容级"证据的办法：
对每个共同张量取若干均匀分布的窗口做字节比对，回答
  * 哪些张量**完全一致**（如 PLE 表、MTP 头）；
  * 哪些张量**被改过**，以及改动的空间分布（按层号/模块名聚合）。

为什么采样而不是全量：全量比对要读 2×142 GiB，会长时间占满同一块 NVMe
（见 AGENTS.md 铁律 12）。采样只读几十 MB~几 GB，代价可控且结论足够定性。

用法：
  ops/tools/cmp_model_dirs.py A_DIR B_DIR [--windows N] [--ple-windows N] [--bytes N] [--json]

  --windows     普通张量的采样窗口数（默认 4）
  --ple-windows PLE 张量的采样窗口数（默认 256，因为 PLE 是 95 GiB 的主项）
  --bytes       每个窗口的字节数（默认 4096）
  --json        机器可读输出

退出码：0 = 正常完成比对（有差异不代表出错）；1 = 缺文件/无法解析
"""
import argparse
import json
import os
import struct
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))
import _cfg  # noqa: E402


def usage(err=None):
    print(__doc__.strip())
    if err:
        print(f"\n错误：{err}", file=sys.stderr)
    raise SystemExit(0 if err is None else 1)


def load_shard(path):
    """返回 {tensor_name: (file_offset, nbytes, dtype)}。"""
    out = {}
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        head = json.loads(f.read(hlen))
    base = 8 + hlen
    for name, m in head.items():
        if name == "__metadata__":
            continue
        a, b = m["data_offsets"]
        if b > size:
            raise ValueError(f"{path.name}: {name} 数据超出文件末尾")
        out[name] = (base + a, b - a, m["dtype"])
    return out


def build_map(d):
    """扫描目录所有 safetensors，返回 {tensor: (path, off, nbytes, dtype)}。"""
    m = {}
    for p in sorted(d.glob("*.safetensors")):
        for name, (off, nb, dt) in load_shard(p).items():
            if name in m:
                raise ValueError(f"{name} 在两个 shard 里都出现（{m[name][0].name} / {p.name}）")
            m[name] = (p, off, nb, dt)
    return m


def sample_shape(start, nbytes, nwin, wsize):
    """在 [start, start+nbytes) 上均匀取 nwin 个长度为 wsize 的窗口（不与前一窗口重叠）。"""
    nwin = max(1, min(nwin, nbytes // wsize or 1))
    if nwin == 1:
        return [(start + nbytes // 2, min(wsize, nbytes))]
    step = nbytes / nwin
    out = []
    for i in range(nwin):
        pos = int(start + i * step)
        out.append((pos, min(wsize, max(1, int(start + nbytes - pos)))))
    return out


def read_windows(fh_cache, path, wins):
    f = fh_cache.get(path)
    if f is None:
        f = fh_cache[path] = open(path, "rb")
    out = []
    for pos, n in wins:
        f.seek(pos)
        out.append(f.read(n))
    return out


def bucket(name):
    """把张量名归成人类可读的类别（用于看"改了哪些部分"）。"""
    if ".ngram_embedding.shard_" in name:
        return "PLE 表"
    if name.startswith("mtp"):
        return "MTP 头"
    for pat, label in (
        (".mlp.", "MLP"),
        (".attn.", "Attention"),
        ("self_attn", "Attention"),
        ("embed_tokens", "Embedding"),
        ("lm_head", "LM head"),
        ("norm", "Norm"),
        (".vision", "Vision"),
    ):
        if pat in name:
            return label
    return "其它"


def main():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--windows", type=int, default=4)
    ap.add_argument("--ple-windows", type=int, default=256)
    ap.add_argument("--bytes", type=int, default=4096)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("-h", "--help", action="store_true")
    a = ap.parse_args()
    if a.help:
        usage()

    A, B = Path(a.a).resolve(), Path(a.b).resolve()
    for d in (A, B):
        if not d.is_dir():
            usage(f"{d} 不是目录")

    ma, mb = build_map(A), build_map(B)
    common = sorted(set(ma) & set(mb))
    only_a, only_b = sorted(set(ma) - set(mb)), sorted(set(mb) - set(ma))
    print(f"A = {A}\nB = {B}\n")
    print(f"张量：A {len(ma):,}  B {len(mb):,}  共同 {len(common):,}  "
          f"仅 A {len(only_a)}  仅 B {len(only_b)}")
    # 注意比的是 nbytes 与 dtype（[2]/[3]），不是文件内偏移（[1]）——
    # 两个模型的 shard 切分方式不同，偏移按定义就不一样。
    geo_bad = [n for n in common if ma[n][2] != mb[n][2] or ma[n][3] != mb[n][3]]
    print(f"几何：同名张量字节数/ dtype 不一致的 {len(geo_bad)} 个"
          + (f"，例如 {geo_bad[:3]}" if geo_bad else "（全部一致）"))
    print(f"采样：普通张量 {a.windows} 窗口，PLE 张量 {a.ple_windows} 窗口，每窗口 {a.bytes} B\n")

    fh = {}
    same, diff = [], []
    dif_bytes = 0
    tot_read = 0
    tot_win = 0
    for n in common:
        pa, oa, na, _ = ma[n]
        pb, ob, nb, _ = mb[n]
        if na != nb:
            diff.append((n, na, nb, None))
            continue
        nwin = a.ple_windows if ".ngram_embedding.shard_" in n else a.windows
        wa = sample_shape(oa, na, nwin, a.bytes)
        wb = [(ob + (p - oa), k) for p, k in wa]
        ba = read_windows(fh, pa, wa)
        bb = read_windows(fh, pb, wb)
        tot_read += sum(len(x) for x in ba) * 2
        tot_win += len(ba)
        nd = sum(1 for x, y in zip(ba, bb) if x != y)
        if nd:
            diff.append((n, na, nb, nd / max(1, len(ba))))
            dif_bytes += nd
        else:
            same.append(n)
    for f in fh.values():
        f.close()

    print(f"读取 {tot_read / 2**20:.1f} MiB（两边合计）\n")
    print(f"## 结果\n")
    print(f"逐窗口**完全一致**的张量：{len(same):,} / {len(common):,}"
          f"  ({100 * len(same) / len(common):.2f}%)")
    print(f"至少一个窗口不同的张量：{len(diff):,} / {len(common):,}"
          f"  ({100 * len(diff) / len(common):.2f}%)")
    if diff:
        print(f"采样窗口中共 {dif_bytes:,} / {tot_win:,} 个不同"
              f"  ({100 * dif_bytes / max(1, tot_win):.2f}% 的被采样字节被改动过)")

    print(f"\n## 按类别统计\n")
    agg = defaultdict(lambda: [0, 0])
    for n in common:
        k = bucket(n)
        agg[k][0] += 1
    difnames = {n for n, *_ in diff}
    for n in common:
        if n in difnames:
            agg[bucket(n)][1] += 1
    print(f"{'类别':12s} {'张量数':>8s} {'不同':>8s} {'占比':>8s}")
    for k, (tot, d) in sorted(agg.items(), key=lambda kv: -kv[1][1]):
        print(f"{k:12s} {tot:8,d} {d:8,d} {100 * d / tot:7.2f}%")

    if only_a:
        print(f"\n仅在 A（{A.name}）里的张量：{len(only_a)} 个，例如 {only_a[:5]}")
    if only_b:
        print(f"仅在 B（{B.name}）里的张量：{len(only_b)} 个，例如 {only_b[:5]}")

    if diff:
        print(f"\n## 不同的张量（最多列 40 个）\n")
        for n, na, nb, frac in diff[:40]:
            extra = f"  尺寸 {na} vs {nb}" if na != nb else f"  采样不同窗口 {frac:.0%}" if frac is not None else ""
            print(f"  {n}{extra}")
        if len(diff) > 40:
            print(f"  … 其余 {len(diff) - 40} 个")
    if a.json:
        print(json.dumps({"a": str(A), "b": str(B), "common": len(common),
                          "identical": len(same), "differing": len(diff)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
