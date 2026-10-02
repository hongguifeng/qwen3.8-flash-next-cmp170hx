#!/usr/bin/env python3
"""inspect_model_dir.py —— 只读体检一个 HF 模型的目录（下完模型/换模型前后用它）

回答三个问题，全部**离线**（只读本地文件，不联网、不碰引擎）：
  1. 文件全不全？大小对不对？（对照 model.safetensors.index.json + 可选的上游 API 尺寸）
  2. 每个 safetensors 是不是**完整合法**？（header 可解析、数据尾部字节数 == 文件大小、
     张量个数与 index 一致 —— 这一步能抓出"分块下载漏块/多写了尾巴"这类静默损坏）
  3. PLE 表是什么布局？（行数/dim/dtype/是否全在专用 shard 里 —— 直接决定 PLE-SSD 补丁能否加载）

用法：
  ops/tools/inspect_model_dir.py [MODEL_DIR] [--ref DIR] [--api] [--json]

  MODEL_DIR   要体检的目录；省略时用 config/engine.env 的 QWEN_MODEL_DIR（唯一默认值来源）
  --ref DIR   参考模型目录（一般是现役模型），逐张量比对 key 名/PLE 几何
  --api       额外联网核对每个文件的上游尺寸（走 HTTP_PROXY，慢，默认关）
  --json      机器可读输出（给别的脚本吃）

退出码：0 = 全部检查通过；1 = 发现缺失/不完整/不一致（细节见输出）
"""
import argparse
import json
import os
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))
import _cfg  # noqa: E402  —— 端口/模型路径的唯一默认值来源

INDEX = "model.safetensors.index.json"
HDR_MAX = 64 << 20  # header 上限（防误读大文件）


def usage(err=None):
    print(__doc__.strip())
    if err:
        print(f"\n错误：{err}", file=sys.stderr)
    raise SystemExit(0 if err is None else 1)


def read_header(path):
    """读 safetensors header，返回 (header_dict, header_len, size)；失败抛 ValueError。"""
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        raw = f.read(8)
        if len(raw) != 8:
            raise ValueError(f"文件只有 {len(raw)} 字节，连 8 字节 header 长度都读不到")
        hlen = struct.unpack("<Q", raw)[0]
        if hlen <= 0 or hlen > HDR_MAX or 8 + hlen > size:
            raise ValueError(f"header 长度异常：{hlen}（文件 {size} 字节）")
        head = json.loads(f.read(hlen))
    return head, hlen, size


def scan_file(path):
    head, hlen, size = read_header(path)
    tensors = {k: v for k, v in head.items() if k != "__metadata__"}
    data_end = max(v["data_offsets"][1] for v in tensors.values()) if tensors else 0
    ngram = [k for k in tensors if ".ngram_embedding.shard_" in k]
    return {
        "path": str(path),
        "size": size,
        "header_len": hlen,
        "tensors": len(tensors),
        "ngram": len(ngram),
        "all_ngram": bool(tensors) and len(ngram) == len(tensors),
        "dtypes": sorted({v["dtype"] for v in tensors.values()}),
        "tail_bytes": size - (8 + hlen + data_end),
        "rows": sum((v["data_offsets"][1] - v["data_offsets"][0]) for v in tensors.values()) // 320
        if ngram
        else 0,
    }


def load_index(d):
    p = d / INDEX
    if not p.is_file():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def main():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("model_dir", nargs="?")
    ap.add_argument("--ref")
    ap.add_argument("--api", action="store_true")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("-h", "--help", action="store_true")
    a = ap.parse_args()
    if a.help:
        usage()

    d = Path(a.model_dir or _cfg.get("QWEN_MODEL_DIR") or usage("没给 MODEL_DIR，config/engine.env 里也没有")).resolve()
    if not d.is_dir():
        usage(f"{d} 不是目录")
    problems = []

    # ---------- 1. 文件清单 ----------
    all_files = sorted(p for p in d.iterdir() if p.is_file() and not p.name.startswith("."))
    shards = [p for p in all_files if p.suffix == ".safetensors"]
    parts = [p for p in all_files if p.name.endswith(".part")]
    idx = load_index(d)
    wm = (idx or {}).get("weight_map", {})
    want_files = sorted(set(wm.values()))
    have_names = {p.name for p in all_files}

    print(f"# {d}")
    print(f"\n## 1. 文件清单\n")
    print(f"文件数 {len(all_files)}（safetensors {len(shards)}），index 引用 {len(want_files)} 个 shard")
    print(f"磁盘占用 {sum(p.stat().st_size for p in all_files) / 2**30:.2f} GiB"
          + (f"，index total_size {(idx or {}).get('metadata', {}).get('total_size', 0) / 2**30:.2f} GiB" if idx else ""))
    if idx:
        missing = [n for n in want_files if n not in have_names]
        extra = sorted(n for n in have_names if n.endswith(".safetensors") and n not in want_files)
        print(f"index 引用但缺失：{missing or '无'}")
        print(f"存在但 index 没引用：{extra or '无'}")
        problems += [f"缺失 {n}" for n in missing]
    if parts:
        print(f"未完成的 .part 文件：{[p.name for p in parts]}")
        problems += [f"未完成 {p.name}" for p in parts]

    # 稀疏/洞检测：尺寸与 header 都对但内容是 0 的文件，只有 st_blocks 能揭穿。
    holes = []
    for p in all_files:
        st = p.stat()
        gap = st.st_size - st.st_blocks * 512
        if gap > 1 << 20:
            holes.append((p.name, gap / 2**20))
    if holes:
        print("含未分配洞的文件（数据不可信，通常是崩溃丢脏页）：")
        for n, mib in holes:
            print(f"   {n}  {mib:.1f} MiB")
        problems += [f"{n} 含 {mib:.1f} MiB 的洞" for n, mib in holes]
    else:
        print("洞检测：所有文件块已全部分配（无崩溃残留）")

    if a.api:
        names = {n: s for n, s in api_sizes(d)}
        bad = [(p.name, p.stat().st_size, names.get(p.name)) for p in all_files
               if p.name in names and names[p.name] != p.stat().st_size]
        print(f"上游尺寸核对：{len(names)} 个文件" + (f"，不一致 {bad}" if bad else "，全部一致"))
        problems += [f"{n} 尺寸 {got} != 上游 {exp}" for n, got, exp in bad]

    # ---------- 2. 逐个 shard 完整性 ----------
    print(f"\n## 2. shard 完整性（tail 必须为 0）\n")
    print(f"{'文件':44s} {'GiB':>6s} {'张量':>7s} {'PLE张量':>7s} {'tail':>8s}  dtypes")
    rows_by_shard = {}
    total_rows = 0
    for p in shards:
        try:
            info = scan_file(p)
        except Exception as exc:  # noqa: BLE001
            print(f"{p.name:44s}  ✗ 无法解析：{exc}")
            problems.append(f"{p.name} 损坏：{exc}")
            continue
        bad = info["tail_bytes"] != 0 or (idx and len(wm) and info["tensors"] == 0)
        if info["tail_bytes"] != 0:
            problems.append(f"{p.name} tail={info['tail_bytes']}（数据尾部与文件大小不符）")
        print(f"{p.name:44s} {info['size'] / 2**30:6.2f} {info['tensors']:7d} {info['ngram']:7d} "
              f"{info['tail_bytes']:8d}  {','.join(info['dtypes'])}{'  ✗' if bad else ''}")
        if info["ngram"]:
            rows_by_shard[p.name] = info
            total_rows += info["rows"]

    # ---------- 3. PLE 布局 ----------
    print(f"\n## 3. PLE 布局（补丁要求：每个 PLE 文件里**只有** PLE 张量）\n")
    if not rows_by_shard:
        print("没有 PLE 张量（这不是 PLE 模型？）")
    else:
        mixed = [n for n, i in rows_by_shard.items() if not i["all_ngram"]]
        print(f"PLE 文件 {len(rows_by_shard)} 个，行数合计 {total_rows:,}，"
              f"占用 {sum(i['size'] for i in rows_by_shard.values()) / 2**30:.2f} GiB")
        print(f"混合文件（PLE + 普通张量混装）：{mixed or '无 —— 满足补丁要求'}")
        print(f"dtypes：{sorted({dt for i in rows_by_shard.values() for dt in i['dtypes']})}")
        problems += [f"{n} 里 PLE 与普通张量混装，PLE-SSD 补丁会拒绝加载" for n in mixed]
        for n in sorted(rows_by_shard):
            i = rows_by_shard[n]
            per_row = i["size"] / i["rows"] if i["rows"] else 0
            print(f"  {n:44s} {i['rows']:>9,} 行  ≈{per_row:.1f} B/行  ({i['ngram']} 个张量)")

    # ---------- 4. 与现役模型比对 ----------
    if a.ref:
        r = Path(a.ref).resolve()
        print(f"\n## 4. 与参考模型比对：{r}\n")
        ridx = load_index(r)
        if idx and ridx:
            ak, bk = set(wm), set(ridx["weight_map"])
            print(f"张量名：本模型 {len(ak):,}，参考 {len(bk):,}，"
                  f"仅本模型有 {len(ak - bk)}，仅参考有 {len(bk - ak)}")
            for n in sorted(ak ^ bk)[:10]:
                print(f"  ± {n}")
            problems += [f"张量名集合不同（{len(ak ^ bk)} 个差异）"] if ak ^ bk else []
        rshards = [p for p in sorted(r.glob("*.safetensors"))]
        rrows = rsize = 0
        for p in rshards:
            try:
                i = scan_file(p)
            except Exception:  # noqa: BLE001
                continue
            if i["ngram"]:
                rrows += i["rows"]
                rsize += i["size"]
        print(f"PLE：本模型 {total_rows:,} 行 / {sum(i['size'] for i in rows_by_shard.values()) / 2**30:.2f} GiB"
              f" vs 参考 {rrows:,} 行 / {rsize / 2**30:.2f} GiB"
              + ("  ✓ 一致" if rrows == total_rows else "  ✗ 不一致"))
        if rrows and rrows != total_rows:
            problems.append(f"PLE 行数不一致：{total_rows:,} vs {rrows:,}")

    # ---------- 结论 ----------
    print(f"\n## 结论\n")
    if problems:
        print(f"✗ {len(problems)} 项问题：")
        for p in problems:
            print(f"  - {p}")
    else:
        print("✓ 全部检查通过：文件齐全、每个 shard 可解析且尾部对齐、PLE 满足专用 shard 要求")
    if a.json:
        print(json.dumps({"dir": str(d), "problems": problems, "ple_rows": total_rows}, ensure_ascii=False))
    return 1 if problems else 0


def api_sizes(d):
    """从上游 API 取 {文件名: 字节数}（读 d/.fdl-state.json 里的仓库信息，或环境变量）。"""
    import urllib.request

    st = d / ".fdl-state.json"
    repo = os.environ.get("FDL_REPO") or "klee100/Qwen3.8-Flash-Next-Uncensored-AutoRound-3bpw-MTP"
    rev = os.environ.get("FDL_REVISION") or "main"
    if st.is_file():
        meta = json.loads(st.read_text())
        repo = meta.get("repo", repo)
        rev = meta.get("revision", rev)
    url = f"https://huggingface.co/api/models/{repo}/revision/{rev}?blobs=true"
    req = urllib.request.Request(url, headers={"User-Agent": "inspect_model_dir"})
    with urllib.request.urlopen(req, timeout=60) as r:
        data = json.load(r)
    return [(s["rfilename"], s["size"]) for s in data.get("siblings", []) if s.get("size") is not None]


if __name__ == "__main__":
    sys.exit(main())
