#!/usr/bin/env python3
"""codebench.py —— 代码能力对比基准（HumanEval / MBPP-sanitized，执行式判定）

为什么用它：这两个是**执行式**（跑测试用例、看 pass/fail）而不是"让评委模型打分"，
同一套题/同一协议跑在两个引擎上就是可直接对比的数字；题目小（164 / 399 题）、
单机离线可跑，适合本机这种"两张 CMP 170HX、各跑一个 checkpoint"的 A/B 场景。

用法：
  # 跑一个引擎（默认协议 = 官方 HumanEval 的裸补全，和公开数字可比）
  ops/bench/codebench.py run --base-url http://127.0.0.1:8001 \\
      --model Qwen3.8-Flash-Next-Uncensored --out /tmp/unc.json

  # 另一个引擎用**完全一样**的协议
  ops/bench/codebench.py run --base-url http://127.0.0.1:8000 \\
      --model Qwen3.8-Flash-Next --out /tmp/cur.json

  # 对比：pass@1 + 四格表（都过 / 只有 A 过 / 只有 B 过 / 都不过）+ 分歧题目
  ops/bench/codebench.py compare /tmp/cur.json /tmp/unc.json

要点：
  * 两边的题序、prompt、停用词、temperature、max_tokens **完全一致**（写进结果 JSON 便于复现）。
  * `--protocol completion` = 官方协议（prompt 直接喂 /v1/completions）；`--protocol chat` =
    走对话模板并要求"只输出代码块"，用于 instruct 模型更公平的对照。**两次对比必须用同一个协议**。
  * 生成的代码**会被真的执行**（这是执行式基准的本意）。本机没有 user namespace（`unshare -n`
    不可用）、也不是 root（`setpriv` 不可用），所以沙箱只能是"尽力而为"：
    临时目录 + 地址空间/CPU/文件大小 ulimit + 5 s 超时 + 一个透明的危险 API 黑名单。
    **被黑名单挡下的题**会单独计数（`blocked`），不计入通过。
  * 数据集缓存在 ops/bench/data/（HumanEval 44 KB、MBPP 430 KB），只下一次。

退出码：0 = 正常跑完；1 = 有请求失败（详情见结果 JSON 的 errors）。
"""
import argparse
import gzip
import http.client
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _cfg  # noqa: E402  —— 端口/模型名的唯一默认值来源

DATA = Path(__file__).resolve().parent / "data"
HE_URL = "https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz"
HEPLUS_URL = ("https://github.com/evalplus/humanevalplus_release/releases/download/"
              "v0.1.10/HumanEvalPlus.jsonl.gz")
MBPP_URL = ("https://raw.githubusercontent.com/google-research/google-research/master/"
            "mbpp/sanitized-mbpp.json")
# 官方 HumanEval 用 5 个停用词，但本引擎的 OpenAI 兼容层把 `stop` 上限卡在 4 个
# （body.stop: "List should have at most 4 items"，实测 400）。这里去掉 "\nprint"；
# **两个模型用的都是这同一组 4 个**，所以对比仍然公平（绝对值可能略偏离官方数字）。
HE_STOPS = ["\nclass", "\ndef", "\n#", "\nif"]
HE_TIMEOUT = 5.0
# 透明黑名单：本机做不了真沙箱，只能挡住"明显会碰系统"的代码（会单独计数并报告）
UNSAFE = re.compile(
    r"os\.system|os\.popen|subprocess\.|shutil\.rmtree|socket\.socket|urllib\.request|"
    r"requests\.(get|post)|__import__\s*\(\s*['\"](os|subprocess|socket)['\"]\s*\)",
    re.I,
)


def usage(err=None):
    print(__doc__.strip())
    if err:
        print(f"\n错误：{err}", file=sys.stderr)
    raise SystemExit(0 if err is None else 1)


# ---------------------------------------------------------------- 数据集
def fetch(url, dest):
    if dest.is_file() and dest.stat().st_size > 1000:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"# 下载数据集 {url}", flush=True)
    with urllib.request.urlopen(url, timeout=180) as r, open(dest, "wb") as f:
        shutil.copyfileobj(r, f)
    return dest


def load_dataset(name):
    """返回统一结构：[{id, prompt, entry_point, tests, kind}]"""
    if name in ("humaneval", "humanevalplus"):
        if name == "humaneval":
            p = fetch(HE_URL, DATA / "HumanEval.jsonl.gz")
        else:
            # HumanEval+：**同样 164 题、同样 prompt**，但每个函数的测试从几条扩到几十条
            # （含边界/负数/重复元素等）⇒ 同题下分辨力高得多，适合 n=164 这种小样本。
            p = fetch(HEPLUS_URL, DATA / "HumanEvalPlus.jsonl.gz")
        rows = [json.loads(l) for l in gzip.open(p, "rt")]
        return [{"id": r["task_id"], "prompt": r["prompt"], "entry_point": r["entry_point"],
                 "tests": r["test"] + f"\ncheck({r['entry_point']})\n", "kind": "he"}
                for r in rows]
    if name == "mbpp":
        p = fetch(MBPP_URL, DATA / "sanitized-mbpp.json")
        rows = json.loads(p.read_text())
        out = []
        for r in rows:
            setup = "\n".join(r.get("test_imports") or []) + "\n"
            tests = setup + "\n".join(r["test_list"]) + "\n"
            # 官方 MBPP 提示词模板（bigcode-evaluation-harness 的 sanitized 配置）：
            # 任务描述 + 要过的测试都在 prompt 里，模型接着写代码。
            comp = (f"You are an expert Python programmer, and here is your task: {r['prompt']}"
                    f" Your code should pass these tests:\n\n{tests}\n[DONE]")
            out.append({"id": f"MBPP/{r['task_id']}", "prompt": r["prompt"],
                        "prompt_completion": comp, "entry_point": None,
                        "tests": tests, "kind": "mbpp"})
        return out
    usage(f"未知数据集 {name}")


# ---------------------------------------------------------------- HTTP
class Client:
    """复用一条 keep-alive 连接（比 urllib 每次新建连接快得多）。"""

    def __init__(self, base_url):
        u = urllib.parse.urlsplit(base_url)
        self.host = u.hostname or "127.0.0.1"
        self.port = u.port or 80
        self.conn = None

    def _connect(self):
        if self.conn is None:
            self.conn = http.client.HTTPConnection(self.host, self.port, timeout=1800)
        return self.conn

    def post(self, path, payload, retries=2):
        body = json.dumps(payload).encode()
        last = None
        for _ in range(retries + 1):
            try:
                c = self._connect()
                c.request("POST", path, body=body, headers={"Content-Type": "application/json"})
                r = c.getresponse()
                data = json.loads(r.read())
                if r.status != 200:
                    raise RuntimeError(f"HTTP {r.status}: {str(data)[:200]}")
                return data
            except Exception as exc:  # noqa: BLE001
                last = exc
                try:
                    if self.conn:
                        self.conn.close()
                finally:
                    self.conn = None
                time.sleep(1.5)
        raise RuntimeError(str(last))


THINK_RE = re.compile(r"<think>.*?</think>", re.S)
CODE_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.S)


def extract_code(text, problem):
    """从模型输出里取出可执行的程序片段。

    * HumanEval 走**官方拼法**：prompt（就是函数签名+docstring，本身是合法 Python）+ 续写；
    * MBPP 的 prompt 是**自然语言+测试**，不能直接拼 ⇒ 只取模型写出的代码
      （优先取 ``` 代码块，否则取去掉 <think> 后的正文）。
    实测：裸补全路径下模型会自己写 <think> 推理（chat 模板的 enable_thinking=false 管不到
    /v1/completions），MBPP 尤其明显 ⇒ 必须剥掉，否则拼出来的是语法错误。
    """
    stripped = THINK_RE.sub("", text).strip()
    if "<think" in text and "</think>" not in text:
        # 预算被推理吃光（裸补全路径下 enable_thinking=false 不起作用）：
        # 取最后一个 ``` 之后的正文当代码。
        tail = text.rsplit("```", 1)[-1]
        tail = re.sub(r"^\s*(python)?\s*\n", "", tail).strip()
        if tail:
            return tail
        return ""  # 什么都没写出来，让它去报错（而不是拼出语法错误）
    m = CODE_RE.search(text)
    block = m.group(1).strip() if m else None
    if problem["kind"] == "he":
        body = stripped or text
        if re.search(rf"\bdef\s+{problem['entry_point']}\b", body):
            return problem["prompt"] + "\n" + body if not body.startswith(problem["prompt"]) else body
        if block and re.search(rf"\bdef\s+{problem['entry_point']}\b", block):
            return problem["prompt"] + "\n" + block
        return problem["prompt"] + text
    return block or stripped or text


def ask(client, model, problem, protocol, max_tokens, temperature, top_p=1.0):
    """返回 (code, n_tokens, prompt_sent)。code 是"可以直接拼测试执行"的程序片段。"""
    if protocol == "completion":
        sent = problem.get("prompt_completion", problem["prompt"])
        body = {"model": model, "prompt": sent, "max_tokens": max_tokens,
                "temperature": temperature, "top_p": top_p, "stop": HE_STOPS,
                "stream": False}
        d = client.post("/v1/completions", body)
        text = d["choices"][0]["text"]
        code = (sent + text) if problem["kind"] == "he" else extract_code(text, problem)
        return code, d["usage"].get("completion_tokens", 0), sent

    if problem["kind"] == "he":
        instr = ("Complete the following Python function. Keep the given signature and "
                 "return a correct implementation. Output only one Python code block.")
        user = instr + "\n\n```python\n" + problem["prompt"] + "```"
    else:
        # MBPP：必须把测试一并给出——函数名由测试决定（如 is_not_prime），不给就会自己起个
        # 别的名字（实测 5/5 都是 NameError）。这也是官方 MBPP 提示词的写法。
        instr = ("Write a Python function that satisfies the requirement below. "
                 "Your code should pass the given tests. Output only one Python code block.")
        user = (instr + "\n\n" + problem["prompt"] + "\n\nTests your code must pass:\n\n"
                + problem["tests"])
    body = {"model": model, "messages": [{"role": "user", "content": user}],
            "max_tokens": max_tokens, "temperature": temperature, "top_p": top_p,
            "stream": False}
    d = client.post("/v1/chat/completions", body)
    text = d["choices"][0]["message"]["content"]
    ntok = d["usage"].get("completion_tokens", 0)
    code = extract_code(text, problem)
    return code, ntok, user


# ---------------------------------------------------------------- 执行判定
def execute(program, timeout=HE_TIMEOUT):
    """尽力而为的沙箱内跑一次；返回 (passed, detail)。"""
    if UNSAFE.search(program):
        return False, "blocked:unsafe-api"
    d = tempfile.mkdtemp(prefix="codebench-", dir="/tmp")
    os.chmod(d, 0o700)
    path = os.path.join(d, "prog.py")
    try:
        with open(path, "w") as f:
            f.write(program)
        # ulimit：地址空间 2 GiB、CPU 10 s、单文件 20 MB；再叠一层墙钟超时
        cmd = ["bash", "-c",
               "ulimit -v 2000000 -t 10 -f 20000 2>/dev/null; exec python3 -I prog.py"]
        r = subprocess.run(cmd, cwd=d, capture_output=True, text=True, timeout=timeout,
                           env={"PATH": "/usr/bin:/bin", "HOME": d, "PYTHONHASHSEED": "0"})
        if r.returncode == 0:
            return True, "pass"
        detail = (r.stderr or r.stdout or "").strip().splitlines()
        return False, (detail[-1][:200] if detail else f"rc={r.returncode}")
    except subprocess.TimeoutExpired:
        return False, "timeout"
    except Exception as exc:  # noqa: BLE001
        return False, f"harness:{exc}"
    finally:
        shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------- run
def cmd_run(a):
    problems = load_dataset(a.dataset)
    if a.limit:
        problems = problems[: a.limit]
    if a.shuffle:
        random.Random(a.seed).shuffle(problems)
    client = Client(a.base_url)
    results, errors = [], []
    t0 = time.time()
    ntok = 0
    passed = blocked = 0
    for i, p in enumerate(problems, 1):
        try:
            code, tok, sent = ask(client, a.model, p, a.protocol, a.max_tokens, a.temperature)
        except Exception as exc:  # noqa: BLE001
            errors.append({"id": p["id"], "error": str(exc)[:300]})
            print(f"[{i}/{len(problems)}] {p['id']} 请求失败：{exc}", flush=True)
            continue
        ntok += tok
        program = code + "\n" + p["tests"]
        ok, detail = execute(program, a.timeout)
        passed += ok
        blocked += detail == "blocked:unsafe-api"
        results.append({"id": p["id"], "pass": ok, "detail": detail, "tokens": tok,
                        "completion": code[len(sent):] if code.startswith(sent) else code})
        mark = "PASS" if ok else "fail"
        print(f"[{i}/{len(problems)}] {p['id']:<16s} {mark}  ({tok} tok)"
              + ("" if ok else f"  {detail[:80]}"), flush=True)
        if i % 20 == 0:
            print(f"    ... 目前 pass@1 = {passed}/{i} = {passed / i:.3f}", flush=True)

    el = time.time() - t0
    n = len(results)
    summary = {
        "model": a.model, "base_url": a.base_url, "dataset": a.dataset, "protocol": a.protocol,
        "temperature": a.temperature, "max_tokens": a.max_tokens,
        "n_problems": len(problems), "n_scored": n, "passed": passed,
        "pass@1": (passed / n) if n else 0.0, "blocked": blocked,
        "errors": len(errors), "seconds": round(el, 1),
        "completion_tokens": ntok, "tok_per_s": round(ntok / el, 1) if el else 0,
        "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps({"summary": summary, "results": results,
                                           "error_list": errors}, ensure_ascii=False, indent=1))
        print(f"\n# 结果已写 {a.out}")
    print(f"\n===== {a.model} @ {a.dataset}/{a.protocol} =====")
    for k in ("n_scored", "passed", "pass@1", "blocked", "errors", "seconds",
              "completion_tokens", "tok_per_s"):
        print(f"  {k:18s} {summary[k]}")
    return 1 if errors else 0


# ---------------------------------------------------------------- compare
def cmd_compare(a):
    A = json.loads(Path(a.a).read_text())
    B = json.loads(Path(a.b).read_text())
    for x, y in ((A, B),):
        if x["summary"]["dataset"] != y["summary"]["dataset"] or \
           x["summary"]["protocol"] != y["summary"]["protocol"]:
            usage("两个结果的 dataset/protocol 不同，不能对比")
    ra = {r["id"]: r["pass"] for r in A["results"]}
    rb = {r["id"]: r["pass"] for r in B["results"]}
    ids = [i for i in ra if i in rb]
    both = [i for i in ids if ra[i] and rb[i]]
    only_a = [i for i in ids if ra[i] and not rb[i]]
    only_b = [i for i in ids if rb[i] and not ra[i]]
    neither = [i for i in ids if not ra[i] and not rb[i]]
    sa, sb = A["summary"], B["summary"]
    print(f"# {sa['model']}  vs  {sb['model']}    题集 {sa['dataset']}/{sa['protocol']}  n={len(ids)}")
    print(f"\n{'':22s} {'A: ' + sa['model']:32s} {'B: ' + sb['model']:32s}")
    print(f"{'pass@1':22s} {sa['pass@1']:<32.4f} {sb['pass@1']:<32.4f}")
    print(f"{'通过/计分':22s} {str(sa['passed']) + '/' + str(sa['n_scored']):<32s} "
          f"{str(sb['passed']) + '/' + str(sb['n_scored']):<32s}")
    print(f"{'被黑名单挡下':22s} {sa['blocked']:<32} {sb['blocked']:<32}")
    print(f"{'耗时 s':22s} {sa['seconds']:<32} {sb['seconds']:<32}")
    print(f"{'completion tok/s':22s} {sa['tok_per_s']:<32} {sb['tok_per_s']:<32}")
    print(f"\n## 四格表（同一批题）\n\n| | B 通过 | B 失败 |\n|---|---|---|\n"
          f"| **A 通过** | {len(both)} | {len(only_a)} |\n"
          f"| **A 失败** | {len(only_b)} | {len(neither)} |\n")
    d = sa["pass@1"] - sb["pass@1"]
    # 配对差异的粗略显著性（McNemar 精确检验，正态近似）
    n_disc = len(only_a) + len(only_b)
    if n_disc:
        z = abs(len(only_a) - len(only_b)) / (n_disc ** 0.5)
        print(f"pass@1 差 = {d:+.4f}（A - B，正=现役更强）；不一致题 {n_disc} 个，"
              f"McNemar |z| = {z:.2f}" + ("（>1.96 ⇒ 5% 水平显著）" if z > 1.96 else "（不显著）"))
    else:
        print("两模型逐题结果完全相同")
    if only_a:
        print(f"\n只有 A 通过（{len(only_a)}）：{', '.join(only_a[:40])}")
    if only_b:
        print(f"\n只有 B 通过（{len(only_b)}）：{', '.join(only_b[:40])}")
    return 0


def main():
    ap = argparse.ArgumentParser(add_help=False)
    sub = ap.add_subparsers(dest="cmd")
    r = sub.add_parser("run", add_help=False)
    r.add_argument("--base-url", default=None, help="默认取 config/engine.env 的 QWEN_PORT")
    r.add_argument("--model", default=None)
    r.add_argument("--dataset", default="humaneval",
                   choices=["humaneval", "humanevalplus", "mbpp"])
    r.add_argument("--protocol", default="completion", choices=["completion", "chat"])
    r.add_argument("--limit", type=int, default=0)
    r.add_argument("--max-tokens", type=int, default=512)
    r.add_argument("--temperature", type=float, default=0.0)
    r.add_argument("--timeout", type=float, default=HE_TIMEOUT)
    r.add_argument("--shuffle", action="store_true")
    r.add_argument("--seed", type=int, default=42)
    r.add_argument("--out")
    c = sub.add_parser("compare", add_help=False)
    c.add_argument("a")
    c.add_argument("b")
    ap.add_argument("-h", "--help", action="store_true")
    a = ap.parse_args()
    if a.help or not a.cmd:
        usage()
    if a.cmd == "run":
        a.base_url = a.base_url or _cfg.api_url()
        a.model = a.model or _cfg.model_name()
        return cmd_run(a)
    return cmd_compare(a)


if __name__ == "__main__":
    sys.exit(main())
