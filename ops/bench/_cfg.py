"""从 config/engine.env 读取默认值 —— 端口/模型名的**唯一来源**，环境变量优先。

为什么有这个小模块：ops/bench/ 下的探针原来各自硬编码 `http://127.0.0.1:<端口>`，
于是"改端口"要改 N 处（曾因此踩坑：bin/ 脚本与引擎的默认端口不一致）。
现在只有两种情况：
  * 由 bin/bench.sh 调用  → 它 export 了 BASE_URL/QWEN_PORT/QWEN_SERVED_NAME，直接用；
  * 自己单跑本脚本        → 从这里读 config/engine.env（仍然只有一处默认值）。
"""
import os
import re
from pathlib import Path

CFG = Path(__file__).resolve().parents[2] / "config" / "engine.env"
_CACHE = {}


def default(key):
    """取 config/engine.env 里 `: "${KEY:=value}"` 的 value；没有就返回 None。"""
    if not _CACHE:
        try:
            text = CFG.read_text(encoding="utf-8")
        except OSError:
            text = ""
        for m in re.finditer(r'^\s*:\s*"\$\{(\w+):=([^}]*)\}"', text, re.M):
            _CACHE[m.group(1)] = m.group(2)
    return _CACHE.get(key)


def get(key, fallback=None):
    """环境变量优先，其次 config/engine.env，最后 fallback。"""
    return os.environ.get(key) or default(key) or fallback


def api_url():
    """引擎 base URL，例如 http://127.0.0.1:8000（不带结尾斜杠）。"""
    base = os.environ.get("BASE_URL") or "http://127.0.0.1:%s" % get("QWEN_PORT", 8000)
    return base.rstrip("/")


def model_name():
    return get("QWEN_SERVED_NAME", "Qwen3.8-Flash-Next")
