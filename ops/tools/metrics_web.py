#!/usr/bin/env python3
"""ops/tools/metrics_web.py —— 本地 /metrics 看板。

只做三件事，全都是只读的：
  1. 把看板页面（metrics_web.html + dashboard.js）用 HTTP 发出去；
  2. 同源转发引擎的 /metrics、/v1/models、/health —— 这样浏览器不依赖引擎的 CORS 设置；
  3. 什么都不改：不重启、不配置、不压测引擎，也不写任何引擎状态。

只用标准库，任何 Python 3.8+ 都能跑（不需要 vLLM 的 venv）。
用法：  ops/tools/metrics_web.py --port 9494 [--host 127.0.0.1] [--upstream http://127.0.0.1:9393]
或用包装脚本： ops/tools/metrics_web.sh start|stop|status|log
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

HERE = os.path.dirname(os.path.abspath(__file__))
CTYPES = {".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
          ".css": "text/css; charset=utf-8", ".json": "application/json", ".txt": "text/plain; charset=utf-8"}


class Handler(BaseHTTPRequestHandler):
    server_version = "metrics-web/1.0"
    protocol_version = "HTTP/1.1"

    # 浏览器轮询很勤，别把访问日志刷满屏；只报错误
    def log_message(self, fmt, *args):
        code = getattr(self, "_code", 200)
        if self.server.verbose or code not in (200, 304):
            sys.stderr.write("看板请求 %s %s → %s\n" % (self.path, fmt % args, code))

    # ---------- 输出小工具 ----------
    def _send(self, code, body, ctype="text/plain; charset=utf-8"):
        data = body.encode("utf-8") if isinstance(body, str) else body
        self._code = code
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")        # 指标绝不能被浏览器缓存
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(data)

    def _json(self, code, obj):
        self._send(code, json.dumps(obj, ensure_ascii=False), "application/json; charset=utf-8")

    def _file(self, name):
        path = os.path.join(HERE, name)
        try:
            with open(path, "rb") as fh:
                data = fh.read()
        except OSError as exc:
            self._send(500, "看板文件缺失：%s（%s）" % (path, exc))
            return
        ext = os.path.splitext(name)[1]
        self._send(200, data, CTYPES.get(ext, "application/octet-stream"))

    # ---------- 上游转发 ----------
    def _upstream(self, path):
        url = self.server.upstream + path
        req = urllib.request.Request(url, headers={"User-Agent": "metrics-web/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=self.server.timeout) as resp:
                return resp.status, resp.read(), resp.headers.get("Content-Type")
        except urllib.error.HTTPError as exc:                      # 引擎回来了但回了错误码
            try:
                body = exc.read()
            except Exception:
                body = b""
            return exc.code, body, exc.headers.get("Content-Type") if exc.headers else None
        except Exception as exc:                                   # 连不上
            return None, ("UPSTREAM_DOWN: %s" % exc).encode(), None

    # ---------- 路由 ----------
    def do_GET(self):
        p = urlparse(self.path).path
        if p in ("/", "/index.html"):
            self._file("metrics_web.html")
        elif p == "/dashboard.js":
            self._file("dashboard.js")
        elif p == "/api/metrics":
            status, body, ctype = self._upstream("/metrics")
            if status is None:
                self._send(502, body, "text/plain; charset=utf-8")
            else:
                self._send(status, body, ctype or "text/plain; version=1.0.0; charset=utf-8")
        elif p == "/api/models":
            status, body, ctype = self._upstream("/v1/models")
            if status is None:
                self._json(503, {"error": "upstream_unreachable", "detail": body.decode("utf-8", "replace")})
            else:
                self._send(status, body, ctype or "application/json")
        elif p == "/api/health":
            status, _body, _ctype = self._upstream("/health")
            self._send(200, str(status) if status is not None else "DOWN")
        elif p == "/api/info":
            self._json(200, {"upstream": self.server.upstream,
                             "dashboard": "http://%s:%d" % (self.server.server_address[0], self.server.server_address[1]),
                             "timeout_s": self.server.timeout})
        elif p == "/favicon.ico":
            self._send(204, b"")
        elif p == "/api/what":                                     # 自解释，省得翻文档
            self._send(200, __doc__.strip())
        else:
            self._send(404, "没有这个路径：%s\n可走：/ · /dashboard.js · /api/metrics · /api/models · /api/health · /api/info" % p)

    do_HEAD = do_GET


class Server(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def main():
    ap = argparse.ArgumentParser(description="Qwen3.8-Flash-Next /metrics 看板（只读）")
    # 不给硬编码默认端口：默认值唯一来源是 config/engine.env，由 ops/tools/metrics_web.sh 注入。
    ap.add_argument("--port", type=int, default=None, help="默认取环境变量 QWEN_METRICS_WEB_PORT")
    ap.add_argument("--host", default=None,
                    help="默认取环境变量 QWEN_METRICS_WEB_HOST；WSL2 是 mirrored 网络，Windows 浏览器直接开 127.0.0.1 即可（要局域网可达就 --host 0.0.0.0）")
    ap.add_argument("--upstream", default=None, help="引擎地址，例如 http://127.0.0.1:9393（默认取 QWEN_UPSTREAM）")
    ap.add_argument("--timeout", type=float, default=4.0, help="转发上游的超时（秒）")
    ap.add_argument("--verbose", action="store_true", help="打印每次请求")
    args = ap.parse_args()

    args.port = args.port or os.environ.get("QWEN_METRICS_WEB_PORT")
    args.host = args.host or os.environ.get("QWEN_METRICS_WEB_HOST")
    args.upstream = args.upstream or os.environ.get("QWEN_UPSTREAM")
    if not (args.port and args.host and args.upstream):
        raise SystemExit(
            "错误：--port/--host/--upstream 没给全，环境变量里也没有。\n"
            "      推荐启动方式：ops/tools/metrics_web.sh start   （它从 config/engine.env 读端口）\n"
            "      或显式指定：metrics_web.py --host 127.0.0.1 --port 9494 --upstream http://127.0.0.1:9393")
    args.port = int(args.port)

    srv = Server((args.host, args.port), Handler)
    srv.upstream = args.upstream.rstrip("/")
    srv.timeout = args.timeout
    srv.verbose = args.verbose

    print("看板已起：http://%s:%d/   （上游 %s）" % (args.host, args.port, srv.upstream))
    print("Windows 浏览器请用 127.0.0.1，不要用 localhost（mirrored 网络不转发 IPv6 回环）。")
    print("Ctrl-C 停止；它只读 /metrics，不会动引擎。")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止")
    finally:
        srv.server_close()


if __name__ == "__main__":
    main()
