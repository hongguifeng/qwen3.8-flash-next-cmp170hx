#!/usr/bin/env bash
# 起/停「/metrics 看板」——只读转发引擎的 /metrics，不重启、不压测、不配置引擎。
#
# 用法:
#   ops/tools/metrics_web.sh start          # 后台起，打印 Windows 浏览器要开的地址
#   ops/tools/metrics_web.sh stop
#   ops/tools/metrics_web.sh status         # 看板活着吗 + 能不能连上引擎
#   ops/tools/metrics_web.sh log [-f]       # 跟踪看板自己的日志
#   ops/tools/metrics_web.sh foreground     # 前台跑（Ctrl-C 停）
#
# 可调（环境变量）：
#   QWEN_PORT=8000          引擎端口（看板从这个上游抓指标；默认值在 config/engine.env）
#   QWEN_METRICS_WEB_PORT=9494  看板端口
#   QWEN_METRICS_WEB_HOST=127.0.0.1  监听地址（要局域网可达就设 0.0.0.0）
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# 端口默认值的唯一来源（不在这里再写一遍默认值）
# shellcheck source=/dev/null
. "$ROOT/config/engine.env"
WEB_PY="$ROOT/ops/tools/metrics_web.py"
LOGDIR="$ROOT/ops/logs"
LOG="$LOGDIR/metrics-web.log"
PIDFILE="$LOGDIR/metrics-web.pid"
PORT="$QWEN_METRICS_WEB_PORT"
HOST="$QWEN_METRICS_WEB_HOST"
UPSTREAM="http://127.0.0.1:${QWEN_PORT}"
PY="${PYTHON:-/usr/bin/python3}"

web_code() { curl -s -o /dev/null -w '%{http_code}' --max-time 3 "http://127.0.0.1:${PORT}/" 2>/dev/null || true; }
api_health() { curl -s --max-time 3 "http://127.0.0.1:${PORT}/api/health" 2>/dev/null || true; }
alive() { [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; }

case "${1:-start}" in
    start)
        mkdir -p "$LOGDIR"
        if alive; then
            echo "看板已在运行 pid=$(cat "$PIDFILE")（http://${HOST}:${PORT}/）"; exit 0
        fi
        setsid nohup "$PY" "$WEB_PY" --port "$PORT" --host "$HOST" --upstream "$UPSTREAM" >>"$LOG" 2>&1 &
        echo $! > "$PIDFILE"
        sleep 1
        code="$(web_code)"
        if [ "$code" = "200" ]; then
            echo "看板已起 pid=$(cat "$PIDFILE")  上游=$UPSTREAM"
            echo "  浏览器打开  http://${HOST}:${PORT}/      ← Windows 侧用 127.0.0.1，别用 localhost"
            echo "  停止        ops/tools/metrics_web.sh stop"
        else
            echo "看板起来了但 / 返回 ${code:-无响应}；日志尾：" >&2
            tail -n 20 "$LOG" >&2; exit 1
        fi
        ;;
    foreground)
        exec "$PY" "$WEB_PY" --port "$PORT" --host "$HOST" --upstream "$UPSTREAM"
        ;;
    stop)
        if ! alive; then echo "没在跑（pid 文件 $PIDFILE）"; rm -f "$PIDFILE"; exit 0; fi
        pid="$(cat "$PIDFILE")"
        kill -TERM "$pid" 2>/dev/null || true
        for _ in $(seq 1 10); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
        kill -0 "$pid" 2>/dev/null && { echo "仍在跑，SIGKILL"; kill -KILL "$pid" 2>/dev/null || true; }
        rm -f "$PIDFILE"; echo "已停止（这个看板随时可起，不需要重启引擎）"
        ;;
    status)
        if alive; then echo "看板 pid=$(cat "$PIDFILE") 运行中  http://${HOST}:${PORT}/"
        else echo "看板未运行"; fi
        echo "  看板自身 /  → $(web_code)"
        echo "  引擎 /health（经看板） → $(api_health)"
        echo "  直连引擎 /health       → $(curl -s -o /dev/null -w '%{http_code}' --max-time 3 "$UPSTREAM/health" 2>/dev/null || echo 无响应)"
        echo "  看板内存 RSS $(ps -o rss= -p "$(cat "$PIDFILE" 2>/dev/null)" 2>/dev/null | awk '{printf "%.0f MiB", $1/1024}' || echo '-')"
        ;;
    log)
        shift || true
        if [ "${1:-}" = "-f" ] || [ "${1:-}" = "--follow" ]; then tail -f "$LOG"; else tail -n 40 "$LOG"; fi
        ;;
    -h|--help|help)
        awk 'NR==1{next} /^#/||/^$/{sub(/^# ?/,""); print; next} {exit}' "$0"
        ;;
    *)
        echo "用法: $0 start|stop|status|log|foreground（--help 看说明）" >&2; exit 2 ;;
esac
