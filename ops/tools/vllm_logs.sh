#!/usr/bin/env bash
# 查看 docker 中 vLLM 容器的实时运行日志
#
# 【遗留工具】这是 Docker 时代的脚本，看的是 **容器** 日志；原生部署请用 bin/logs.sh。
# 端口 9393 是它的默认值（容器路径），不与 config/engine.env 共享。
#
# 用法:
#   ./vllm_logs.sh                 # 实时跟踪日志 (tail -f, 默认最近 200 行)
#   ./vllm_logs.sh -n 1000         # 实时跟踪, 先显示最近 1000 行
#   ./vllm_logs.sh -s 10m          # 只显示最近 10 分钟起的日志
#   ./vllm_logs.sh -g "error|warn" # 只看匹配关键字的行 (大小写不敏感)
#   ./vllm_logs.sh --grep-engine   # 常用过滤: 引擎/请求相关
#   ./vllm_logs.sh -t              # 显示 docker 时间戳
#   ./vllm_logs.sh --no-follow     # 打印完即退出 (不加 -f)
#   ./vllm_logs.sh -c hong-pc      # 指定容器名(默认自动探测)
#   ./vllm_logs.sh --status        # 查看容器状态 + /health + 最近日志尾部
#
# 环境变量:
#   CONTAINER=hong-pc      容器名(覆盖自动探测; 也接受 run_container.sh 的 CONTAINER_NAME)
#   PORT=9393              宿主端口(用于 --status 健康检查)
set -uo pipefail

CONTAINER="${CONTAINER:-${CONTAINER_NAME:-hong-pc}}"
PORT="${PORT:-9393}"
LINES=200
SINCE=""
GREP=""
FOLLOW=1
TIMESTAMPS=""
STATUS_ONLY=0

usage() { sed -n '2,20p' "$0" | sed 's/^# \?//'; exit "${1:-0}"; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    -n|--lines)   LINES="$2"; shift 2 ;;
    -s|--since)   SINCE="$2"; shift 2 ;;
    -g|--grep)    GREP="$2"; shift 2 ;;
    -c|--container) CONTAINER="$2"; shift 2 ;;
    -t|--timestamps) TIMESTAMPS="-t"; shift ;;
    --no-follow)  FOLLOW=0; shift ;;
    --status)     STATUS_ONLY=1; shift ;;
    --grep-engine) GREP='engine|request|Throughput|Avg|KV cache|gpu|cuda|OOM|error|warn'; shift ;;
    -h|--help)    usage 0 ;;
    *) echo "未知参数: $1" >&2; usage 1 ;;
  esac
done

# 1) 探测容器: 优先指定名, 否则找镜像里带 vllm 的容器
if ! docker inspect "$CONTAINER" >/dev/null 2>&1; then
  CAND=$(docker ps --format '{{.Names}} {{.Image}}' | grep -i vllm | head -1 | awk '{print $1}')
  if [[ -z "${CAND:-}" ]]; then
    echo "找不到容器 '$CONTAINER'，也没有镜像名含 vllm 的运行中容器。" >&2
    echo "当前运行中的容器:" >&2
    docker ps --format '  {{.Names}}\t{{.Image}}\t{{.Status}}' >&2
    exit 1
  fi
  echo "自动选中容器: $CAND" >&2
  CONTAINER="$CAND"
fi

# 2) --status: 状态 + health + 最近日志尾部
if [[ "$STATUS_ONLY" == 1 ]]; then
  echo "=== 容器状态 ($CONTAINER) ==="
  docker ps -a --filter "name=^/${CONTAINER}$" \
    --format 'Name: {{.Names}}\nImage: {{.Image}}\nStatus: {{.Status}}\nPorts: {{.Ports}}'
  echo
  echo "=== 健康检查 (http://127.0.0.1:${PORT}/health) ==="
  if curl -sf -m 5 "http://127.0.0.1:${PORT}/health" >/dev/null; then
    echo "OK (服务已就绪)"
  else
    echo "FAIL / 无响应 (可能仍在加载模型)"
  fi
  echo
  echo "=== 最近 40 行日志 ==="
  docker logs --tail 40 -t "$CONTAINER" 2>&1
  exit 0
fi

# 3) 组装 docker logs 参数
ARGS=(--tail "$LINES")
[[ -n "$SINCE" ]] && ARGS+=(--since "$SINCE")
[[ -n "$TIMESTAMPS" ]] && ARGS+=("$TIMESTAMPS")
[[ "$FOLLOW" == 1 ]] && ARGS+=(-f)
ARGS+=("$CONTAINER")

echo ">>> docker logs ${ARGS[*]}  (Ctrl-C 退出)" >&2
if [[ -n "$GREP" ]]; then
  echo ">>> 过滤: $GREP" >&2
  exec docker logs "${ARGS[@]}" 2>&1 | grep --line-buffered -iE "$GREP"
else
  exec docker logs "${ARGS[@]}" 2>&1
fi
