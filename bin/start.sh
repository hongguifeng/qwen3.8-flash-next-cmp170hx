#!/usr/bin/env bash
# 启动 Qwen3.8-Flash-Next 推理服务（原生 WSL，后台）。根目录的 ./start.sh 就是这个脚本。
#
# 用法:
#   bin/start.sh                 # 启动 → 等就绪 → 回收宿主内存
#   bin/start.sh --wait 900      # 就绪等待上限（秒），默认 600（见 QWEN_START_WAIT）
#   bin/start.sh --keep-cache    # 不回收 WSL 页缓存（默认回收，防整机发卡）
#   bin/start.sh --foreground    # 前台运行（调试用，Ctrl-C 退出）
#   bin/start.sh --params        # 只打印生效参数，不启动
#   bin/start.sh --help
#
# 参数默认值只在 config/engine.env 一处定义，环境变量优先：
#   QWEN_MTP=1 QWEN_GPU_MEMORY=0.94 QWEN_CONTEXT=262144 QWEN_SEQS=4 QWEN_BATCH_TOKENS=2048 bin/start.sh
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

WAIT="$QWEN_START_WAIT"; KEEP_CACHE=0; FG=0
while [ $# -gt 0 ]; do
    case "$1" in
        --foreground) FG=1 ;;
        --keep-cache) KEEP_CACHE=1 ;;
        --wait) WAIT="${2:-$QWEN_START_WAIT}"; shift ;;
        --params) print_params; exit 0 ;;
        -h|--help) help_exit "$0" ;;
        *) echo "未知参数: $1（--help 看用法）" >&2; exit 2 ;;
    esac
    shift
done

if [ "$FG" = 1 ]; then
    exec "$LAUNCHER" foreground
fi

# 已在运行就不动它（探针用 QWEN_PORT，与引擎实际端口同源）
if [ "$(health_code)" = "200" ]; then
    echo "服务已在运行（pid $(engine_pid)，$(c_ok 'health=200')）。要重启请用 bin/stop.sh && bin/start.sh"
    exit 0
fi

echo "启动原生引擎（pid 记在 $PID_FILE，日志 $LOG_FILE）…"
"$LAUNCHER" start || exit 1
PID="$(engine_pid)"

printf '等待就绪（最多 %ss，首次要加载 143 GB 权重，约 5~7 分钟）…' "$WAIT"
t0=$(date +%s); code=""
while :; do
    code="$(health_code)"
    [ "$code" = "200" ] && break
    if ! kill -0 "$PID" 2>/dev/null; then
        echo; c_err "进程 $PID 已退出！"; hr; tail -n 40 "$LOG_FILE"; exit 1
    fi
    now=$(date +%s); el=$((now - t0))
    [ "$el" -ge "$WAIT" ] && break
    printf '\r等待就绪… 已用 %ss / %ss（health=%s）   ' "$el" "$WAIT" "${code:-无响应}"
    sleep 10
done
el=$(( $(date +%s) - t0 ))

if [ "$code" != "200" ]; then
    echo; c_err "超时：${WAIT}s 内 /health 未返回 200"; hr
    echo "日志尾 40 行："; tail -n 40 "$LOG_FILE"
    echo; echo "排查：bin/logs.sh -e"
    exit 1
fi

echo; hr
printf '  %s  启动完成，耗时 %ss\n' "$(c_ok '服务已就绪 ✔')" "$el"
printf '  接口        : %s/v1   （模型名 %s）\n' "$BASE_URL" "$MODEL_NAME"
printf '  浏览器/客户端: http://127.0.0.1:%s/v1  ← Windows 侧必须用 127.0.0.1，不能用 localhost\n' "$PORT"
printf '  进程/日志   : pid %s / %s\n' "$PID" "$LOG_FILE"
printf '  显存        : %s\n' "$(gpu_mem)"
printf '  参数来源    : %s（改端口/参数只改它；--params 看生效值）\n' "${CFG_FILE#"$ROOT"/}"

if [ "$KEEP_CACHE" = 0 ]; then
    echo; echo "回收 WSL 页缓存（宿主内存，避免整机发卡）…"
    "$ROOT/bin/drop_host_cache.sh" || echo "  （回收失败，可稍后手动跑 bin/drop_host_cache.sh）"
fi
echo; echo "查看日志: bin/logs.sh -f      状态: bin/status.sh      停止: bin/stop.sh"
