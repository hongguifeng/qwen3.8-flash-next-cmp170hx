#!/usr/bin/env bash
# 优雅停止 Qwen3.8-Flash-Next 推理服务（SIGTERM，最多等 60s）。
#
# 用法:
#   bin/stop.sh --check          # 【先跑这个】只报告"现在停会不会打断用户请求"，不做任何事
#   bin/stop.sh                  # 停止引擎
#   bin/stop.sh --params         # 只打印生效参数
#   bin/stop.sh --help
#
# 停止逻辑全在 vllm-native/bin/run_native.sh 里（唯一实现），本脚本只做检查与收尾。
# 不会自动 SIGKILL：60s 未退出时它会报错并让你先看日志（真要强杀：QWEN_FORCE=1 bin/stop.sh）。
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

CHECK_ONLY=0
while [ $# -gt 0 ]; do
    case "$1" in
        --check) CHECK_ONLY=1 ;;
        --params) print_params; exit 0 ;;
        -h|--help) help_exit "$0" ;;
        *) echo "未知参数: $1（--help 看用法）" >&2; exit 2 ;;
    esac
    shift
done

running="$(metric num_requests_running)"
waiting="$(metric num_requests_waiting)"
# 指标读不到时不能瞎判：服务未就绪 / 端口刚改过还没重启时，
# 旧逻辑会把空值当成"有请求在跑"（"-" ≠ "0"），既吓人又误导。
if [ "$CHECK_ONLY" = 1 ]; then
    if [ -z "$running" ]; then
        echo "运行中请求=读不到  排队=读不到"
        c_warn "读不到 $BASE_URL/metrics ⇒ 无法判断有没有人在用"
        echo "  常见原因：服务未就绪，或端口与引擎实际监听不一致"
        echo "  核对：bin/status.sh --short     与     ss -ltn | grep ":$PORT""
        exit 1
    fi
    echo "运行中请求=$running  排队=$waiting"
    if [ "${running%%.*}" != "0" ]; then
        echo "$(c_warn '有请求在跑，现在停会中断它们')"
        exit 1
    fi
    echo "$(c_ok '空闲，可以安全停止')"
    exit 0
fi

if [ -n "$running" ] && [ "${running%%.*}" != "0" ]; then
    c_warn "警告：当前有 $running 个请求在跑（排队 $waiting），继续会中断它们。"
    printf '仍然继续? [y/N] '
    read -r ans
    case "$ans" in y|Y) ;; *) echo "已取消"; exit 0 ;; esac
fi

"$LAUNCHER" stop || exit 1

# 确认端口释放（端口来自 config/engine.env，唯一来源）
for _ in $(seq 1 20); do port_listening "$PORT" || break; sleep 1; done
if port_listening "$PORT"; then
    c_err "警告：端口 $PORT 仍被占用："; ss -ltnp 2>/dev/null | grep ":$PORT" | sed 's/^/  /'
else
    echo "端口 $PORT 已释放 ✔"
fi

echo "显存: $(gpu_mem)"

echo
echo "重启: bin/start.sh        回退到 Docker: $ROOT/ops/legacy-docker/run_container.sh"
