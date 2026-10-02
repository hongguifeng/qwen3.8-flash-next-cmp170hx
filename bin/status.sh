#!/usr/bin/env bash
# 查看 Qwen3.8-Flash-Next 服务状态（健康、进程、请求、显存、宿主内存、投机解码接受率）。
#
# 用法:
#   bin/status.sh            # 完整状态
#   bin/status.sh --model unc # 看档位 unc（另一个实例）
#   bin/status.sh --watch 5  # 每 5 秒刷新一次（Ctrl-C 退出）
#   bin/status.sh --short    # 只输出一行（脚本友好，字段顺序固定）
#   bin/status.sh --params   # 只打印生效参数（不探服务）
#   bin/status.sh --help
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

WATCH=0; SHORT=0; MODEL_ARG=""; PARAMS_ONLY=0
while [ $# -gt 0 ]; do
    case "$1" in
        --watch) WATCH="${2:-5}"; shift ;;
        --short) SHORT=1 ;;
        --model) MODEL_ARG="${2:-}"; shift ;;
        --params) PARAMS_ONLY=1 ;;          # 不直接输出：要先让 --model 生效
        -h|--help) help_exit "$0" ;;
        *) echo "未知参数: $1（--help 看用法）" >&2; exit 2 ;;
    esac
    shift
done

# ---- 选档位（必须在探活/打印之前）------
case "${MODEL_ARG:-}" in
    '') ;;
    list|ls) variant_list; exit 0 ;;
    *) set_variant "$MODEL_ARG" || exit 2 ;;
esac
if [ "$PARAMS_ONLY" = 1 ]; then print_params; exit 0; fi

report() {
    local code pid up rss run wait kvraw kv draft acc accpct
    code="$(health_code)"
    pid="$(engine_pid)"
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
        read -r up rss < <(ps -o etime=,rss= -p "$pid" | awk '{print $1, $2}')
        rss="$(( ${rss:-0} / 1024 ))"
    else
        up="-"; rss=0
    fi
    run="$(metric num_requests_running)"; wait="$(metric num_requests_waiting)"
    # vLLM 0.29 已把 vllm:gpu_cache_usage_perc 改名为 vllm:kv_cache_usage_perc（值域 0~1）
    kvraw="$(metric kv_cache_usage_perc)"
    if [ -n "$kvraw" ]; then
        kv="$(awk -v v="$kvraw" 'BEGIN{printf "%.1f%%", 100*v}')"
    else
        kv="-"
    fi
    draft="$(metric spec_decode_num_draft_tokens_total)"
    acc="$(metric spec_decode_num_accepted_tokens_total)"
    if [ -n "$draft" ] && [ -n "$acc" ] && awk "BEGIN{exit !($draft>0)}"; then
        accpct="$(awk -v a="$acc" -v d="$draft" 'BEGIN{printf "%.1f%%", 100*a/d}')"
    else
        accpct="-"
    fi

    if [ "$SHORT" = 1 ]; then
        printf 'health=%s pid=%s up=%s running=%s waiting=%s kv=%s acc=%s\n' \
            "${code:-DOWN}" "${pid:--}" "$up" "${run:--}" "${wait:--}" "${kv:--}" "$accpct"
        return
    fi

    printf '  %-14s %s\n' "健康 /health" "$( [ "$code" = 200 ] && c_ok "200 ✔" || c_err "${code:-无响应} ✘" )"
    printf '  %-14s %s\n' "档位 / 显卡" "$VARIANT / GPU$CUDA_VISIBLE_DEVICES"
    printf '  %-14s %s\n' "模型名" "$MODEL_NAME"
    printf '  %-14s %s\n' "接口" "$BASE_URL/v1"
    printf '  %-14s %s\n' "进程 pid" "${pid:--}  运行时长 ${up}  RSS ${rss} MiB"
    printf '  %-14s %s\n' "当前请求" "运行中 ${run:--} / 排队 ${wait:--}"
    printf '  %-14s %s\n' "KV 缓存占用" "${kv:--}"
    printf '  %-14s %s\n' "MTP 接受率" "$accpct  (累计 draft=${draft:--} accepted=${acc:--})"
    printf '  %-14s %s\n' "显存" "$(gpu_mem)"
    printf '  %-14s %s\n' "宿主内存" "$(windows_host_mem)"
    printf '  %-14s %s\n' "guest 内存" "$(free -m | sed -n 2p | awk '{printf "used=%sMB free=%sMB cached=%sMB", $3, $4, $6}')"
    printf '  %-14s %s\n' "MTP 设置" "QWEN_MTP=$QWEN_MTP（改值：编辑 ${CFG_FILE#"$ROOT"/}）"
    printf '  %-14s %s\n' "日志" "$LOG_FILE ($(du -h "$LOG_FILE" 2>/dev/null | cut -f1))"
}

if [ "$WATCH" != 0 ]; then
    while :; do
        printf '\033[2J\033[H'; hr; echo "  Qwen3.8-Flash-Next 状态  [档位 $VARIANT]  $(date '+%F %T')"; hr; report
        sleep "$WATCH"
    done
else
    if [ "$SHORT" != 1 ]; then hr; echo "  Qwen3.8-Flash-Next 状态  [档位 $VARIANT]"; hr; fi
    report
    [ "$SHORT" = 1 ] || hr
fi
