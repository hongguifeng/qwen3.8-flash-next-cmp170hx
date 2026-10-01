#!/usr/bin/env bash
# Qwen3.8-Flash-Next 原生 WSL 引擎启动器（唯一直接调用 `vllm serve` 的地方）。
#
# 用法:
#   run_native.sh start        # 后台启动引擎（setsid nohup 分离，本脚本退出后继续跑）
#   run_native.sh stop         # 优雅停止（SIGTERM，最多等 60s；不给 SIGKILL，除非 QWEN_FORCE=1）
#   run_native.sh restart      # stop + start
#   run_native.sh status       # 端口 /health + 进程 + 运行中请求
#   run_native.sh print-cmd    # 【干跑】只打印将执行的 vllm 命令行，不启动任何东西
#   run_native.sh params       # 只打印参数来源与当前生效值
#   run_native.sh foreground   # 前台运行（调试用，Ctrl-C 退出）
#   run_native.sh --help
#
# 参数默认值全在 config/engine.env（**唯一来源**），环境变量优先：
#   QWEN_PORT=9393 QWEN_MTP=2 QWEN_BATCH_TOKENS=2048 run_native.sh start
#
# 目录布局：本脚本在 <repo>/vllm-native/bin/ 下，$ENGINE_DIR=<repo>/vllm-native，
# 引擎自身所需的一切（venv / 已打补丁的 vLLM 源码 / Triton 缓存 / 日志）都在其下，
# 唯一的外部依赖是只读的模型权重（QWEN_MODEL_DIR）与 <repo>/config/engine.env。
set -euo pipefail

ENGINE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # <repo>/vllm-native
ROOT="$(cd "$ENGINE_DIR/.." && pwd)"                            # 仓库根（config 里 $ROOT 的语义）
CFG_FILE="$ROOT/config/engine.env"
if [ ! -r "$CFG_FILE" ]; then
    echo "错误：找不到 $CFG_FILE（引擎参数的唯一默认值来源）" >&2
    exit 1
fi
. "$CFG_FILE"

VLLM_VENV="$ENGINE_DIR/opt/vllm/.venv"
LOGDIR="$ENGINE_DIR/logs"
PIDFILE="$LOGDIR/server.pid"
TRITON_CACHE="$QWEN_TRITON_CACHE_DIR"
PLE_LIB="$QWEN_PLE_LIB"
MODEL_DIR="$QWEN_MODEL_DIR"
PORT="$QWEN_PORT"

# ---- 环境（等价于镜像里的 /opt/entrypoint.sh，每一项都有踩坑依据）--------------
export PATH="$VLLM_VENV/bin:$PATH"
export OMP_NUM_THREADS="$QWEN_OMP_THREADS"
export MKL_NUM_THREADS="$QWEN_OMP_THREADS"
export TOKENIZERS_PARALLELISM=false
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES
export VLLM_USE_BREAKABLE_CUDAGRAPH="$QWEN_BREAKABLE_CUDAGRAPH"
export HF_HUB_OFFLINE=1
# 不要去掉：关掉之后解码掉 30~35%（见 ops/OPS.md）
export VLLM_WSL2_ENABLE_PIN_MEMORY="$QWEN_PIN_MEMORY"
export TRITON_CACHE_DIR="$TRITON_CACHE"
# 分配器治愈线程（qsa.py 已是 heal build）：关掉后长上下文会让 prefill 慢 4.5×
export QSA_ALLOC_HEAL="$QWEN_ALLOC_HEAL"
export QSA_ALLOC_HEAL_FREE_MB="$QWEN_ALLOC_HEAL_FREE_MB"
export QSA_ALLOC_HEAL_CACHED_MB="$QWEN_ALLOC_HEAL_CACHED_MB"
export QSA_ALLOC_HEAL_COOLDOWN_S="$QWEN_ALLOC_HEAL_COOLDOWN_S"
export PYTHONUNBUFFERED=1

# ---- 拼命令行（唯一实现；print-cmd 打印的就是它）-------------------------------
serve_cmd=(
    "$VLLM_VENV/bin/vllm" serve "$MODEL_DIR"
    --served-model-name "$QWEN_SERVED_NAME"
    --host "$QWEN_HOST" --port "$PORT"
    --dtype bfloat16
    --max-model-len "$QWEN_CONTEXT"
    --max-num-seqs "$QWEN_SEQS"
    --max-num-batched-tokens "$QWEN_BATCH_TOKENS"
    --gpu-memory-utilization "$QWEN_GPU_MEMORY"
    --safetensors-load-strategy lazy
    --mamba-cache-mode align
    --enable-prefix-caching
    --enable-chunked-prefill
    --default-chat-template-kwargs "{\"enable_thinking\": ${QWEN_ENABLE_THINKING}}"
    --reasoning-parser qwen3
    --enable-auto-tool-choice --tool-call-parser qwen3_xml
    --additional-config "{\"ple_ssd_offload\":true,\"ple_ssd_workers\":${QWEN_SSD_WORKERS},\"ple_ssd_cache_mb\":${QWEN_SSD_CACHE_MB},\"ple_ssd_native_library\":\"${PLE_LIB}\",\"ple_ssd_io_depth\":${QWEN_SSD_DEPTH},\"ple_ssd_prefetch_tokens\":${QWEN_SSD_PREFETCH}}"
)
if [ "${QWEN_EAGER:-0}" = "1" ]; then
    serve_cmd+=(--enforce-eager)
else
    serve_cmd+=(--compilation-config "{\"cudagraph_mode\":\"${QWEN_GRAPH_MODE}\",\"cudagraph_capture_sizes\":${QWEN_CAPTURE_SIZES}}")
fi
if (( QWEN_MTP > 0 )); then
    serve_cmd+=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${QWEN_MTP}}")
fi

is_up()    { curl -sf -o /dev/null --max-time 3 "http://127.0.0.1:${PORT}/health"; }
running()  { [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; }

case "${1:-start}" in
    start)
        mkdir -p "$LOGDIR" "$TRITON_CACHE"
        if [ ! -w "$TRITON_CACHE" ]; then
            echo "错误：TRITON_CACHE_DIR=$TRITON_CACHE 不可写（是不是 Docker(root) 建的那份？）" >&2
            echo "修复：cp -r $ROOT/ops/legacy-docker/triton_cache/. $TRITON_CACHE/" >&2
            exit 1
        fi
        # 日志轮转：超过 QWEN_LOG_MAX_MB 就换名保存，只留最近 QWEN_LOG_KEEP 份
        if [ -f "$LOGDIR/server.log" ] && [ "$(stat -c%s "$LOGDIR/server.log")" -gt "$(( QWEN_LOG_MAX_MB * 1048576 ))" ]; then
            mv "$LOGDIR/server.log" "$LOGDIR/server-$(date +%Y%m%d-%H%M%S).log"
            ls -1t "$LOGDIR"/server-*.log 2>/dev/null | tail -n "+$(( QWEN_LOG_KEEP + 1 ))" | xargs -r rm -f
        fi
        if running; then
            echo "已在运行 (pid $(cat "$PIDFILE"))；要重启请用 bin/stop.sh && bin/start.sh"
            exit 0
        fi
        echo "启动原生引擎：$VLLM_VENV/bin/vllm  port=$PORT  model=$MODEL_DIR"
        echo "日志：$LOGDIR/server.log（追加写入 O_APPEND；轮转见上）"
        # 用 >> （O_APPEND）而不是 > ：
        #   1) 轮转后不会把旧日志截掉，日志成为带轮转的连续记录（>100MB 自动归档，留 5 份）；
        #   2) 只有 O_APPEND 的写入者才能安全地被 logs.sh --clean 就地截断
        #      （非 O_APPEND 时进程记得自己的偏移量，截断后下次写入会在文件头留下 NUL 空洞）。
        setsid nohup "${serve_cmd[@]}" >>"$LOGDIR/server.log" 2>&1 &
        echo $! > "$PIDFILE"
        echo "pid=$(cat "$PIDFILE")  就绪后 /health 返回 200（首次需 5~7 分钟加载权重）"
        ;;
    stop)
        if ! running; then
            echo "引擎未在运行"
            rm -f "$PIDFILE"
            exit 0
        fi
        pid="$(cat "$PIDFILE")"
        echo "优雅停止 pid=$pid（SIGTERM，最多等 60s）…"
        kill -TERM "$pid" 2>/dev/null || true
        for _ in $(seq 1 60); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
        if kill -0 "$pid" 2>/dev/null; then
            # 默认不用 SIGKILL：强杀可能留下未落盘的缓存/半截状态（见 AGENTS.md 铁律 8）。
            if [ "${QWEN_FORCE:-0}" = "1" ]; then
                echo "60s 未退出且 QWEN_FORCE=1 ⇒ SIGKILL" >&2
                kill -KILL "$pid" 2>/dev/null || true
                sleep 2
            else
                echo "错误：pid $pid 60s 内未退出。请先 bin/logs.sh -n 80 看它在等什么；" >&2
                echo "      确认无请求在跑后再 QWEN_FORCE=1 bin/stop.sh" >&2
                exit 1
            fi
        fi
        rm -f "$PIDFILE"
        echo "已停止"
        ;;
    restart)
        "$0" stop
        exec "$0" start
        ;;
    status)
        echo "port $PORT  /health: $(is_up && echo 200 || echo DOWN)   pid: $(cat "$PIDFILE" 2>/dev/null || echo '-')"
        if running; then
            ps -o pid,etime,rss,cmd -p "$(cat "$PIDFILE")" | tail -1
            curl -s --max-time 3 "http://127.0.0.1:${PORT}/metrics" \
                | awk '/^vllm:num_requests_running/{print "running="$2}' || true
        fi
        ;;
    print-cmd)
        # 干跑：只输出将执行的参数，一行一个（便于和 /proc/<pid>/cmdline 逐项比对）
        printf '%s\n' "${serve_cmd[@]}"
        ;;
    params)
        echo "参数来源: $CFG_FILE   （环境变量优先，下面按字母序）"
        printf '  %-24s %s\n' QWEN_PORT "$QWEN_PORT" QWEN_HOST "$QWEN_HOST" QWEN_MTP "$QWEN_MTP" \
            QWEN_CONTEXT "$QWEN_CONTEXT" QWEN_SEQS "$QWEN_SEQS" QWEN_BATCH_TOKENS "$QWEN_BATCH_TOKENS" \
            QWEN_GPU_MEMORY "$QWEN_GPU_MEMORY" QWEN_MODEL_DIR "$QWEN_MODEL_DIR" \
            QWEN_TRITON_CACHE_DIR "$TRITON_CACHE" QWEN_ALLOC_HEAL "$QWEN_ALLOC_HEAL"
        ;;
    foreground)
        mkdir -p "$LOGDIR" "$TRITON_CACHE"
        exec "${serve_cmd[@]}"
        ;;
    -h|--help|help)
        awk 'NR==1{next} /^#/||/^$/{sub(/^# ?/,""); print; next} {exit}' "$0"
        ;;
    *)
        echo "用法: $0 start|stop|restart|status|print-cmd|params|foreground（--help 看说明）" >&2
        exit 2
        ;;
esac
