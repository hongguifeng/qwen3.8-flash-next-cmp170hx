#!/usr/bin/env bash
# 启动 Qwen3.8-Flash-Next 推理服务（原生 WSL，后台）。根目录的 ./start.sh 就是这个脚本。
#
# 用法:
#   bin/start.sh                         # 启动默认档位 main（GPU0 / :8000）→ 等就绪 → 回收宿主内存
#   bin/start.sh unc                     # ← 简写：第一个位置参数就是档位
#   bin/start.sh unc 0                   # ← 简写：第二个位置参数是显卡编号（两个参数搞定）
#   bin/start.sh 1                       # ← 简写：只给数字 = 换卡（main 换到 GPU1）
#   bin/start.sh --model unc             # 长写法（等价于 `bin/start.sh unc`）
#   bin/start.sh --model unc --gpu 0     # 档位 unc 但换到 GPU0（端口仍用该档位默认 8001）
#   bin/start.sh --gpu 1                 # 默认档位 main 换到 GPU1
#   bin/start.sh --model list            # 只列出所有档位（模型目录 / 端口 / 显卡），不启动（简写：bin/start.sh list）
#   bin/start.sh --wait 900              # 就绪等待上限（秒），默认 600（见 QWEN_START_WAIT）
#   bin/start.sh --keep-cache            # 不回收 WSL 页缓存（默认回收，防整机发卡）
#   bin/start.sh --foreground            # 前台运行（调试用，Ctrl-C 退出）
#   bin/start.sh --params                # 只打印生效参数，不启动
#   bin/start.sh --help
#
# 档位 = 一套「模型目录 + 对外模型名 + 端口 + 显卡」，清单与默认值只在 config/engine.env
# 一处定义（QWEN_MODELS / QWEN_<大写档位>_{MODEL_DIR,SERVED_NAME,PORT,GPU}），加模型不用改脚本：
#   QWEN_MTP=1 QWEN_MODEL=unc ./start.sh      # 环境变量同样生效，且优先于档位默认值
#   QWEN_B_PORT=8002 ./start.sh --model unc   # 旧名 QWEN_B_* / QWEN_INSTANCE=b 也仍然有效
#
# 相关：bin/stop.sh --model unc / bin/status.sh --model unc / bin/logs.sh -f --model unc
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

WAIT="$QWEN_START_WAIT"; KEEP_CACHE=0; FG=0; MODEL_ARG=""; GPU_ARG=""; PARAMS_ONLY=0
POS=()   # 位置参数：<档位> [显卡编号]
while [ $# -gt 0 ]; do
    case "$1" in
        --model)      MODEL_ARG="${2:-}"; shift ;;
        --gpu)        GPU_ARG="${2:-}";   shift ;;
        --foreground) FG=1 ;;
        --keep-cache) KEEP_CACHE=1 ;;
        --wait)       WAIT="${2:-$QWEN_START_WAIT}"; shift ;;
        # --params 不在这里直接输出：必须先走完 --model 的切档，否则会打印出默认档位的值
        --params)     PARAMS_ONLY=1 ;;
        -h|--help)    help_exit "$0" ;;
        --)           shift; while [ $# -gt 0 ]; do POS+=("$1"); shift; done; break ;;
        -*)           echo "未知参数: $1（--help 看用法）" >&2; exit 2 ;;
        *)            POS+=("$1") ;;
    esac
    shift
done

# ---- 位置参数（简写）---------------------------------------------------------
#   ./start.sh unc      → 档位 unc（用它自己的默认卡）
#   ./start.sh unc 0    → 档位 unc + GPU0          （两个参数搞定）
#   ./start.sh 1        → 档位不变，换到 GPU1
# 与 --model/--gpu 混用时，显式长选项优先（下面用 ${VAR:-…} 实现）。
if [ "${#POS[@]}" -gt 2 ]; then
    echo "错误：位置参数最多两个 —— <档位> [显卡编号]（收到：${POS[*]}）" >&2; exit 2
fi
if [ "${#POS[@]}" -gt 0 ]; then
    p1="${POS[0]}"; p2="${POS[1]:-}"
    case "$p1" in
        *[!0-9,]*)  # 第一个是档位名：全部字符不都是数字/逗号
            MODEL_ARG="${MODEL_ARG:-$p1}"; [ -n "$p2" ] && GPU_ARG="${GPU_ARG:-$p2}" ;;
        *)          # 第一个是显卡编号，此时第二个（若有）才是档位
            GPU_ARG="${GPU_ARG:-$p1}";     [ -n "$p2" ] && MODEL_ARG="${MODEL_ARG:-$p2}" ;;
    esac
fi

# ---- 选档位 / 选卡（必须在下面所有检查之前）------------------------------------
case "${MODEL_ARG:-}" in
    '') ;;                                                    # 用默认档位（engine.env 里 QWEN_MODELS 的第一个）
    list|ls) variant_list; exit 0 ;;
    *)   set_variant "$MODEL_ARG" || exit 2 ;;
esac
if [ -n "$GPU_ARG" ]; then
    case "$GPU_ARG" in
        *[!0-9,]*|'') echo "错误：--gpu 只接受编号或逗号分隔的编号（例：--gpu 1 / --gpu 0,1）" >&2; exit 2 ;;
    esac
    case "$GPU_ARG" in
        *,*) echo "提示：--gpu 传了多个编号，但本脚本不设 --tensor-parallel-size，" >&2
             echo "      vllm 只会用其中第一张（要真多卡 TP 请改 config/engine.env 里的 serve 参数）。" >&2 ;;
    esac
    gpu_tot="$(gpu_count)"
    first_gpu="${GPU_ARG%%,*}"
    if [ -n "$gpu_tot" ] && [ "$gpu_tot" -gt 0 ] 2>/dev/null && [ "$first_gpu" -ge "$gpu_tot" ] 2>/dev/null; then
        c_err "错误：--gpu $GPU_ARG 超范围（本机 $gpu_tot 张卡，编号 0..$((gpu_tot - 1))）"
        exit 2
    fi
    set_gpu "$GPU_ARG"
fi

# --params：切档/换卡之后再打印，才反映真实生效值
if [ "$PARAMS_ONLY" = 1 ]; then
    print_params; exit 0
fi

if [ "$FG" = 1 ]; then
    exec "$LAUNCHER" foreground
fi

# 已在运行就不动它（探针用档位自己的端口与 pid 文件，与引擎实际监听的同源）
if [ "$(health_code)" = "200" ]; then
    echo "档位 $VARIANT 已在运行（pid $(engine_pid)，$(c_ok 'health=200')，端口 $PORT，GPU$CUDA_VISIBLE_DEVICES）"
    echo "要重启：bin/stop.sh --model $VARIANT && bin/start.sh --model $VARIANT"
    exit 0
fi

# 端口被占但不是本档位的引擎 ⇒ 拒绝启动（否则会起第二个引擎抢显存/撞端口）
if port_listening "$PORT"; then
    c_err "错误：端口 $PORT 已被占用，但它不是档位 $VARIANT 的引擎（/health ≠ 200）"
    ss -ltnp 2>/dev/null | grep ":$PORT" | sed 's/^/  /'
    echo "  别的档位在跑？ bin/status.sh --model <档位>；或换个端口：QWEN_PORT=8002 bin/start.sh --model $VARIANT"
    exit 1
fi

# 目标显卡已有大量显存占用 ⇒ 起不来（会 OOM），先提醒（不阻止，可能你就是要共享）
gpu_used="$(gpu_usage "$CUDA_VISIBLE_DEVICES")"
if [ -n "$gpu_used" ]; then
    used_mib="${gpu_used%%/*}"
    if [ "${used_mib:-0}" -gt 1024 ] 2>/dev/null; then
        c_warn "警告：GPU$CUDA_VISIBLE_DEVICES 已被占用 $gpu_used"
        echo "      本档位需要约 60 GiB 显存；先确认另一个实例是不是该停（bin/status.sh --model <档位>）"
    fi
fi

echo "启动档位 $VARIANT：模型 $MODEL_DIR"
echo "  端口 $PORT ／ 显卡 GPU$CUDA_VISIBLE_DEVICES ／ pid $PID_FILE"
echo "  日志 $LOG_FILE（当前该卡占用：${gpu_used:-读不到}）"
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
    echo; echo "排查：bin/logs.sh -e --model $VARIANT"
    exit 1
fi

echo; hr
printf '  %s  启动完成，耗时 %ss\n' "$(c_ok '服务已就绪 ✔')" "$el"
printf '  档位/模型   : %s（%s）\n' "$VARIANT" "$MODEL_NAME"
printf '  接口        : %s/v1   （模型名 %s）\n' "$BASE_URL" "$MODEL_NAME"
printf '  浏览器/客户端: http://127.0.0.1:%s/v1  ← Windows 侧必须用 127.0.0.1，不能用 localhost\n' "$PORT"
printf '  显卡        : GPU%s        进程/日志: pid %s / %s\n' "$CUDA_VISIBLE_DEVICES" "$PID" "$LOG_FILE"
printf '  显存        : %s\n' "$(gpu_mem)"
printf '  参数来源    : %s（改端口/参数/加档位只改它；--params 看生效值）\n' "${CFG_FILE#"$ROOT"/}"

if [ "$KEEP_CACHE" = 0 ]; then
    echo; echo "回收 WSL 页缓存（宿主内存，避免整机发卡）…"
    "$ROOT/bin/drop_host_cache.sh" || echo "  （回收失败，可稍后手动跑 bin/drop_host_cache.sh）"
fi
echo; echo "查看日志: bin/logs.sh -f --model $VARIANT      状态: bin/status.sh --model $VARIANT      停止: bin/stop.sh --model $VARIANT"
