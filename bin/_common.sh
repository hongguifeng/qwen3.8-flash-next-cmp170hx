#!/usr/bin/env bash
# 共享库：被 bin/ 下所有脚本 source，不要直接执行。
# 职责只有四件：① 定位仓库路径 ② 加载唯一默认值来源 config/engine.env ③ 档位（模型/卡/端口）解析
# ④ 提供公共函数。
# 这里**不放默认值**——默认值全在 config/engine.env（改端口/参数/加模型只改那一个文件）。
# shellcheck shell=bash

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CFG_FILE="$ROOT/config/engine.env"
if [ ! -r "$CFG_FILE" ]; then
    echo "错误：找不到 $CFG_FILE（端口与引擎参数的唯一默认值来源）" >&2
    exit 1
fi
# 记下调用方是否**显式**传了这几个变量（source 之后就分不清了；档位解析靠它定优先级）
_ORIG_QWEN_PORT="${QWEN_PORT-}"
_ORIG_QWEN_SERVED_NAME="${QWEN_SERVED_NAME-}"
_ORIG_QWEN_MODEL_DIR="${QWEN_MODEL_DIR-}"
_ORIG_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-}"

# shellcheck source=/dev/null
. "$CFG_FILE"

# ---- 派生路径（全部由上面那份配置 + $ROOT 推出，不再有第二处默认值）------------
ENGINE_DIR="$ROOT/vllm-native"
LAUNCHER="$ENGINE_DIR/bin/run_native.sh"
DISTRO_NAME="${WSL_DISTRO_NAME:-Ubuntu}"

# ---- 档位（variant）：一个名字 = 一套「模型目录 + 模型名 + 端口 + 显卡」----------
# 清单/默认值全在 config/engine.env（QWEN_MODELS、QWEN_<档位>_*），新增档位不用改脚本。
indirect() { local n="$1"; printf '%s' "${!n:-}"; }

# 别名/大小写 → 档位名；认不出来返回 1
variant_canon() {
    local want lower alias m
    want="${1:-main}"
    lower="$(printf '%s' "$want" | tr 'A-Z' 'a-z')"
    for alias in $QWEN_ALIASES; do
        case "$alias" in "${lower}="*) printf '%s' "${alias#*=}"; return 0 ;; esac
    done
    for m in $QWEN_MODELS; do
        [ "$lower" = "$m" ] && { printf '%s' "$m"; return 0; }
    done
    return 1
}

# 档位变量名：variant_var unc MODEL_DIR → QWEN_UNC_MODEL_DIR
variant_var() { printf 'QWEN_%s_%s' "$(printf '%s' "$1" | tr 'a-z' 'A-Z')" "$2"; }

# 根据 QWEN_MODEL / 历史 QWEN_INSTANCE 算出所有随档位变化的变量：
#   VARIANT INSTANCE_ID MODEL_DIR PORT BASE_URL MODEL_NAME LOG_DIR LOGNAME LOG_FILE PID_FILE CUDA_VISIBLE_DEVICES
# 显式传的环境变量优先（_ORIG_* 就是为此存在）。幂等：脚本解析完 --model 后再调一次即可切档。
derive_variant() {
    local v mv nv pv gv
    if ! v="$(variant_canon "${QWEN_MODEL:-${QWEN_INSTANCE:-main}}")"; then
        echo "错误：未知档位 '${QWEN_MODEL:-${QWEN_INSTANCE:-}}'（可用：$QWEN_MODELS；别名：$QWEN_ALIASES）" >&2
        return 2
    fi
    VARIANT="$v"
    if [ "$v" = main ]; then INSTANCE_ID=""; else INSTANCE_ID="$v"; fi

    if [ "$v" != main ]; then
        mv="$(indirect "$(variant_var "$v" MODEL_DIR)")"
        nv="$(indirect "$(variant_var "$v" SERVED_NAME)")"
        pv="$(indirect "$(variant_var "$v" PORT)")"
        gv="$(indirect "$(variant_var "$v" GPU)")"
        [ -n "$_ORIG_QWEN_MODEL_DIR" ]       || QWEN_MODEL_DIR="$mv"
        [ -n "$_ORIG_QWEN_SERVED_NAME" ]     || QWEN_SERVED_NAME="$nv"
        [ -n "$_ORIG_QWEN_PORT" ]            || QWEN_PORT="$pv"
        [ -n "$_ORIG_CUDA_VISIBLE_DEVICES" ] || CUDA_VISIBLE_DEVICES="$gv"
    fi

    MODEL_DIR="$QWEN_MODEL_DIR"
    PORT="$QWEN_PORT"
    BASE_URL="http://127.0.0.1:${PORT}"
    MODEL_NAME="$QWEN_SERVED_NAME"
    LOG_DIR="$ENGINE_DIR/logs"
    if [ -n "$INSTANCE_ID" ]; then LOGNAME="server-$INSTANCE_ID"; else LOGNAME="server"; fi
    LOG_FILE="$LOG_DIR/$LOGNAME.log"
    PID_FILE="$LOG_DIR/$LOGNAME.pid"
    export CUDA_VISIBLE_DEVICES
}

# 切档（--model 解析后调）：成功返回 0，未知档位返回 2
derive_variant || exit 2

# 切档（脚本解析完 --model 后调）：成功返回 0，未知档位返回 2
# 导出 QWEN_MODEL，子进程（run_native.sh）才认得到——它就是靠环境变量继承的。
set_variant() {
    local v
    if ! v="$(variant_canon "$1")"; then
        echo "错误：未知档位 '$1'（可用：$QWEN_MODELS；别名：$QWEN_ALIASES）" >&2
        return 2
    fi
    QWEN_MODEL="$v"; export QWEN_MODEL
    QWEN_INSTANCE=""
    derive_variant
}

# 显式指定显卡（--gpu）：优先级高于档位默认值
set_gpu() { CUDA_VISIBLE_DEVICES="$1"; _ORIG_CUDA_VISIBLE_DEVICES="$1"; export CUDA_VISIBLE_DEVICES; }

# 列出可用档位（./start.sh --model list）
variant_list() {
    local v dir port gpu
    printf '  %-14s %-6s %-8s %s\n' 档位 端口 显卡 模型目录
    for v in $QWEN_MODELS; do
        if [ "$v" = main ]; then
            dir="$MODEL_DIR"; port="$PORT"; gpu="$CUDA_VISIBLE_DEVICES"
        else
            dir="$(indirect "$(variant_var "$v" MODEL_DIR)")"
            port="$(indirect "$(variant_var "$v" PORT)")"
            gpu="$(indirect "$(variant_var "$v" GPU)")"
        fi
        printf '  %-14s %-6s %-8s %s\n' "$v" "$port" "GPU$gpu" "$dir"
    done
    printf '\n  别名：%s（例：--model uncensored；历史 QWEN_INSTANCE=b 也一样）\n' "$QWEN_ALIASES"
}
PY="${PYTHON:-/usr/bin/python3}"
PS_EXE="/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"
WSL_EXE="/mnt/c/Windows/System32/wsl.exe"

# ---- 输出 ---------------------------------------------------------------------
c_ok()   { printf '\033[32m%s\033[0m' "$*"; }
c_warn() { printf '\033[33m%s\033[0m' "$*"; }
c_err()  { printf '\033[31m%s\033[0m' "$*"; }
hr()     { printf '%s\n' "------------------------------------------------------------"; }

# 用法帮助：把脚本头部的 `#` 注释块原样打印。
# 这样加/删注释不再需要数行号（以前的 `sed -n '2,17p'` 一加注释就错位）。
usage() { awk 'NR==1{next} /^#/||/^$/{sub(/^# ?/,""); print; next} {exit}' "${1:-$0}"; }
help_exit() { usage "$1"; exit 0; }

# 打印此刻实际生效的参数（不启动、不连接任何东西）。
# 刻意用 KEY=VALUE 逐行输出：可以 grep、可以直接对比，也不受中文宽度影响对齐。
print_params() {
    hr; printf '  生效参数（唯一来源 %s；环境变量优先）\n' "$CFG_FILE"; hr
    printf 'QWEN_MODEL=%s   (档位；可用 %s)\n' "$VARIANT" "$QWEN_MODELS"
    printf 'MODEL_DIR=%s\n'          "$MODEL_DIR"
    printf 'QWEN_PORT=%s\n'            "$QWEN_PORT"
    printf 'CUDA_VISIBLE_DEVICES=%s\n' "$CUDA_VISIBLE_DEVICES"
    printf 'QWEN_HOST=%s\n'            "$QWEN_HOST"
    printf 'QWEN_SERVED_NAME=%s\n'     "$QWEN_SERVED_NAME"
    printf 'QWEN_MODEL_DIR=%s\n'       "$QWEN_MODEL_DIR"
    printf 'QWEN_CONTEXT=%s\n'         "$QWEN_CONTEXT"
    printf 'QWEN_SEQS=%s\n'            "$QWEN_SEQS"
    printf 'QWEN_BATCH_TOKENS=%s\n'    "$QWEN_BATCH_TOKENS"
    printf 'QWEN_GPU_MEMORY=%s\n'      "$QWEN_GPU_MEMORY"
    printf 'QWEN_MTP=%s\n'             "$QWEN_MTP"
    printf 'QWEN_ENABLE_THINKING=%s\n' "$QWEN_ENABLE_THINKING"
    printf 'QWEN_SSD_WORKERS=%s\n'     "$QWEN_SSD_WORKERS"
    printf 'QWEN_SSD_CACHE_MB=%s\n'    "$QWEN_SSD_CACHE_MB"
    printf 'QWEN_SSD_DEPTH=%s\n'       "$QWEN_SSD_DEPTH"
    printf 'QWEN_SSD_PREFETCH=%s\n'    "$QWEN_SSD_PREFETCH"
    printf 'QWEN_ALLOC_HEAL=%s\n'      "$QWEN_ALLOC_HEAL"
    printf 'QWEN_TRITON_CACHE_DIR=%s\n' "$QWEN_TRITON_CACHE_DIR"
    printf 'BASE_URL=%s\n'             "$BASE_URL"
    printf 'PID_FILE=%s\n'             "$PID_FILE"
    printf 'LOG_FILE=%s\n'             "$LOG_FILE"
    hr
    printf '  改默认值 → 编辑 %s      临时覆盖 → QWEN_MTP=1 ./start.sh\n' "${CFG_FILE#"$ROOT"/}"
    printf '  换模型/换卡 → ./start.sh --model <档位> --gpu <n>   （列表：--model list）\n'
}

# ---- 探针 ---------------------------------------------------------------------
health_code() { curl -s -o /dev/null -w '%{http_code}' --max-time 5 "$BASE_URL/health" 2>/dev/null || true; }

engine_pid()   { [ -f "$PID_FILE" ] && cat "$PID_FILE" 2>/dev/null || true; }
engine_alive() { local p; p="$(engine_pid)"; [ -n "$p" ] && kill -0 "$p" 2>/dev/null; }
port_listening() { ss -ltn 2>/dev/null | grep -q ":${1}\b"; }

# 从 /metrics 取一个标量：metric <名字>
metric() {
    curl -s --max-time 5 "$BASE_URL/metrics" 2>/dev/null |
        awk -v k="vllm:$1" 'index($1, k "{") == 1 { print $2; exit }'
}

# 宿主机（Windows）内存：可用内存 / vmmemWSL 工作集
windows_host_mem() {
    "$PS_EXE" -NoProfile -Command \
        "\$o=Get-CimInstance Win32_OperatingSystem; 'free={0:N1}GB' -f (\$o.FreePhysicalMemory/1MB); \$p=Get-Process vmmemWSL -ErrorAction SilentlyContinue; if(\$p){'vmmem={0:N1}GB' -f (\$p.WorkingSet64/1GB)}" \
        2>/dev/null | tr -d '\000' | paste -sd' ' -
}

gpu_mem() {
    /mnt/c/Windows/System32/nvidia-smi.exe --query-gpu=index,memory.used,memory.total,utilization.gpu \
        --format=csv,noheader 2>/dev/null | paste -sd' | ' -
}

gpu_count() { /mnt/c/Windows/System32/nvidia-smi.exe --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l; }

# 指定显卡的占用（"used/total MiB, util N%"）；读不到则输出空。
# nvidia-smi 的 index 与 CUDA_VISIBLE_DEVICES 编号同序（CUDA_DEVICE_ORDER=PCI_BUS_ID）。
gpu_usage() {
    /mnt/c/Windows/System32/nvidia-smi.exe --query-gpu=index,memory.used,memory.total,utilization.gpu \
        --format=csv,noheader,nounits 2>/dev/null | tr -d '\r' |
        awk -F', *' -v want="${1%%,*}" '$1 == want { printf "%s/%s MiB, util %s%%", $2, $3, $4 }'
}
