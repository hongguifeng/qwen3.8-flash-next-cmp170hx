#!/usr/bin/env bash
# 共享库：被 bin/ 下所有脚本 source，不要直接执行。
# 职责只有三件：① 定位仓库路径 ② 加载唯一默认值来源 config/engine.env ③ 提供公共函数。
# 这里**不放默认值**——默认值全在 config/engine.env（改端口/参数只改那一个文件）。
# shellcheck shell=bash

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CFG_FILE="$ROOT/config/engine.env"
if [ ! -r "$CFG_FILE" ]; then
    echo "错误：找不到 $CFG_FILE（端口与引擎参数的唯一默认值来源）" >&2
    exit 1
fi
# shellcheck source=/dev/null
. "$CFG_FILE"

# ---- 派生路径（全部由上面那份配置 + $ROOT 推出，不再有第二处默认值）------------
ENGINE_DIR="$ROOT/vllm-native"
LAUNCHER="$ENGINE_DIR/bin/run_native.sh"
LOG_DIR="$ENGINE_DIR/logs"
LOG_FILE="$LOG_DIR/server.log"
PID_FILE="$LOG_DIR/server.pid"

PORT="$QWEN_PORT"
BASE_URL="http://127.0.0.1:${PORT}"
MODEL_NAME="$QWEN_SERVED_NAME"
DISTRO_NAME="${WSL_DISTRO_NAME:-Ubuntu}"
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
    printf 'QWEN_PORT=%s\n'            "$QWEN_PORT"
    printf 'QWEN_HOST=%s\n'            "$QWEN_HOST"
    printf 'QWEN_SERVED_NAME=%s\n'     "$QWEN_SERVED_NAME"
    printf 'QWEN_MODEL_DIR=%s\n'       "$QWEN_MODEL_DIR"
    printf 'QWEN_CONTEXT=%s\n'         "$QWEN_CONTEXT"
    printf 'QWEN_SEQS=%s\n'            "$QWEN_SEQS"
    printf 'QWEN_BATCH_TOKENS=%s\n'    "$QWEN_BATCH_TOKENS"
    printf 'QWEN_GPU_MEMORY=%s\n'      "$QWEN_GPU_MEMORY"
    printf 'QWEN_MTP=%s\n'             "$QWEN_MTP"
    printf 'QWEN_SSD_WORKERS=%s\n'     "$QWEN_SSD_WORKERS"
    printf 'QWEN_SSD_CACHE_MB=%s\n'    "$QWEN_SSD_CACHE_MB"
    printf 'QWEN_SSD_DEPTH=%s\n'       "$QWEN_SSD_DEPTH"
    printf 'QWEN_SSD_PREFETCH=%s\n'    "$QWEN_SSD_PREFETCH"
    printf 'QWEN_ALLOC_HEAL=%s\n'      "$QWEN_ALLOC_HEAL"
    printf 'QWEN_TRITON_CACHE_DIR=%s\n' "$QWEN_TRITON_CACHE_DIR"
    printf 'BASE_URL=%s\n'             "$BASE_URL"
    printf 'LOG_FILE=%s\n'             "$LOG_FILE"
    hr
    printf '  改默认值 → 编辑 %s      临时覆盖 → QWEN_MTP=1 ./start.sh\n' "${CFG_FILE#"$ROOT"/}"
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
