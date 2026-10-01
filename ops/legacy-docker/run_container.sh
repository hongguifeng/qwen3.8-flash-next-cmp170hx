#!/usr/bin/env bash
# Wait for the checkpoint download, start the patched vLLM container, stream its
# log, and only return once the API server answers /health + /v1/models.
# Exit codes: 0 = ready, 1 = startup failed, 2 = readiness timeout.
#
# Usage: run_container.sh [--now] [--name <container>]   (see usage())
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-$HOME/models/Qwen3.8-Flash-Next-AutoRound-3bpw-MTP}"
IMAGE="${IMAGE:-18gogogo/170hx1-qwen38nextf:sm80}"
# Extra image entrypoint knobs (see /opt/entrypoint.sh in the image).  Only the
# ones actually set in the environment are forwarded, so the image defaults stay
# authoritative; this exists so A/B runs can be `QWEN_BATCH_TOKENS=4096 ./run_container.sh`.
#
# 2026-09-28: leave QWEN_BATCH_TOKENS at the image default (2048).  Measured with 2048:
# 2.17 s / 3779 tok/s for a fresh 8K prompt, 10.09 s / 3247 tok/s for 32K.  With 8192 the
# long steady state is 2.0-2.35x slower (1609/1620 tok/s) because an 8192-token chunk is
# larger than every captured cudagraph size (max 2048) and therefore runs eager.
#   QWEN_CAPTURE_SIZES / QWEN_GRAPH_MODE / QWEN_EAGER  -- cudagraph tuning
#   QWEN_SSD_{WORKERS,CACHE_MB,DEPTH,PREFETCH}         -- PLE SSD tuning
#   QWEN_OMP_THREADS / QWEN_HOST / QWEN_MODEL_DIR / QWEN_SERVED_NAME
EXTRA_ENV_VARS=(
    QWEN_CAPTURE_SIZES QWEN_GRAPH_MODE QWEN_EAGER
    QWEN_SSD_WORKERS QWEN_SSD_CACHE_MB QWEN_SSD_DEPTH QWEN_SSD_PREFETCH
    QWEN_OMP_THREADS QWEN_HOST QWEN_MODEL_DIR QWEN_SERVED_NAME
    # allocator policy: PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True cures the
    # "free=0MiB / reserved=63.16GiB" fragmentation state left by a >=96K prefill
    PYTORCH_CUDA_ALLOC_CONF
    QSA_ALLOC_HEAL QSA_ALLOC_HEAL_FREE_MB QSA_ALLOC_HEAL_CACHED_MB QSA_ALLOC_HEAL_COOLDOWN_S
    # diagnostics only (see the CG_PATCH / QSA_*_PATCH knobs below)
    CG_INSTR CG_INSTR_MAX_PAIRS CG_INSTR_LOG_EVERY
)
extra_env_args=()
QWEN_MODEL_DIR="${QWEN_MODEL_DIR:-/model}"
# 2026-09-29 (OPS.md 9.19): allocator healer.  A >=96K-token prefill drives the caching
# allocator to device free=0 with a fragmented cache; every op that then needs a new block
# calls cudaMalloc while the device is full and the driver reclaims/evicts (~40 ms each),
# which made 2048-token prefills jump 0.65 s -> 2.1-2.6 s and long prefills drop to ~1000 tok/s.
# A background thread in the engine process releases cached blocks only in that regime
# (measured: full 131K/196K/262143 ladders stay at 0.66-0.69 s and ~2500 tok/s).
# Set QSA_ALLOC_HEAL=0 to disable, QSA_OPS_PATCH="" to run the stock qsa module.
#
# 2026-09-29 (OPS.md 9.20): default VRAM utilisation 0.96 -> 0.94.  At 0.96 a long prefill drove
# the device to free=0; the host-side driver then had to reclaim/evict, and the user's desktop
# (physical monitor on the Radeon RX550) stalled for several seconds while the machine was already
# low on host RAM.  0.94 + the healer still passes the full 131K/196K/262143 ladder
# (0.664/0.684/0.693 s post-probe, 2485-2526 tok/s, constant).
QSA_OPS_PATCH="${QSA_OPS_PATCH:-/home/hong/code/qwen3.8-flash-next-cmp170hx/ops/legacy-docker/qsa_ops_heal.py}"
QSA_ALLOC_HEAL="${QSA_ALLOC_HEAL:-1}"
QWEN_GPU_MEMORY="${QWEN_GPU_MEMORY:-0.94}"
for v in "${EXTRA_ENV_VARS[@]}"; do
    [[ -n "${!v:-}" ]] && extra_env_args+=(-e "$v=${!v}")
done

# Container name: comes from CONTAINER_NAME or --name, never from the generic NAME
# env var -- this host's shell injects NAME=Code (VS Code / pi extension sets it),
# which used to silently create a container named "Code" instead of recreating
# hong-pc.  IGNORED_NAME only feeds the hint printed by warn_ignored_name().
IGNORED_NAME="${NAME-}"
NAME="${CONTAINER_NAME:-hong-pc}"
PORT="${PORT:-8000}"
LOG="${LOG:-/home/hong/code/qwen3.8-flash-next-cmp170hx/ops/legacy-docker/fdl.log}"
MARKER="${MARKER:-/home/hong/code/qwen3.8-flash-next-cmp170hx/ops/legacy-docker/download-complete}"
SERVED_NAME="${SERVED_NAME:-Qwen3.8-Flash-Next}"

# Startup supervision knobs.
READY_TIMEOUT="${READY_TIMEOUT:-1800}"  # give up after this many seconds (cold start ~7 min)
POLL="${POLL:-5}"                       # seconds between readiness probes
STREAM_LOGS="${STREAM_LOGS:-1}"         # 1: stream `docker logs -f` while loading
TAIL_LINES="${TAIL_LINES:-40}"          # log lines replayed on failure / before streaming

# Triton JIT kernel cache (~/.triton/cache in the container, i.e. /root/.triton).
# By default it is container-local, and this script always `docker rm -f`s the
# old container, so every restart threw the compiled kernels away.  The next
# request that needs an uncompiled shape (e.g. the GPU top-k/top-p sampler's
# _gumbel_sample_kernel with the model's default temperature=1.0/top_k=20/top_p=0.95)
# then cold-compiles *inside an engine step*, and the engine loop stops logging
# for minutes -- 2026-09-28 it blocked EngineCore for ~7.5 min with no output.
# Persisting the cache on the host means only the very first run pays that cost.
PERSIST_TRITON_CACHE="${PERSIST_TRITON_CACHE:-1}"          # 1: mount the cache from the host
TRITON_CACHE_HOST_DIR="${TRITON_CACHE_HOST_DIR:-/home/hong/code/qwen3.8-flash-next-cmp170hx/ops/legacy-docker/triton_cache}"

HEALTH_URL="http://127.0.0.1:${PORT}/health"
MODELS_URL="http://127.0.0.1:${PORT}/v1/models"
STREAM_PID=""

# Diagnostics go to stderr, and stdout carries the container log stream (plus the
# final summary), so `docker logs` output stays greppable on its own.
log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*" >&2; }

die() { log "$*"; exit 1; }

usage() {
    cat >&2 <<'EOF'
用法: run_container.sh [--now] [--name <容器名>]
  --now             跳过等待模型下载标记文件
  --name <容器名>   容器名（默认 hong-pc，等价于 CONTAINER_NAME=...）
环境变量: IMAGE MODEL_DIR CONTAINER_NAME PORT LOG MARKER SERVED_NAME READY_TIMEOUT
          POLL STREAM_LOGS TAIL_LINES QWEN_CONTEXT QWEN_SEQS QWEN_BATCH_TOKENS
          QWEN_GPU_MEMORY QWEN_MTP CUDA_VISIBLE_DEVICES VLLM_WSL2_ENABLE_PIN_MEMORY
          QWEN_CAPTURE_SIZES QWEN_GRAPH_MODE QWEN_EAGER QWEN_SSD_WORKERS QWEN_SSD_CACHE_MB
          QWEN_SSD_DEPTH QWEN_SSD_PREFETCH QWEN_OMP_THREADS QWEN_HOST QWEN_MODEL_DIR QWEN_SERVED_NAME
          PROFILE PERSIST_TRITON_CACHE TRITON_CACHE_HOST_DIR
EOF
}

# `NAME` is far too generic to read from the environment (see the top of the file),
# but say so when it looks like someone meant to set the container name.
warn_ignored_name() {
    if [[ -n "$IGNORED_NAME" && "$IGNORED_NAME" != "$NAME" ]]; then
        log "提示：已忽略环境变量 NAME=$IGNORED_NAME（本机 shell 会注入它）；改容器名请用 CONTAINER_NAME=$IGNORED_NAME 或 --name $IGNORED_NAME"
    fi
}

check_prereqs() {
    local c missing=()
    for c in docker python3 curl; do
        command -v "$c" >/dev/null 2>&1 || missing+=("$c")
    done
    (( ${#missing[@]} == 0 )) || die "缺少命令：${missing[*]}"
    docker info >/dev/null 2>&1 || die "docker 不可用（daemon 没起来？sudo service docker start）"
}

wait_for_download() {
    if [[ -f "$MARKER" ]]; then
        log "下载已完成（$MARKER），跳过等待"
        return 0
    fi
    log "等待模型下载完成（标记文件 $MARKER）..."
    while [[ ! -f "$MARKER" ]]; do
        sleep 60
        grep -q "^ALL DONE" "$LOG" 2>/dev/null && break
    done
}

verify_checkpoint() {
    [[ -d "$MODEL_DIR" ]] || die "模型目录不存在：$MODEL_DIR"
    log "校验 checkpoint：$MODEL_DIR"
    python3 - "$MODEL_DIR" <<'PY'
import os, sys
d = sys.argv[1]
parts = [f for f in os.listdir(d) if f.endswith((".part", ".incomplete"))]
if parts:
    sys.exit(f"incomplete files remain: {parts[:3]}")
shards = sorted(f for f in os.listdir(d) if f.endswith(".safetensors"))
total = sum(os.path.getsize(os.path.join(d, f)) for f in shards)
print(f"{len(shards)} safetensors files, {total/2**30:.2f} GiB")
if len(shards) != 13:
    sys.exit(f"expected 13 safetensors files, found {len(shards)}")
PY
}

# The CDI spec lives in /var/run/cdi (tmpfs), so every WSL reboot needs it
# regenerated; nvidia-cdi-refresh.service often loses the boot race because NVML
# is not ready yet, and it gives up after 5 retries.  Fail fast with the fix
# instead of letting docker run die with "unresolvable CDI devices".
check_cdi() {
    if ! command -v nvidia-ctk >/dev/null 2>&1; then
        log "警告：找不到 nvidia-ctk，跳过 CDI 预检"
        return 0
    fi
    nvidia-ctk cdi list 2>/dev/null | grep -qx 'nvidia.com/gpu=all' && return 0
    cat >&2 <<'EOF'
[!] CDI spec 缺失：docker run --device nvidia.com/gpu=all 会失败。
    spec 位于 tmpfs，WSL 重启后丢失；开机自动生成的 service 常因 NVML 未就绪失败。
    任选一条执行后重跑本脚本：
      sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml   # 持久化，推荐
      sudo systemctl restart nvidia-cdi-refresh.service
EOF
    return 1
}

# Only the container id goes to stdout.
start_container() {
    local -a profile_args=() triton_mount=() ple_patch_mount=() cg_patch_mount=()
    local -a qsa_patch_mounts=()
    # PLE_PATCH=<host file> mounts a modified ple_ssd.py over the engine's copy
    # (used for temporary diagnostics/A-B runs; unset = stock image behaviour).
    if [[ -n "${PLE_PATCH:-}" ]]; then
        [[ -f "$PLE_PATCH" ]] || die "PLE_PATCH 指定的文件不存在：$PLE_PATCH"
        log "PLE_PATCH=$PLE_PATCH 会覆盖容器内的 ple_ssd.py（仅用于诊断）"
        ple_patch_mount=(-v "$PLE_PATCH:/opt/vllm/src/vllm/models/qwen4_exp/nvidia/ple_ssd.py:ro")
    fi
    # CG_PATCH=<host file> mounts cg_instr.py (CUPTI-free GPU-time attribution).
    # Only useful together with CG_INSTR=1 and the instrumented ple_ssd.py.
    if [[ -n "${CG_PATCH:-}" ]]; then
        [[ -f "$CG_PATCH" ]] || die "CG_PATCH 指定的文件不存在：$CG_PATCH"
        log "CG_PATCH=$CG_PATCH 会挂载为容器内 vllm/compilation/cg_instr.py（仅用于诊断）"
        cg_patch_mount=(-v "$CG_PATCH:/opt/vllm/src/vllm/compilation/cg_instr.py:ro")
    fi
    if [[ -n "${QSA_CACHE_PATCH:-}" ]]; then
        [[ -f "$QSA_CACHE_PATCH" ]] || die "QSA_CACHE_PATCH 指定的文件不存在：$QSA_CACHE_PATCH"
        qsa_patch_mounts+=(-v "$QSA_CACHE_PATCH:/opt/vllm/src/vllm/models/qwen4_exp/common/qsa_cache.py:ro")
    fi
    if [[ -n "${QSA_OPS_PATCH:-}" ]]; then
        [[ -f "$QSA_OPS_PATCH" ]] || die "QSA_OPS_PATCH 指定的文件不存在：$QSA_OPS_PATCH"
        qsa_patch_mounts+=(-v "$QSA_OPS_PATCH:/opt/vllm/src/vllm/models/qwen4_exp/nvidia/ops/qsa.py:ro")
    fi
    if [[ -n "${QSA_OPS_INDEXER_PATCH:-}" ]]; then
        [[ -f "$QSA_OPS_INDEXER_PATCH" ]] || die "QSA_OPS_INDEXER_PATCH 指定的文件不存在：$QSA_OPS_INDEXER_PATCH"
        qsa_patch_mounts+=(-v "$QSA_OPS_INDEXER_PATCH:/opt/vllm/src/vllm/models/qwen4_exp/nvidia/ops/qsa_indexer.py:ro")
    fi
    if [[ -n "${QSA_PRE_INDEXER_PATCH:-}" ]]; then
        [[ -f "$QSA_PRE_INDEXER_PATCH" ]] || die "QSA_PRE_INDEXER_PATCH 指定的文件不存在：$QSA_PRE_INDEXER_PATCH"
        qsa_patch_mounts+=(-v "$QSA_PRE_INDEXER_PATCH:/opt/vllm/src/vllm/models/qwen4_exp/nvidia/ops/qsa_pre_indexer.py:ro")
    fi
    # GDN_PATCH=<host file> mounts an instrumented copy of the GatedDeltaNet
    # attention layer (the QSA/GDN cache path is the other poisoning suspect).
    if [[ -n "${GDN_PATCH:-}" ]]; then
        [[ -f "$GDN_PATCH" ]] || die "GDN_PATCH 指定的文件不存在：$GDN_PATCH"
        qsa_patch_mounts+=(-v "$GDN_PATCH:/opt/vllm/src/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py:ro")
    fi
    # file).  The container runs as root, so the host dir must be world-writable
    # for both sides; on the very first run it is empty and Triton still has to
    # cold-compile everything once.
    if [[ "$PERSIST_TRITON_CACHE" == 1 ]]; then
        mkdir -p "$TRITON_CACHE_HOST_DIR"
        chmod 777 "$TRITON_CACHE_HOST_DIR" 2>/dev/null || true
        if [[ -n "$(ls -A "$TRITON_CACHE_HOST_DIR" 2>/dev/null | head -1)" ]]; then
            log "复用 Triton 缓存 $TRITON_CACHE_HOST_DIR（$(du -sh "$TRITON_CACHE_HOST_DIR" 2>/dev/null | cut -f1)）"
        else
            log "Triton 缓存 $TRITON_CACHE_HOST_DIR 为空：本次为冷启动，引擎首次用到新 shape 时仍会编译（之后会复用）"
        fi
        triton_mount=(-v "$TRITON_CACHE_HOST_DIR:/root/.triton/cache")
    else
        log "PERSIST_TRITON_CACHE=0：Triton 缓存仅存在于容器内，重建容器后丢失"
    fi
    # PROFILE=1 exposes the torch profiler (/start_profile, /stop_profile) inside
    # the container, writing kineto traces to /home/hong/code/qwen3.8-flash-next-cmp170hx/ops/prof.  The entrypoint wrapper
    # adds the marker-gated --profiler-config flag the image entrypoint lacks; the
    # profiler itself only records while /start_profile has been called.
    if [[ "${PROFILE:-0}" == 1 ]]; then
        profile_args=(--entrypoint /prof_enable.sh
                      -v "/home/hong/code/qwen3.8-flash-next-cmp170hx/ops/prof:/prof"
                      -v "/home/hong/code/qwen3.8-flash-next-cmp170hx/ops/diagnostics/enable_profiler_entrypoint.sh:/prof_enable.sh:ro")
    fi

    if docker inspect "$NAME" >/dev/null 2>&1; then
        log "移除同名容器 $NAME（它此前的日志会一起丢失）"
        # 2026-09-29 (OPS.md 9.21): graceful stop first.  An abrupt `rm -f` of a *running* engine
        # tears down a ~60 GiB CUDA context while kernels are still in flight; the host driver logs
        # that as a GPU fault (nvlddmkm "Error occurred on GPUID: b00"), which can make dxgkrnl
        # capture a WATCHDOG live kernel dump (freezes the whole machine for seconds) and makes WSL
        # write a 1-5 GB crash dump of the engine to the host disk.
        docker stop -t 30 "$NAME" >/dev/null 2>&1 || true
        docker rm -f "$NAME" >/dev/null 2>&1 || true
    fi

    # This WSL2 host exposes the GPUs through CDI (the legacy nvidia runtime hook
    # hangs), so pass the CDI device instead of --gpus.
    #
    # VLLM_WSL2_ENABLE_PIN_MEMORY=1 is REQUIRED for performance here.  Without it
    # vllm.platforms.cuda.CudaPlatformBase.is_pin_memory_available() returns False
    # on WSL2, so is_uva_available() is False and every UvaBufferPool becomes a
    # NonUvaBuffer (pageable): each copy_to_uva() then does a synchronous
    # CPU->GPU copy (48us for 16B, 473us for 1MiB) instead of publishing a
    # zero-copy UVA view (1.6us / 25us).  During decode that was ~76% of
    # EngineCore CPU and cost ~15ms per step.
    docker run -d \
        --name "$NAME" \
        --device nvidia.com/gpu=all \
        -e CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
        --ipc host \
        -p "${PORT}:8000" \
        -e QWEN_CONTEXT="${QWEN_CONTEXT:-262144}" \
        -e QWEN_SEQS="${QWEN_SEQS:-4}" \
        -e QWEN_BATCH_TOKENS="${QWEN_BATCH_TOKENS:-2048}" \
        -e QWEN_GPU_MEMORY="${QWEN_GPU_MEMORY:-0.94}" \
        -e QWEN_MTP="${QWEN_MTP:-1}" \
        -e VLLM_WSL2_ENABLE_PIN_MEMORY="${VLLM_WSL2_ENABLE_PIN_MEMORY:-1}" \
        "${extra_env_args[@]}" \
        "${profile_args[@]}" \
        "${triton_mount[@]}" \
        "${ple_patch_mount[@]}" \
        "${cg_patch_mount[@]}" \
        "${qsa_patch_mounts[@]}" \
        -v "${MODEL_DIR}:/model:ro" \
        --restart no \
        "$IMAGE"
}

stop_stream() {
    if [[ -n "$STREAM_PID" ]] && kill -0 "$STREAM_PID" 2>/dev/null; then
        kill "$STREAM_PID" 2>/dev/null || true
        wait "$STREAM_PID" 2>/dev/null || true
    fi
    STREAM_PID=""
}

on_interrupt() {
    stop_stream
    log "已中断等待；容器 $NAME 仍在后台加载（docker logs -f $NAME 继续看进度）"
    exit 130
}

http_code() { curl -s -o /dev/null -m 3 -w '%{http_code}' "$1" 2>/dev/null || true; }

# 0 = serving, 1 = container gone, 2 = timeout.
wait_for_ready() {
    local deadline=$((SECONDS + READY_TIMEOUT)) status exit_code probes=0
    while :; do
        if [[ "$(http_code "$HEALTH_URL")" == 200 && "$(http_code "$MODELS_URL")" == 200 ]]; then
            return 0
        fi

        status="$(docker inspect -f '{{.State.Status}}' "$NAME" 2>/dev/null || true)"
        if [[ -z "$status" ]]; then
            log "容器 $NAME 不见了"
            return 1
        fi
        if [[ "$status" != "running" ]]; then
            exit_code="$(docker inspect -f '{{.State.ExitCode}}' "$NAME" 2>/dev/null || true)"
            log "容器已退出（status=$status, exit=${exit_code:-?}）"
            return 1
        fi

        (( SECONDS >= deadline )) && { log "等待 ${READY_TIMEOUT}s 仍未就绪，超时"; return 2; }

        # Quiet mode: one heartbeat a minute with the last log line as progress.
        if (( STREAM_LOGS == 0 )) && (( probes % 12 == 0 )); then
            log "加载中… $(docker logs --tail 1 "$NAME" 2>/dev/null | tr -d '\r' | cut -c1-120)"
        fi
        probes=$((probes + 1))
        sleep "$POLL"
    done
}

print_ready() {
    local cid="$1" secs=$SECONDS
    cat <<EOF

============================================================
vLLM 服务已就绪（启动耗时 $((secs / 60))m$((secs % 60))s）
  容器   : $NAME (${cid:0:12})
  接口   : http://127.0.0.1:${PORT}/v1
  模型名 : $SERVED_NAME
  自检   : curl -sf http://127.0.0.1:${PORT}/health
           curl -s  http://127.0.0.1:${PORT}/v1/models
  日志   : docker logs -f $NAME
  停止   : docker stop -t 30 $NAME && docker rm -f $NAME
EOF
    if [[ "$PERSIST_TRITON_CACHE" == 1 ]]; then
        printf '  Triton : %s → /root/.triton/cache（%s）\n' \
            "$TRITON_CACHE_HOST_DIR" "$(du -sh "$TRITON_CACHE_HOST_DIR" 2>/dev/null | cut -f1 || echo '?')"
    fi
    if [[ "${PROFILE:-0}" == 1 ]]; then
        cat <<EOF
  剖析   : curl -X POST http://127.0.0.1:${PORT}/start_profile
           curl -X POST http://127.0.0.1:${PORT}/stop_profile   # trace → /home/hong/code/qwen3.8-flash-next-cmp170hx/ops/prof
EOF
    fi
    echo "============================================================"
}

main() {
    local start_now=0
    while (( $# )); do
        case "$1" in
            --now) start_now=1; shift ;;
            --name) [[ -n "${2:-}" ]] || die "--name 需要一个容器名"; NAME="$2"; shift 2 ;;
            -h|--help) usage; return 0 ;;
            *) usage; die "未知参数：$1" ;;
        esac
    done
    warn_ignored_name
    check_prereqs
    (( start_now )) || wait_for_download
    verify_checkpoint
    check_cdi || exit 1

    local cid
    SECONDS=0
    log "启动容器 $NAME（镜像 $IMAGE，端口 ${PORT}→8000）"
    if ! cid="$(start_container)"; then
        die "docker run 失败（见上面的 docker 报错）"
    fi
    log "容器已启动：${cid:0:12}（加载权重+抓图约 3~7 分钟）"

    trap stop_stream EXIT
    trap on_interrupt INT TERM
    if (( STREAM_LOGS )); then
        docker logs -f --tail "$TAIL_LINES" "$NAME" 2>&1 &
        STREAM_PID=$!
    fi

    local rc=0
    wait_for_ready || rc=$?
    stop_stream

    if (( rc == 0 )); then
        print_ready "$cid"
        return 0
    fi

    log "启动失败，最后 $TAIL_LINES 行日志："
    docker logs --tail "$TAIL_LINES" "$NAME" 2>&1 || true
    if (( rc == 2 )); then
        log "提示：可加长等待（READY_TIMEOUT=3600 ...）或先看进度 docker logs --tail 20 $NAME"
    else
        log "提示：docker inspect $NAME -f '{{.State.ExitCode}} {{.State.Error}}'；CDI 缺 spec 见脚本 check_cdi"
    fi
    return 1
}

# Sourceable for smoke tests; only runs when executed directly.
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
