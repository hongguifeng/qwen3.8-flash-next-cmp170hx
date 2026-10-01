#!/usr/bin/env bash
# WSL2 部署 / 自检脚本（根目录的 ./deploy.sh 是这个脚本的薄封装）。
#
# 用法:
#   ops/deploy/deploy_wsl2.sh check            # 【默认】只读体检：环境/模型/运行时/服务逐项 PASS|WARN|FAIL
#   ops/deploy/deploy_wsl2.sh check --json     # 同上，输出机器可读 JSON（CI/脚本用）
#   ops/deploy/deploy_wsl2.sh plan             # 只打印部署步骤（不做任何动作）
#   ops/deploy/deploy_wsl2.sh install          # 只打印"缺什么、要怎么装"（不动手）
#   ops/deploy/deploy_wsl2.sh install --yes    # 幂等补齐：建目录 / 打补丁 / 编 ple_ssd_io.so
#   ops/deploy/deploy_wsl2.sh start            # 拉起服务（委托 bin/start.sh，唯一启动实现）
#   ops/deploy/deploy_wsl2.sh verify           # 验收：health=200 + 新鲜 2048 预填（对比基线）
#   ops/deploy/deploy_wsl2.sh verify --full    # 追加 bin/bench.sh --full（131072 长上下文 + 中毒检查，约 3 分钟）
#   ops/deploy/deploy_wsl2.sh all              # check → start → verify
#
# 选项: --yes（install 才真动手） --download-model（install 时允许拉 142.5 GiB 权重） --json --full
#
# 设计约束（对应 docs/SCRIPTS.md 第 4 节四条约定）：
#   ① 不写默认值：source bin/_common.sh ⇒ config/engine.env 是唯一来源（端口/模型/PLE 库路径都取自它）；
#   ② help 走 help_exit（不数行号）；
#   ③ ROOT/BASE_URL/PORT/LOG_FILE 全部来自 _common.sh；
#   ④ 不重启、不强杀：启动一律 exec bin/start.sh；install 在服务运行中拒绝改源码树。
#
# 安全默认：本脚本**默认只读**。会改磁盘的动作只有 `install --yes`，且引擎在跑时拒绝打补丁。
# 完整部署说明见 docs/DEPLOY-WSL2.md。
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/../../bin/_common.sh"

# ---- 路径（全部由 $ROOT / config/engine.env 推出，无第二处默认值）---------------
ENGINE_DIR_LOCAL="$ROOT/vllm-native"
VENVDIR="$ENGINE_DIR_LOCAL/opt/vllm/.venv"
VENVSP="$VENVDIR/lib/python3.12/site-packages"
VENVPY="$VENVDIR/bin/python"
SRC="$ENGINE_DIR_LOCAL/opt/vllm/src"
MODEL_DIR="$QWEN_MODEL_DIR"
PLE_LIB="$QWEN_PLE_LIB"

# ---- 期望值（从仓库既有文件读，不重复定义）------------------------------------
CONSTRAINTS="$ROOT/requirements/tested-constraints.txt"
VLLM_REVISION="$(sed -n 's/^VLLM_REVISION=//p' "$ROOT/scripts/common.sh" | head -1)"
MODEL_REVISION="$(sed -n 's/^MODEL_REVISION=//p' "$ROOT/scripts/common.sh" | head -1)"
MODEL_SHARDS=13            # 11 × model-* + 2 × mtp-model-*（本机实测）
PLE_TABLE_MIN_GB=100       # 最大分片必须 ≥ 100 GB（95.4 GiB BF16 PLE 表，O_DIRECT 直读）
DISK_MIN_GB=250            # 全新部署：142.5(模型) + 8.3(运行时) + 余量
DISK_MIN_GB_IDLE=30        # 权重已在位：只需要运行余量

pinned() { sed -n "s/^$1==//p" "$CONSTRAINTS" 2>/dev/null | head -1; }
dist_version() { ls -d "$VENVSP/$1"-*.dist-info 2>/dev/null | head -1 | xargs -r basename | sed 's/\.dist-info$//'; }

# ---- 结果收集 -----------------------------------------------------------------
MODE=check; JSON=0; YES=0; FULL=0; ALLOW_DL=0
RESULT_FILE="$(mktemp)"; trap 'rm -f "$RESULT_FILE"' EXIT
N_PASS=0; N_WARN=0; N_FAIL=0

badge() {
    case "$1" in
        PASS) c_ok   '[PASS]' ;;
        WARN) c_warn '[WARN]' ;;
        FAIL) c_err  '[FAIL]' ;;
    esac
}
_rec() { # _rec <名字> <PASS|WARN|FAIL> <详情> [修复建议]
    local name="$1" st="$2" det="${3//$'\t'/ }" fix="${4:-}"
    printf '%s\t%s\t%s\t%s\n' "$name" "$st" "$det" "$fix" >>"$RESULT_FILE"
    case "$st" in PASS) N_PASS=$((N_PASS+1));; WARN) N_WARN=$((N_WARN+1));; FAIL) N_FAIL=$((N_FAIL+1));; esac
    [ "$JSON" = 1 ] && return 0
    printf '  %s %-20s %s\n' "$(badge "$st")" "$name" "$det"
    if [ -n "$fix" ] && [ "$st" != PASS ]; then
        printf '         %s %s\n' "$(c_warn '→ 修复:')" "$fix"
    fi
    return 0
}
ok()   { _rec "$1" PASS "${2:-}"; }
warn() { _rec "$1" WARN "${2:-}" "${3:-}"; }
bad()  { _rec "$1" FAIL "${2:-}" "${3:-}"; }
have() { command -v "$1" >/dev/null 2>&1; }

json_esc() { printf '%s' "$1" | sed -e 's/\\/\\\\/g' -e 's/"/\\"/g'; }

# ---- Windows 侧路径 -----------------------------------------------------------
WIN_NVIDIA="/mnt/c/Windows/System32/nvidia-smi.exe"
wslconfig_path() {
    local p
    p="$("$PS_EXE" -NoProfile -Command '$env:USERPROFILE' 2>/dev/null | tr -d '\r\n\000')"
    if [ -n "$p" ]; then
        p="$(wslpath "$p" 2>/dev/null || true)/.wslconfig"
        [ -r "$p" ] && { printf '%s' "$p"; return 0; }
    fi
    for p in /mnt/c/Users/*/.wslconfig; do [ -r "$p" ] && { printf '%s' "$p"; return 0; }; done
    return 1
}

# ============================== 体检项 ========================================
chk_host() {
    local kern; kern="$(uname -r)"
    if ! grep -qi microsoft /proc/version; then
        bad host.distro "内核 $kern 不是 WSL2" "本部署依赖 WSL2 的 /dev/dxg；重跑: wsl --install / wsl --update"
    else
        ok host.distro "$(. /etc/os-release; printf '%s' "$PRETTY_NAME") / WSL 内核 $kern / $(nproc) vCPU"
    fi

    local cfg mem net
    if cfg="$(wslconfig_path)"; then
        mem="$(sed -n 's/^[[:space:]]*memory[[:space:]]*=[[:space:]]*\([0-9]*\).*/\1/pi' "$cfg" | tail -1)"
        net="$(sed -n 's/^[[:space:]]*networkingMode[[:space:]]*=[[:space:]]*\([A-Za-z]*\).*/\1/p' "$cfg" | tail -1)"
        if [ -z "$mem" ]; then
            warn host.wslconfig "$cfg 未设 memory=" "建议 memory=48GB（参考 config 里本机实测值，见 docs/DEPLOY-WSL2.md §2.1）"
        elif [ "$mem" -lt 32 ]; then
            bad host.wslconfig "memory=${mem}GB 偏小" "引擎加载 143 GB 权重需要宿主余量；设 memory=48GB 后 wsl --shutdown 生效（先征得用户同意）"
        elif [ "$mem" -lt 44 ]; then
            warn host.wslconfig "memory=${mem}GB（本机用 48GB）" "设 memory=48GB 后 wsl --shutdown 生效（先征得用户同意）"
        else
            ok host.wslconfig "memory=${mem}GB / networkingMode=${net:-未设}（$cfg）"
        fi
        if [ "${net:-}" != "mirrored" ]; then
            warn host.netmode "networkingMode=${net:-未设}" "非 mirrored 时 Windows 侧不能用 127.0.0.1，要改用 WSL 的 IP（铁律 2）"
        fi
        if grep -qiE '^[[:space:]]*autoMemoryReclaim' "$cfg"; then
            warn host.autoreclaim "存在 autoMemoryReclaim" "本机已删除该行（ops/OPS.md §9.22.1）；改 .wslconfig 后需 wsl --shutdown"
        fi
    else
        warn host.wslconfig "找不到 .wslconfig" "在 Windows 用户目录建 %USERPROFILE%\\.wslconfig，内容见 docs/DEPLOY-WSL2.md §2.1"
    fi
}

chk_gpu() {
    if [ ! -e /dev/dxg ]; then
        bad host.gpu "/dev/dxg 不存在（WSL 里看不到 GPU）" "更新 Windows NVIDIA 驱动（≥616.92）+ wsl --update；不要装 Linux 内核驱动"
        return 0
    fi
    local line name total drv
    line="$("$WIN_NVIDIA" --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null | head -1 | tr -d '\r')"
    if [ -z "$line" ]; then
        warn host.gpu "/dev/dxg 在，但读不到 Windows 侧 nvidia-smi" "确认 $WIN_NVIDIA 存在；或看 /usr/lib/wsl/lib/nvidia-smi"
        return 0
    fi
    name="$(printf '%s' "$line" | cut -d, -f1 | xargs)"; total="$(printf '%s' "$line" | cut -d, -f2 | xargs)"; drv="$(printf '%s' "$line" | cut -d, -f3 | xargs)"
    local mib; mib="$(printf '%s' "$total" | tr -dc '0-9')"
    if [ -n "$mib" ] && [ "$mib" -lt 60000 ]; then
        bad host.gpu "$name $total（driver $drv）" "权重 47 GiB + KV + CUDA 图需要 ≥64 GiB；本机为 65 536 MiB"
    else
        ok host.gpu "$name $total / driver $drv / /dev/dxg OK"
    fi
}

chk_mem() {
    local tot; tot="$(free -g | awk 'NR==2{print $2}')"
    if [ "${tot:-0}" -ge 40 ]; then ok host.ram "guest ${tot}GiB 可用"
    elif [ "${tot:-0}" -ge 24 ]; then warn host.ram "guest ${tot}GiB" "建议 .wslconfig memory=48GB"
    else bad host.ram "guest ${tot}GiB" "内存不足：.wslconfig memory=48GB + wsl --shutdown（先征得同意）"; fi

    local hm; hm="$(windows_host_mem)"
    if [ -z "$hm" ]; then warn host.vmmem "读不到宿主内存（PowerShell 不可用？）" "手动看任务管理器 / Get-Process vmmemWSL"
    else
        local free_g vmmem_g
        free_g="$(printf '%s' "$hm" | sed -n 's/.*free=\([0-9.]*\)GB.*/\1/p')"
        vmmem_g="$(printf '%s' "$hm" | sed -n 's/.*vmmem=\([0-9.]*\)GB.*/\1/p')"
        if [ -n "$vmmem_g" ] && awk "BEGIN{exit !($vmmem_g > 32)}"; then
            warn host.vmmem "宿主 free=${free_g:-?}GB, vmmemWSL=${vmmem_g}GB（>32GB）" "跑 bin/drop_host_cache.sh 回收页缓存；解码测速前必须 <32GB（铁律 3/7）"
        elif [ -n "$free_g" ] && awk "BEGIN{exit !($free_g < 5)}"; then
            warn host.vmmem "宿主 free=${free_g}GB" "宿主内存紧张，整机会发卡；跑 bin/drop_host_cache.sh"
        else
            ok host.vmmem "宿主 free=${free_g:-?}GB, vmmemWSL=${vmmem_g:-?}GB"
        fi
    fi
}

chk_disk() {
    local avail_kb tot_kb avail_gb tot_gb need hard
    avail_kb="$(df -Pk "$ROOT" | awk 'NR==2{print $4}')"; tot_kb="$(df -Pk "$ROOT" | awk 'NR==2{print $2}')"
    avail_gb=$((avail_kb/1048576)); tot_gb=$((tot_kb/1048576))
    # 权重已在位时不再需要那 142.5 GiB 的下载空间，只留运行余量
    if [ -r "$MODEL_DIR/model.safetensors.index.json" ]; then need=$DISK_MIN_GB_IDLE; else need=$DISK_MIN_GB; fi
    hard=$(( need / 2 ))
    if [ "$avail_gb" -ge "$need" ]; then
        ok disk.space "可用 ${avail_gb}GiB / 共 ${tot_gb}GiB（${ROOT}）；本机需要 ≥${need}GiB"
    elif [ "$avail_gb" -ge "$hard" ]; then
        warn disk.space "可用 ${avail_gb}GiB（需要 ≥${need}GiB）" "腾空间；不要在引擎运行时做 GB 级删除（铁律 12）"
    else
        bad disk.space "可用 ${avail_gb}GiB（需要 ≥${need}GiB）" "先 bin/stop.sh 再做清理（铁律 12）"
    fi
}

chk_model() {
    if [ ! -d "$MODEL_DIR" ]; then
        bad model.dir "目录不存在：$MODEL_DIR" "改 config/engine.env 的 QWEN_MODEL_DIR，或下载权重（见 §4.3）"
        return 0
    fi
    local files=() f
    shopt -s nullglob; files=("$MODEL_DIR"/*.safetensors); shopt -u nullglob
    if [ "${#files[@]}" -eq 0 ]; then
        bad model.dir "$MODEL_DIR 里没有 safetensors" "用 ops/tools/fdl.py 下载（可续传），revision 钉 $MODEL_REVISION"
        return 0
    fi
    local bytes gib
    bytes="$(stat -c%s "${files[@]}" | awk '{s+=$1} END{print s}')"
    gib="$(awk -v b="$bytes" 'BEGIN{printf "%.1f", b/1073741824}')"
    local big bigsize biggb bigname
    big="$(stat -c '%s %n' "${files[@]}" | sort -rn | head -1)"
    bigsize="${big%% *}"; bigname="${big#* }"
    biggb="$(awk -v b="$bigsize" 'BEGIN{printf "%.1f", b/1000000000}')"

    if [ ! -r "$MODEL_DIR/model.safetensors.index.json" ]; then
        bad model.index "缺 model.safetensors.index.json" "不完整的快照；重跑下载器（.fdl-state.json 会续传）"
    else
        ok model.index "index json 存在（PLE 表按 weight_map 定位分片）"
    fi
    if [ "${#files[@]}" -eq "$MODEL_SHARDS" ] && awk "BEGIN{exit !($gib > 140 && $gib < 145)}"; then
        ok model.shards "${#files[@]} 个分片 / ${gib}GiB（= 142.5 GiB 目标）"
    else
        bad model.shards "${#files[@]} 个分片 / ${gib}GiB（期望 ${MODEL_SHARDS} 个 / ≈142.5GiB）" \
            "用 ops/tools/fdl.py 续传（状态在 $MODEL_DIR/.fdl-state.json）；校验用 ops/tools/verify_ckpt.py"
    fi
    if awk "BEGIN{exit !($biggb >= $PLE_TABLE_MIN_GB)}"; then
        ok model.ple "最大分片 $(basename "$bigname") = ${biggb} GB（BF16 PLE 表，O_DIRECT 直读 SSD）"
    else
        bad model.ple "最大分片只有 ${biggb} GB" "PLE 表分片（102.4 GB）缺失或截断 ⇒ 重下该分片"
    fi
}

chk_runtime() {
    if [ ! -x "$VENVPY" ]; then
        bad runtime.venv "缺 $VENVPY" "按 docs/DEPLOY-WSL2.md §4.4 部署运行时（路线 A 提取 / 路线 B 构建）"
        return 0
    fi
    local pyv; pyv="$("$VENVPY" -V 2>&1 | awk '{print $2}')"
    local torch triton vllm tf
    torch="$(dist_version torch)"; triton="$(dist_version triton)"; vllm="$(dist_version vllm)"; tf="$(dist_version transformers)"
    local exp_torch exp_triton exp_tf
    exp_torch="$(pinned torch)"; exp_triton="$(pinned triton)"; exp_tf="$(pinned transformers)"

    if [ "${pyv%%.*}" = "3" ] && [ "${pyv#3.}" = "12.14" ]; then
        ok runtime.python "Python $pyv（venv → uv-python/cpython-3.12-linux-x86_64-gnu）"
    else
        warn runtime.python "Python ${pyv:-未知}" "本机实测 3.12.14；换解释器要重建 venv（editable 安装 + cu130 轮子）"
    fi
    if [ "$torch" = "torch-${exp_torch}+cu130" ] || [ "${torch#torch-}" = "${exp_torch}+cu130" ]; then
        ok runtime.torch "${torch#torch-}（tested-constraints 钉 $exp_torch）"
    else
        warn runtime.torch "torch=${torch#torch-}（期望 ${exp_torch}+cu130）" "版本不一致不必然坏，但基线数字（docs/RESULTS-WSL2.md）只在钉住的版本上成立"
    fi
    if [ "${triton#triton-}" = "$exp_triton" ]; then ok runtime.triton "triton $exp_triton"
    else warn runtime.triton "triton=${triton#triton-}（期望 $exp_triton）" "Triton 变化会改变 GDN/QSA kernel 编译结果，需重新测基线"; fi
    if [ "${tf#transformers-}" = "$exp_tf" ]; then ok runtime.transformers "transformers $exp_tf"
    else warn runtime.transformers "transformers=${tf#transformers-}（期望 $exp_tf）" "以 requirements/tested-constraints.txt 为准"; fi

    if [ -n "$vllm" ] && printf '%s' "$vllm" | grep -q 'ple1'; then
        ok runtime.vllm "${vllm#vllm-}（已打补丁的 editable 构建）"
    else
        bad runtime.vllm "vllm=${vllm:-未安装}" "需要 ple1 构建：patches/qwen38-ple-ssd.patch + editable 安装（§4.4/§4.5）"
    fi
    if ls "$VENVSP"/__editable__.vllm-*.pth >/dev/null 2>&1 && [ -f "$SRC/vllm/__init__.py" ]; then
        ok runtime.editable "editable 指向 $SRC（改源码即生效，无需重装）"
    else
        bad runtime.editable "没有 editable 的 vllm 安装" "按 §4.4 重装：uv pip install -e $SRC（VLLM_PRECOMPILED_WHEEL_LOCATION 见 §4.4）"
    fi
}

chk_patches() {
    if [ ! -d "$SRC/.git" ]; then
        bad patch.src "找不到 $SRC/.git" "vLLM 源码树缺失（§4.4）；补丁必须在 checkout 上打"
        return 0
    fi
    local head; head="$(git -C "$SRC" rev-parse HEAD 2>/dev/null)"
    if [ "$head" = "$VLLM_REVISION" ]; then ok patch.rev "src HEAD = ${VLLM_REVISION:0:12}（与 scripts/common.sh 钉住的一致）"
    else warn patch.rev "src HEAD = ${head:0:12}（期望 ${VLLM_REVISION:0:12}）" "≠ 上游锚点时补丁可能打不上；git -C $SRC checkout $VLLM_REVISION"; fi

    local applied=0 p total=0
    for p in "$ROOT/patches/qwen38-ple-ssd.patch" "$ROOT/ops/patches/qsa-alloc-heal.patch"; do
        total=$((total+1))
        if git -C "$SRC" apply --check --reverse "$p" >/dev/null 2>&1; then
            applied=$((applied+1))
            ok "patch.$(basename "$p" .patch)" "已应用（反向校验通过）"
        else
            bad "patch.$(basename "$p" .patch)" "未应用或与源码树不匹配" "deploy.sh install --yes（引擎运行时拒绝改源码树，先 bin/stop.sh）"
        fi
    done
    [ "$applied" = "$total" ] && ok patch.all "$applied/$total 个补丁全部就位（PLE SSD offload + 分配器治愈线程）"
}

chk_plelib() {
    if [ ! -f "$PLE_LIB" ]; then
        bad ple.lib "缺 $PLE_LIB" "编排: scripts/build-ple-io.sh（cc -O3 -shared -fPIC，秒级）；或 deploy.sh install --yes"
        return 0
    fi
    local out
    out="$("$VENVPY" -c "import ctypes;ctypes.CDLL('$PLE_LIB');print('dlopen OK')" 2>&1 | tail -1)"
    if [ "$out" = "dlopen OK" ]; then
        ok ple.lib "$(stat -c '%s' "$PLE_LIB") 字节，dlopen OK（$PLE_LIB）"
    else
        bad ple.lib "dlopen 失败: $out" "重新编译：scripts/build-ple-io.sh（缺 cc 时 apt install build-essential）"
    fi
}

chk_cache() {
    if [ ! -d "$QWEN_TRITON_CACHE_DIR" ]; then
        warn cache.triton "目录不存在（启动时会自动建）" "首次启动要多花几分钟编译 kernel；若已有 Docker 时代缓存可 cp -r ops/legacy-docker/triton_cache/. $QWEN_TRITON_CACHE_DIR/"
    elif [ ! -w "$QWEN_TRITON_CACHE_DIR" ]; then
        bad cache.triton "$QWEN_TRITON_CACHE_DIR 不可写（root 建的？）" "sudo chown -R \$(id -un):\$QWEN_TRITON_CACHE_DIR（run_native.sh start 会直接报错退出）"
    else
        local n; n="$(ls -1 "$QWEN_TRITON_CACHE_DIR" 2>/dev/null | wc -l)"
        ok cache.triton "可写，$n 个已编译 kernel 目录（复用可省一次冷启动编译）"
    fi
}

chk_config() {
    ok config.file "唯一参数来源 $CFG_FILE（PORT=$(printf '%s' "$PORT") MODEL=$(printf '%s' "$QWEN_MODEL_DIR")）"
    local missing=() s
    for s in start stop status logs bench drop_host_cache _common; do
        [ -f "$ROOT/bin/$s.sh" ] || missing+=("bin/$s.sh")
    done
    [ -x "$LAUNCHER" ] || missing+=("vllm-native/bin/run_native.sh(可执行)")
    if [ "${#missing[@]}" -eq 0 ]; then ok config.scripts "bin/ 六个入口 + run_native.sh 均在位"
    else bad config.scripts "缺: ${missing[*]}" "从仓库重新检出（vllm-native/bin/run_native.sh 是最容易被 .gitignore 误伤的那个）"; fi

    # 端口一致性：脚本默认值 vs 正在跑的进程（docs/SCRIPTS.md 的干跑比对法）
    local pid; pid="$(engine_pid)"
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null && [ -r "/proc/$pid/cmdline" ]; then
        local live; live="$(tr '\0' '\n' <"/proc/$pid/cmdline" | awk '/^--port$/{getline; print; exit}')"
        if [ "$live" = "$PORT" ]; then ok config.port "引擎实际 --port=$live 与 config/engine.env 一致"
        else bad config.port "引擎实际 --port=${live:-?} 但配置是 $PORT" "改端口只改 config/engine.env，然后 bin/stop.sh && bin/start.sh"; fi
    fi
}

chk_service() {
    local code; code="$(health_code)"
    if [ "$code" = "200" ]; then
        ok service.health "health=200 pid=$(engine_pid)（$(metric num_requests_running 2>/dev/null | sed 's/^/running=/')）"
    else
        warn service.health "health=${code:-000}（服务未运行）" "拉起：./deploy.sh start（首次加载 143 GB 权重 5~7 分钟）"
    fi
}

run_checks() {
    [ "$JSON" = 0 ] && { hr; printf '  WSL2 部署体检 —— %s\n' "$ROOT"; hr; }
    chk_host; chk_gpu; chk_mem; chk_disk
    chk_model; chk_runtime; chk_patches; chk_plelib; chk_cache; chk_config; chk_service
    if [ "$JSON" = 1 ]; then
        printf '{"root":"%s","port":"%s","model_dir":"%s","pass":%d,"warn":%d,"fail":%d,"checks":[' \
            "$(json_esc "$ROOT")" "$PORT" "$(json_esc "$MODEL_DIR")" "$N_PASS" "$N_WARN" "$N_FAIL"
        local first=1 name st det fix
        while IFS=$'\t' read -r name st det fix; do
            [ "$first" = 0 ] && printf ','
            first=0
            printf '{"name":"%s","status":"%s","detail":"%s","fix":"%s"}' \
                "$(json_esc "$name")" "$st" "$(json_esc "$det")" "$(json_esc "$fix")"
        done <"$RESULT_FILE"
        printf ']}\n'
    else
        hr
        printf '  汇总: %s %d 项   %s %d 项   %s %d 项\n' \
            "$(c_ok PASS)" "$N_PASS" "$(c_warn WARN)" "$N_WARN" "$(c_err FAIL)" "$N_FAIL"
        if [ "$N_FAIL" -gt 0 ]; then
            printf '  %s\n' "$(c_err '有 FAIL ⇒ 按每条的「→ 修复」处理；不知道从哪下手看 docs/DEPLOY-WSL2.md §7')"
        else
            printf '  %s\n' "$(c_ok '没有 FAIL；WARN 多为"服务未运行/宿主内存偏高"这类状态性提示')"
        fi
        hr
    fi
    [ "$N_FAIL" -gt 0 ] && return 1
    return 0
}

# ============================== plan / install =================================
do_plan() {
    cat <<EOF
$(hr)
  部署步骤（本机已部署完成；换机器/重装时按序执行，每步的验证命令都在括号里）
$(hr)
  0. 前置：Windows 驱动 ≥616.92、WSL ≥2.7.11、.wslconfig（memory=48GB / networkingMode=mirrored）
     (wsl --version; nvidia-smi.exe --query-gpu=driver_version --format=csv,noheader)     §2
  1. 取仓库 + 读 config/engine.env（端口与全部参数的唯一来源）
     (./start.sh --params)                                                               §3
  2. 模型权重 142.5 GiB / 13 分片（PLE 表 = model-00001-of-00011，102.4 GB，O_DIRECT）
     (ops/tools/verify_ckpt.py)                                                          §4.3
  3. 运行时：vllm-native/opt/vllm/{.venv,src,optimization} + uv-python/ 基解释器
     (deploy.sh check → runtime.*)                                                       §4.4
  4. 打补丁：patches/qwen38-ple-ssd.patch + ops/patches/qsa-alloc-heal.patch
     (git -C vllm-native/opt/vllm/src apply --check --reverse <patch>)                   §4.5
  5. 编 ple_ssd_io.so：cc -O3 -shared -fPIC（scripts/build-ple-io.sh）
     (deploy.sh check → ple.lib)                                                         §4.6
  6. Triton 缓存可写（root 缓存会让 run_native.sh start 直接失败）                        §4.7
  7. 启动：./start.sh（等 health=200，约 250~320 s；之后自动回收 Windows 页缓存）          §4.8
  8. 验收：./deploy.sh verify（health + 新鲜 2048 预填对比基线）                            §5
$(hr)
EOF
}

do_install() {
    local todo=() act=0
    echo "=== install（$( [ "$YES" = 1 ] && echo '--yes：会真的动手' || echo '默认：只报告，不动手' )）==="

    # ① 目录
    if [ ! -d "$LOG_DIR" ] || [ ! -d "$QWEN_TRITON_CACHE_DIR" ]; then
        todo+=("建目录 $LOG_DIR / $QWEN_TRITON_CACHE_DIR")
        if [ "$YES" = 1 ]; then mkdir -p "$LOG_DIR" "$QWEN_TRITON_CACHE_DIR" && { echo "  ✔ 目录已建"; act=1; }; fi
    fi
    # ② Triton 缓存属主
    if [ -d "$QWEN_TRITON_CACHE_DIR" ] && [ ! -w "$QWEN_TRITON_CACHE_DIR" ]; then
        echo "  ✖ $QWEN_TRITON_CACHE_DIR 不可写（root 属主）⇒ 需要你手动 sudo chown -R $(id -un) （本脚本不自作 sudo）"
    fi
    # ③ 补丁
    local p
    for p in "$ROOT/patches/qwen38-ple-ssd.patch" "$ROOT/ops/patches/qsa-alloc-heal.patch"; do
        if git -C "$SRC" apply --check --reverse "$p" >/dev/null 2>&1; then continue; fi
        todo+=("应用补丁 $(basename "$p")")
        if [ "$YES" = 1 ]; then
            if [ "$(health_code)" = "200" ]; then
                echo "  ✖ 拒绝在引擎运行时改源码树（铁律：不要动正在跑的服务）⇒ 先 bin/stop.sh（需用户同意）"
                continue
            fi
            if git -C "$SRC" apply --check "$p" >/dev/null 2>&1; then
                git -C "$SRC" apply "$p" && { echo "  ✔ 已应用 $(basename "$p")"; act=1; }
            else
                echo "  ✖ $(basename "$p") 打不上（源码树版本不对？期望 HEAD ${VLLM_REVISION:0:12}）"
            fi
        fi
    done
    # ④ ple_ssd_io.so
    if [ ! -f "$PLE_LIB" ]; then
        todo+=("编译 $PLE_LIB")
        if [ "$YES" = 1 ]; then
            if have cc || have gcc; then
                "$ROOT/scripts/build-ple-io.sh" && { echo "  ✔ ple_ssd_io.so 已编译"; act=1; }
            else
                echo "  ✖ 缺 cc/gcc ⇒ sudo apt install -y build-essential"
            fi
        fi
    fi
    # ⑤ 模型
    if [ ! -r "$MODEL_DIR/model.safetensors.index.json" ]; then
        todo+=("下载权重到 $MODEL_DIR（142.5 GiB）")
        if [ "$YES" = 1 ] && [ "$ALLOW_DL" = 1 ]; then
            local py="$HOME/dlvenv/bin/python"; have "$py" || py=python3
            echo "  开始下载（可续传，小时级；中断后重跑即可）：$py $ROOT/ops/tools/fdl.py"
            "$py" "$ROOT/ops/tools/fdl.py"
        elif [ "$ALLOW_DL" = 0 ]; then
            echo "  ℹ 下载需显式加 --download-model（142.5 GiB，先确认磁盘与网络）"
        fi
    fi

    if [ "${#todo[@]}" -eq 0 ]; then
        echo "  ✔ 无需补齐（目录/补丁/PLE 库/模型都在位）"
    else
        echo "  待办："
        printf '    - %s\n' "${todo[@]}"
        [ "$YES" = 0 ] && echo "  加 --yes 执行上面这些动作（幂等，可反复跑）"
    fi
    if [ "$YES" = 1 ] && [ "$act" = 1 ]; then
        echo; echo "  改动后请重跑体检： ./deploy.sh check"
    fi
}

# ============================== start / verify =================================
do_start() {
    local args=("$@")
    echo "→ 委托 bin/start.sh（唯一启动实现）${args[*]:+ 参数: ${args[*]}}"
    "$ROOT/bin/start.sh" ${args[@]+"${args[@]}"}
}

# 只取 $CUDA_VISIBLE_DEVICES 指定那张卡的显存行（本机有两张 170HX，混排会看不懂）
gpu_mem_one() {
    local idx="${CUDA_VISIBLE_DEVICES%%,*}"
    /mnt/c/Windows/System32/nvidia-smi.exe --query-gpu=index,memory.used,memory.total,utilization.gpu \
        --format=csv,noheader 2>/dev/null | tr -d '\r' | awk -v i="$idx" -F, '$1+0==i+0{printf "GPU%s: used%s / total%s util%s", $1, $2, $3, $4}'
}

do_verify() {
    local code; code="$(health_code)"
    if [ "$code" != "200" ]; then
        echo "$(c_err "服务未就绪（health=${code:-000}）")" >&2
        echo "先跑：./deploy.sh start   （日志：bin/logs.sh -e）" >&2
        return 1
    fi
    hr
    printf '  验收 · 服务 %s (health=200, pid %s)\n' "$BASE_URL" "$(engine_pid)"
    hr

    # 新鲜 2048 预填 ×3（warmup.py 用全新 token id，永不吃前缀缓存 —— 铁律 4）。
    # 取 best-of-3 作为稳态值，与 docs/RESULTS-WSL2.md 的"3 次取中位数 0.693 s"口径一致。
    local py="$VENVPY"; [ -x "$py" ] || py=/usr/bin/python3
    local tmp; tmp="$(mktemp)"
    echo "  1) 新鲜 2048 预填 ×3（基线 0.64~0.73 s，docs/RESULTS-WSL2.md）"
    if ! BASE_URL="$BASE_URL" QWEN_PORT="$PORT" QWEN_SERVED_NAME="$QWEN_SERVED_NAME" \
         "$py" "$ROOT/ops/bench/warmup.py" 2048 --reps 3 >"$tmp" 2>&1; then
        sed 's/^/     /' "$tmp"; rm -f "$tmp"
        bad verify.prefill "预填请求失败" "看 bin/logs.sh -e；接口 $BASE_URL"
        hr; printf '  汇总: %s %d 项   %s %d 项   %s %d 项\n' "$(c_ok PASS)" "$N_PASS" "$(c_warn WARN)" "$N_WARN" "$(c_err FAIL)" "$N_FAIL"; hr
        return 1
    fi
    grep -E '^fresh prefill|best of' "$tmp" | sed 's/^/     /'
    local secs tps
    read -r secs tps < <(awk '/best of/{s=$6; t=$8} /^fresh prefill/{ls=$6; lt=$8} END{if(s==""){s=ls;t=lt} if(s!="")printf "%s %s", s, t}' "$tmp")
    rm -f "$tmp"
    if [ -z "${secs:-}" ]; then
        bad verify.prefill "没能解析 warmup.py 输出" "手动跑：$py ops/bench/warmup.py 2048 --reps 3"
    elif awk "BEGIN{exit !($secs <= 0.80)}"; then
        ok verify.prefill "2048 稳态 ${secs} s（${tps} tok/s）— 命中基线 0.64~0.73 s"
    elif awk "BEGIN{exit !($secs <= 1.15)}"; then
        warn verify.prefill "2048 稳态 ${secs} s（${tps} tok/s）— 比基线慢约 $(awk -v s="$secs" 'BEGIN{printf "%.2f", s/0.693}')×，不算故障但已偏离" \
            "① bin/drop_host_cache.sh（宿主页缓存）② 确认 QSA_ALLOC_HEAL=1 ③ 长上下文后跑 bin/bench.sh --full 的中毒检查（铁律 5）"
    else
        bad verify.prefill "2048 稳态 ${secs} s（${tps} tok/s）— 约为基线的 $(awk -v s="$secs" 'BEGIN{printf "%.2f", s/0.693}')×" \
            "① bin/drop_host_cache.sh ② QSA_ALLOC_HEAL=1 ③ 长上下文后中毒需重启（铁律 5，重启前先征得同意）"
    fi

    echo
    echo "  2) 进程与显存"
    local acc accpct draft
    acc="$(metric spec_decode_num_accepted_tokens_total)"; draft="$(metric spec_decode_num_draft_tokens_total)"
    accpct="-"
    if [ -n "$draft" ] && [ -n "$acc" ] && awk "BEGIN{exit !($draft>0)}"; then
        accpct="$(awk -v a="$acc" -v d="$draft" 'BEGIN{printf "%.1f%%", 100*a/d}')"
    fi
    ok verify.engine "pid=$(engine_pid)（$(engine_alive && printf '进程存活' || printf '进程已退出')）$(gpu_mem_one)；KV usage $(metric kv_cache_usage_perc)%；MTP 接受率 $accpct"

    if [ "$FULL" = 1 ]; then
        echo; echo "  3) bin/bench.sh --full（131072 长上下文 + 事后中毒检查，约 3 分钟）"
        if "$ROOT/bin/bench.sh" --full; then ok verify.bench "bin/bench.sh --full 通过（已存 ops/measurements/perf-history.csv）"
        else bad verify.bench "bin/bench.sh --full 失败" "看 ops/measurements/ 下最新 bench-*.log"; fi
    else
        echo "  （加 --full 会追加 131072 长上下文 + 中毒检查）"
    fi

    hr
    printf '  汇总: %s %d 项   %s %d 项   %s %d 项\n' "$(c_ok PASS)" "$N_PASS" "$(c_warn WARN)" "$N_WARN" "$(c_err FAIL)" "$N_FAIL"
    hr
    [ "$N_FAIL" -gt 0 ] && return 1
    return 0
}

# ============================== 入口 ==========================================
case "${1:-}" in
    -h|--help) help_exit "$0" ;;
esac

SUB="${1:-check}"; [ $# -gt 0 ] && shift
ARGS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --json) JSON=1 ;;
        --full) FULL=1 ;;
        --yes|-y) YES=1 ;;
        --download-model) ALLOW_DL=1 ;;
        -h|--help) help_exit "$0" ;;
        *) ARGS+=("$1") ;;
    esac
    shift
done

case "$SUB" in
    check)   run_checks ;;
    plan)    do_plan ;;
    install) do_install ;;
    start)   do_start ${ARGS[@]+"${ARGS[@]}"} ;;
    verify)  do_verify ;;
    all)     run_checks && do_start && do_verify ;;
    help|-h|--help) help_exit "$0" ;;
    *) echo "未知子命令: $SUB（check|plan|install|start|verify|all；--help 看用法）" >&2; exit 2 ;;
esac
