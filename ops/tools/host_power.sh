#!/usr/bin/env bash
# host_power.sh — 宿主（Windows）电源管理：诊断 / 修复 / 回滚
#
# 为什么需要它：0x9F(DRIVER_POWER_STATE_FAILURE) 的直接前因是 AC 空闲 60 分钟
# 自动进入 S3（04:29 开机 → 05:29 睡眠 → 07:59 固件 S3 恢复超时 → 08:01 崩溃）；
# 0x7A(KERNEL_DATA_INPAGE_ERROR, STATUS_NO_SUCH_DEVICE) 前有 stornvme 129 控制器
# 重置密集出现。两者都与宿主的电源管理设置直接相关。
#
# 用法（本脚本不碰引擎、不重启、不动 .wslconfig）：
#   ops/tools/host_power.sh            # 只读诊断（等同 --diag）
#   ops/tools/host_power.sh --diag     # 只读：设置 / 磁盘 / 唤醒源 / bugcheck / 事件时间线
#   ops/tools/host_power.sh --apply    # 空跑：打印将要执行的命令、需要管理员
#   ops/tools/host_power.sh --apply --yes
#                                      # 真改：禁自动睡眠/休眠、关快速启动、ASPM=Off、
#                                      # 硬盘空闲=Never、USB 选择性挂起=Off（可 --rollback 回退）
#   ops/tools/host_power.sh --apply --yes --elevate
#                                      # 同上，但通过 UAC 弹窗提权（会弹 Windows 确认框）
#   ops/tools/host_power.sh --apply --yes --disable-wake
#                                      # 额外禁用两块网卡的唤醒权限（避免“睡 3 秒就醒”抖动）
#   ops/tools/host_power.sh --rollback --yes
#                                      # 恢复 Windows 默认省电行为（并恢复休眠）
#
# 产物：/tmp/host-power-<时间戳>.{before,after}.txt（powercfg 全量转储，作为证据留存）
#
# 注意：本脚本**只做不需要重启的事**。BIOS 里的 PCIe ASPM / L1 substates /
# Power Supply Idle Control / M.2 插槽选择，以及设备管理器里每个设备的
# “允许计算机关闭此设备以节约电源”，都不是本脚本能改的（需要重启，见 OPS.md §9.34）。
# shellcheck shell=bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck source=/dev/null
. "$ROOT/bin/_common.sh"

MODE="diag"; YES=0; ELEVATE=0; DISABLE_WAKE=0
while [ $# -gt 0 ]; do
    case "$1" in
        --diag)         MODE="diag" ;;
        --apply)        MODE="apply" ;;
        --rollback)     MODE="rollback" ;;
        --yes|-y)       YES=1 ;;
        --elevate)      ELEVATE=1 ;;
        --disable-wake) DISABLE_WAKE=1 ;;
        -h|--help)      help_exit "$0" ;;
        *) echo "未知参数：$1（-h 看用法）" >&2; exit 2 ;;
    esac
    shift
done

PS1="$ROOT/ops/tools/host_power.ps1"
[ -r "$PS1" ] || { echo "错误：找不到 $PS1" >&2; exit 1; }
[ -x "$PS_EXE" ] || { echo "错误：找不到 Windows PowerShell：$PS_EXE" >&2; exit 1; }

# 需要管理员的模式：先确认，再决定怎么提权。
if [ "$MODE" != "diag" ]; then
    if [ "$MODE" = "apply" ] && [ "$YES" = "0" ]; then
        hr; printf '  空跑（什么都不会改）。要真的执行，加 %s\n' "$(c_warn --yes)"
        printf '  将要执行：禁自动睡眠/休眠、%s、ASPM=Off、硬盘空闲=Never、USB 选择性挂起=Off\n' "$(c_warn 'powercfg /h off')"
        hr
        MODE="dryrun"
    fi
fi

# 把模式注入脚本开头，再整体 UTF-16LE+base64 → -EncodedCommand。
# 这样绕开两件事：① stdin 逐行解析对多行块不友好；② 中文注释被代码页搞坏导致 ParserError。
build_cmd() {
    local body="$1" prefix="$2" enc
    enc="$( { printf '%s\n' "$prefix"; cat "$body"; } | iconv -f UTF-8 -t UTF-16LE | base64 -w0 )"
    printf '%s' "$enc"
}

if [ "$MODE" = "dryrun" ]; then
    # 空跑：不调用 powershell，直接把将执行的命令列出来（apply 分支里的原文）。
    sed -n '/^function Apply-All/,/^    if (\$env:HOST_POWER_DISABLE_WAKE/p' "$PS1" \
        | grep -E '^\s+(powercfg|Write-Output)' | sed 's/^[[:space:]]*/  /'
    exit 0
fi

ENC="$(build_cmd "$PS1" "\$Mode = '$MODE'")"

if [ "$ELEVATE" = "1" ] && [ "$MODE" != "diag" ]; then
    # 通过 UAC 提权：把 EncodedCommand 交给一个 elevated 的 powershell 子进程。
    "$PS_EXE" -NoProfile -Command "Start-Process -FilePath 'powershell.exe' -Verb RunAs -ArgumentList '-NoProfile','-EncodedCommand','$ENC' -Wait"
    echo "（已通过 UAC 提权执行；输出在弹窗里，本终端看不到。诊断请看 host_power.sh --diag）"
    exit 0
fi

export HOST_POWER_DISABLE_WAKE="$DISABLE_WAKE"
export HOST_POWER_DUMP_DIR="${TMPDIR:-/tmp}"

rc=0
"$PS_EXE" -NoProfile -EncodedCommand "$ENC" 2>&1 | tr -d '\r' || rc=$?
exit "$rc"
