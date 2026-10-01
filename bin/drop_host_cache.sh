#!/usr/bin/env bash
# 把 WSL 里积累的干净页缓存丢掉，内存立刻还给 Windows。
# 背景：每次引擎启动都要读 143 GB 权重，WSL 会攒下约 50 GB 干净页缓存，
#       于是 vmmemWSL 工作集涨到 60+ GB，Windows 只剩个位数 GB 可用 + 页面文件狂换，
#       整机（含客户端流式渲染）就发卡。丢缓存不重启、不碰引擎。
# 为什么能提权：wsl.exe -u root 由 WSL 启动器直接指定用户，不需要 sudo 密码。
# 为什么不影响引擎：权重已在显存里；PLE 走 O_DIRECT + 自己的 512 MiB 行缓存，不走页缓存。
set -u
WSLD="/mnt/c/Windows/System32/wsl.exe"
PS="/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"
DISTRO="${WSL_DISTRO_NAME:-Ubuntu}"

"$WSLD" -d "$DISTRO" -u root -- sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches' || { echo "丢弃缓存失败" >&2; exit 1; }
echo "guest: $(free -m | sed -n 2p | awk '{print "used="$3"MB free="$4"MB cached="$6"MB"}')"
"$PS" -NoProfile -Command "\$os=Get-CimInstance Win32_OperatingSystem; 'Windows 可用内存 = {0:N1} GB' -f (\$os.FreePhysicalMemory/1MB); \$p=Get-Process vmmemWSL -ErrorAction SilentlyContinue; if(\$p){'vmmemWSL 工作集 = {0:N1} GB' -f (\$p.WorkingSet64/1GB)}" 2>/dev/null | tr -d '\000'
