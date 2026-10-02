# ============================================================================
# host_power.ps1 — Windows 宿主「电源管理」诊断 / 修复（由 ops/tools/host_power.sh 调用）
#
# 模式（由 wrapper 在开头注入 $Mode 变量）：
#   diag      只读：电源计划、ASPM/空闲/睡眠设置、磁盘、唤醒源、bugcheck 与
#             stornvme/disk/Kernel-Power 事件时间线。不需要管理员。
#   apply     修复第一层（全部可逆、无需重启）：禁自动睡眠/休眠、关快速启动、
#             ASPM=Off、硬盘空闲=Never、USB 选择性挂起=Off。需要管理员。
#   rollback  恢复 Windows 出厂式的省电默认（并提示恢复休眠）。需要管理员。
#
# 输出刻意全部使用英文标签：避免 WSL ↔ Windows PowerShell 之间的代码页乱码
# （事件消息体本身是中文，保持原样）。
#
# 背景（2026-10-02 实测，见 ops/OPS.md §9.34）：
#   0x9F DRIVER_POWER_STATE_FAILURE(Arg1=3) 的直接前因是 AC 空闲 60 分钟自动进入
#   S3 —— 04:29 开机 → 05:29:17 Kernel-Power 42（进入睡眠）→ 07:59:23
#   Kernel-Power 131（固件 S3 恢复超时）→ 08:01:45 崩溃。
#   0x7A KERNEL_DATA_INPAGE_ERROR(Arg2=0xc000000e STATUS_NO_SUCH_DEVICE) ×3 的直接
#   前因是 KIOXIA EXCERIA G2（D:，DRAM-less）控制器重置：stornvme 129 ×2~3 分钟内
#   密集出现 → stornvme 11 → 18×disk 51（分页写入失败）→ 崩溃。
# ============================================================================

$ErrorActionPreference = 'SilentlyContinue'
$ProgressPreference = 'SilentlyContinue'   # 否则管道里会混入 #< CLIXML 的进度对象
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8

if (-not (Get-Variable Mode -Scope Script -ErrorAction SilentlyContinue)) { $Mode = 'diag' }
$RepoHint = if ($env:HOST_POWER_DUMP_DIR) { $env:HOST_POWER_DUMP_DIR } else { $env:TEMP }

# ---- 常量：powercfg 的 GUID 与别名 ----------------------------------------
# 组：SUB_PCIEXPRESS(ASPM) / SUB_DISK(DISKIDLE) / SUB_SLEEP / SUB_USB
$ASPM_GUID     = 'ee12f906-d277-404b-b6da-e5fa1a576df5'   # 链接状态电源管理
$DISKIDLE_GUID = '6738e2c4-e8a5-4a42-b16a-e040e769756e'   # 在此时间后关闭硬盘
$STANDBY_GUID  = '29f6c1db-86da-48c5-9fdb-f2b67b1f44da'   # 在此时间后睡眠
$HIBER_GUID    = '9d7815a6-7ee4-497e-8888-515a05f02364'   # 在此时间后休眠
$USB_GUID      = '48e6b7a6-50f5-4782-a5d4-53bb8f07e226'   # USB 选择性挂起
$SUB_PCIEXPRESS = '501a4d13-42af-4429-9fd1-a8218c268e20'
$SUB_DISK       = '0012ee47-9041-4b5d-9b77-535fba8b1442'
$SUB_SLEEP      = '238c9fa8-0aad-41ed-83f4-97be242c8f20'
$SUB_USB        = '2a737441-1930-4402-8d77-b2bebba308a3'

function Sec($t) { Write-Output ""; Write-Output "===== $t =====" }
function Is-Admin {
    $id = [Security.Principal.WindowsIdentity]::GetCurrent()
    (New-Object Security.Principal.WindowsPrincipal $id).IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}
function Show-Val($g, $s, $name) {
    $o = (powercfg /q SCHEME_CURRENT $g $s | Out-String)
    $ac = [regex]::Match($o, '(?m)^\s*(?:Current AC Power Setting Index|当前交流电源设置索引)\s*:\s*(0x[0-9a-f]+)')
    $dc = [regex]::Match($o, '(?m)^\s*(?:Current DC Power Setting Index|当前直流电源设置索引)\s*:\s*(0x[0-9a-f]+)')
    $acv = if ($ac.Success) { [Convert]::ToInt64($ac.Groups[1].Value,16) } else { -1 }
    $dcv = if ($dc.Success) { [Convert]::ToInt64($dc.Groups[1].Value,16) } else { -1 }
    $acl = if ($acv -eq 0) { 'Off/Never' } elseif ($acv -eq 0xffffffff) { 'n/a' } else { "$acv s / level $acv" }
    $dcl = if ($dcv -eq 0) { 'Off/Never' } elseif ($dcv -eq 0xffffffff) { 'n/a' } else { "$dcv s / level $dcv" }
    Write-Output ("{0,-26} AC=0x{1:x8} ({2,-14})  DC=0x{3:x8} ({4})" -f $name, $acv, $acl, $dcv, $dcl)
}

function Diag-All {
    Sec 'Machine / OS'
    Get-CimInstance Win32_BaseBoard | Select-Object Manufacturer,Product | Format-List
    Get-CimInstance Win32_ComputerSystem | Select-Object Manufacturer,Model,SystemFamily | Format-List
    $os = Get-CimInstance Win32_OperatingSystem
    Write-Output ("OS: {0}  build {1}.{2}" -f $os.Caption, $os.Version.Split('.')[2], (Get-ItemProperty 'HKLM:\SOFTWARE\Microsoft\Windows NT\CurrentVersion').UBR)

    Sec 'Active power scheme'
    powercfg /getactivescheme

    Sec 'Power settings (AC / DC)'
    Show-Val $SUB_PCIEXPRESS $ASPM_GUID     'ASPM (link state)'
    Show-Val $SUB_DISK       $DISKIDLE_GUID 'disk idle timeout'
    Show-Val $SUB_SLEEP      $STANDBY_GUID  'standby (sleep) timeout'
    Show-Val $SUB_SLEEP      $HIBER_GUID    'hibernate timeout'
    Show-Val $SUB_USB        $USB_GUID      'USB selective suspend'
    $hb = (Get-ItemProperty 'HKLM:\SYSTEM\CurrentControlSet\Control\Session Manager\Power').HiberbootEnabled
    Write-Output ("Fast Startup (HiberbootEnabled) : {0}   [1=on -> disable]" -f $hb)

    Sec 'Available sleep states'
    powercfg /a

    Sec 'Disks + which one holds D: (WSL vhdx lives there)'
    Get-Disk | Select-Object Number,FriendlyName,FirmwareVersion,BusType,@{n='GiB';e={[int]($_.Size/1GB)}} | Format-Table -AutoSize
    Get-Partition | Where-Object DriveLetter | Select-Object DriveLetter,DiskNumber,@{n='GiB';e={[int]($_.Size/1GB)}} | Sort-Object DriveLetter | Format-Table -AutoSize

    Sec 'Devices whose Device Manager power tab matters'
    Get-PnpDevice -Class DiskDrive,SCSIAdapter,Display -Status OK | Select-Object Class,FriendlyName,InstanceId | Sort-Object Class | Format-Table -AutoSize -Wrap

    Sec 'Devices allowed to wake the machine (wake_armed)'
    powercfg /devicequery wake_armed
    Sec 'Last wake source'
    powercfg /lastwake

    Sec 'BugCheck history (last 14 days)'
    Get-WinEvent -FilterHashtable @{LogName='System'; ProviderName='Microsoft-Windows-WER-SystemErrorReporting'; StartTime=(Get-Date).AddDays(-14)} |
      Sort-Object TimeCreated |
      ForEach-Object { Write-Output ($_.TimeCreated.ToString('MM-dd HH:mm:ss') + "  " + (($_.Message -replace '\s+',' ').Substring(0,[Math]::Min(160,$_.Message.Length)))) }

    Sec 'stornvme controller resets (129/11) and disk page-write errors (51), last 14 days'
    Get-WinEvent -FilterHashtable @{LogName='System'; StartTime=(Get-Date).AddDays(-14)} |
      Where-Object { $_.ProviderName -in 'stornvme','disk' } |
      Sort-Object TimeCreated |
      ForEach-Object { Write-Output ($_.TimeCreated.ToString('MM-dd HH:mm:ss') + "  [" + $_.ProviderName + "] id=" + $_.Id) }
    Write-Output '(counts by provider/id)'
    Get-WinEvent -FilterHashtable @{LogName='System'; StartTime=(Get-Date).AddDays(-14)} |
      Where-Object { $_.ProviderName -in 'stornvme','disk' } |
      Group-Object ProviderName,Id | ForEach-Object { Write-Output ("  " + $_.Name + " x " + $_.Count) }

    Sec 'Sleep / wake / firmware-S3-timeout events, last 14 days'
    Get-WinEvent -FilterHashtable @{LogName='System'; ProviderName='Microsoft-Windows-Kernel-Power'; StartTime=(Get-Date).AddDays(-14)} |
      Where-Object { $_.Id -in 40,41,42,107,130,131 } |
      Sort-Object TimeCreated |
      ForEach-Object { Write-Output ($_.TimeCreated.ToString('MM-dd HH:mm:ss') + "  id=" + $_.Id) }
    Get-WinEvent -FilterHashtable @{LogName='System'; ProviderName='Microsoft-Windows-Kernel-Power'; StartTime=(Get-Date).AddDays(-14)} |
      Where-Object { $_.Id -in 40,41,42,107,130,131 } |
      Group-Object Id | Sort-Object {[int]$_.Name} | ForEach-Object { Write-Output ("  id " + $_.Name + " x " + $_.Count) }
}

function Apply-All {
    if (-not (Is-Admin)) { Write-Output 'ERROR: needs Administrator. Run from an elevated PowerShell (see host_power.sh --apply --elevate).'; return 2 }
    $stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
    $before = Join-Path $RepoHint "host-power-$stamp.before.txt"
    (powercfg /q SCHEME_CURRENT) | Out-File -Encoding utf8 $before
    Write-Output "before dump -> $before"

    Write-Output '--- 1) never sleep / never hibernate on AC and DC'
    powercfg /change standby-timeout-ac 0
    powercfg /change standby-timeout-dc 0
    powercfg /change hibernate-timeout-ac 0
    powercfg /change hibernate-timeout-dc 0

    Write-Output '--- 2) disable hibernation (also disables Fast Startup, removes hiberfil.sys)'
    powercfg /h off

    Write-Output '--- 3) ASPM off, disk idle never, USB selective suspend off (AC + DC)'
    powercfg /setacvalueindex SCHEME_CURRENT SUB_PCIEXPRESS $ASPM_GUID 0
    powercfg /setdcvalueindex SCHEME_CURRENT SUB_PCIEXPRESS $ASPM_GUID 0
    powercfg /setacvalueindex SCHEME_CURRENT SUB_DISK $DISKIDLE_GUID 0
    powercfg /setdcvalueindex SCHEME_CURRENT SUB_DISK $DISKIDLE_GUID 0
    powercfg /setacvalueindex SCHEME_CURRENT SUB_SLEEP $STANDBY_GUID 0
    powercfg /setdcvalueindex SCHEME_CURRENT SUB_SLEEP $STANDBY_GUID 0
    powercfg /setacvalueindex SCHEME_CURRENT SUB_USB $USB_GUID 0
    powercfg /setdcvalueindex SCHEME_CURRENT SUB_USB $USB_GUID 0
    powercfg /setactive SCHEME_CURRENT

    if ($env:HOST_POWER_DISABLE_WAKE -eq '1') {
        Write-Output '--- 4) stop NICs from waking the machine (keeps keyboard/mouse wake)'
        foreach ($d in @('Realtek Gaming 2.5GbE Family Controller','Intel(R) I211 Gigabit Network Connection')) {
            powercfg /devicedisablewake $d | Out-Null
            Write-Output ("    disabled wake: " + $d)
        }
    }

    $after = Join-Path $RepoHint "host-power-$stamp.after.txt"
    (powercfg /q SCHEME_CURRENT) | Out-File -Encoding utf8 $after
    Write-Output "after dump -> $after"
    Write-Output ''
    Write-Output '--- verify'
    Diag-All
    Write-Output ''
    Write-Output 'NOTE: BIOS-side items (PCIe ASPM / L1 substates / Power Supply Idle Control / M.2 slot)'
    Write-Output '      still require a reboot + BIOS change; device-level "allow the computer to turn off'
    Write-Output '      this device to save power" checkboxes are per-device in Device Manager.'
    return 0
}

function Rollback-All {
    if (-not (Is-Admin)) { Write-Output 'ERROR: needs Administrator.'; return 2 }
    Write-Output '--- restore Windows default power-saving behaviour'
    powercfg /h on
    powercfg /change standby-timeout-ac 60
    powercfg /change standby-timeout-dc 10
    powercfg /change hibernate-timeout-ac 0
    powercfg /setacvalueindex SCHEME_CURRENT SUB_PCIEXPRESS $ASPM_GUID 0
    powercfg /setdcvalueindex SCHEME_CURRENT SUB_PCIEXPRESS $ASPM_GUID 2
    powercfg /setacvalueindex SCHEME_CURRENT SUB_DISK $DISKIDLE_GUID 0
    powercfg /setdcvalueindex SCHEME_CURRENT SUB_DISK $DISKIDLE_GUID 600
    powercfg /setacvalueindex SCHEME_CURRENT SUB_USB $USB_GUID 1
    powercfg /setdcvalueindex SCHEME_CURRENT SUB_USB $USB_GUID 1
    powercfg /setactive SCHEME_CURRENT
    Diag-All
    return 0
}

switch ($Mode) {
    'diag'     { Diag-All }
    'apply'    { Apply-All }
    'rollback' { Rollback-All }
    default    { Write-Output "unknown mode: $Mode" }
}
