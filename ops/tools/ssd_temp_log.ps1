# Log NVMe SSD temperature / reliability counters (ASCII only, run in an ADMIN PowerShell).
# Usage: powershell -ExecutionPolicy Bypass -File ssd_temp_log.ps1 [-DeviceId 1] [-Minutes 30] [-IntervalSec 10] [-Out ssd-temp.csv]
param(
  [int]$DeviceId = 1,
  [int]$Minutes = 30,
  [int]$IntervalSec = 10,
  [string]$Out = "ssd-temp.csv"
)
"[time] device=$DeviceId every ${IntervalSec}s for ${Minutes}min -> $Out"
"time,temp_c,temp_max_c,wear,read_err_corrected,flush_latency_max_ms,read_latency_max_ms" | Set-Content -Encoding ascii $Out
$end = (Get-Date).AddMinutes($Minutes)
while ((Get-Date) -lt $end) {
  try {
    $c = Get-PhysicalDisk | Where-Object DeviceId -eq $DeviceId | Get-StorageReliabilityCounter -ErrorAction Stop
    $line = "{0:HH:mm:ss},{1},{2},{3},{4},{5},{6}" -f (Get-Date), $c.Temperature, $c.TemperatureMax, $c.Wear, $c.ReadErrorsCorrected, $c.FlushLatencyMax, $c.ReadLatencyMax
    $line | Add-Content -Encoding ascii $Out
    Write-Host ("{0:HH:mm:ss}  temp={1}C  max={2}C  wear={3}" -f (Get-Date), $c.Temperature, $c.TemperatureMax, $c.Wear)
  } catch { Write-Host ("{0:HH:mm:ss}  read failed: {1}" -f (Get-Date), $_.Exception.Message) }
  Start-Sleep -Seconds $IntervalSec
}
"done -> $Out"
