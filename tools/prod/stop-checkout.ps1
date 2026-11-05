param(
    [Parameter(Mandatory)][string]$Root,
    [Parameter(Mandatory)][ValidateRange(1,65535)][int]$Port,
    [string]$Token,
    [switch]$DryRun
)
$ErrorActionPreference = 'Stop'
$prefix = [IO.Path]::GetFullPath($Root).TrimEnd('\') + '\'
function OwnEngine {
    Get-CimInstance Win32_Process -Filter "Name = 'ace-server.exe'" | Where-Object {
        $_.ExecutablePath -and $_.ExecutablePath.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)
    }
}
$listeners = @(Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue |
    Select-Object -ExpandProperty OwningProcess -Unique)
foreach ($ownerId in $listeners) {
    $process = Get-CimInstance Win32_Process -Filter "ProcessId=$ownerId"
    if ($process.Name -ne 'node.exe' -or -not $process.CommandLine -or
        $process.CommandLine.IndexOf($prefix, [StringComparison]::OrdinalIgnoreCase) -lt 0) {
        throw "Port $Port belongs to PID $ownerId outside this checkout. Refusing shutdown."
    }
}
if ($DryRun) {
    OwnEngine | ForEach-Object { "Would stop engine PID $($_.ProcessId): $($_.ExecutablePath)" }
    "Shutdown scope verified for $prefix on port $Port. Dry run; no processes stopped."
    exit 0
}
if ($listeners.Count) {
    $headers = @{}
    if ($Token) { $headers['x-hotstep-prod-token'] = $Token }
    Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:$Port/api/shutdown" -Headers $headers -TimeoutSec 10 | Out-Null
}
for ($n = 0; $n -lt 20; $n++) {
    $engines = @(OwnEngine)
    $servers = @($listeners | Where-Object { Get-Process -Id $_ -ErrorAction SilentlyContinue })
    if (-not $engines.Count -and -not $servers.Count) { exit 0 }
    if ($n -eq 15 -and -not $servers.Count) {
        $engines | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction Stop }
    }
    Start-Sleep -Seconds 1
}
throw 'Checkout server or engine did not stop. No build or file replacement is safe.'
