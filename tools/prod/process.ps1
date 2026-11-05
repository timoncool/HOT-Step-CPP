param([int]$ProcessId, [switch]$CheckPorts, [switch]$CheckServerPort)
$ErrorActionPreference = 'Stop'
if ($CheckPorts -or $CheckServerPort) {
    $ports = if ($CheckServerPort) { @(3101) } else { @(3101,8186) }
    if (Get-NetTCPConnection -LocalPort $ports -State Listen -ErrorAction SilentlyContinue) {
        throw 'Production ports 3101/8186 are occupied. Refusing to start or replace an unowned listener.'
    }
    exit 0
}
$process = Get-CimInstance Win32_Process -Filter "ProcessId=$ProcessId"
if (-not $process) { 'null'; exit 0 }
@{ command = $process.CommandLine; started = $process.CreationDate.ToUniversalTime().ToString('o') } | ConvertTo-Json -Compress
