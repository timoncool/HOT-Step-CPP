$ErrorActionPreference = 'Stop'
$root = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$dummyDir = $root + '-isolation-' + [Guid]::NewGuid()
New-Item -ItemType Directory -Force $dummyDir | Out-Null
& git init --quiet $dummyDir
if ($LASTEXITCODE) { throw 'Could not initialize the disposable fixture checkout.' }
# ping is an inert stand-in. It uses no model or GPU and is never a server.
$dummyExe = Join-Path $dummyDir 'ace-server.exe'
Copy-Item "$env:SystemRoot/System32/ping.exe" $dummyExe
$dummy = Start-Process $dummyExe -ArgumentList @('-t','127.0.0.1') -PassThru -WindowStyle Hidden -RedirectStandardOutput (Join-Path $dummyDir 'dummy.log')
try {
    Start-Sleep -Milliseconds 300
    if ($dummy.HasExited) { throw 'Inert engine fixture failed to start.' }
    # Use a different checkout root with a sibling-prefix engine path.
    & pwsh -NoProfile -File (Join-Path $PSScriptRoot 'stop-checkout.ps1') -Root ($dummyDir + '-other') -Port 62199 -DryRun
    if ($LASTEXITCODE) { throw 'Scope dry run failed.' }
    $scope = & (Join-Path $root 'dev-rebuild.bat') --dry-run
    if ($LASTEXITCODE) { throw 'Rebuild dry run failed.' }
    if (($scope | Out-String) -match "engine PID $($dummy.Id):") { throw 'Sibling engine entered the rebuild scope.' }
    if (-not (Get-Process -Id $dummy.Id -ErrorAction SilentlyContinue)) { throw 'Second engine was killed.' }
    $lockFile = Join-Path $root '.tmp/engine-artifact.lock'
    $handle = [IO.File]::Open($lockFile, 'OpenOrCreate', 'ReadWrite', 'None')
    try {
        & (Join-Path $root 'dev-rebuild.bat') --dry-run 2>&1 | Out-String | Write-Host
        if ($LASTEXITCODE -eq 0) { throw 'Concurrent rebuild was allowed.' }
    } finally { $handle.Dispose() }
    'ISOLATION PASS: second ace-server survived rebuild dry run; concurrent rebuild refused.'
} finally {
    # Only reap the fixture PID that this test started.
    if (Get-Process -Id $dummy.Id -ErrorAction SilentlyContinue) { Stop-Process -Id $dummy.Id -Force }
}
