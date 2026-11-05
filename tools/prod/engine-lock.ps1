param(
    [string]$Script,
    [switch]$Promote,
    [Parameter(ValueFromRemainingArguments)][string[]]$ScriptArgs
)
$ErrorActionPreference = 'Stop'
$root = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$lockDir = Join-Path $root '.tmp'
New-Item -ItemType Directory -Force $lockDir | Out-Null
try {
    $handle = [IO.File]::Open((Join-Path $lockDir 'engine-artifact.lock'), 'OpenOrCreate', 'ReadWrite', 'None')
} catch {
    throw 'Engine rebuild/update or production promotion already holds the engine artifact lock. Retry after it finishes.'
}
try {
    $env:HOTSTEP_ENGINE_LOCK_ROOT = $root
    if ($Promote) {
        & node (Join-Path $PSScriptRoot 'promote.mjs') @ScriptArgs
    } else {
        & $Script @ScriptArgs
    }
    $result = $LASTEXITCODE
} finally {
    Remove-Item Env:HOTSTEP_ENGINE_LOCK_ROOT -ErrorAction SilentlyContinue
    $handle.Dispose()
}
exit $result
