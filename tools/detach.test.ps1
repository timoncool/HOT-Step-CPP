# Argument checks for detach.ps1. Run: pwsh -NoProfile -File tools\detach.test.ps1
$ErrorActionPreference = 'Stop'
$detach = Join-Path $PSScriptRoot 'detach.ps1'
$dir = Join-Path $PSScriptRoot '..\logs\detached\test'
New-Item -ItemType Directory -Force $dir | Out-Null
$bat = Join-Path $dir 'echo-args.bat'
$out = Join-Path $dir 'echo-args.out'
Set-Content $bat "@echo off`r`necho [%~1][%~2][%~3]> `"$out`""
$fail = 0

# Rejected before anything launches.
foreach ($bad in 'alpha&echo.X', 'a|b', 'a>b', 'a<b', 'a^b', 'a"b', '%PATH%', '') {
  Remove-Item $out -ErrorAction SilentlyContinue
  $threw = $false
  try { & $detach $bat $bad | Out-Null } catch { $threw = $_.Exception.Message -like 'Unsupported argument*' }
  Start-Sleep -Milliseconds 300
  if (-not $threw -or (Test-Path $out)) { "FAIL: '$bad' was not rejected"; $fail++ }
}

# Plain and spaced arguments arrive intact.
Remove-Item $out -ErrorAction SilentlyContinue
& $detach $bat 'two words' plain '--flag' | Out-Null
for ($i = 0; $i -lt 50 -and -not (Test-Path $out); $i++) { Start-Sleep -Milliseconds 100 }
$got = if (Test-Path $out) { (Get-Content $out).Trim() } else { '<no output>' }
if ($got -ne '[two words][plain][--flag]') { "FAIL: round trip gave $got"; $fail++ }

Remove-Item $dir -Recurse -Force
if ($fail) { exit 1 } else { 'detach.ps1 argument checks: ok' }
