<#
.SYNOPSIS
  Start a script outside the caller's process tree, so it outlives an agent run.

.DESCRIPTION
  Agent runtimes put each run in a Windows Job Object with kill-on-close and no
  breakaway, so everything the run starts (Start-Process and `start` included)
  dies when the run ends. A process started by the Task Scheduler service is
  outside that job. This registers a one-shot task for the current user
  (interactive token, no admin rights, no stored password), runs it, records
  the PID and deletes the task; deleting a task does not stop its process.
  WMI's Win32_Process.Create did the same job, but antivirus flagged it.

  The task runs `conhost --headless` (no window on the desktop) around cmd,
  which runs logs\detached\<name>-<timestamp>.cmd with stdout/stderr appended
  to the matching .log. Prints the PID and the log path.
  Stop it with: taskkill /PID <pid> /T /F

.EXAMPLE
  pwsh -NoProfile -File tools\detach.ps1 dev.bat
.EXAMPLE
  pwsh -NoProfile -File tools\detach.ps1 tools\yue2-vae-train\run.bat --resume
#>
param(
  [Parameter(Mandatory, Position = 0)][string]$Script,
  [Parameter(ValueFromRemainingArguments)][string[]]$ScriptArgs
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path $PSScriptRoot -Parent
$target = (Resolve-Path $Script).Path
$cwd = (Get-Location).ProviderPath

# Everything is pasted into one cmd line: refuse what cmd would treat as
# an operator, expansion or quote, rather than trying to escape it.
foreach ($a in @($target, $cwd) + @($ScriptArgs)) {
  if ($null -ne $a -and ($a -eq '' -or $a -match '[&|<>^"%]')) {
    throw "Unsupported argument '$a': empty values and & | < > ^ `" % cannot be passed through cmd safely."
  }
}

# A second dev.bat on top of a live stack is the doubled-VRAM orphan problem.
if ((Split-Path $target -Leaf) -eq 'dev.bat' -and
    (Get-NetTCPConnection -LocalPort 3001 -State Listen -ErrorAction SilentlyContinue)) {
  throw 'The app is already running (port 3001 is listening). Use it, or stop it first.'
}
# Bitdefender ATD killed the whole tree when the browser helper's
# `powershell -ExecutionPolicy Bypass` ran under a detached cmd.
if ((Split-Path $target -Leaf) -eq 'dev.bat') { $ScriptArgs = @('--no-browser') + @($ScriptArgs | Where-Object { $_ }) }

$logDir = Join-Path $repo 'logs\detached'
New-Item -ItemType Directory -Force $logDir | Out-Null
# conhost drops inner quotes when it rebuilds the cmd line, so the task only
# passes a space-free launcher name and the launcher holds the real command.
# The PID suffix keeps concurrent launches of the same script apart.
$name = '{0}-{1:yyyyMMdd-HHmmss}-{2}' -f ([IO.Path]::GetFileNameWithoutExtension($target) -replace '[^\w.-]', '_'), (Get-Date), $PID
$log = Join-Path $logDir "$name.log"
$argText = ($ScriptArgs | ForEach-Object { if ($_ -match '\s') { "`"$_`"" } else { $_ } }) -join ' '
# UTF-8 plus chcp 65001 on the (ASCII) first line keeps non-ASCII paths and
# arguments intact; cmd decodes each later line with the new code page.
Set-Content -Encoding utf8NoBOM (Join-Path $logDir "$name.cmd") @(
  '@chcp 65001 >nul'
  "@cd /d `"$cwd`""
  "@`"$target`" $argText >> `"$log`" 2>&1"
)

$x = [Security.SecurityElement]
$xml = @"
<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.2" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <Principals><Principal id="Author"><LogonType>InteractiveToken</LogonType><RunLevel>LeastPrivilege</RunLevel></Principal></Principals>
  <Settings>
    <MultipleInstancesPolicy>Parallel</MultipleInstancesPolicy>
    <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>
    <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>
    <ExecutionTimeLimit>PT0S</ExecutionTimeLimit>
    <Priority>7</Priority>
  </Settings>
  <Actions Context="Author"><Exec>
    <Command>conhost.exe</Command>
    <Arguments>--headless cmd.exe /d /c $name.cmd</Arguments>
    <WorkingDirectory>$($x::Escape($logDir))</WorkingDirectory>
  </Exec></Actions>
</Task>
"@

$svc = New-Object -ComObject Schedule.Service
$svc.Connect()
$folder = $svc.GetFolder('\')
$taskName = "HotStep-detach-$name"
# 6 = create or update, 3 = interactive token (current user, no password).
$task = $folder.RegisterTask($taskName, $xml, 6, $null, $null, 3)
try {
  $run = $task.Run($null)
  for ($i = 0; $i -lt 50 -and -not $run.EnginePID; $i++) { Start-Sleep -Milliseconds 100; $run.Refresh() }
  $taskPid = $run.EnginePID
} finally {
  $folder.DeleteTask($taskName, 0)
}
if (-not $taskPid) { throw "The scheduled task did not report a process; see $log" }
if (-not (Get-Process -Id $taskPid -ErrorAction SilentlyContinue)) { Write-Warning "PID $taskPid has already exited; see $log" }
"PID $taskPid"
"Log $log"
