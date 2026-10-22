@echo off
REM Existing work-channel users reserve the app and GPU for this entire command.
REM A guarded child inherits this marker; it is a recursion guard, not authentication.
if "%HOTSTEP_WORK_GUARD_ACTIVE%"=="1" (
    echo ,%HOTSTEP_WORK_GUARD_RESOURCES%,| findstr /L /C:",app-server," >nul
    if errorlevel 1 goto :missing_work_resource
    echo ,%HOTSTEP_WORK_GUARD_RESOURCES%,| findstr /L /C:",gpu," >nul
    if errorlevel 1 goto :missing_work_resource
    echo ,%HOTSTEP_WORK_GUARD_RESOURCES%,| findstr /L /C:",engine-build," >nul
    if errorlevel 1 goto :missing_work_resource
    goto :work_guarded
)
REM Work-channel reservations are opt-in; the database alone does not enable them.
if not "%HOTSTEP_WORK_CHANNEL%"=="1" goto :work_guarded
REM Repositories that have never used collaboration retain the standalone flow.
if not exist "%~dp0data\collaboration.db" goto :work_guarded
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\mcp-lyricstudio\work-run.ps1" --channel HOT-Step --name Rebuild --resource app-server --resource gpu --resource engine-build --reason "Engine rebuild" -- "%~f0"
exit /b %errorlevel%
:missing_work_resource
echo [dev-rebuild] The outer work guard must reserve app-server, gpu and engine-build.
exit /b 1
:work_guarded
REM dev-rebuild.bat — Gracefully shut down the running app, rebuild the engine, done.
REM
REM Usage: dev-rebuild.bat
REM
REM Sends a shutdown request to the Node server (port 3001), which kills
REM ace-server and itself cleanly (no orphan restarting). Then rebuilds.

echo [dev-rebuild] Requesting graceful shutdown...
curl -s -X POST http://localhost:3001/api/shutdown >nul 2>&1

REM Wait for ace-server to actually die (the shutdown route kills by port)
echo [dev-rebuild] Waiting for ace-server to exit...
set /A retries=0
:wait_loop
tasklist /FI "IMAGENAME eq ace-server.exe" 2>nul | find /I "ace-server.exe" >nul
if %errorlevel% neq 0 goto :build
timeout /t 1 /nobreak >nul
set /A retries+=1
if %retries% GEQ 10 (
    echo [dev-rebuild] Force-killing ace-server after 10s timeout...
    taskkill /F /IM ace-server.exe /T >nul 2>&1
    timeout /t 2 /nobreak >nul
)
if %retries% GEQ 15 (
    echo [dev-rebuild] FATAL: could not stop ace-server. Aborting.
    exit /b 1
)
goto :wait_loop

:build
echo [dev-rebuild] ace-server stopped. Building engine...
call "%~dp0engine\build.cmd"
if errorlevel 1 (
    echo [dev-rebuild] BUILD FAILED - the binaries on disk are STALE.
    echo [dev-rebuild] Scroll up for the first 'error C' or 'error LNK' line.
    exit /b 1
)
echo [dev-rebuild] Done. Start the app with LAUNCH.bat to pick up changes.
