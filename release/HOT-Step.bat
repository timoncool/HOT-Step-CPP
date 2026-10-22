@echo off
echo =============================================
echo   HOT-Step 9000 CPP
echo   High-Performance Music Generation
echo =============================================
echo.

REM Set the distribution root for portable mode detection.
REM The server reads HOT_STEP_ROOT to resolve all paths.
set "HOT_STEP_ROOT=%~dp0"

REM Resolve the stable profile marker with the bundled runtime.
set "RESTART_MARKER="
for /f "delims=" %%P in ('""%~dp0runtime\node.exe" "%~dp0profile-path.mjs""') do set "RESTART_MARKER=%%P"
if not defined RESTART_MARKER exit /b 1

REM Open browser if no existing tab is found
start /MIN "" "%~dp0runtime\node.exe" "%~dp0open-browser-if-needed.mjs" "http://localhost:3001/"

REM ── Restart loop ──────────────────────────────────────────
REM The server writes the profile restart marker when the user clicks
REM "Restart" in the UI. After node exits, we check for the
REM marker — if it exists, we delete it and relaunch.
:start
echo Starting server...
echo.
"%~dp0runtime\node.exe" "%~dp0server\server.mjs"

REM Check for restart marker
if exist "%RESTART_MARKER%" (
    del "%RESTART_MARKER%"
    echo.
    echo [HOT-Step] Restarting...
    echo.
    goto start
)

echo.
echo [HOT-Step] Server stopped. Press any key to exit.
pause >nul
