@echo off
setlocal
REM HOT-Step engine build. Optional argument: all (default), cuda, vulkan, cpu.
REM Uses vswhere to find any Visual Studio edition automatically.

set "BUILD_BACKEND=%~1"
if not defined BUILD_BACKEND set "BUILD_BACKEND=all"
set "BUILD_CUDA=OFF"
set "BUILD_VULKAN=OFF"
if /i "%BUILD_BACKEND%"=="all" goto :all
if /i "%BUILD_BACKEND%"=="cuda" goto :cuda
if /i "%BUILD_BACKEND%"=="vulkan" goto :vulkan
if /i "%BUILD_BACKEND%"=="cpu" goto :setup
echo ERROR: Unknown backend "%BUILD_BACKEND%". Use all, cuda, vulkan or cpu.
exit /b 1
:all
set "BUILD_VULKAN=ON"
:cuda
set "BUILD_CUDA=ON"
goto :setup
:vulkan
set "BUILD_VULKAN=ON"
:setup
echo HOT-Step engine build: %BUILD_BACKEND%

REM Skip vcvars if already sourced (prevents PATH overflow on repeated runs)
if defined VSCMD_VER goto :deps

set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" (
    echo ERROR: vswhere.exe not found. Is Visual Studio or Build Tools installed?
    exit /b 1
)

set "VCVARS_TMP=%TEMP%\vcvars_path.txt"
"%VSWHERE%" -latest -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -find "VC\Auxiliary\Build\vcvars64.bat" > "%VCVARS_TMP%" 2>nul

set "VCVARS="
for /f "usebackq tokens=*" %%i in ("%VCVARS_TMP%") do set "VCVARS=%%i"
del "%VCVARS_TMP%" 2>nul

if not defined VCVARS (
    echo ERROR: Could not find vcvars64.bat via vswhere.
    echo        Install the "Desktop development with C++" workload.
    exit /b 1
)

echo Using: %VCVARS%
call "%VCVARS%"
if errorlevel 1 exit /b %errorlevel%

:deps
:build
cd /d "%~dp0"
if errorlevel 1 exit /b %errorlevel%
rem rd /s /q build 2>nul
mkdir build 2>nul
cd build
if errorlevel 1 exit /b %errorlevel%

REM CUDA uses the installed toolkit; this does not select a CI CUDA version.
cmake .. -DGGML_CPU_ALL_VARIANTS=ON -DGGML_CUDA=%BUILD_CUDA% -DGGML_VULKAN=%BUILD_VULKAN% -DGGML_BACKEND_DL=ON %RELEASE_CMAKE_EXTRA%
if errorlevel 1 exit /b %errorlevel%
cmake --build . --config Release -j %NUMBER_OF_PROCESSORS%
if errorlevel 1 exit /b %errorlevel%

cd ..
exit /b 0
