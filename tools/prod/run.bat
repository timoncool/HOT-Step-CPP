@echo off
pwsh -NoProfile -File "%~dp0run.ps1"
exit /b %errorlevel%
