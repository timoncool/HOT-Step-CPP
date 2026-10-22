@echo off
node "%~dp0codegraph\launcher.cjs" %*
exit /b %ERRORLEVEL%
