@echo off
rem Double-click this file. Keep the matching .ps1 beside it.
"%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe" -NoProfile -ExecutionPolicy Bypass -File "%~dp0Train GRU and Generate 6.ps1" %*
set "jobExit=%ERRORLEVEL%"
echo.
if not "%jobExit%"=="0" echo The job stopped with an error. Read the message above.
pause
exit /b %jobExit%
