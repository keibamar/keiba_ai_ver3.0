@echo off
REM Double-click this file to (re)register all MAR scheduled tasks.
REM
REM "Run with PowerShell" on .ps1 files fails silently on this PC because the
REM PowerShell execution policy is Restricted. This launcher bypasses the policy
REM for this run only (no system setting is changed). The .ps1 then re-launches
REM itself as Administrator (UAC prompt) and waits for Enter before closing.

powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup_task_scheduler.ps1"
if errorlevel 1 pause
