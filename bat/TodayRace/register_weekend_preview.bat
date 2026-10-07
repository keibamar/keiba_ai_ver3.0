@echo off
REM ============================================================
REM Register weekend preview task to Windows Task Scheduler
REM Run as Administrator
REM ============================================================

echo Registering weekend preview task...

schtasks /create ^
  /tn "\keiba\generate_weekend_preview" ^
  /tr "cmd /c \"C:\keiba_ai\keiba_ai_ver3.0\bat\TodayRace\generate_weekend_preview.bat\" >> \"C:\keiba_ai\keiba_ai_ver3.0\logs\weekend_preview.log\" 2>&1" ^
  /sc weekly ^
  /d THU ^
  /st 20:00 ^
  /f

if %errorlevel% == 0 (
    echo.
    echo Task registered successfully!
    echo Task: \keiba\generate_weekend_preview
    echo Schedule: every Thursday 20:00
    echo Log: C:\keiba_ai\keiba_ai_ver3.0\logs\weekend_preview.log
) else (
    echo.
    echo Error: Please run as Administrator.
)

pause