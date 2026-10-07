@echo off
REM ============================================================
REM Register daily preview task to Windows Task Scheduler
REM Run as Administrator
REM ============================================================

echo Registering daily preview task...

schtasks /create ^
  /tn "\keiba\generate_daily_preview" ^
  /tr "cmd /c \"C:\keiba_ai\keiba_ai_ver3.0\bat\TodayRace\generate_daily_preview.bat\" >> \"C:\keiba_ai\keiba_ai_ver3.0\logs\daily_preview.log\" 2>&1" ^
  /sc weekly ^
  /d FRI,SAT,SUN ^
  /st 21:00 ^
  /f

if %errorlevel% == 0 (
    echo.
    echo Task registered successfully!
    echo Task: \keiba\generate_daily_preview
    echo Schedule: every Friday/Saturday/Sunday 21:00
    echo Log: C:\keiba_ai\keiba_ai_ver3.0\logs\daily_preview.log
) else (
    echo.
    echo Error: Please run as Administrator.
)

pause