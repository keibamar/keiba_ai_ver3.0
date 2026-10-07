@echo off
REM ============================================================
REM Auto-generate weekly review article (every TUE 21:00 by Task Scheduler)
REM Run as Administrator
REM ============================================================

echo Registering weekly review task...

schtasks /create ^
  /tn "\keiba\generate_weekly_review" ^
  /tr "cmd /c \"C:\keiba_ai\keiba_ai_ver3.0\bat\TodayRace\generate_weekly_review.bat\" >> \"C:\keiba_ai\keiba_ai_ver3.0\logs\weekly_review.log\" 2>&1" ^
  /sc weekly ^
  /d TUE ^
  /st 21:00 ^
  /f

if %errorlevel% == 0 (
    echo.
    echo Task registered successfully!
    echo Task: \keiba\generate_weekly_review
    echo Schedule: every Tuesday 21:00
    echo Log: C:\keiba_ai\keiba_ai_ver3.0\logs\weekly_review.log
) else (
    echo.
    echo Error: Please run as Administrator.
)

pause