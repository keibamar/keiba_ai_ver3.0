@echo off
chcp 65001 > nul
REM ============================================================
REM generate_daily_preview: Fri/Sat 20:00 (task scheduler)
REM
REM Steps:
REM   1. Refresh next-day odds/popularity  -> update race card HTML
REM   2. Generate daily preview article via Claude API
REM   3. git commit
REM   4. ConoHa upload
REM ============================================================

cd /d C:\keiba_ai\keiba_ai_ver3.0
set PYTHONIOENCODING=utf-8

echo === [1/3] odds refresh ===
python scripts\run_refresh_prev_day_odds.py
if %errorlevel% neq 0 (
    echo [ERROR] odds refresh failed ^(errorlevel=%errorlevel%^)
    exit /b 1
)

echo.
echo === [2/3] daily preview generation ===
python scripts\run_generate_daily_preview.py
if %errorlevel% neq 0 (
    echo [ERROR] daily preview generation failed ^(errorlevel=%errorlevel%^)
    exit /b 1
)

echo.
echo === [3/3] commit and upload ===
call C:\keiba_ai\keiba_ai_ver3.0\bat\Commit\commit_for_daily_preview.bat
call C:\keiba_ai\keiba_ai_ver3.0\bat\Deploy\upload_to_conoha_auto.bat

echo.
echo done: %date% %time%
