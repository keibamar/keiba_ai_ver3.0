@echo off
REM ============================================================
REM Weekly review article auto-generation (every TUE 21:00)
REM ============================================================

cd C:\keiba_ai\keiba_ai_ver3.0
set PYTHONIOENCODING=utf-8

echo === [1/3] Generate weekly review ===
python scripts\run_make_weekly_review.py
if %errorlevel% neq 0 (
    echo Error: weekly review generation failed.
    exit /b 1
)

echo.
echo === [2/3] Commit ===
call C:\keiba_ai\keiba_ai_ver3.0\bat\Commit\commit_for_weekly_review.bat

echo.
echo === [3/3] Upload to ConoHa ===
call C:\keiba_ai\keiba_ai_ver3.0\bat\Deploy\upload_to_conoha_auto.bat

echo.
echo Done: %date% %time%