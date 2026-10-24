@echo off
REM ============================================================
REM Weekend graded race preview auto-generation (every THU 20:00)
REM ============================================================

cd C:\keiba_ai\keiba_ai_ver3.0
set PYTHONIOENCODING=utf-8

echo === [1/4] Make time id list ===
python scripts\run_make_time_id_list.py
if %errorlevel% neq 0 (
    echo Error: make_time_id_list failed.
    exit /b 1
)

echo.
echo === [2/4] Generate weekend preview ===
python scripts\run_generate_weekend_preview.py
if %errorlevel% neq 0 (
    echo Error: weekend preview generation failed.
    exit /b 1
)

echo.
echo === [3/4] Commit ===
call C:\keiba_ai\keiba_ai_ver3.0\bat\Commit\commit_for_weekend_preview.bat

echo.
echo === [4/4] Upload to ConoHa ===
call C:\keiba_ai\keiba_ai_ver3.0\bat\Deploy\upload_to_conoha_auto.bat

echo.
echo Done: %date% %time%