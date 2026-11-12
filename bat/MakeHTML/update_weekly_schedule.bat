@echo off
REM ============================================================
REM Weekly schedule + weekend preview (every THU 20:00)
REM   1. race_time_id_list + weekend race card HTML (entries)
REM   2. weekend graded race preview (needs the entries from step 1)
REM ============================================================

cd C:\keiba_ai\keiba_ai_ver3.0
set PYTHONIOENCODING=utf-8

echo === [1/5] Make time id list + weekend HTML ===
python scripts\run_weekly_schedule_update.py

echo.
echo === [2/5] Commit (time id list) ===
call C:\keiba_ai\keiba_ai_ver3.0\bat\Commit\commit_for_time_id_list.bat

echo.
echo === [3/5] Generate weekend preview ===
python scripts\run_generate_weekend_preview.py
if %errorlevel% neq 0 echo Error: weekend preview generation failed.

echo.
echo === [4/5] Commit (weekend preview) ===
call C:\keiba_ai\keiba_ai_ver3.0\bat\Commit\commit_for_weekend_preview.bat

echo.
echo === [5/5] Upload to ConoHa ===
call C:\keiba_ai\keiba_ai_ver3.0\bat\Deploy\upload_to_conoha_auto.bat

echo.
echo Done: %date% %time%
