@echo off
REM ============================================================
REM 前日展望記事の自動生成（毎週金・土 21:00 タスクスケジューラで実行）
REM
REM 処理順:
REM   1. 翌日のオッズ・人気を再取得して出馬表HTML更新
REM   2. Claude API で前日展望記事を生成
REM   3. git コミット
REM   4. ConoHa アップロード
REM ============================================================

cd C:\keiba_ai\keiba_ai_ver3.0
set PYTHONIOENCODING=utf-8

echo === [1/3] 翌日オッズ更新 ===
python scripts\run_refresh_prev_day_odds.py
if %errorlevel% neq 0 (
    echo オッズ更新でエラーが発生しました。
    exit /b 1
)

echo.
echo === [2/3] 前日展望記事生成 ===
python scripts\run_generate_daily_preview.py
if %errorlevel% neq 0 (
    echo 前日展望生成でエラーが発生しました。
    exit /b 1
)

echo.
echo === [3/3] コミット＆アップロード ===
call C:\keiba_ai\keiba_ai_ver3.0\bat\Commit\commit_for_daily_preview.bat
call C:\keiba_ai\keiba_ai_ver3.0\bat\Deploy\upload_to_conoha_auto.bat

echo.
echo 完了: %date% %time%
