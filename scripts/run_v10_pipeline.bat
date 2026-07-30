@echo off
setlocal

cd /d C:\keiba_ai\keiba_ai_ver3.0
set LOG_DIR=logs
set DATE_STR=%date:~0,4%%date:~5,2%%date:~8,2%

echo ======================================
echo v10 パイプライン実行 (%DATE_STR%)
echo Step 1: v10データセット生成 (6-10時間)
echo Step 2: v19モデル学習 (2-4時間)
echo ======================================

:: Step 1: v10データセット生成
echo.
echo [Step 1] v10データセット生成中...
python scripts/make_v10_datasets_no26.py > %LOG_DIR%\make_v10_datasets_%DATE_STR%.log 2>&1
if errorlevel 1 (
    echo ERROR: v10データセット生成に失敗しました
    echo ログ: %LOG_DIR%\make_v10_datasets_%DATE_STR%.log
    exit /b 1
)
echo [Step 1] 完了

:: Step 2: v19モデル学習
echo.
echo [Step 2] v19モデル学習中...
python scripts/train_v19_no26.py > %LOG_DIR%\train_v19_no26_%DATE_STR%.log 2>&1
if errorlevel 1 (
    echo ERROR: v19モデル学習に失敗しました
    echo ログ: %LOG_DIR%\train_v19_no26_%DATE_STR%.log
    exit /b 1
)
echo [Step 2] 完了

echo.
echo ======================================
echo パイプライン完了
echo 次のステップ: python scripts/eval_v19_no26.py
echo ======================================
endlocal
