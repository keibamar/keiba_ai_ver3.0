@echo off
setlocal

cd /d C:\keiba_ai\keiba_ai_ver3.0
set LOG_DIR=logs
set DATE_STR=%date:~0,4%%date:~5,2%%date:~8,2%

echo ======================================================
echo v12birth パイプライン実行 (%DATE_STR%)
echo  Step 1: v12データセット生成 (30〜60分)
echo  Step 2: v12birth学習 (binary+odds, 2〜4時間)
echo  Step 3: v12birth_ev学習 (EV回帰, 2〜4時間)
echo  Step 4: 評価 (v11ev vs v12birth vs v12birth_ev)
echo ======================================================

:: Step 1: v12データセット生成（v11にbirth_month付加）
echo.
echo [Step 1] v12 データセット生成中...
python scripts/make_v12_datasets_no26.py > %LOG_DIR%\make_v12_datasets_%DATE_STR%.log 2>&1
if errorlevel 1 (
    echo ERROR: v12データセット生成に失敗しました
    echo ログ: %LOG_DIR%\make_v12_datasets_%DATE_STR%.log
    exit /b 1
)
echo [Step 1] 完了

:: Step 2: v12birth 学習 (binary + odds weight)
echo.
echo [Step 2] v12birth モデル学習中 (binary)...
python scripts/train_v12birth_no26.py > %LOG_DIR%\train_v12birth_%DATE_STR%.log 2>&1
if errorlevel 1 (
    echo ERROR: v12birth学習に失敗しました
    echo ログ: %LOG_DIR%\train_v12birth_%DATE_STR%.log
    exit /b 1
)
echo [Step 2] 完了

:: Step 3: v12birth_ev 学習 (EV回帰)
echo.
echo [Step 3] v12birth_ev モデル学習中 (EV回帰)...
python scripts/train_v12birth_ev_no26.py > %LOG_DIR%\train_v12birth_ev_%DATE_STR%.log 2>&1
if errorlevel 1 (
    echo ERROR: v12birth_ev学習に失敗しました
    echo ログ: %LOG_DIR%\train_v12birth_ev_%DATE_STR%.log
    exit /b 1
)
echo [Step 3] 完了

:: Step 4: 評価
echo.
echo [Step 4] 評価中 (v11ev vs v12birth vs v12birth_ev)...
python scripts/eval_v12birth_no26.py > %LOG_DIR%\eval_v12birth_%DATE_STR%.log 2>&1
if errorlevel 1 (
    echo WARNING: 評価でエラーが発生しました (ログ確認推奨)
    echo ログ: %LOG_DIR%\eval_v12birth_%DATE_STR%.log
) else (
    echo [Step 4] 完了
    echo.
    echo === 評価結果 ===
    type %LOG_DIR%\eval_v12birth_%DATE_STR%.log
)

echo.
echo ======================================================
echo パイプライン完了
echo ログ確認:
echo   %LOG_DIR%\eval_v12birth_%DATE_STR%.log
echo ======================================================
endlocal
