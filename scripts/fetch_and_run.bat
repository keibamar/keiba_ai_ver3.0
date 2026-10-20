@echo off
chcp 65001 > nul
cd /d C:\keiba_ai\keiba_ai_ver3.0
echo === 調教データ取得 + 実験自動実行 ===

:FETCH_LOOP
echo [%time%] 取得開始...
python scripts/fetch_training_2026horses.py --no-deadline
echo [%time%] プロセス終了。残り確認...

python -c "
import os
from datetime import datetime
d = r'C:\keiba_ai\keiba_ai_ver3.0\data\training'
start = datetime(2026,10,19,22,0,0)
done = sum(1 for f in os.scandir(d) if f.is_file() and f.name.endswith('.csv') and datetime.fromtimestamp(f.stat().st_mtime) >= start)
print(f'done={done}')
" > %TEMP%\fetch_count.txt
for /f "tokens=2 delims==" %%i in (%TEMP%\fetch_count.txt) do set DONE=%%i

echo [%time%] 取得済み: %DONE% / 9186
if %DONE% GEQ 9186 goto EXPERIMENTS
echo 未完了。5秒後に再起動...
timeout /t 5 /nobreak > nul
goto FETCH_LOOP

:EXPERIMENTS
echo.
echo === 実験開始 ===
echo [%time%] [1/3] 新馬モデル再評価...
python scripts/eval_shinba_training.py > logs\eval_shinba_after_fetch.log 2>&1
echo [%time%] 完了

echo [%time%] [2/3] v19tr 全コース再学習...
python scripts/train_v19_training_no26.py > logs\train_v19tr_after_fetch.log 2>&1
echo [%time%] 完了

echo [%time%] [3/3] v19tr バックテスト...
python scripts/eval_v19_training_no26.py > logs\eval_v19tr_after_fetch.log 2>&1
echo [%time%] 完了

echo.
echo === 全ステップ完了 ===
echo ログ: logs\eval_shinba_after_fetch.log
echo ログ: logs\train_v19tr_after_fetch.log
echo ログ: logs\eval_v19tr_after_fetch.log
pause
