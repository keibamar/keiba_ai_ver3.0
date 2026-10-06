@echo off
chcp 65001 > nul
cd /d C:\keiba_ai\keiba_ai_ver3.0
echo ========================================================
echo 新馬戦専用モデル学習
echo ========================================================
python scripts\train_shinba_model.py
pause
