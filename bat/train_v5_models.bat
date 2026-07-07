@echo off
cd C:\keiba_ai\keiba_ai_ver3.0
set PYTHONIOENCODING=utf-8
echo ================================
echo v5 モデル学習（血統カテゴリカル特徴量）
echo 全10場 2020-2026年
echo ================================
python scripts\train_v5_models.py
pause
