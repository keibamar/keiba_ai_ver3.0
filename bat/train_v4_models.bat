@echo off
cd C:\keiba_ai\keiba_ai_ver3.0
set PYTHONIOENCODING=utf-8
echo ================================
echo v4 モデル学習（的中率重視 + オッズ/人気）
echo 函館・福島・小倉 2020-2026年
echo ================================
python scripts\train_v4_models.py
pause
