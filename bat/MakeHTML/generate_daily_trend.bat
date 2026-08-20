cd C:\keiba_ai\keiba_ai_ver3.0
python scripts\generate_daily_trend.py %1
call bat\Deploy\upload_to_conoha_auto.bat
