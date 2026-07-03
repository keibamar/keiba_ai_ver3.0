cd C:\keiba_ai\keiba_ai_ver3.0
set PYTHONIOENCODING=utf-8
python scripts\run_refresh_prev_day_odds.py
call C:\keiba_ai\keiba_ai_ver3.0\bat\Commit\commit_for_make_html_prev_day.bat
call C:\keiba_ai\keiba_ai_ver3.0\bat\Deploy\upload_to_conoha_auto.bat
