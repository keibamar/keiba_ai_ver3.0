cd C:\keiba_ai\keiba_ai_ver3.0
python scripts\run_weekly_update.py
python scripts\run_weekly_update_html.py
call C:\keiba_ai\keiba_ai_ver3.0\bat\Commit\commit_for_weekly_update.bat
call C:\keiba_ai\keiba_ai_ver3.0\bat\Deploy\upload_to_conoha_auto.bat
