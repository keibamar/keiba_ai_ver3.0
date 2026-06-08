cd C:\keiba_ai\keiba_ai_ver2.0
git add . 
git commit -m  "%date:~2,2%%date:~5,2%%date:~8,2%_make_time_id_list"
timeout /t 20 /nobreak >nul
git push