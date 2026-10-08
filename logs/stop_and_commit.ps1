Start-Sleep -Seconds 1215
# fetch_horse_profiles を停止
Stop-Process -Id 4652 -Force -ErrorAction SilentlyContinue
Add-Content 'C:\keiba_ai\keiba_ai_ver3.0\logs\fetch_profiles_stop.log' ("Stopped at " + (Get-Date))
Start-Sleep -Seconds 3
# git commit & push
Set-Location 'C:\keiba_ai\keiba_ai_ver3.0'
git add data/horse/horse_profile/
$count = (git diff --cached --name-only | Measure-Object -Line).Lines
$msg = "261008_bulk_fetch_horse_profiles_checkpoint`n`n一括取得バックグラウンド処理の中間コミット。horse_profile ${count}ファイル追加。`n`nCo-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"
git commit -m $msg
git push origin main
Add-Content 'C:\keiba_ai\keiba_ai_ver3.0\logs\fetch_profiles_stop.log' ("Git push done at " + (Get-Date))
