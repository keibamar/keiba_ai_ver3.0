# タスクスケジューラの update_today_content を正しい bat に修正する
# 管理者権限の PowerShell で実行してください

$xmlPath = "$env:TEMP\update_today_content_fix.xml"
schtasks /Query /TN "\keiba\update_today_content" /XML | Out-File -FilePath $xmlPath -Encoding utf8

$xml = Get-Content $xmlPath -Raw
$xml = $xml -replace [regex]::Escape("C:\keiba_ai\keiba_ai_ver3.0\bat\TodayRace\post_weekend_summary.bat"),
                                      "C:\keiba_ai\keiba_ai_ver3.0\bat\MakeHTML\update_today_content.bat"
$xml | Out-File -FilePath $xmlPath -Encoding utf8

Register-ScheduledTask -TaskName "update_today_content" -TaskPath "\keiba\" -Xml (Get-Content $xmlPath -Raw) -Force

# 確認
$t = Get-ScheduledTask -TaskName "update_today_content"
Write-Host "修正後のAction: $($t.Actions[0].Execute)"
