# Weekly update automation - Task Scheduler setup script
# Run this in an elevated (Administrator) PowerShell:
#   Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
#   cd C:\keiba_ai\keiba_ai_ver3.0
#   .\bat\Deploy\setup_task_scheduler.ps1

$root = "C:\keiba_ai\keiba_ai_ver3.0"
$user = $env:USERNAME   # 実行ユーザーを自動取得（移行先PCでも変更不要）

# --- Remove all stale tasks under \keiba\ ---
Get-ScheduledTask -TaskPath "\keiba\" -ErrorAction SilentlyContinue | ForEach-Object {
    Unregister-ScheduledTask -TaskName $_.TaskName -TaskPath "\keiba\" -Confirm:$false -ErrorAction SilentlyContinue
}

# logsディレクトリ作成
New-Item -ItemType Directory -Force -Path "$root\logs" | Out-Null

$principal = New-ScheduledTaskPrincipal -UserId $user -LogonType S4U -RunLevel Highest

function New-Settings([switch]$LongRunning) {
    if ($LongRunning) {
        New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
            -StartWhenAvailable -WakeToRun -ExecutionTimeLimit (New-TimeSpan -Hours 14)
    } else {
        New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
            -StartWhenAvailable -WakeToRun -ExecutionTimeLimit (New-TimeSpan -Hours 2)
    }
}

# --- 1. weekly_update: Wed 00:30 ---
$action = New-ScheduledTaskAction -Execute "$root\bat\Datasets\update_weekly.bat"
$trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Wednesday -At "00:30"
Register-ScheduledTask -TaskName "weekly_update" -TaskPath "\keiba\" -Action $action `
    -Trigger $trigger -Principal $principal -Settings (New-Settings) -Force

# --- 2. weekly_schedule: Thu 20:00 ---
$action = New-ScheduledTaskAction -Execute "$root\bat\MakeHTML\update_weekly_schedule.bat"
$trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Thursday -At "20:00"
Register-ScheduledTask -TaskName "weekly_schedule" -TaskPath "\keiba\" -Action $action `
    -Trigger $trigger -Principal $principal -Settings (New-Settings) -Force

# --- 3. make_html_prev_day: Fri 18:00 + Sat 16:00 ---
$action = New-ScheduledTaskAction -Execute "$root\bat\MakeHTML\make_html_prev_day.bat"
$triggers = @(
    (New-ScheduledTaskTrigger -Weekly -DaysOfWeek Friday -At "18:00"),
    (New-ScheduledTaskTrigger -Weekly -DaysOfWeek Saturday -At "16:00")
)
Register-ScheduledTask -TaskName "make_html_prev_day" -TaskPath "\keiba\" -Action $action `
    -Trigger $triggers -Principal $principal -Settings (New-Settings) -Force

# --- 4. post_today_race: Sat 09:00 + Sun 09:00 (long-running live loop) ---
$action = New-ScheduledTaskAction -Execute "$root\bat\TodayRace\post_today_race.bat"
$triggers = @(
    (New-ScheduledTaskTrigger -Weekly -DaysOfWeek Saturday -At "09:00"),
    (New-ScheduledTaskTrigger -Weekly -DaysOfWeek Sunday -At "09:00")
)
Register-ScheduledTask -TaskName "post_today_race" -TaskPath "\keiba\" -Action $action `
    -Trigger $triggers -Principal $principal -Settings (New-Settings -LongRunning) -Force

# --- 5. post_today_race_mon: Mon 09:00 (long-running live loop) ---
$action = New-ScheduledTaskAction -Execute "$root\bat\TodayRace\post_today_race.bat"
$trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday -At "09:00"
Register-ScheduledTask -TaskName "post_today_race_mon" -TaskPath "\keiba\" -Action $action `
    -Trigger $trigger -Principal $principal -Settings (New-Settings -LongRunning) -Force

# --- 6. today_race_returns: Sat 18:00 + Sun 18:00 ---
$action = New-ScheduledTaskAction -Execute "$root\bat\TodayRace\today_race_rerturns.bat"
$triggers = @(
    (New-ScheduledTaskTrigger -Weekly -DaysOfWeek Saturday -At "18:00"),
    (New-ScheduledTaskTrigger -Weekly -DaysOfWeek Sunday -At "18:00")
)
Register-ScheduledTask -TaskName "today_race_returns" -TaskPath "\keiba\" -Action $action `
    -Trigger $triggers -Principal $principal -Settings (New-Settings) -Force

# --- 7. today_race_returns_mon: Mon 18:00 ---
$action = New-ScheduledTaskAction -Execute "$root\bat\TodayRace\today_race_rerturns.bat"
$trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday -At "18:00"
Register-ScheduledTask -TaskName "today_race_returns_mon" -TaskPath "\keiba\" -Action $action `
    -Trigger $trigger -Principal $principal -Settings (New-Settings) -Force

# --- 8. update_daily_html: Sat 20:00 + Sun 20:00 ---
$action = New-ScheduledTaskAction -Execute "$root\bat\MakeHTML\update_daily_html.bat"
$triggers = @(
    (New-ScheduledTaskTrigger -Weekly -DaysOfWeek Saturday -At "20:00"),
    (New-ScheduledTaskTrigger -Weekly -DaysOfWeek Sunday -At "20:00")
)
Register-ScheduledTask -TaskName "update_daily_html" -TaskPath "\keiba\" -Action $action `
    -Trigger $triggers -Principal $principal -Settings (New-Settings) -Force

# --- 9. update_daily_html_mon: Mon 20:00 ---
$action = New-ScheduledTaskAction -Execute "$root\bat\MakeHTML\update_daily_html.bat"
$trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday -At "20:00"
Register-ScheduledTask -TaskName "update_daily_html_mon" -TaskPath "\keiba\" -Action $action `
    -Trigger $trigger -Principal $principal -Settings (New-Settings) -Force

# --- 10. post_weekend_preview: Fri 12:00 ---
$action = New-ScheduledTaskAction -Execute "$root\bat\TodayRace\post_weekend_preview.bat"
$trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Friday -At "12:00"
Register-ScheduledTask -TaskName "post_weekend_preview" -TaskPath "\keiba\" -Action $action `
    -Trigger $trigger -Principal $principal -Settings (New-Settings) -Force

# --- 11. post_weekend_summary: Tue 20:00 ---
$action = New-ScheduledTaskAction -Execute "$root\bat\TodayRace\post_weekend_summary.bat"
$trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Tuesday -At "20:00"
Register-ScheduledTask -TaskName "post_weekend_summary" -TaskPath "\keiba\" -Action $action `
    -Trigger $trigger -Principal $principal -Settings (New-Settings) -Force

# --- 12. update_today_content: every day 00:05 ---
$action = New-ScheduledTaskAction -Execute "$root\bat\MakeHTML\update_today_content.bat"
$trigger = New-ScheduledTaskTrigger -Daily -At "00:05"
Register-ScheduledTask -TaskName "update_today_content" -TaskPath "\keiba\" -Action $action `
    -Trigger $trigger -Principal $principal -Settings (New-Settings) -Force

# --- 13. generate_weekend_preview: Thu 20:00 (with log) ---
$action = New-ScheduledTaskAction -Execute "cmd" `
    -Argument "/c `"$root\bat\TodayRace\generate_weekend_preview.bat`" >> `"$root\logs\weekend_preview.log`" 2>&1"
$trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Thursday -At "20:00"
Register-ScheduledTask -TaskName "generate_weekend_preview" -TaskPath "\keiba\" -Action $action `
    -Trigger $trigger -Principal $principal -Settings (New-Settings) -Force

# --- 14. generate_weekly_review: Tue 21:00 (with log) ---
$action = New-ScheduledTaskAction -Execute "cmd" `
    -Argument "/c `"$root\bat\TodayRace\generate_weekly_review.bat`" >> `"$root\logs\weekly_review.log`" 2>&1"
$trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Tuesday -At "21:00"
Register-ScheduledTask -TaskName "generate_weekly_review" -TaskPath "\keiba\" -Action $action `
    -Trigger $trigger -Principal $principal -Settings (New-Settings) -Force

# --- 15. generate_daily_preview: Fri 20:00 + Sat 20:00 (with log) ---
$action = New-ScheduledTaskAction -Execute "cmd" `
    -Argument "/c `"$root\bat\TodayRace\generate_daily_preview.bat`" >> `"$root\logs\daily_preview.log`" 2>&1"
$triggers = @(
    (New-ScheduledTaskTrigger -Weekly -DaysOfWeek Friday -At "20:00"),
    (New-ScheduledTaskTrigger -Weekly -DaysOfWeek Saturday -At "20:00")
)
Register-ScheduledTask -TaskName "generate_daily_preview" -TaskPath "\keiba\" -Action $action `
    -Trigger $triggers -Principal $principal -Settings (New-Settings) -Force

Write-Output "Done. Current tasks under \keiba\:"
Get-ScheduledTask -TaskPath "\keiba\" | Select-Object TaskName, State | Format-Table -AutoSize
