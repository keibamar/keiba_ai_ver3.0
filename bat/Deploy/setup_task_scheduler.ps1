# Weekly update automation - Task Scheduler setup script
# Run this in an elevated (Administrator) PowerShell:
#   Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
#   cd C:\keiba_ai\keiba_ai_ver3.0
#   .\bat\Deploy\setup_task_scheduler.ps1
#
# All tasks run through scripts\run_task_with_notify.py, which executes the bat,
# sends Gmail notifications on start / finish, and saves logs to logs\tasks\.

$root = "C:\keiba_ai\keiba_ai_ver3.0"
$user = $env:USERNAME   # current user (auto-detected; no change needed on a new PC)

# Python executable: always register with a full path.
# Python is often only on the *user* PATH, so an elevated PowerShell may not find
# it via Get-Command. Fall back to the per-user default install location.
$python = (Get-Command python -ErrorAction SilentlyContinue |
    Where-Object { $_.Source -notlike "*\WindowsApps\*" } | Select-Object -First 1).Source
if (-not $python) {
    $python = Get-ChildItem "$env:LOCALAPPDATA\Programs\Python\Python3*\python.exe" -ErrorAction SilentlyContinue |
        Sort-Object FullName -Descending | Select-Object -First 1 -ExpandProperty FullName
}
if (-not $python) {
    Write-Error "python.exe not found. Set `$python manually in this script."
    exit 1
}
Write-Output "python: $python"

# --- Remove all stale tasks under \keiba\ ---
Get-ScheduledTask -TaskPath "\keiba\" -ErrorAction SilentlyContinue | ForEach-Object {
    Unregister-ScheduledTask -TaskName $_.TaskName -TaskPath "\keiba\" -Confirm:$false -ErrorAction SilentlyContinue
}

# create logs directory
New-Item -ItemType Directory -Force -Path "$root\logs\tasks" | Out-Null

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

# action: run the bat through the notify wrapper
function New-NotifyAction([string]$name, [string]$batRel) {
    New-ScheduledTaskAction -Execute $python `
        -Argument "`"$root\scripts\run_task_with_notify.py`" $name `"$root\bat\$batRel`"" `
        -WorkingDirectory $root
}

# --- 1. weekly_update: Wed 00:30 ---
$action = New-NotifyAction "weekly_update" "Datasets\update_weekly.bat"
$trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Wednesday -At "00:30"
Register-ScheduledTask -TaskName "weekly_update" -TaskPath "\keiba\" -Action $action `
    -Trigger $trigger -Principal $principal -Settings (New-Settings) -Force

# --- 2. weekly_schedule: Thu 20:00 ---
$action = New-NotifyAction "weekly_schedule" "MakeHTML\update_weekly_schedule.bat"
$trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Thursday -At "20:00"
Register-ScheduledTask -TaskName "weekly_schedule" -TaskPath "\keiba\" -Action $action `
    -Trigger $trigger -Principal $principal -Settings (New-Settings) -Force

# --- 3. make_html_prev_day: Fri 18:00 + Sat 16:00 ---
$action = New-NotifyAction "make_html_prev_day" "MakeHTML\make_html_prev_day.bat"
$triggers = @(
    (New-ScheduledTaskTrigger -Weekly -DaysOfWeek Friday -At "18:00"),
    (New-ScheduledTaskTrigger -Weekly -DaysOfWeek Saturday -At "16:00")
)
Register-ScheduledTask -TaskName "make_html_prev_day" -TaskPath "\keiba\" -Action $action `
    -Trigger $triggers -Principal $principal -Settings (New-Settings) -Force

# --- 4. post_today_race: Sat 09:00 + Sun 09:00 (long-running live loop) ---
$action = New-NotifyAction "post_today_race" "TodayRace\post_today_race.bat"
$triggers = @(
    (New-ScheduledTaskTrigger -Weekly -DaysOfWeek Saturday -At "09:00"),
    (New-ScheduledTaskTrigger -Weekly -DaysOfWeek Sunday -At "09:00")
)
Register-ScheduledTask -TaskName "post_today_race" -TaskPath "\keiba\" -Action $action `
    -Trigger $triggers -Principal $principal -Settings (New-Settings -LongRunning) -Force

# --- 5. post_today_race_mon: Mon 09:00 (long-running live loop) ---
$action = New-NotifyAction "post_today_race_mon" "TodayRace\post_today_race.bat"
$trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday -At "09:00"
Register-ScheduledTask -TaskName "post_today_race_mon" -TaskPath "\keiba\" -Action $action `
    -Trigger $trigger -Principal $principal -Settings (New-Settings -LongRunning) -Force

# --- 6. today_race_returns: Sat 18:00 + Sun 18:00 ---
$action = New-NotifyAction "today_race_returns" "TodayRace\today_race_rerturns.bat"
$triggers = @(
    (New-ScheduledTaskTrigger -Weekly -DaysOfWeek Saturday -At "18:00"),
    (New-ScheduledTaskTrigger -Weekly -DaysOfWeek Sunday -At "18:00")
)
Register-ScheduledTask -TaskName "today_race_returns" -TaskPath "\keiba\" -Action $action `
    -Trigger $triggers -Principal $principal -Settings (New-Settings) -Force

# --- 7. today_race_returns_mon: Mon 18:00 ---
$action = New-NotifyAction "today_race_returns_mon" "TodayRace\today_race_rerturns.bat"
$trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday -At "18:00"
Register-ScheduledTask -TaskName "today_race_returns_mon" -TaskPath "\keiba\" -Action $action `
    -Trigger $trigger -Principal $principal -Settings (New-Settings) -Force

# --- 8. update_daily_html: Sat 20:00 + Sun 20:00 ---
$action = New-NotifyAction "update_daily_html" "MakeHTML\update_daily_html.bat"
$triggers = @(
    (New-ScheduledTaskTrigger -Weekly -DaysOfWeek Saturday -At "20:00"),
    (New-ScheduledTaskTrigger -Weekly -DaysOfWeek Sunday -At "20:00")
)
Register-ScheduledTask -TaskName "update_daily_html" -TaskPath "\keiba\" -Action $action `
    -Trigger $triggers -Principal $principal -Settings (New-Settings) -Force

# --- 9. update_daily_html_mon: Mon 20:00 ---
$action = New-NotifyAction "update_daily_html_mon" "MakeHTML\update_daily_html.bat"
$trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday -At "20:00"
Register-ScheduledTask -TaskName "update_daily_html_mon" -TaskPath "\keiba\" -Action $action `
    -Trigger $trigger -Principal $principal -Settings (New-Settings) -Force

# --- 10. post_weekend_preview: Fri 12:00 ---
$action = New-NotifyAction "post_weekend_preview" "TodayRace\post_weekend_preview.bat"
$trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Friday -At "12:00"
Register-ScheduledTask -TaskName "post_weekend_preview" -TaskPath "\keiba\" -Action $action `
    -Trigger $trigger -Principal $principal -Settings (New-Settings) -Force

# --- 11. post_weekend_summary: Tue 20:00 ---
$action = New-NotifyAction "post_weekend_summary" "TodayRace\post_weekend_summary.bat"
$trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Tuesday -At "20:00"
Register-ScheduledTask -TaskName "post_weekend_summary" -TaskPath "\keiba\" -Action $action `
    -Trigger $trigger -Principal $principal -Settings (New-Settings) -Force

# --- 12. update_today_content: every day 00:05 ---
$action = New-NotifyAction "update_today_content" "MakeHTML\update_today_content.bat"
$trigger = New-ScheduledTaskTrigger -Daily -At "00:05"
Register-ScheduledTask -TaskName "update_today_content" -TaskPath "\keiba\" -Action $action `
    -Trigger $trigger -Principal $principal -Settings (New-Settings) -Force

# --- 13. generate_weekend_preview: Thu 20:00 ---
$action = New-NotifyAction "generate_weekend_preview" "TodayRace\generate_weekend_preview.bat"
$trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Thursday -At "20:00"
Register-ScheduledTask -TaskName "generate_weekend_preview" -TaskPath "\keiba\" -Action $action `
    -Trigger $trigger -Principal $principal -Settings (New-Settings) -Force

# --- 14. generate_weekly_review: Tue 21:00 ---
$action = New-NotifyAction "generate_weekly_review" "TodayRace\generate_weekly_review.bat"
$trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Tuesday -At "21:00"
Register-ScheduledTask -TaskName "generate_weekly_review" -TaskPath "\keiba\" -Action $action `
    -Trigger $trigger -Principal $principal -Settings (New-Settings) -Force

# --- 15. generate_daily_preview: Fri 20:00 + Sat 20:00 ---
$action = New-NotifyAction "generate_daily_preview" "TodayRace\generate_daily_preview.bat"
$triggers = @(
    (New-ScheduledTaskTrigger -Weekly -DaysOfWeek Friday -At "20:00"),
    (New-ScheduledTaskTrigger -Weekly -DaysOfWeek Saturday -At "20:00")
)
Register-ScheduledTask -TaskName "generate_daily_preview" -TaskPath "\keiba\" -Action $action `
    -Trigger $triggers -Principal $principal -Settings (New-Settings) -Force

Write-Output "Done. Current tasks under \keiba\:"
Get-ScheduledTask -TaskPath "\keiba\" | Select-Object TaskName, State | Format-Table -AutoSize

# --- 傾向分析・オッズ更新タスク（ルート \ フォルダ） ---
# \keiba\ 外のタスク。既存があれば上書き。
foreach ($t in @("KeibaTrend_Daily_Mon","KeibaTrend_Daily_Sat","KeibaTrend_Daily_Sun","keiba_refresh_prev_day_odds")) {
    Unregister-ScheduledTask -TaskName $t -Confirm:$false -ErrorAction SilentlyContinue
}

# 月曜 20:00
$a = New-ScheduledTaskAction -Execute "cmd.exe" `
    -Argument "/c `"$root\bat\MakeHTML\generate_daily_trend.bat`"" `
    -WorkingDirectory $root
$tr = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Monday -At "20:00"
Register-ScheduledTask -TaskName "KeibaTrend_Daily_Mon" -Action $a `
    -Trigger $tr -Principal $principal -Settings (New-Settings) -Force | Out-Null
Write-Output "Registered: KeibaTrend_Daily_Mon (Mon 20:00)"

# 土曜 20:00
$a = New-ScheduledTaskAction -Execute "cmd.exe" `
    -Argument "/c `"$root\bat\MakeHTML\generate_daily_trend.bat`"" `
    -WorkingDirectory $root
$tr = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Saturday -At "20:00"
Register-ScheduledTask -TaskName "KeibaTrend_Daily_Sat" -Action $a `
    -Trigger $tr -Principal $principal -Settings (New-Settings) -Force | Out-Null
Write-Output "Registered: KeibaTrend_Daily_Sat (Sat 20:00)"

# 日曜 20:00
$a = New-ScheduledTaskAction -Execute "cmd.exe" `
    -Argument "/c `"$root\bat\MakeHTML\generate_daily_trend.bat`"" `
    -WorkingDirectory $root
$tr = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Sunday -At "20:00"
Register-ScheduledTask -TaskName "KeibaTrend_Daily_Sun" -Action $a `
    -Trigger $tr -Principal $principal -Settings (New-Settings) -Force | Out-Null
Write-Output "Registered: KeibaTrend_Daily_Sun (Sun 20:00)"

# 金・土 20:00（前日オッズ更新）
$a = New-ScheduledTaskAction -Execute "cmd.exe" `
    -Argument "/c `"$root\bat\MakeHTML\refresh_prev_day_odds.bat`"" `
    -WorkingDirectory $root
$tr = New-ScheduledTaskTrigger -Weekly -DaysOfWeek Friday, Saturday -At "20:00"
Register-ScheduledTask -TaskName "keiba_refresh_prev_day_odds" -Action $a `
    -Trigger $tr -Principal $principal -Settings (New-Settings) -Force | Out-Null
Write-Output "Registered: keiba_refresh_prev_day_odds (Fri+Sat 20:00)"

Write-Output ""
Write-Output "Done. Root-folder tasks:"
Get-ScheduledTask | Where-Object { $_.TaskPath -eq "\" -and $_.TaskName -like "*eiba*" } |
    Select-Object TaskName, State | Format-Table -AutoSize
