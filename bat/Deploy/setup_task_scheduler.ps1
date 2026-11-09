# Task Scheduler setup script for MAR (17 tasks under \keiba\)
#
# How to run (either works):
#   - Right-click this file -> "Run with PowerShell"  (it re-launches itself as Administrator)
#   - powershell -ExecutionPolicy Bypass -File C:\keiba_ai\keiba_ai_ver3.0\bat\Deploy\setup_task_scheduler.ps1
#
# The window stays open at the end so you can read the result.
#
# All tasks run through scripts\run_task_with_notify.py, which executes the bat,
# sends Gmail notifications on start / finish, and saves logs to logs\tasks\.
#
# NOTE: keep this file ASCII-only. Windows PowerShell 5.1 reads BOM-less files
# as Shift-JIS, and non-ASCII text can break parsing.

$root = "C:\keiba_ai\keiba_ai_ver3.0"

function Wait-AndExit([int]$code) {
    Write-Output ""
    Read-Host "Press Enter to close"
    exit $code
}

# --- Re-launch as Administrator if needed ---
$identity = [Security.Principal.WindowsIdentity]::GetCurrent()
$isAdmin = ([Security.Principal.WindowsPrincipal]$identity).IsInRole(
    [Security.Principal.WindowsBuiltInRole]::Administrator)
if (-not $isAdmin) {
    Write-Output "Not running as Administrator. Re-launching elevated..."
    try {
        Start-Process powershell.exe -Verb RunAs -ArgumentList `
            "-NoProfile -ExecutionPolicy Bypass -File `"$PSCommandPath`"" -ErrorAction Stop
    } catch {
        Write-Output "ERROR: could not elevate (UAC was cancelled?): $_"
        Wait-AndExit 1
    }
    exit 0
}

$user = $env:USERNAME   # current user (auto-detected; no change needed on a new PC)
Write-Output "user  : $user"

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
    Write-Output "ERROR: python.exe not found. Set `$python manually in this script."
    Wait-AndExit 1
}
Write-Output "python: $python"
Write-Output ""

# --- Remove all stale tasks under \keiba\ ---
Get-ScheduledTask -TaskPath "\keiba\" -ErrorAction SilentlyContinue | ForEach-Object {
    Unregister-ScheduledTask -TaskName $_.TaskName -TaskPath "\keiba\" -Confirm:$false -ErrorAction SilentlyContinue
}

# --- Remove legacy root-folder tasks (old PC registered these outside \keiba\) ---
# keiba_refresh_prev_day_odds is intentionally NOT re-created: generate_daily_preview
# already runs run_refresh_prev_day_odds.py as its first step at the same time.
foreach ($t in @("KeibaTrend_Daily_Mon", "KeibaTrend_Daily_Sat", "KeibaTrend_Daily_Sun", "keiba_refresh_prev_day_odds")) {
    Get-ScheduledTask -TaskPath "\" -TaskName $t -ErrorAction SilentlyContinue |
        Unregister-ScheduledTask -Confirm:$false -ErrorAction SilentlyContinue
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

$expected = @()
$failed = @()

# register one task under \keiba\ (output suppressed; failures collected)
function Add-KeibaTask([string]$name, [string]$batRel, $triggers, [switch]$LongRunning) {
    $script:expected += $name
    try {
        Register-ScheduledTask -TaskName $name -TaskPath "\keiba\" `
            -Action (New-NotifyAction $name $batRel) -Trigger $triggers `
            -Principal $principal -Settings (New-Settings -LongRunning:$LongRunning) `
            -Force -ErrorAction Stop | Out-Null
        Write-Output ("  OK    {0}" -f $name)
    } catch {
        $script:failed += $name
        Write-Output ("  FAIL  {0}: {1}" -f $name, $_.Exception.Message)
    }
}

function W([string]$day, [string]$at) { New-ScheduledTaskTrigger -Weekly -DaysOfWeek $day -At $at }

Write-Output "Registering tasks under \keiba\ ..."

# 1. weekly_update: Wed 00:30
Add-KeibaTask "weekly_update" "Datasets\update_weekly.bat" (W Wednesday "00:30")
# 2. weekly_schedule: Thu 20:00
Add-KeibaTask "weekly_schedule" "MakeHTML\update_weekly_schedule.bat" (W Thursday "20:00")
# 3. make_html_prev_day: Fri 18:00 + Sat 16:00
Add-KeibaTask "make_html_prev_day" "MakeHTML\make_html_prev_day.bat" @((W Friday "18:00"), (W Saturday "16:00"))
# 4. post_today_race: Sat 09:00 + Sun 09:00 (long-running live loop)
Add-KeibaTask "post_today_race" "TodayRace\post_today_race.bat" @((W Saturday "09:00"), (W Sunday "09:00")) -LongRunning
# 5. post_today_race_mon: Mon 09:00 (long-running live loop)
Add-KeibaTask "post_today_race_mon" "TodayRace\post_today_race.bat" (W Monday "09:00") -LongRunning
# 6. today_race_returns: Sat 18:00 + Sun 18:00
Add-KeibaTask "today_race_returns" "TodayRace\today_race_rerturns.bat" @((W Saturday "18:00"), (W Sunday "18:00"))
# 7. today_race_returns_mon: Mon 18:00
Add-KeibaTask "today_race_returns_mon" "TodayRace\today_race_rerturns.bat" (W Monday "18:00")
# 8. update_daily_html: Sat 20:00 + Sun 20:00
Add-KeibaTask "update_daily_html" "MakeHTML\update_daily_html.bat" @((W Saturday "20:00"), (W Sunday "20:00"))
# 9. update_daily_html_mon: Mon 20:00
Add-KeibaTask "update_daily_html_mon" "MakeHTML\update_daily_html.bat" (W Monday "20:00")
# 10. post_weekend_preview: Fri 12:00
Add-KeibaTask "post_weekend_preview" "TodayRace\post_weekend_preview.bat" (W Friday "12:00")
# 11. post_weekend_summary: Tue 20:00
Add-KeibaTask "post_weekend_summary" "TodayRace\post_weekend_summary.bat" (W Tuesday "20:00")
# 12. update_today_content: every day 00:05
Add-KeibaTask "update_today_content" "MakeHTML\update_today_content.bat" (New-ScheduledTaskTrigger -Daily -At "00:05")
# 13. generate_weekend_preview: Thu 20:00
Add-KeibaTask "generate_weekend_preview" "TodayRace\generate_weekend_preview.bat" (W Thursday "20:00")
# 14. generate_weekly_review: Tue 21:00
Add-KeibaTask "generate_weekly_review" "TodayRace\generate_weekly_review.bat" (W Tuesday "21:00")
# 15. generate_daily_preview: Fri 20:00 + Sat 20:00 (also refreshes next-day odds)
Add-KeibaTask "generate_daily_preview" "TodayRace\generate_daily_preview.bat" @((W Friday "20:00"), (W Saturday "20:00"))
# 16-17. daily trend analysis: Sat / Sun / Mon 20:00 (skips by itself on non-race days)
Add-KeibaTask "daily_trend" "MakeHTML\generate_daily_trend.bat" @((W Saturday "20:00"), (W Sunday "20:00"))
Add-KeibaTask "daily_trend_mon" "MakeHTML\generate_daily_trend.bat" (W Monday "20:00")

# --- Summary ---
Write-Output ""
$registered = @(Get-ScheduledTask -TaskPath "\keiba\" -ErrorAction SilentlyContinue | Select-Object -ExpandProperty TaskName)
$missing = @($expected | Where-Object { $_ -notin $registered })
Get-ScheduledTask -TaskPath "\keiba\" -ErrorAction SilentlyContinue |
    ForEach-Object {
        $info = $_ | Get-ScheduledTaskInfo
        [pscustomobject]@{ TaskName = $_.TaskName; State = $_.State; NextRun = $info.NextRunTime }
    } | Sort-Object NextRun | Format-Table -AutoSize | Out-String | Write-Output

if ($missing.Count -eq 0 -and $failed.Count -eq 0) {
    Write-Output ("SUCCESS: all {0} tasks registered under \keiba\." -f $expected.Count)
    Wait-AndExit 0
} else {
    Write-Output ("ERROR: {0} task(s) not registered: {1}" -f $missing.Count, ($missing -join ", "))
    Wait-AndExit 1
}
