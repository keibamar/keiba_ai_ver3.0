# 調教データ取得（自動再起動ループ）→ 完了後に実験自動実行
# 使い方: powershell -ExecutionPolicy Bypass -File scripts/fetch_and_run.ps1

Set-Location "C:\keiba_ai\keiba_ai_ver3.0"
$python = (Get-Command python).Source
$target = 9186
$sessionStart = Get-Date "2026-10-19 22:00:00"

function Get-DoneCount {
    (Get-ChildItem "data\training" -Filter "*.csv" |
     Where-Object { $_.LastWriteTime -gt $sessionStart }).Count
}

Write-Host "=== 調教データ取得 自動再起動ループ ===" -ForegroundColor Cyan

# ---- フェーズ1: 取得ループ ----
$attempt = 0
while ($true) {
    $done = Get-DoneCount
    Write-Host "[$(Get-Date -Format 'HH:mm:ss')] 取得済み: $done / $target 頭" -ForegroundColor Yellow

    if ($done -ge $target) {
        Write-Host "全頭取得完了！" -ForegroundColor Green
        break
    }

    $attempt++
    Write-Host "[$(Get-Date -Format 'HH:mm:ss')] 取得開始 (試行 $attempt 回目)..." -ForegroundColor Cyan
    & $python scripts/fetch_training_2026horses.py --no-deadline
    Write-Host "[$(Get-Date -Format 'HH:mm:ss')] プロセス終了。残り確認..." -ForegroundColor Yellow

    $done = Get-DoneCount
    if ($done -ge $target) {
        Write-Host "全頭取得完了！" -ForegroundColor Green
        break
    }
    Write-Host "  残り $($target - $done) 頭。5秒後に再起動..." -ForegroundColor Yellow
    Start-Sleep -Seconds 5
}

# ---- フェーズ2: 実験自動実行 ----
Write-Host "`n=== 実験開始 ===" -ForegroundColor Cyan

Write-Host "`n[1/3] 新馬モデル再評価..." -ForegroundColor Yellow
& $python scripts/eval_shinba_training.py 2>&1 | Tee-Object -FilePath "logs\eval_shinba_after_fetch.log"

Write-Host "`n[2/3] v19tr 全コース再学習..." -ForegroundColor Yellow
& $python scripts/train_v19_training_no26.py 2>&1 | Tee-Object -FilePath "logs\train_v19tr_after_fetch.log"

Write-Host "`n[3/3] v19tr バックテスト..." -ForegroundColor Yellow
& $python scripts/eval_v19_training_no26.py 2>&1 | Tee-Object -FilePath "logs\eval_v19tr_after_fetch.log"

Write-Host "`n=== 全ステップ完了 ===" -ForegroundColor Green
Write-Host "ログ: logs/eval_shinba_after_fetch.log"
Write-Host "ログ: logs/train_v19tr_after_fetch.log"
Write-Host "ログ: logs/eval_v19tr_after_fetch.log"
