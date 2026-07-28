@echo off
schtasks /create /tn "\keiba\night_sleep" /tr "rundll32.exe powrprof.dll,SetSuspendState 0,1,0" /sc daily /st 22:00 /rl highest /f
if %errorlevel% == 0 (
    echo.
    echo タスク登録完了！毎日22:00に自動スリープします。
) else (
    echo.
    echo エラーが発生しました。このファイルを右クリック→管理者として実行してください。
)
pause
