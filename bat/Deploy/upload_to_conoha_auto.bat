@echo off
REM Upload public_html/ to the ConoHa document root (sync, changed files only).
REM Same as upload_to_conoha.bat, but without the trailing "pause" so it can be
REM called unattended from the other automation bats (タスクスケジューラから
REM HTML更新系batの最後に呼ばれる)。WinSCPサイトのセットアップはupload_to_conoha.bat
REM 側のコメントを参照。

set WINSCP_PATH="C:\Program Files (x86)\WinSCP\WinSCP.com"
set SITE_NAME=mar-keiba
set REMOTE_PATH=/public_html/mar-keiba.com

cd C:\keiba_ai\keiba_ai_ver3.0

%WINSCP_PATH% /command ^
  "open %SITE_NAME%" ^
  "synchronize remote public_html %REMOTE_PATH%" ^
  "exit"

echo.
echo Upload finished.
