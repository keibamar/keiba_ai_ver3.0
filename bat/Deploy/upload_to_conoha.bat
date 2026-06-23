@echo off
REM Upload public_html/ to the ConoHa document root (sync, changed files only).
REM
REM One-time setup:
REM   1. Install WinSCP (https://winscp.net/).
REM   2. In WinSCP, create a "New Site" with ConoHa's SFTP/FTP info
REM      (host, username, password, port) and save it.
REM      Use a site name that matches SITE_NAME below.
REM      (The password is stored by WinSCP, not written in this file.)
REM   3. Update WINSCP_PATH / SITE_NAME / REMOTE_PATH below if needed.
REM
REM "synchronize remote" only transfers changed/new files (not everything
REM every time). -delete is intentionally omitted until this is verified
REM to work, since a wrong REMOTE_PATH + -delete could remove remote files
REM that have no local counterpart. Add -delete later once confirmed safe.

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
pause
