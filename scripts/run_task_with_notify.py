"""タスクスケジューラ用ラッパー：bat を実行し、開始・完了をメール通知する

使い方:
    python scripts\\run_task_with_notify.py <タスク名> <batのパス>
    python scripts\\run_task_with_notify.py --test      # テストメール送信のみ

- 開始時に「▶開始」メールを送信
- bat の出力を logs\\tasks\\<タスク名>_YYYYMMDD_HHMMSS.log に保存
- 終了時に判定結果（✅成功 / ⚠要確認 / ❌失敗）とログ末尾を付けてメール送信
  - ❌失敗  : bat の終了コードが 0 以外
  - ⚠要確認: 終了コードは 0 だが、ログにエラーらしき行がある
    （既存 bat は途中で失敗しても終了コード 0 になることが多いため、ログ内容でも判定）
- メール送信に失敗しても bat の実行自体は止めない
- 終了コードは bat のものをそのまま返す（タスクスケジューラの「前回の実行結果」に反映）

標準ライブラリ + python-dotenv のみ使用（重いモジュールを import しない）。
"""

import datetime
import os
import smtplib
import socket
import ssl
import subprocess
import sys
import time
from email.mime.text import MIMEText
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
LOG_DIR = ROOT / "logs" / "tasks"
LOG_KEEP_DAYS = 60
TAIL_LINES = 40
MAX_MATCH_LINES = 20

load_dotenv(ROOT / ".env")

# ログ中にあれば「要確認」とする文字列（大文字小文字は区別する）
ERROR_PATTERNS = [
    "Traceback (most recent call last)",
    "[ERROR]",
    "Error:",           # Python例外名（KeyError: 等）/ bat の "Error: ..."
    "error:",           # git
    "fatal:",           # git
    "! [rejected]",     # git push 拒否
    "エラー",            # WinSCP / 各スクリプトの日本語エラー表示
    "見つかりません",     # WinSCP（サイト名不一致など）
    "接続できません",
    "認証に失敗",
    "Authentication failed",
    "Access is denied",
    "アクセスが拒否",
]


def _decode_line(raw: bytes) -> str:
    """cmd/git/WinSCP は cp932、Python は utf-8 で出力するため両対応でデコードする"""
    for enc in ("utf-8", "cp932"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def send_mail(subject: str, body: str) -> None:
    account = os.environ.get("GMAIL_ACCOUNT", "")
    password = os.environ.get("GMAIL_APP_PASSWORD", "")
    send_to = os.environ.get("GMAIL_SEND_TO", "")
    if not (account and password and send_to):
        print("[notify] GMAIL_* が .env に設定されていないためメール送信をスキップ")
        return

    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = subject
    msg["From"] = account
    msg["To"] = send_to

    for attempt in range(1, 4):
        try:
            with smtplib.SMTP_SSL(
                "smtp.gmail.com", 465,
                context=ssl.create_default_context(), timeout=30,
            ) as server:
                server.login(account, password)
                server.send_message(msg)
            return
        except Exception as e:  # ネットワーク瞬断などはリトライ
            print(f"[notify] メール送信失敗 ({attempt}/3): {e}")
            time.sleep(10 * attempt)


def cleanup_old_logs() -> None:
    limit = time.time() - LOG_KEEP_DAYS * 86400
    for p in LOG_DIR.glob("*.log"):
        try:
            if p.stat().st_mtime < limit:
                p.unlink()
        except OSError:
            pass


def fmt_duration(sec: float) -> str:
    sec = int(sec)
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}時間{m}分"
    if m:
        return f"{m}分{s}秒"
    return f"{s}秒"


def run(task_name: str, bat_path: str) -> int:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    cleanup_old_logs()

    host = socket.gethostname()
    start = datetime.datetime.now()
    log_path = LOG_DIR / f"{task_name}_{start:%Y%m%d_%H%M%S}.log"

    send_mail(
        f"[MAR] ▶開始 {task_name} ({start:%m/%d %H:%M})",
        f"タスク: {task_name}\n"
        f"PC    : {host}\n"
        f"開始  : {start:%Y/%m/%d %H:%M:%S}\n"
        f"bat   : {bat_path}\n"
        f"ログ  : {log_path}\n",
    )

    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"

    lines: list[str] = []
    exit_code = -1
    launch_error = ""
    with open(log_path, "w", encoding="utf-8") as log:
        log.write(f"# {task_name} start {start:%Y/%m/%d %H:%M:%S} on {host}\n")
        log.write(f"# bat: {bat_path}\n\n")
        log.flush()
        try:
            proc = subprocess.Popen(
                ["cmd", "/c", bat_path],
                cwd=str(ROOT), env=env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
            )
            for raw in proc.stdout:
                line = _decode_line(raw).rstrip("\r\n")
                lines.append(line)
                log.write(line + "\n")
                log.flush()
            exit_code = proc.wait()
        except Exception as e:
            launch_error = f"bat の起動に失敗しました: {e}"
            log.write(launch_error + "\n")

    end = datetime.datetime.now()
    duration = fmt_duration((end - start).total_seconds())

    matched = [ln for ln in lines if any(p in ln for p in ERROR_PATTERNS)]

    if launch_error or exit_code != 0:
        status = "❌失敗"
    elif matched:
        status = "⚠要確認"
    else:
        status = "✅成功"

    body = [
        f"結果  : {status}",
        f"タスク: {task_name}",
        f"PC    : {host}",
        f"開始  : {start:%Y/%m/%d %H:%M:%S}",
        f"終了  : {end:%Y/%m/%d %H:%M:%S}（所要 {duration}）",
        f"終了コード: {exit_code}",
        f"ログ  : {log_path}",
    ]
    if launch_error:
        body += ["", launch_error]
    if matched:
        body += ["", f"--- エラーらしき行（{len(matched)}件、先頭{MAX_MATCH_LINES}件まで） ---"]
        body += matched[:MAX_MATCH_LINES]
    body += ["", f"--- ログ末尾 {TAIL_LINES} 行 ---"]
    body += lines[-TAIL_LINES:]

    send_mail(
        f"[MAR] {status} {task_name} ({start:%m/%d %H:%M}, {duration})",
        "\n".join(body) + "\n",
    )
    return exit_code if not launch_error else 1


def main() -> int:
    if len(sys.argv) == 2 and sys.argv[1] == "--test":
        now = datetime.datetime.now()
        send_mail(
            f"[MAR] テスト通知 ({now:%m/%d %H:%M})",
            f"PC: {socket.gethostname()}\nタスク通知メールのテスト送信です。\n",
        )
        print("テストメールを送信しました")
        return 0
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    return run(sys.argv[1], sys.argv[2])


if __name__ == "__main__":
    sys.exit(main())
