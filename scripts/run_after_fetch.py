"""調教データ取得完了を待って実験を自動実行するスクリプト

fetch_training_2026horses.py (PID 20244) の終了を待ち、
以下を順番に実行する:
  1. eval_shinba_training.py  — 新馬モデル再評価
  2. train_v19_training_no26.py — v19tr 全コース再学習
  3. eval_v19_training_no26.py  — v19tr バックテスト

実行:
    python scripts/run_after_fetch.py
"""

import os
import subprocess
import sys
import time
from datetime import datetime

SCRIPTS_DIR    = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR    = os.path.dirname(SCRIPTS_DIR)
PYTHON         = sys.executable
LOG_DIR        = os.path.join(PROJECT_DIR, "logs")
TRAINING_DIR   = os.path.join(PROJECT_DIR, "data", "training")
TARGET_COUNT   = 9186   # 2026年出走馬総数
IDLE_MINUTES   = 8      # この分間更新がなければ「完了」と判定
os.makedirs(LOG_DIR, exist_ok=True)


def fetch_done() -> bool:
    """取得済み頭数が TARGET_COUNT に達したか、IDLE_MINUTES 間更新がなければ True。"""
    session_start = datetime(2026, 10, 19, 22, 0, 0)
    files = [
        f for f in os.scandir(TRAINING_DIR)
        if f.is_file() and f.name.endswith(".csv")
        and datetime.fromtimestamp(f.stat().st_mtime) >= session_start
    ]
    done_count = len(files)
    if done_count >= TARGET_COUNT:
        return True
    if not files:
        return False
    latest_mtime = max(f.stat().st_mtime for f in files)
    idle_sec = time.time() - latest_mtime
    return idle_sec >= IDLE_MINUTES * 60


def run(script: str, log_name: str):
    log_path = os.path.join(LOG_DIR, log_name)
    print(f"\n{'='*60}")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 実行開始: {script}")
    print(f"  ログ: {log_path}")
    print('='*60)
    with open(log_path, "w", encoding="utf-8") as flog:
        proc = subprocess.run(
            [PYTHON, os.path.join(SCRIPTS_DIR, script)],
            cwd=PROJECT_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        flog.write(proc.stdout)
        # 末尾20行を画面にも表示
        lines = proc.stdout.strip().splitlines()
        for line in lines[-30:]:
            print(line)
    if proc.returncode != 0:
        print(f"\n[ERROR] {script} が終了コード {proc.returncode} で失敗しました")
        print("  ログを確認してください:", log_path)
        return False
    print(f"\n[OK] {script} 完了")
    return True


def main():
    print("=" * 60)
    print("調教データ取得完了待ち → 実験自動実行")
    print(f"完了判定: {TARGET_COUNT}頭到達 または {IDLE_MINUTES}分間更新なし")
    print("=" * 60)

    # 取得完了を待つ（ファイル更新ベース）
    check_interval = 60
    waited = 0
    while not fetch_done():
        session_start = datetime(2026, 10, 19, 22, 0, 0)
        done_count = sum(
            1 for f in os.scandir(TRAINING_DIR)
            if f.is_file() and f.name.endswith(".csv")
            and datetime.fromtimestamp(f.stat().st_mtime) >= session_start
        )
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 取得中... {done_count}/{TARGET_COUNT}頭 ({waited//60}分経過)", flush=True)
        time.sleep(check_interval)
        waited += check_interval

    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] 取得完了を検知しました")
    time.sleep(10)  # ファイル書き込みの完了を待つ

    # ---- Step 1: 新馬モデル再評価 ----
    ok = run("eval_shinba_training.py", "eval_shinba_after_fetch.log")
    if not ok:
        print("新馬評価に失敗しましたが、v19tr 実験は続行します")

    # ---- Step 2: v19tr 全コース再学習 ----
    ok = run("train_v19_training_no26.py", "train_v19tr_after_fetch.log")
    if not ok:
        print("v19tr 学習に失敗しました。終了します")
        sys.exit(1)

    # ---- Step 3: v19tr バックテスト ----
    run("eval_v19_training_no26.py", "eval_v19tr_after_fetch.log")

    print("\n" + "=" * 60)
    print("全ステップ完了")
    print(f"  新馬評価ログ:    logs/eval_shinba_after_fetch.log")
    print(f"  v19tr学習ログ:   logs/train_v19tr_after_fetch.log")
    print(f"  v19trバックテスト: logs/eval_v19tr_after_fetch.log")
    print("=" * 60)


if __name__ == "__main__":
    main()
