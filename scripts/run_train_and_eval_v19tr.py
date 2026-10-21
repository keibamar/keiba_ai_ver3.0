"""v19tr 再学習 → バックテスト評価 シーケンシャル実行"""
import subprocess
import sys
import time
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(ROOT, "logs")

def run(script_name, log_name):
    log_path = os.path.join(LOG_DIR, log_name)
    cmd = [sys.executable, os.path.join(ROOT, "scripts", script_name)]
    print(f"\n{'='*60}")
    print(f"開始: {script_name}")
    print(f"ログ: {log_path}")
    print(f"{'='*60}")
    t0 = time.time()
    with open(log_path, "w", encoding="utf-8") as f:
        result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, cwd=ROOT)
    elapsed = (time.time() - t0) / 60
    print(f"完了: {script_name}  exit={result.returncode}  経過={elapsed:.1f}分")
    return result.returncode

if __name__ == "__main__":
    rc1 = run("train_v19_training_no26.py", "train_v19tr_no26.log")
    if rc1 != 0:
        print(f"再学習でエラー(exit={rc1})。評価はスキップします。")
        sys.exit(rc1)
    rc2 = run("eval_v19_training_no26.py", "eval_v19tr_after_horseid.log")
    sys.exit(rc2)
