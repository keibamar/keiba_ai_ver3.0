"""
fetch_and_run.py  --  training data fetch loop + auto experiment

Usage:
    python scripts/fetch_and_run.py
"""

import os
import subprocess
import sys
import time
from datetime import datetime

PROJECT_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR   = os.path.join(PROJECT_DIR, "scripts")
TRAINING_DIR  = os.path.join(PROJECT_DIR, "data", "training")
LOG_DIR       = os.path.join(PROJECT_DIR, "logs")
PYTHON        = sys.executable
TARGET_COUNT  = 9186
SESSION_START = datetime(2026, 10, 19, 22, 0, 0)

os.makedirs(LOG_DIR, exist_ok=True)


def done_count():
    n = 0
    for f in os.scandir(TRAINING_DIR):
        if f.is_file() and f.name.endswith(".csv"):
            if datetime.fromtimestamp(f.stat().st_mtime) >= SESSION_START:
                n += 1
    return n


def run_script(script, log_name):
    log_path = os.path.join(LOG_DIR, log_name)
    print(f"\n[{datetime.now():%H:%M:%S}] Starting: {script}")
    print(f"  log -> {log_path}")
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
        for line in proc.stdout.strip().splitlines()[-40:]:
            print(line)
    ok = proc.returncode == 0
    print(f"[{datetime.now():%H:%M:%S}] {'OK' if ok else 'ERROR (rc='+str(proc.returncode)+')'}: {script}")
    return ok


# ---- Phase 1: fetch loop ----------------------------------------

print("=" * 60)
print("Phase 1: fetch training data (auto-restart loop)")
print(f"  target: {TARGET_COUNT} horses")
print("=" * 60)

attempt = 0
while True:
    n = done_count()
    print(f"\n[{datetime.now():%H:%M:%S}] done={n}/{TARGET_COUNT}")
    if n >= TARGET_COUNT:
        print("All horses fetched!")
        break
    attempt += 1
    print(f"[{datetime.now():%H:%M:%S}] Run attempt #{attempt}  (remaining ~{TARGET_COUNT - n})")
    subprocess.run(
        [PYTHON, os.path.join(SCRIPTS_DIR, "fetch_training_2026horses.py"), "--no-deadline"],
        cwd=PROJECT_DIR,
    )
    n = done_count()
    if n >= TARGET_COUNT:
        print("All horses fetched!")
        break
    print(f"  still {TARGET_COUNT - n} remaining, restarting in 5s...")
    time.sleep(5)

# ---- Phase 2: experiments ----------------------------------------

print("\n" + "=" * 60)
print("Phase 2: experiments")
print("=" * 60)

print("\n[1/3] Shinba model re-evaluation")
run_script("eval_shinba_training.py", "eval_shinba_after_fetch.log")

print("\n[2/3] v19tr re-training (all courses)")
ok = run_script("train_v19_training_no26.py", "train_v19tr_after_fetch.log")
if not ok:
    print("ERROR: training failed. Stopping.")
    sys.exit(1)

print("\n[3/3] v19tr backtest")
run_script("eval_v19_training_no26.py", "eval_v19tr_after_fetch.log")

print("\n" + "=" * 60)
print("All done.")
print(f"  logs/eval_shinba_after_fetch.log")
print(f"  logs/train_v19tr_after_fetch.log")
print(f"  logs/eval_v19tr_after_fetch.log")
print("=" * 60)
