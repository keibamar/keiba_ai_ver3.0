"""2026年出走馬の調教データを再取得するスクリプト

2026年 race_results CSV から horse_id を収集し、
data/training/{horse_id}.csv を上書き保存する。

既存CSVがあっても上書きするので、2026年分の調教データが追加される。

実行例:
    python scripts/fetch_training_2026horses.py
    python scripts/fetch_training_2026horses.py --deadline 08:50 --sleep 3.0

オプション:
    --deadline HH:MM  停止時刻 (デフォルト: 08:50)
    --no-deadline     時刻停止なし
    --sleep           リクエスト間隔秒数 (デフォルト: 3.0)
    --dry-run         対象馬数確認のみ (スクレイピングしない)
"""

import argparse
import os
import sys
import time
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

import pandas as pd

from src.config import paths
from src.logic.scraping.training_scraper import login_netkeiba, scrape_horse_training

TRAINING_DATA_PATH = paths.TRAINING_DATA_PATH
RACE_RESULTS_BASE  = r"C:\keiba_ai\keiba_ai_ver2.0\data\RaceResults"

CONSECUTIVE_ERROR_THRESHOLD = 5
LONG_WAIT_SEC = 60


def collect_2026_horse_ids() -> list[str]:
    horse_ids = set()
    for place_dir in sorted(os.listdir(RACE_RESULTS_BASE)):
        path = os.path.join(RACE_RESULTS_BASE, place_dir, "2026_race_results.csv")
        if not os.path.isfile(path):
            continue
        try:
            df = pd.read_csv(path, index_col=0, dtype=str, low_memory=False)
            if "horse_id" not in df.columns:
                continue
            ids = df["horse_id"].dropna().str.strip()
            ids = ids[ids != "nan"]
            horse_ids.update(ids.tolist())
        except Exception:
            pass
    return sorted(horse_ids)


def calc_deadline_dt(deadline_time: str) -> datetime:
    h, m = map(int, deadline_time.split(":"))
    now = datetime.now()
    candidate = now.replace(hour=h, minute=m, second=0, microsecond=0)
    if candidate <= now:
        candidate += timedelta(days=1)
    return candidate


def main():
    parser = argparse.ArgumentParser(description="2026年出走馬の調教データ再取得")
    parser.add_argument("--deadline", default="08:50")
    parser.add_argument("--no-deadline", action="store_true")
    parser.add_argument("--sleep", type=float, default=3.0)
    parser.add_argument("--dry-run", action="store_true",
                        help="対象馬数確認のみ (実際には取得しない)")
    args = parser.parse_args()

    print("=== 2026年出走馬 調教データ再取得 ===")

    print("horse_id 収集中...")
    all_targets = collect_2026_horse_ids()

    # 今回の取得セッション（2026-10-19 22:00以降）で更新済みのファイルはスキップ
    session_start = pd.Timestamp("2026-10-19 22:00:00")
    already_done = set()
    for hid in all_targets:
        path = os.path.join(TRAINING_DATA_PATH, f"{hid}.csv")
        if os.path.isfile(path):
            mtime = pd.Timestamp(os.path.getmtime(path), unit="s", tz="Asia/Tokyo").tz_localize(None)
            if mtime >= session_start:
                already_done.add(hid)

    targets = [hid for hid in all_targets if hid not in already_done]
    print(f"全対象: {len(all_targets)} 頭  取得済みスキップ: {len(already_done)} 頭  残り: {len(targets)} 頭")

    eta_min = len(targets) * args.sleep / 60
    print(f"推定時間: 約 {eta_min:.0f} 分 ({eta_min/60:.1f} 時間) @ {args.sleep}秒/頭")

    if args.dry_run:
        print("\n[dry-run] ここで終了します")
        return

    deadline_dt = None if args.no_deadline else calc_deadline_dt(args.deadline)
    if deadline_dt:
        print(f"停止予定: {deadline_dt.strftime('%Y-%m-%d %H:%M')}")

    login_id = os.environ.get("NETKEIBA_LOGIN_ID", "")
    password  = os.environ.get("NETKEIBA_PASSWORD", "")
    if not login_id or not password:
        print("ERROR: .env に NETKEIBA_LOGIN_ID / NETKEIBA_PASSWORD がありません")
        sys.exit(1)

    print("netkeibaにログイン中...")
    try:
        session = login_netkeiba(login_id, password)
    except Exception as e:
        print(f"ERROR: ログイン失敗: {e}")
        sys.exit(1)
    print("ログイン完了\n")

    os.makedirs(TRAINING_DATA_PATH, exist_ok=True)

    ok = ng = 0
    consecutive_errors = 0
    start_time = datetime.now()

    for i, horse_id in enumerate(targets):
        if deadline_dt is not None and datetime.now() >= deadline_dt:
            print(f"\n停止時刻 {args.deadline} に達しました。終了します。")
            break

        if i % 100 == 0:
            elapsed = (datetime.now() - start_time).seconds
            per_horse = elapsed / max(i, 1)
            remaining = len(targets) - i
            eta_sec = remaining * per_horse
            print(
                f"[{i+1}/{len(targets)}] 経過: {elapsed//60}分  "
                f"ok={ok} ng={ng}  残り推定: {eta_sec//60:.0f}分"
            )

        try:
            df = scrape_horse_training(horse_id, session)
            out_path = os.path.join(TRAINING_DATA_PATH, f"{horse_id}.csv")
            if not df.empty:
                df.to_csv(out_path, index=False, encoding="utf-8-sig")
            else:
                # データなし = 空ファイルを保存してスキップ対象にする
                open(out_path, "w").close()
            ok += 1
            consecutive_errors = 0
        except Exception as e:
            ng += 1
            consecutive_errors += 1
            print(f"  ERROR [{horse_id}]: {e}")
            if consecutive_errors >= CONSECUTIVE_ERROR_THRESHOLD:
                print(f"  連続エラー {CONSECUTIVE_ERROR_THRESHOLD} 回。{LONG_WAIT_SEC}秒待機...")
                time.sleep(LONG_WAIT_SEC)
                consecutive_errors = 0

        time.sleep(args.sleep)

    elapsed = (datetime.now() - start_time).seconds
    print(f"\n=== 完了 ===")
    print(f"取得: {ok} 頭  エラー: {ng} 頭  経過: {elapsed//60}分")


if __name__ == "__main__":
    main()
