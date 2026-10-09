"""データセット全馬の調教タイムを一括スクレイピングするスクリプト

- data/horse/horse_peds/ 配下の全 horse_id を対象
- data/training/{horse_id}.csv が既に存在する馬はスキップ
- 連続エラー時は長めに待機してから再試行（レート制限対策）
- --deadline オプション指定時刻（デフォルト 08:50）になったら安全に終了

実行例:
    python scripts/fetch_all_training_data.py
    python scripts/fetch_all_training_data.py --deadline 08:50 --sleep 2.0
"""

import argparse
import glob
import os
import sys
import time
from datetime import datetime, date, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from src.config import paths
from src.logic.scraping.training_scraper import login_netkeiba, scrape_horse_training


# ──────────────────────────────────────────────
# 設定
# ──────────────────────────────────────────────
TRAINING_DATA_PATH = paths.TRAINING_DATA_PATH
HORSE_PEDS_PATH = os.path.join(paths.HORSE_DATA_PATH, "horse_peds")

# 連続エラー閾値: これを超えたら長待機
CONSECUTIVE_ERROR_THRESHOLD = 5
LONG_WAIT_SEC = 60


def get_all_peds_horse_ids() -> list[str]:
    """horse_peds キャッシュに存在する全 horse_id を返す（.csv のみ）。"""
    files = glob.glob(os.path.join(HORSE_PEDS_PATH, "*.csv"))
    ids = [os.path.splitext(os.path.basename(f))[0] for f in files]
    return sorted(ids)


def already_fetched(horse_id: str) -> bool:
    """data/training/{horse_id}.csv が存在すれば True。"""
    return os.path.exists(os.path.join(TRAINING_DATA_PATH, f"{horse_id}.csv"))


def calc_deadline_dt(deadline_time: str) -> datetime:
    """HH:MM 文字列から「次に来るその時刻」の datetime を返す。

    例: 現在 22:00、deadline_time="08:50" → 翌日 08:50 の datetime
    """
    h, m = map(int, deadline_time.split(":"))
    now = datetime.now()
    candidate = now.replace(hour=h, minute=m, second=0, microsecond=0)
    if candidate <= now:
        candidate += timedelta(days=1)
    return candidate


def deadline_reached(deadline_dt: datetime) -> bool:
    """現在時刻が deadline_dt を過ぎていれば True。"""
    return datetime.now() >= deadline_dt


def main():
    parser = argparse.ArgumentParser(description="全馬調教タイム一括スクレイピング")
    parser.add_argument(
        "--deadline", default="08:50",
        help="この時刻（HH:MM）になったら停止（デフォルト: 08:50）"
    )
    parser.add_argument(
        "--sleep", type=float, default=2.0,
        help="リクエスト間隔秒数（デフォルト: 2.0）"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="既存 CSV を上書きして再取得"
    )
    args = parser.parse_args()

    # 認証情報
    login_id = os.environ.get("NETKEIBA_LOGIN_ID", "")
    password = os.environ.get("NETKEIBA_PASSWORD", "")
    if not login_id or not password:
        print("ERROR: NETKEIBA_LOGIN_ID / NETKEIBA_PASSWORD が .env に設定されていません")
        sys.exit(1)

    deadline_dt = calc_deadline_dt(args.deadline)

    print(f"=== 全馬調教タイム取得 ===")
    print(f"停止予定時刻: {deadline_dt.strftime('%Y-%m-%d %H:%M')}  |  sleep: {args.sleep}s  |  force: {args.force}")

    # horse_id リスト収集
    all_ids = get_all_peds_horse_ids()
    if not args.force:
        targets = [hid for hid in all_ids if not already_fetched(hid)]
    else:
        targets = all_ids

    print(f"対象: {len(targets)}頭 / 全{len(all_ids)}頭（既取得スキップ: {len(all_ids) - len(targets)}頭）")

    if not targets:
        print("取得対象なし。終了します。")
        return

    # ログイン
    print("netkeibaにログイン中...")
    try:
        session = login_netkeiba(login_id, password)
    except Exception as e:
        print(f"ERROR: ログイン失敗: {e}")
        sys.exit(1)
    print("ログイン完了")

    os.makedirs(TRAINING_DATA_PATH, exist_ok=True)

    ok = 0
    ng = 0
    skip = 0
    consecutive_errors = 0
    start_time = datetime.now()

    for i, horse_id in enumerate(targets):
        # 期限チェック（各馬の前に確認）
        if deadline_reached(deadline_dt):
            print(f"\n⏰ 停止時刻 {args.deadline} に達しました。終了します。")
            break

        # 進捗表示（100頭ごと）
        if i % 50 == 0:
            elapsed = (datetime.now() - start_time).seconds
            per_horse = elapsed / max(i, 1)
            remaining = len(targets) - i
            eta_sec = remaining * per_horse
            print(
                f"[{i+1}/{len(targets)}] 経過: {elapsed//60}分  "
                f"進捗: ok={ok} ng={ng}  "
                f"残り推定: {eta_sec//60:.0f}分"
            )

        # 取得済みチェック（forceでなければ）
        if not args.force and already_fetched(horse_id):
            skip += 1
            continue

        try:
            df = scrape_horse_training(horse_id, session)
            if not df.empty:
                out_path = os.path.join(TRAINING_DATA_PATH, f"{horse_id}.csv")
                df.to_csv(out_path, index=False, encoding="utf-8-sig")
                ok += 1
                consecutive_errors = 0
            else:
                # データなし（引退馬・新馬等）は空ファイルで記録してスキップ扱い
                out_path = os.path.join(TRAINING_DATA_PATH, f"{horse_id}.csv")
                open(out_path, "w").close()
                skip += 1
                consecutive_errors = 0
        except Exception as e:
            print(f"  [ERROR] {horse_id}: {e}")
            ng += 1
            consecutive_errors += 1
            # 連続エラー: レート制限の可能性があるため長待機後に続行
            if consecutive_errors >= CONSECUTIVE_ERROR_THRESHOLD:
                print(f"  連続エラー {consecutive_errors} 回 → {LONG_WAIT_SEC}s 待機してから再開")
                time.sleep(LONG_WAIT_SEC)
                consecutive_errors = 0
            else:
                time.sleep(args.sleep * 2)
            continue

        time.sleep(args.sleep)

    elapsed_total = (datetime.now() - start_time).seconds
    print(f"\n=== 完了 ===")
    print(f"成功: {ok}頭  スキップ/データなし: {skip}頭  エラー: {ng}頭")
    print(f"所要時間: {elapsed_total // 60}分 {elapsed_total % 60}秒")
    print(f"保存先: {TRAINING_DATA_PATH}")


if __name__ == "__main__":
    main()
