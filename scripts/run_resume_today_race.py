"""PC再起動などで post_today_race が中断された場合のリカバリスクリプト

X投稿・メール送信なし。レース結果収集とHTML更新のみ行う。
最終レース終了後 30 分で自動終了。

Usage:
    python scripts/run_resume_today_race.py
"""

import os
import sys
import warnings
import subprocess
import datetime
from time import sleep
from datetime import date, timedelta

warnings.simplefilter("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import paths
from src.config.constants import PLACE_LIST
from src.managers import (
    race_card_dataset_manager,
    race_result_dataset_manager,
    race_info_dataset_manager,
)
from src.logic.html_generator import race_page_generator, daily_index_generator
from src.logic.scraping import netkeiba_scraper
from src.logic.scheduler.race_day_scheduler import (
    _scrape_with_retry,
    _update_race_card_from_result,
)


def _has_result(race_id: str) -> bool:
    """レース結果がすでに保存済みかチェック"""
    year = race_id[:4]
    place_id = int(race_id[4:6])
    place_dir = PLACE_LIST[place_id - 1]
    result_path = os.path.join(
        paths.RACE_RESULT_DATA_PATH, place_dir, year, f"{race_id}.csv"
    )
    return os.path.isfile(result_path) and os.path.getsize(result_path) > 100


def _collect_result(race_id: str, race_day: date) -> None:
    """レース結果・払戻を取得して保存する"""
    try:
        results_df = _scrape_with_retry(netkeiba_scraper.scrape_day_race_result, race_id)
        if not results_df.empty:
            race_result_dataset_manager.save_race_result_for_race_id(race_id, results_df)
            _update_race_card_from_result(race_day, race_id, results_df)
            print(f"  結果保存: {race_id}")
        else:
            print(f"  結果なし（まだ確定していない可能性）: {race_id}")
    except Exception:
        print(f"  結果取得エラー: {race_id}", sys.exc_info()[1])

    try:
        df_return = _scrape_with_retry(netkeiba_scraper.scrape_day_race_returns, race_id)
        if not df_return.empty:
            race_info_dataset_manager.save_race_return_for_race_id(race_id, df_return)
            print(f"  払戻保存: {race_id}")
    except Exception:
        print(f"  払戻取得エラー: {race_id}", sys.exc_info()[1])


def _commit():
    """git add . && commit && push（タイムアウト付き）"""
    bat = os.path.join(paths.PROJECT_ROOT, "bat", "Commit", "commit_for_race_cards.bat")
    try:
        subprocess.run([bat], shell=True, check=False, timeout=120)
    except subprocess.TimeoutExpired:
        print("  commit timeout — スキップ")
    except Exception as e:
        print(f"  commit エラー: {e}")


if __name__ == "__main__":
    race_day = date.today()
    date_str = race_day.strftime("%Y%m%d")

    time_id_list = race_card_dataset_manager.get_time_id_list(race_day)
    if not time_id_list:
        print(f"race_time_id_list なし: {race_day}")
        sys.exit(1)

    print(f"[resume_today_race] {race_day} 結果収集再開（X投稿なし）")
    print(f"  対象: {len(time_id_list)} レース / 最終 {time_id_list[-1][0][:2]}:{time_id_list[-1][0][2:]}")

    # 最終レース発走時刻 + 30分 で打ち切り
    last_race_time_str = time_id_list[-1][0]
    last_hour = int(last_race_time_str[:2])
    last_min = int(last_race_time_str[2:])
    deadline = datetime.datetime(
        race_day.year, race_day.month, race_day.day, last_hour, last_min
    ) + timedelta(minutes=30)

    last_race_by_place: dict[int, str] = {}

    while time_id_list:
        now = datetime.datetime.now()
        if now >= deadline:
            print("終了時刻を過ぎたためループを終了します")
            break

        race_time_str, race_id = time_id_list[0]
        race_hour = int(race_time_str[:2])
        race_min = int(race_time_str[2:])
        race_start = datetime.datetime(
            race_day.year, race_day.month, race_day.day, race_hour, race_min
        )

        # 発走+30分経過した時点で結果収集を試みる
        if now >= race_start + timedelta(minutes=30):
            place_id = int(race_id[4:6])

            # 直前レースの結果を取得
            prev_id = last_race_by_place.get(place_id)
            if prev_id and not _has_result(prev_id):
                _collect_result(prev_id, race_day)
                race_page_generator.make_race_card_html(date_str, place_id, prev_id)

            # 今回のレースが発走済みでかつ結果確定していれば収集
            if not _has_result(race_id):
                _collect_result(race_id, race_day)
            else:
                print(f"  スキップ（取得済み）: {race_id}")

            race_page_generator.make_race_card_html(date_str, place_id, race_id)
            daily_index_generator.make_daily_index_page(race_day)

            last_race_by_place[place_id] = race_id
            time_id_list.pop(0)
            print(f"  処理完了: {race_id}  残り {len(time_id_list)} レース")

            # 数レースまとめて処理できた場合のみコミット
            if len(time_id_list) % 3 == 0 or len(time_id_list) == 0:
                _commit()
        else:
            remain = int((race_start + timedelta(minutes=30) - now).total_seconds() // 60)
            print(
                f"  待機中: {race_id} ({race_time_str[:2]}:{race_time_str[2:]}) まで約{remain}分 | 現在 {now.strftime('%H:%M')}"
            )
            sleep(60)

    # 最終処理：最終レースの結果取得
    print("\n最終処理中...")
    for place_id, last_id in last_race_by_place.items():
        if not _has_result(last_id):
            _collect_result(last_id, race_day)
            race_page_generator.make_race_card_html(date_str, place_id, last_id)

    race_page_generator.make_daily_race_card_html(race_day)
    print("HTML全体を再生成しました")

    _commit()
    print("\n[resume_today_race] 完了")
    print("※ ConoHaへのアップロードは手動で bat/Deploy/upload_to_conoha.bat を実行してください")
