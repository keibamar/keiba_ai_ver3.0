"""race_returns（配当結果）データセットの更新オーケストレーション

旧 src/legacy_datasets/race_returns.py の更新系関数群を移植したもの。
race_idの算出は src/managers/race_schedule_dataset_manager.py（Chronicle）、
スクレイピングは src/logic/scraping/netkeiba_scraper.py、
CSV保存は src/managers/race_info_dataset_manager.py を利用する。

旧実装の montly_update_race_returns は monthly_update_race_returns に改名している
（旧実装にはこの新スケジューラを呼び出している箇所はないため、呼び出し元への影響はない）。

旧実装の weekly_update_race_returns はrace_id単位の分割（split_race_returns_csv相当）を
呼んでいなかったが、新実装ではrace_result_scheduler.weekly_update_race_resultsと同様に
update後にsplit_race_returns_csvを呼び出す。
"""

from datetime import date

import pandas as pd

from src.config.constants import PLACE_LIST
from src.logic.scraping import netkeiba_scraper
from src.managers import race_info_dataset_manager, race_schedule_dataset_manager


def update_race_returns_dataset(place_id, day=date.today()):
    """開催コースと日にちを指定して、過去1週間分のrace_returnsデータセットを更新する

    Args:
        place_id (int): 開催コースid
        day (date): 日（初期値：今日）
    """
    race_id_list = race_schedule_dataset_manager.get_past_weekly_id(place_id, day)

    old_race_returns_df = race_info_dataset_manager.get_race_returns_csv(place_id, day.year)
    new_race_returns_df = netkeiba_scraper.scrape_race_returns_dataframe(race_id_list)

    if new_race_returns_df.empty:
        return

    try:
        new_race_returns_df = pd.concat([old_race_returns_df, new_race_returns_df], axis=0)
        race_info_dataset_manager.save_race_returns_dataset(place_id, day.year, new_race_returns_df)
    except Exception as e:
        print(f"{e.__class__.__name__}: {e}")


def make_yearly_race_returns_dataset(place_id, year=date.today().year):
    """開催コースと年を指定して、1年間のrace_returnsデータセットを作成する

    Args:
        place_id (int): 開催コースid
        year (int): 年（初期値：今年）
    """
    race_id_list = race_schedule_dataset_manager.get_year_id_all(place_id, year)
    race_returns_df = netkeiba_scraper.scrape_race_returns_dataframe(race_id_list)
    race_info_dataset_manager.save_race_returns_dataset(place_id, year, race_returns_df)


def make_up_to_day_dataset(place_id, day=date.today()):
    """指定日までの、年間のrace_returnsデータセットを作成する

    Args:
        place_id (int): 開催コースid
        day (date): 日（初期値：今日）
    """
    race_id_list = race_schedule_dataset_manager.get_past_year_id(place_id, day)
    race_returns_df = netkeiba_scraper.scrape_race_returns_dataframe(race_id_list)

    try:
        race_info_dataset_manager.save_race_returns_dataset(place_id, day.year, race_returns_df)
    except Exception as e:
        print(f"{e.__class__.__name__}: {e}")


def weekly_update_race_returns(day=date.today()):
    """指定した日にちから、1週間分のrace_returnsデータセットを更新する

    Args:
        day (date): 日（初期値：今日）
    """
    for place_id in range(1, len(PLACE_LIST) + 1):
        print("[WeeklyUpdate]" + PLACE_LIST[place_id - 1] + " RaceReturns")
        update_race_returns_dataset(place_id, day)
        race_info_dataset_manager.split_race_returns_csv(place_id, day.year)


def monthly_update_race_returns(day=date.today()):
    """指定した日にちまでの、その年のrace_returnsデータセットを更新する

    Args:
        day (date): 日（初期値：今日）
    """
    for place_id in range(1, len(PLACE_LIST) + 1):
        print("[MonthlyUpdate]" + PLACE_LIST[place_id - 1] + " RaceReturns")
        make_up_to_day_dataset(place_id, day)


def make_all_race_returns(year=date.today().year):
    """指定した年までの、すべてのrace_returnsデータセットを作成する

    Args:
        year (int): 年（初期値：今年）
    """
    for y in range(2019, year + 1):
        for place_id in range(1, len(PLACE_LIST) + 1):
            print("[NewMake]" + str(y) + ":" + PLACE_LIST[place_id - 1] + " RaceReturns")
            make_yearly_race_returns_dataset(place_id, y)
