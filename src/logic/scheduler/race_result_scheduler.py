"""race_result（レース結果）データセットの更新オーケストレーション

旧 src/legacy_datasets/race_results.py の更新系関数群を移植したもの。
race_idの算出は src/managers/race_schedule_dataset_manager.py（Chronicle）、
スクレイピングは src/logic/scraping/netkeiba_scraper.py、
CSV保存は src/managers/race_result_dataset_manager.py を利用する。

旧実装の montly_update_race_results は monthly_update_race_results に改名している
（旧実装にはこの新スケジューラを呼び出している箇所はないため、呼び出し元への影響はない）。
"""

from datetime import date

import numpy as np
import pandas as pd

from src.config.constants import PLACE_LIST
from src.datasets.race_result import transform
from src.logic.scraping import netkeiba_scraper
from src.managers import race_result_dataset_manager, race_schedule_dataset_manager


def update_race_results_dataset(place_id, day=date.today()):
    """開催コースと日にちを指定して、過去1週間分のrace_resultsデータセットを更新する

    Args:
        place_id (int): 開催コースid
        day (date): 日（初期値：今日）
    """
    race_id_list = race_schedule_dataset_manager.get_past_weekly_id(place_id, day)

    old_race_results_df = race_result_dataset_manager.get_race_results_csv(place_id, day.year)
    new_race_results_df = netkeiba_scraper.scrape_race_results_dataframe(race_id_list)

    if new_race_results_df.empty:
        return

    try:
        if not old_race_results_df.empty:
            base_columns = old_race_results_df.columns
            new_race_results_df = new_race_results_df.reindex(columns=base_columns)
        new_race_results_df = new_race_results_df.replace(r"^\s*$", np.nan, regex=True)
        new_race_results_df.fillna(np.nan)
        new_race_results_df = pd.concat([old_race_results_df, new_race_results_df], axis=0)
        new_race_results_df["course_len"] = new_race_results_df["course_len"].apply(transform.to_int)
        race_result_dataset_manager.save_race_results_dataset(place_id, day.year, new_race_results_df)
    except Exception as e:
        print(f"{e.__class__.__name__}: {e}")


def make_yearly_race_results_dataset(place_id, year=date.today().year):
    """開催コースと年を指定して、1年間のrace_resultsデータセットを作成する

    Args:
        place_id (int): 開催コースid
        year (int): 年（初期値：今年）
    """
    race_id_list = race_schedule_dataset_manager.get_year_id_all(place_id, year)

    race_results_df = netkeiba_scraper.scrape_race_results_dataframe(race_id_list)
    race_results_df["course_len"] = race_results_df["course_len"].apply(transform.to_int)

    race_result_dataset_manager.save_race_results_dataset(place_id, year, race_results_df)


def make_up_to_day_dataset(place_id, day=date.today()):
    """指定日までの、年間のrace_resultsデータセットを作成する

    Args:
        place_id (int): 開催コースid
        day (date): 日（初期値：今日）
    """
    race_id_list = race_schedule_dataset_manager.get_past_year_id(place_id, day)

    race_results_df = netkeiba_scraper.scrape_race_results_dataframe(race_id_list)
    race_results_df["course_len"] = race_results_df["course_len"].apply(transform.to_int)

    try:
        race_result_dataset_manager.save_race_results_dataset(place_id, day.year, race_results_df)
    except Exception as e:
        print(f"{e.__class__.__name__}: {e}")


def weekly_update_race_results(day=date.today()):
    """指定した日にちから、1週間分のrace_resultsデータセットを更新する

    Args:
        day (date): 日（初期値：今日）
    """
    for place_id in range(1, len(PLACE_LIST) + 1):
        print("[WeeklyUpdate]" + PLACE_LIST[place_id - 1] + " RaceResults")
        update_race_results_dataset(place_id, day)
        race_result_dataset_manager.export_race_info_per_race(place_id, day.year)
        race_result_dataset_manager.split_race_results_by_year(place_id, day.year)


def monthly_update_race_results(day=date.today()):
    """指定した日にちまでの、その年のrace_resultsデータセットを更新する

    Args:
        day (date): 日（初期値：今日）
    """
    for place_id in range(1, len(PLACE_LIST) + 1):
        print("[MonthlyUpdate]" + PLACE_LIST[place_id - 1] + " RaceResults")
        make_up_to_day_dataset(place_id, day)


def make_all_race_results(year=date.today().year):
    """指定した年までの、すべてのrace_resultsデータセットを作成する

    Args:
        year (int): 年（初期値：今年）
    """
    for y in range(2019, year + 1):
        for place_id in range(1, len(PLACE_LIST) + 1):
            print("[NewMake]" + str(y) + ":" + PLACE_LIST[place_id - 1] + " RaceResults")
            make_yearly_race_results_dataset(place_id, y)
