"""horse（血統・血統別レース結果・過去成績）データセットの更新オーケストレーション

旧 src/legacy_datasets/horse_peds.py / peds_results.py / past_performance.py の
更新系関数群を移植したもの。
race_idの算出は src/managers/race_schedule_dataset_manager.py（Chronicle）、
horse_pedsの取得・保存は src/managers/horse_peds_dataset_manager.py、
peds_resultsの取得・保存・集計は src/managers/peds_results_dataset_manager.py、
past_performanceの取得・保存は src/managers/past_performance_dataset_manager.py を利用する。

past_performanceの更新系関数は、旧実装の「netkeiba.comの馬個別ページをスクレイピングして
作成する」方式ではなく、data/race_resultから再構築する
past_performance_dataset_manager.update_past_performance を馬ごとに呼び出す方式に統一している。
"""

from datetime import date

from tqdm import tqdm

from src.config.constants import PLACE_LIST
from src.managers import (
    horse_peds_dataset_manager,
    past_performance_dataset_manager,
    peds_results_dataset_manager,
    race_schedule_dataset_manager,
)


def weekly_update_horse_peds(day=date.today()):
    """指定した日にちから、1週間分のhorse_pedsデータセットを更新する

    Args:
        day (date): 日（初期値：今日）
    """
    for place_id in tqdm(range(1, len(PLACE_LIST) + 1)):
        print("[WeeklyUpdate]" + PLACE_LIST[place_id - 1] + "HorsePeds")
        race_id_list = race_schedule_dataset_manager.get_past_weekly_id(place_id, day)
        horse_id_list = past_performance_dataset_manager.get_horse_id_list_from_race_id_list(race_id_list)
        horse_peds_dataset_manager.make_horse_peds_datasets_from_horse_id_list(horse_id_list)


def monthly_update_horse_peds(day=date.today()):
    """指定した日にちまでの、その年のhorse_pedsデータセットを更新する

    Args:
        day (date): 日（初期値：今日）
    """
    for place_id in tqdm(range(1, len(PLACE_LIST) + 1)):
        print("[MonthlyUpdate]" + PLACE_LIST[place_id - 1] + "HorsePeds")
        race_id_list = race_schedule_dataset_manager.get_past_year_id(place_id, day)
        horse_id_list = past_performance_dataset_manager.get_horse_id_list_from_race_id_list(race_id_list)
        horse_peds_dataset_manager.make_horse_peds_datasets_from_horse_id_list(horse_id_list)


def make_all_horse_peds(year=date.today().year):
    """指定した年までの、すべてのhorse_pedsデータセットを作成する

    Args:
        year (int): 年（初期値：今年）
    """
    for y in range(2019, year + 1):
        for place_id in range(1, len(PLACE_LIST) + 1):
            print("[NewMake]" + str(y) + ":" + PLACE_LIST[place_id - 1] + " HorsePeds")
            race_id_list = race_schedule_dataset_manager.get_year_id_all(place_id, y)
            horse_id_list = past_performance_dataset_manager.get_horse_id_list_from_race_id_list(race_id_list)
            horse_peds_dataset_manager.make_horse_peds_datasets_from_horse_id_list(horse_id_list)


def weekly_update_pedsdata(day=date.today()):
    """指定した日にちから、1週間分のpeds_resultsデータセットを更新する

    Args:
        day (date): 日（初期値：今日）
    """
    for place_id in range(1, len(PLACE_LIST) + 1):
        print("[WeeklyUpdate]" + PLACE_LIST[place_id - 1] + " RaceResults")
        peds_results_dataset_manager.update_peds_dataset(place_id, day)
        peds_results_dataset_manager.merge_pedsdata_with_race_results(place_id, day.year)
        # 集計結果を更新
        peds_results_dataset_manager.aggregate_peds_results(place_id, day.year)
        peds_results_dataset_manager.aggregate_total_peds_results(place_id=place_id, end_year=day.year)


def monthly_update_pedsdata(day=date.today()):
    """指定した日にちまでの、その年のpeds_resultsデータセットを更新する

    Args:
        day (date): 日（初期値：今日）
    """
    for place_id in range(1, len(PLACE_LIST) + 1):
        print("[MonthlyUpdate]" + PLACE_LIST[place_id - 1] + " PedsResults")
        peds_results_dataset_manager.make_peds_dataset_from_race_results(place_id, day.year)
        peds_results_dataset_manager.merge_pedsdata_with_race_results(place_id, day.year)
        # 集計結果を更新
        peds_results_dataset_manager.aggregate_peds_results(place_id, day.year)
        peds_results_dataset_manager.aggregate_total_peds_results(place_id=place_id, end_year=day.year)


def make_all_pedsdata(year=date.today().year):
    """指定した年までの、すべてのpeds_resultsデータセットを作成する

    Args:
        year (int): 年（初期値：今年）
    """
    for y in range(2019, year + 1):
        for place_id in range(1, len(PLACE_LIST) + 1):
            print("[NewMake]" + str(y) + ":" + PLACE_LIST[place_id - 1] + " PedsResults")
            peds_results_dataset_manager.make_peds_dataset_from_race_results(place_id, y)
            peds_results_dataset_manager.merge_pedsdata_with_race_results(place_id, y)
            # 集計結果を更新
            peds_results_dataset_manager.aggregate_peds_results(place_id, y)
            peds_results_dataset_manager.aggregate_total_peds_results(place_id=place_id, end_year=y)


def make_peds_results(year=date.today().year):
    """指定した年のpeds_results集計結果のみを作成する

    Args:
        year (int): 年（初期値：今年）
    """
    for place_id in range(1, len(PLACE_LIST) + 1):
        peds_results_dataset_manager.aggregate_peds_results(place_id, year)


def weekly_update_past_performance(day=date.today()):
    """指定した日にちから、1週間分のpast_performanceデータセットを更新する

    Args:
        day (date): 日（初期値：今日）
    """
    for place_id in tqdm(range(1, len(PLACE_LIST) + 1)):
        print("[WeeklyUpdate]", PLACE_LIST[place_id - 1], "PastPerformance")
        race_id_list = race_schedule_dataset_manager.get_past_weekly_id(place_id, day)
        horse_id_list = past_performance_dataset_manager.get_horse_id_list_from_race_id_list(race_id_list)
        for horse_id in horse_id_list:
            past_performance_dataset_manager.update_past_performance(horse_id)


def monthly_update_past_performance(day=date.today()):
    """指定した日にちまでの、その年のpast_performanceデータセットを更新する

    Args:
        day (date): 日（初期値：今日）
    """
    for place_id in tqdm(range(1, len(PLACE_LIST) + 1)):
        print("[MonthlyUpdate]", PLACE_LIST[place_id - 1], "PastPerformance")
        race_id_list = race_schedule_dataset_manager.get_past_year_id(place_id, day)
        horse_id_list = past_performance_dataset_manager.get_horse_id_list_from_race_id_list(race_id_list)
        for horse_id in horse_id_list:
            past_performance_dataset_manager.update_past_performance(horse_id)


def make_all_past_performance(year=date.today().year):
    """指定した年までの、すべてのpast_performanceデータセットを作成する

    Args:
        year (int): 年（初期値：今年）
    """
    for y in range(2019, year + 1):
        for place_id in range(1, len(PLACE_LIST) + 1):
            print("[NewMake]" + str(y) + ":" + PLACE_LIST[place_id - 1] + " PastPerformance")
            race_id_list = race_schedule_dataset_manager.get_year_id_all(place_id, y)
            horse_id_list = past_performance_dataset_manager.get_horse_id_list_from_race_id_list(race_id_list)
            for horse_id in horse_id_list:
                past_performance_dataset_manager.update_past_performance(horse_id)
