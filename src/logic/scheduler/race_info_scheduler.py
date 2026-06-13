"""race_info（レース基本情報・集計）データセットの更新オーケストレーション

旧 src/legacy_datasets/analysis_race_info.py の weekly_update_* 関数群を移植したもの。
集計の取得・保存は src/managers/race_info_dataset_manager.py を利用する。
"""

from datetime import date

from src.config.constants import PLACE_LIST
from src.managers import race_info_dataset_manager


def weekly_update_horse_name_id_map(year=date.today().year):
    """全開催場について、race_resultsから horse_id_map.csv を更新する

    Args:
        year (int): 年（初期値：今年）
    """
    for place_id in range(1, len(PLACE_LIST) + 1):
        race_info_dataset_manager.update_horse_name_id_map_from_results(place_id, year)


def weekly_update_average_pops(year=date.today().year):
    """全開催場について、平均人気の集計結果を更新する

    Args:
        year (int): 年（初期値：今年）
    """
    for place_id in range(1, len(PLACE_LIST) + 1):
        race_info_dataset_manager.update_average_pops(place_id, year)


def weekly_update_winners_weight(year=date.today().year):
    """全開催場について、勝ち馬の平均馬体重の集計結果を更新する

    Args:
        year (int): 年（初期値：今年）
    """
    for place_id in range(1, len(PLACE_LIST) + 1):
        race_info_dataset_manager.update_winners_weight(place_id, year)


def weekly_update_average_frame_and_horse(year=date.today().year):
    """全開催場について、枠番・馬番の集計結果を更新する

    Args:
        year (int): 年（初期値：今年）
    """
    for place_id in range(1, len(PLACE_LIST) + 1):
        race_info_dataset_manager.update_average_frame_and_horse(place_id, year)


def weekly_update_winner_time(year=date.today().year):
    """全開催場について、勝ち馬の上り/通過の集計結果を更新する

    Args:
        year (int): 年（初期値：今年）
    """
    for place_id in range(1, len(PLACE_LIST) + 1):
        race_info_dataset_manager.update_winner_time(place_id, year)


def weekly_update_average_time(year=date.today().year):
    """全開催場について、平均タイムの集計結果（年間・全期間）を更新する

    Args:
        year (int): 年（初期値：今年）
    """
    for place_id in range(1, len(PLACE_LIST) + 1):
        race_info_dataset_manager.update_annual_average_time(place_id, year)
        race_info_dataset_manager.update_total_average_time(place_id, year)
