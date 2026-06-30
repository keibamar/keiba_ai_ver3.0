"""race_card（出馬表+score/rank・レース情報・race_time_id_list）のManager層

CSVの読み書きのみを担う。スクレイピング・予想ロジック自体は
src/RacePrediction/race_card.py, make_time_id_list.py に残置する。

旧 src/RacePrediction/race_card.py の save_race_cards/save_race_info_df/get_race_cards、
make_time_id_list.py の save_time_id_list/get_time_id_list の移植先。
"""

import os

import pandas as pd

from src.config import paths
from src.config.constants import PLACE_LIST
from src.utils.file_utils import read_csv_or_empty


def save_race_cards(race_card_df, race_day, race_id):
    """出馬表+score/rankを保存する"""
    str_day = race_day.strftime("%Y%m%d")
    out_dir = os.path.join(paths.RACE_CARD_DATA_PATH, str_day)
    os.makedirs(out_dir, exist_ok=True)
    race_card_df.to_csv(os.path.join(out_dir, f"{race_id}.csv"))


def get_race_cards(race_day, race_id):
    """出馬表+score/rankを取得する"""
    str_day = race_day.strftime("%Y%m%d")
    path = os.path.join(paths.RACE_CARD_DATA_PATH, str_day, f"{race_id}.csv")
    return read_csv_or_empty(path, dtype=None, index_col=0)


def save_race_info_df(race_info_df, race_day, race_id):
    """レース情報（race_type, course_len, weather, ground_state, class等）を保存する"""
    year = race_day.strftime("%Y")
    place_id = int(str(race_id)[4:6])
    out_dir = os.path.join(paths.RACE_INFO_DATA_PATH, PLACE_LIST[place_id - 1], year)
    os.makedirs(out_dir, exist_ok=True)
    race_info_df.to_csv(os.path.join(out_dir, f"{race_id}.csv"))


def get_race_info_csv(race_id):
    """レース情報（race_type, course_len, weather, ground_state, class等）を取得する"""
    year = str(race_id)[:4]
    place_id = int(str(race_id)[4:6])
    path = os.path.join(paths.RACE_INFO_DATA_PATH, PLACE_LIST[place_id - 1], year, f"{race_id}.csv")
    return read_csv_or_empty(path, dtype=str)


def save_time_id_list(race_day, time_id_list):
    """race_time_id_list（race_time, race_id, race_name, grade）を保存する

    grade（G1/G2/G3。対象外はNone）はmake_time_id_listが追加した列。古い保存済み
    CSV（gradeが無い時期のもの）と区別する必要はなく、get_race_time_id_list_dfは
    read_csv_or_emptyでそのまま読むため、grade列が無い古いCSVはgradeを持たない
    DataFrameとして読まれる（呼び出し側はdf.get("grade")相当の存在確認をする）。
    """
    if not any(time_id_list):
        return
    str_day = race_day.strftime("%Y%m%d")
    os.makedirs(paths.RACE_TIME_ID_LIST_PATH, exist_ok=True)
    time_id_list_df = pd.DataFrame(time_id_list, columns=["race_time", "race_id", "race_name", "grade"])
    time_id_list_df.to_csv(os.path.join(paths.RACE_TIME_ID_LIST_PATH, f"{str_day}.csv"))


def get_time_id_list(race_day):
    """race_time_id_listを[race_time, race_id]のリストで取得する"""
    df = get_race_time_id_list_df(race_day)
    if df.empty:
        return []
    return df[["race_time", "race_id"]].values.tolist()


def get_race_time_id_list_df(race_day):
    """race_time_id_listをDataFrameで取得する"""
    str_day = race_day.strftime("%Y%m%d")
    path = os.path.join(paths.RACE_TIME_ID_LIST_PATH, f"{str_day}.csv")
    return read_csv_or_empty(path, dtype=str)
