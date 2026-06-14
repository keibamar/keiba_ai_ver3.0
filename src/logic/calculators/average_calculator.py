"""平均タイム・タイム差の計算ロジック（純粋関数のみ・I/Oなし）

旧 src/datasets/race_info/transform.py（さらに遡ると旧
src/legacy_datasets/average_time.py）の平均タイム計算ロジックを移植したもの。
"""

import datetime
import math
import re

import numpy as np
import pandas as pd

from src.config.constants import GROUND_STATE_LIST
from src.datasets.race_info import model


# --- 平均タイム ---------------------------------------------------------------


def extract_course_race_results(race_type, course_len, race_results_df):
    """race_resultsデータセットから該当コース・距離の勝ち馬の行を抽出する"""
    course_race_results = race_results_df[race_results_df["race_type"] == str(race_type)]
    course_race_results = course_race_results[course_race_results["course_len"] == str(course_len)]
    return course_race_results[course_race_results["着順"] == "1"]


def calc_avg_time(time_data):
    """走破時計（文字列）のSeriesから平均タイム(ms)を算出する。データが無い場合はNaTを返す"""
    if len(time_data) == 0:
        return np.timedelta64("NaT")

    time_format = "%H:%M:%S.%f"
    for i in range(len(time_data)):
        time_data[i] = datetime.datetime.strptime(time_data.iloc[i], time_format)

    time_data = time_data.astype("datetime64[ms]").to_numpy()
    base_time = np.datetime64(0, "ms")
    avg_time = ((time_data - base_time) % np.timedelta64(1, "D")).mean()
    return avg_time.astype(int)


def get_avg_time_list_from_race_results_df(df_course_race_results):
    """全馬場状態・各馬場状態（GROUND_STATE_LIST）ごとの平均タイムのリストを作成する"""
    avg_time_list = []

    all_time = df_course_race_results["タイム"].reset_index(drop=True)
    avg_time_list.append(calc_avg_time(all_time))

    for ground_state in GROUND_STATE_LIST:
        df_temp = df_course_race_results[df_course_race_results["ground_state"] == ground_state]
        time_temp = df_temp["タイム"].reset_index(drop=True)
        avg_time_list.append(calc_avg_time(time_temp))

    return avg_time_list


def make_avg_time_dataset(race_type, course_len, class_name, avg_time_list):
    """平均タイムのリストから avg_time データセットを作成する

    ground_state列は model.GROUNDS（"全"+GROUND_STATE_LIST）に対応する。
    旧実装ではこの列が ["全","良","稍重","重","不"] とハードコードされ"不良"であるべき
    箇所が"不"になっていたが、新実装ではGROUNDSに合わせて"不良"に修正している。
    """
    avg_time = pd.DataFrame({"avg_time": avg_time_list})
    course_data = pd.DataFrame({
        "race_type": [str(race_type)] * len(model.GROUNDS),
        "course_len": [str(course_len)] * len(model.GROUNDS),
        "ground_state": model.GROUNDS,
        "class": [class_name] * len(model.GROUNDS),
    })
    return pd.concat([course_data, avg_time], axis=1)


def make_average_time_datasets(df_race_results, courses):
    """race_resultsからaverage_timeデータセットを作成する"""
    df_avg_time = pd.DataFrame()
    for race_type, course_len in courses:
        df_course = extract_course_race_results(race_type, course_len, df_race_results)

        all_avg_time = get_avg_time_list_from_race_results_df(df_course)
        df_avg_time = pd.concat([df_avg_time, make_avg_time_dataset(race_type, course_len, "all", all_avg_time)])

        for class_name in model.CLASSES[1:]:
            df_class = df_course[df_course["class"] == class_name]
            class_avg_time = get_avg_time_list_from_race_results_df(df_class)
            df_avg_time = pd.concat(
                [df_avg_time, make_avg_time_dataset(race_type, course_len, class_name, class_avg_time)]
            )

    return df_avg_time.reset_index(drop=True)


# --- タイム差 -----------------------------------------------------------------


def get_race_time_msec(time_str):
    """走破時計をmsecに変換する

    Args:
        time_str (str): race_resultの走破時計

    Returns:
        race_time(float): race_time(msec)
    """
    if type(time_str) is str:
        time_format = "%M%S%f"
        time_str = re.sub(r"\D", "", "0" + time_str)
        race_time = datetime.datetime.strptime(time_str, time_format)
        return race_time.minute * 60 * 1000 + race_time.second * 1000 + race_time.microsecond / 100000

    elif math.isnan(time_str):
        return np.nan


def calc_time_diff(base_time, time):
    """基準タイムとの差分を計算する(msec)

    Args:
        base_time (str): 基準タイム
        time (float): 走破タイム

    Returns:
        diff_time(float): 基準との差分タイム
    """
    if type(base_time) is str:
        base_time = int(base_time)

    if math.isnan(base_time) or math.isnan(time):
        return np.nan
    else:
        base_time = int(base_time)
        return float((base_time - time) / base_time)
