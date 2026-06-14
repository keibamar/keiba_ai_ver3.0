"""src/logic/calculators/average_calculator.py の出力が旧実装と一致することを
確認するテスト（オフライン）。

旧 src/legacy_datasets/average_time.py の対応関数と同一結果を返すことを確認する。
"""

import os

import numpy as np
import pandas as pd
import pytest

from src.config import paths
from src.config.lists import COURSE_LISTS
from src.legacy_datasets import average_time as old_avg_time
from src.logic.calculators import average_calculator

SAMPLE_PLACE_ID = 2
SAMPLE_PLACE = "02_hakodate"


# --- 平均タイム（average_time.py） --------------------------------------------------


def test_make_average_time_datasets_matches_old():
    path = os.path.join(paths.RACE_RESULT_DATA_PATH, SAMPLE_PLACE, "2019_race_results.csv")
    df_race_results = pd.read_csv(path, dtype=str, index_col=0)

    old_result = old_avg_time.make_average_time_datasets(df_race_results, SAMPLE_PLACE_ID).reset_index(drop=True)
    new_result = average_calculator.make_average_time_datasets(df_race_results, COURSE_LISTS[SAMPLE_PLACE_ID - 1])

    assert not new_result.empty

    # 旧実装の ground_state 列は ["全","良","稍重","重","不"] とハードコードされており
    # "不良"であるべき箇所が"不"になっていた（新実装ではmodel.GROUNDSに合わせて"不良"に修正）
    old_result["ground_state"] = old_result["ground_state"].replace("不", "不良")
    assert old_result.equals(new_result)


# --- タイム差（average_time.py） -----------------------------------------------------


@pytest.mark.parametrize("time_str", ["1:11.2", "1:39.8", "2:00.0"])
def test_get_race_time_msec_matches_old(time_str):
    old_result = old_avg_time.get_race_time_msec(time_str)
    new_result = average_calculator.get_race_time_msec(time_str)

    assert old_result == new_result


def test_get_race_time_msec_nan_matches_old():
    old_result = old_avg_time.get_race_time_msec(np.nan)
    new_result = average_calculator.get_race_time_msec(np.nan)

    assert np.isnan(old_result) and np.isnan(new_result)


@pytest.mark.parametrize(("base_time", "race_time"), [("70000", 71000.0), (70000.0, 69000.0)])
def test_calc_time_diff_matches_old(base_time, race_time):
    old_result = old_avg_time.calc_time_diff(base_time, race_time)
    new_result = average_calculator.calc_time_diff(base_time, race_time)

    assert old_result == new_result


def test_calc_time_diff_nan_matches_old():
    old_result = old_avg_time.calc_time_diff(np.nan, 70000.0)
    new_result = average_calculator.calc_time_diff(np.nan, 70000.0)

    assert np.isnan(old_result) and np.isnan(new_result)
