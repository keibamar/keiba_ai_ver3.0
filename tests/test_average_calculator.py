"""src/logic/calculators/average_calculator.py のテスト（オフライン）。

src/logic/calculators/average_calculator.py の各関数を新実装単体で検証する。
"""

import os

import numpy as np
import pandas as pd
import pytest

from src.config import paths
from src.config.lists import COURSE_LISTS
from src.logic.calculators import average_calculator

SAMPLE_PLACE_ID = 2
SAMPLE_PLACE = "02_hakodate"


# --- 平均タイム（make_average_time_datasets） ---------------------------------------


def test_make_average_time_datasets_returns_expected():
    path = os.path.join(paths.RACE_RESULT_DATA_PATH, SAMPLE_PLACE, "2019_race_results.csv")
    df_race_results = pd.read_csv(path, dtype=str, index_col=0)

    result = average_calculator.make_average_time_datasets(df_race_results, COURSE_LISTS[SAMPLE_PLACE_ID - 1])

    print(f"\n--- make_average_time_datasets(place={SAMPLE_PLACE}, year=2019) ---")
    print(f"  shape: {result.shape}")
    print(result.head().to_string())

    assert result.shape == (280, 5)
    assert result.columns.tolist() == ["race_type", "course_len", "ground_state", "class", "avg_time"]
    assert result["ground_state"].unique().tolist() == ["全", "良", "稍重", "重", "不良"]

    first = result.iloc[0]
    assert first[["race_type", "course_len", "ground_state", "class"]].tolist() == ["芝", "1000", "全", "all"]
    assert first["avg_time"] == 58100
    assert pd.isna(result.iloc[2]["avg_time"])


# --- タイム差 ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("time_str", "expected"),
    [("1:11.2", 71002.0), ("1:39.8", 99008.0), ("2:00.0", 120000.0)],
)
def test_get_race_time_msec_returns_expected(time_str, expected):
    assert average_calculator.get_race_time_msec(time_str) == expected


def test_get_race_time_msec_nan_returns_nan():
    assert np.isnan(average_calculator.get_race_time_msec(np.nan))


@pytest.mark.parametrize(
    ("base_time", "race_time", "expected"),
    [("70000", 71000.0, -0.014285714285714285), (70000.0, 69000.0, 0.014285714285714285)],
)
def test_calc_time_diff_returns_expected(base_time, race_time, expected):
    assert average_calculator.calc_time_diff(base_time, race_time) == expected


def test_calc_time_diff_nan_returns_nan():
    assert np.isnan(average_calculator.calc_time_diff(np.nan, 70000.0))
