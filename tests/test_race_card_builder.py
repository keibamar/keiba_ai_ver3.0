"""src/datasets/race_card/transform.py, src/logic/prediction/race_card_builder.py のテスト

parse_race_card_info_tokens / fill_race_info_defaults / extract_peds_for_display は
純粋関数なのでオフラインで検証する。extract_peds_for_display は旧
src/RacePrediction/race_card.py の同名関数と出力が一致することを確認する。

make_race_card は netkeiba.com への実通信が必要なため @pytest.mark.network を付与し、
確定済みの固定race_idを使って旧 src/RacePrediction/race_card.make_race_card と
新 race_card_builder.make_race_card の出力が一致することを確認する。
"""

import os
import sys

import pandas as pd
import pytest

from src.datasets.race_card import transform
from src.logic.prediction import race_card_builder

# old_race_card (src/RacePrediction/race_card.py) は内部で `import day_race_prediction` という
# sibling importを行うため、src/RacePrediction自体をsys.pathに追加する必要がある
RACE_PREDICTION_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src", "RacePrediction")
if RACE_PREDICTION_PATH not in sys.path:
    sys.path.insert(0, RACE_PREDICTION_PATH)

from src.RacePrediction import race_card as old_race_card  # noqa: E402

# 2024年1月27日 東京1回1日目1R（確定済みのレース。出走馬の過去成績・血統データはキャッシュ済み）
FIXED_RACE_ID = "202405010101"


# --- parse_race_card_info_tokens -------------------------------------------------


def test_parse_race_card_info_tokens():
    info = ["3歳未勝利", "10", "05発走", "ダ1400m", "左", "天候", "晴", "馬場", "良", "サラ系3歳", "未勝利", "混", "指", "馬齢", "16頭"]

    result = transform.parse_race_card_info_tokens(info)

    assert result == {
        "class": "未勝利",
        "race_type": "ダート",
        "course_len": 1400,
        "ground_state": "良",
        "weather": "晴",
    }


def test_parse_race_card_info_tokens_turf_and_obstacle():
    assert transform.parse_race_card_info_tokens(["芝", "2000m"])["race_type"] == "芝"
    assert transform.parse_race_card_info_tokens(["障害", "3000m"])["race_type"] == "障害"


# --- fill_race_info_defaults ------------------------------------------------------


def test_fill_race_info_defaults_adds_missing_columns():
    race_info_df = pd.DataFrame([{"race_type": "芝", "course_len": 2000, "class": "オープン"}])

    result = transform.fill_race_info_defaults(race_info_df)

    assert result.at[0, "weather"] == "-"
    assert result.at[0, "ground_state"] == "良"


def test_fill_race_info_defaults_fills_nan_values():
    race_info_df = pd.DataFrame([{"race_type": "芝", "course_len": 2000, "weather": None, "ground_state": None}])

    result = transform.fill_race_info_defaults(race_info_df)

    assert result.at[0, "weather"] == "-"
    assert result.at[0, "ground_state"] == "良"


# --- extract_peds_for_display ------------------------------------------------------


def _sample_horse_peds_df():
    columns = ["2021107090", "2021106245"]
    index = [f"peds_{i}" for i in range(62)]
    data = {col: [f"{col}_{i}" for i in range(62)] for col in columns}
    return pd.DataFrame(data, index=index)


def test_extract_peds_for_display_matches_old():
    horse_peds_df = _sample_horse_peds_df()

    old_result = old_race_card.extract_peds_for_display(horse_peds_df.copy())
    new_result = transform.extract_peds_for_display(horse_peds_df.copy())

    assert old_result.equals(new_result)
    # 文字列インデックス("peds_0"等)に対するrenameはno-opのため、行ラベルは変化しない
    assert new_result.index.tolist() == ["peds_0", "peds_1", "peds_4"]


# --- make_race_card（エンドツーエンド、実データ） -----------------------------------


@pytest.mark.network
def test_make_race_card_matches_old():
    old_result = old_race_card.make_race_card(FIXED_RACE_ID)
    new_result = race_card_builder.make_race_card(FIXED_RACE_ID)

    assert isinstance(old_result, tuple)
    assert isinstance(new_result, tuple)

    old_card_df, old_info_df = old_result
    new_card_df, new_info_df = new_result

    assert not old_card_df.empty
    assert not new_card_df.empty
    assert old_info_df.equals(new_info_df)
    assert old_card_df.columns.tolist() == new_card_df.columns.tolist()
    assert old_card_df.equals(new_card_df)
