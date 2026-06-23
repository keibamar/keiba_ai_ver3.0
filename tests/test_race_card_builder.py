"""src/datasets/race_card/transform.py, src/logic/prediction/race_card_builder.py のテスト

parse_race_card_info_tokens / fill_race_info_defaults / extract_peds_for_display は
純粋関数なのでオフラインで検証する。

make_race_card は netkeiba.com への実通信が必要なため @pytest.mark.network を付与し、
確定済みの固定race_idについて、既知の期待値との比較で検証する。
"""

import pandas as pd
import pytest

from src.datasets.race_card import transform
from src.logic.prediction import race_card_builder

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


# --- is_waku_decided / blank_rank_df -----------------------------------------------


def test_is_waku_decided_true_when_waku_column_has_values():
    race_card_df = pd.DataFrame({"枠": [1, 2], "馬番": [1, 2]})

    assert transform.is_waku_decided(race_card_df) is True


def test_is_waku_decided_false_when_waku_column_missing():
    race_card_df = pd.DataFrame({"馬番": [1, 2]})

    assert transform.is_waku_decided(race_card_df) is False


def test_is_waku_decided_false_when_waku_column_all_nan():
    race_card_df = pd.DataFrame({"枠": [None, None], "馬番": [1, 2]})

    assert transform.is_waku_decided(race_card_df) is False


def test_blank_rank_df_returns_nan_score_and_rank():
    result = transform.blank_rank_df(3)

    assert result.shape == (3, 2)
    assert list(result.columns) == ["score", "rank"]
    assert result["score"].isna().all()
    assert result["rank"].isna().all()


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


def test_extract_peds_for_display_returns_expected():
    horse_peds_df = _sample_horse_peds_df()

    result = transform.extract_peds_for_display(horse_peds_df.copy())

    assert result.shape == (3, 2)
    # 文字列インデックス("peds_0"等)に対するrenameはno-opのため、行ラベルは変化しない
    assert result.index.tolist() == ["peds_0", "peds_1", "peds_4"]
    assert result.columns.tolist() == ["2021107090", "2021106245"]


# --- make_race_card（エンドツーエンド、実データ） -----------------------------------


@pytest.mark.network
def test_make_race_card_returns_expected():
    race_card_df, race_info_df = race_card_builder.make_race_card(FIXED_RACE_ID)

    print(f"\n--- make_race_card({FIXED_RACE_ID}) ---")
    print(f"race_info_df:\n{race_info_df.to_string()}")
    print(f"race_card_df (shape={race_card_df.shape}):\n{race_card_df.to_string()}")

    assert race_info_df.to_dict("records") == [
        {"race_type": "ダート", "course_len": 1400, "weather": "晴", "ground_state": "良", "class": "未勝利"}
    ]

    assert race_card_df.shape == (16, 16)
    assert race_card_df.columns.tolist() == [
        "枠", "馬番", "馬名", "性齢", "斤量", "騎手", "厩舎", "馬体重(増減)", "所属",
        "horse_id", "jockey_id", "peds_0", "peds_1", "peds_4", "score", "rank",
    ]

    first = race_card_df.iloc[0]
    assert first[["馬名", "horse_id", "peds_0", "peds_1", "peds_4"]].tolist() == [
        "アフロマン", "2021107090", "アルアイン", "リュイールスター", "キングカメハメハ",
    ]
    assert first["rank"] == 16
    assert sorted(race_card_df["rank"].tolist()) == list(range(1, 17))
