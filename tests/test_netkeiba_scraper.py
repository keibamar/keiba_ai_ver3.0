"""src/logic/scraping/netkeiba_scraper.py のテスト

parse_race_info_tokens は純粋関数なのでオフラインで検証する。
scrape_race_results / scrape_day_race_result / scrape_race_returns_dataframe /
scrape_race_card はnetkeiba.comへの実通信が必要なため @pytest.mark.network を付与し、
確定済みの固定race_id（FIXED_RACE_ID）について、既知の期待値との比較で検証する。

scrape_race_returns_dataframeについては、旧 src/legacy_datasets/race_returns.py の
同名関数はformat_type_returns_dataframeの戻り値に対する.set_index(0)がKeyErrorとなり
処理が失敗するため（詳細はnetkeiba_scraper.scrape_race_returns_dataframeのdocstring参照）、
新旧比較ではなく、data/RaceReturns/05_tokyo/2024_race_returns.csv に保存済みの結果から
判明している期待値との比較で検証する。
"""

import pandas as pd
import pytest

from src.datasets.race_info import model as race_info_model
from src.datasets.race_result import transform
from src.logic.scraping import netkeiba_scraper

# 2024年1月27日 東京1回1日目1R（確定済みのレース結果ページ）
FIXED_RACE_ID = "202405010101"


def test_parse_race_info_tokens():
    tokens = ["3歳未勝利", "ダート", "1400m", "晴", "良", "2024年01月27日", "10時05分"]

    info = transform.parse_race_info_tokens(tokens)

    assert info["race_type"] == "ダート"
    assert info["course_len"] == 1400
    assert info["class"] == "未勝利"
    assert info["weather"] == "晴"
    assert info["ground_state"] == "良"
    assert info["date"] == "2024年01月27日"


def test_parse_race_info_tokens_handles_turf_and_obstacle():
    assert transform.parse_race_info_tokens(["芝", "2000m", "3歳以上オープン"])["race_type"] == "芝"
    assert transform.parse_race_info_tokens(["障害3000m", "障害3歳以上オープン"])["race_type"] == "障害"


@pytest.mark.network
def test_scrape_race_results_returns_expected():
    df = netkeiba_scraper.scrape_race_results(FIXED_RACE_ID)

    print(f"\n--- scrape_race_results({FIXED_RACE_ID}) ---")
    print(f"shape: {df.shape}")
    print(f"columns: {df.columns.tolist()}")
    print(df.to_string())

    assert df.shape == (16, 23)
    assert df.columns.tolist() == [
        "着順", "枠番", "馬番", "馬名", "性齢", "斤量", "騎手", "タイム", "着差", "通過", "上り",
        "単勝", "人気", "馬体重", "調教師", "course_len", "weather", "race_type", "ground_state",
        "date", "class", "horse_id", "jockey_id",
    ]
    assert df.index.unique().tolist() == [FIXED_RACE_ID]

    first = df.iloc[0]
    assert first[["着順", "馬番", "馬名", "タイム", "単勝", "course_len", "race_type", "horse_id", "jockey_id"]].tolist() == [
        "1", "7", "エースアビリティ", "0:1:26.9", "1.4", 1400, "ダート", "2021102098", "05339",
    ]


@pytest.mark.network
def test_scrape_day_race_result_returns_expected():
    df = netkeiba_scraper.scrape_day_race_result(FIXED_RACE_ID)

    print(f"\n--- scrape_day_race_result({FIXED_RACE_ID}) ---")
    print(f"shape: {df.shape}")
    print(f"columns: {df.columns.tolist()}")
    print(df.to_string())

    assert df.shape == (16, 15)
    assert df.columns.tolist() == [
        "着順", "枠", "馬番", "馬名", "性齢", "斤量", "騎手", "タイム", "着差", "人気", "単勝", "上り", "通過", "厩舎", "馬体重",
    ]
    assert df.index.unique().tolist() == [FIXED_RACE_ID]

    first = df.iloc[0]
    assert first[["着順", "馬番", "馬名", "タイム", "単勝", "厩舎", "馬体重"]].tolist() == [
        1, 7, "エースアビリティ", "0:1:26.9", 1.4, "美浦 堀内", "480(+2)",
    ]


@pytest.mark.network
def test_scrape_race_returns_dataframe_matches_known_result():
    # data/RaceReturns/05_tokyo/2024_race_returns.csv に保存済みの確定結果から判明している期待値
    expected = pd.DataFrame(
        [
            ["単勝", "7", "140", "1"],
            ["複勝", "7", "110", "1"],
            ["複勝", "6", "230", "3"],
            ["複勝", "13", "400", "6"],
            ["枠連", "3-4", "790", "2"],
            ["馬連", "6-7", "760", "2"],
            ["ワイド", "6-7", "350", "2"],
            ["ワイド", "7-13", "620", "7"],
            ["ワイド", "6-13", "2310", "17"],
            ["馬単", "7→6", "1130", "4"],
            ["三連複", "6-7-13", "4240", "11"],
            ["三連単", "7→6→13", "10490", "30"],
        ],
        columns=race_info_model.RACE_RETURNS_COLUMNS,
        index=[FIXED_RACE_ID] * 12,
    )

    new_result = netkeiba_scraper.scrape_race_returns_dataframe([FIXED_RACE_ID])

    print(f"\n--- scrape_race_returns_dataframe([{FIXED_RACE_ID}]) ---")
    print(f"shape: {new_result.shape}")
    print(new_result.to_string())

    assert new_result.reset_index(drop=True).equals(expected.reset_index(drop=True))
    assert new_result.index.tolist() == expected.index.tolist()


@pytest.mark.network
def test_scrape_race_card_returns_expected():
    info, race_info_df, race_card_df = netkeiba_scraper.scrape_race_card(FIXED_RACE_ID)

    print(f"\n--- scrape_race_card({FIXED_RACE_ID}) ---")
    print(f"info: {info}")
    print(f"race_info_df:\n{race_info_df.to_string()}")
    print(f"race_card_df (shape={race_card_df.shape}):\n{race_card_df.to_string()}")

    assert info == [
        "3歳未勝利", "10", "05発走", "ダ1400m", "左", "天候", "晴", "馬場", "良",
        "サラ系３歳", "未勝利", "混", "指", "馬齢", "16頭",
    ]
    assert race_info_df.to_dict("records") == [
        {"race_type": "ダート", "course_len": 1400, "weather": "晴", "ground_state": "良", "class": "未勝利"}
    ]

    assert race_card_df.shape == (16, 11)
    assert race_card_df.columns.tolist() == [
        "枠", "馬番", "馬名", "性齢", "斤量", "騎手", "厩舎", "馬体重(増減)", "所属", "horse_id", "jockey_id",
    ]
    assert race_card_df.index.unique().tolist() == [FIXED_RACE_ID]
    assert race_card_df.iloc[0].tolist() == [
        1, 1, "アフロマン", "牡3", 57.0, "木幡育", "萱野", "410(+16)", "美浦", "2021107090", "01167",
    ]
