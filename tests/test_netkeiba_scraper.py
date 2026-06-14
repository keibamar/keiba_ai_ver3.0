"""src/logic/scraping/netkeiba_scraper.py のテスト

parse_race_info_tokens は純粋関数なのでオフラインで検証する。
scrape_race_results はnetkeiba.comへの実通信が必要なため @pytest.mark.network を付与し、
確定済みの固定race_idを使って旧 libs/scraping.scrape_race_results と
新 netkeiba_scraper.scrape_race_results の出力が一致することを確認する。

scrape_race_returns_dataframeについても、旧 src/legacy_datasets/race_returns.py の
同名関数はformat_type_returns_dataframeの戻り値に対する.set_index(0)がKeyErrorとなり
処理が失敗するため、新旧比較ではなく、確定済みのレース（FIXED_RACE_ID）について
data/RaceReturns/05_tokyo/2024_race_returns.csv に保存済みの結果から判明している
期待値との比較で検証する。
"""

import pandas as pd
import pytest

import scraping as old_scraping
from src.datasets.race_info import model as race_info_model
from src.datasets.race_result import transform
from src.legacy_datasets import race_returns as old_returns
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
def test_scrape_race_results_matches_old():
    old_df = old_scraping.scrape_race_results(FIXED_RACE_ID)
    new_df = netkeiba_scraper.scrape_race_results(FIXED_RACE_ID)

    assert not old_df.empty
    assert not new_df.empty
    assert old_df.columns.tolist() == new_df.columns.tolist()
    assert len(old_df) == len(new_df)
    assert old_df.equals(new_df)


@pytest.mark.network
def test_scrape_day_race_result_matches_old():
    old_df = old_scraping.scrape_day_race_results(FIXED_RACE_ID)
    new_df = netkeiba_scraper.scrape_day_race_result(FIXED_RACE_ID)

    assert not old_df.empty
    assert not new_df.empty
    assert old_df.columns.tolist() == new_df.columns.tolist()
    assert len(old_df) == len(new_df)
    assert old_df.equals(new_df)


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

    assert new_result.reset_index(drop=True).equals(expected.reset_index(drop=True))
    assert new_result.index.tolist() == expected.index.tolist()


@pytest.mark.network
def test_scrape_race_card_matches_old():
    old_info, old_race_info_df, old_race_card_df = old_scraping.scrape_race_card(FIXED_RACE_ID)
    new_info, new_race_info_df, new_race_card_df = netkeiba_scraper.scrape_race_card(FIXED_RACE_ID)

    assert old_info == new_info
    assert not old_race_card_df.empty
    assert not new_race_card_df.empty
    assert old_race_info_df.equals(new_race_info_df)
    assert old_race_card_df.equals(new_race_card_df)


@pytest.mark.network
def test_old_scrape_race_returns_dataframe_is_broken():
    # 旧実装は format_type_returns_dataframe の戻り値（列名 "式別","馬番","配当","人気"）に
    # 対して .set_index(0) を呼び出すため、KeyErrorで処理全体が失敗する
    with pytest.raises(KeyError):
        old_returns.scrape_race_returns_dataframe([FIXED_RACE_ID])
