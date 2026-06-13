"""src/logic/scraping/netkeiba_scraper.py のテスト

parse_race_info_tokens は純粋関数なのでオフラインで検証する。
scrape_race_results はnetkeiba.comへの実通信が必要なため @pytest.mark.network を付与し、
確定済みの固定race_idを使って旧 libs/scraping.scrape_race_results と
新 netkeiba_scraper.scrape_race_results の出力が一致することを確認する。
"""

import pytest

import scraping as old_scraping
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
def test_scrape_race_results_matches_old():
    old_df = old_scraping.scrape_race_results(FIXED_RACE_ID)
    new_df = netkeiba_scraper.scrape_race_results(FIXED_RACE_ID)

    assert not old_df.empty
    assert not new_df.empty
    assert old_df.columns.tolist() == new_df.columns.tolist()
    assert len(old_df) == len(new_df)
    assert old_df.equals(new_df)
