"""src/logic/scraping/jra_calendar_scraper.py のテスト

parse_meeting_name / assign_days は純粋関数なのでオフラインで検証する。
get_jra_calendar はJRA公式サイトへの実通信が必要なため @pytest.mark.network を付与する。
"""

import pytest

from src.datasets.race_schedule import model, validate
from src.logic.scraping import jra_calendar_scraper
from src.managers import race_schedule_dataset_manager


def test_parse_meeting_name():
    assert jra_calendar_scraper.parse_meeting_name("4回東京") == (4, 5)
    assert jra_calendar_scraper.parse_meeting_name("1回中山") == (1, 6)


def test_parse_meeting_name_returns_none_for_unknown_place():
    assert jra_calendar_scraper.parse_meeting_name("ばんえい帯広") == (None, None)


def test_assign_days_assigns_sequential_days_per_course_and_times():
    entries = [
        (1, 5, 6, 1),
        (1, 5, 7, 1),
        (1, 6, 6, 1),
        (3, 1, 6, 2),
    ]

    rows = jra_calendar_scraper.assign_days(entries)

    assert rows == [
        {"month": 1, "day": 5, "course": 6, "times": 1, "days": 1},
        {"month": 1, "day": 5, "course": 7, "times": 1, "days": 1},
        {"month": 1, "day": 6, "course": 6, "times": 1, "days": 2},
        {"month": 3, "day": 1, "course": 6, "times": 2, "days": 1},
    ]


@pytest.mark.network
def test_get_jra_calendar_has_calendar_columns():
    calendar_df = jra_calendar_scraper.get_jra_calendar(2025)
    assert validate.has_calendar_columns(calendar_df)
    assert not calendar_df.empty


@pytest.mark.network
def test_get_jra_calendar_matches_existing_2025_calendar():
    """スクレイパーが取得した2025年のカレンダーが、既存のdata/race_schedule/2025_race_calendar.csv
    とほぼ一致することを確認する。

    以下の4行はJRA公式カレンダーJSONで確認した結果、既存CSV側の誤りと考えられる
    既知の差分のため、ここでは許容する:
      - 京都(course=8) 1回: 既存は(2/8, 2/9)をdays=3,4としているが、
        JRA公式の開催日は(2/9, 2/10)であり、新実装はこちらに合致する。
      - 福島(course=3)・小倉(course=10) 2回: 既存は6/29をtimes=3としているが、
        JRA公式では"2回福島"/"2回小倉"(times=2)であり、新実装はこちらに合致する。
    """
    scraped = jra_calendar_scraper.get_jra_calendar(2025)
    existing = race_schedule_dataset_manager.get_race_calendar(2025)

    scraped_rows = set(scraped[model.CALENDAR_COLUMNS].itertuples(index=False, name=None))
    existing_rows = set(existing[model.CALENDAR_COLUMNS].itertuples(index=False, name=None))

    known_legacy_data_errors = {
        ("2", "8", "8", "1", "3"),
        ("2", "9", "8", "1", "4"),
        ("6", "29", "3", "3", "2"),
        ("6", "29", "10", "3", "2"),
    }
    known_new_values = {
        ("2", "9", "8", "1", "3"),
        ("2", "10", "8", "1", "4"),
        ("6", "29", "3", "2", "2"),
        ("6", "29", "10", "2", "2"),
    }

    assert existing_rows - scraped_rows == known_legacy_data_errors
    assert scraped_rows - existing_rows == known_new_values
