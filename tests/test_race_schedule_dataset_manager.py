"""src/managers/race_schedule_dataset_manager.py のrace_id算出系のテスト（オフライン）。

get_year_id_calendar と get_next_weekly_id は、カレンダーの各行ごとのtimes/days
からrace_idを生成する／当日から6日後までの7日分のget_daily_idを連結する、
という新実装の組み立て方が正しいことを src.datasets.race_schedule.transform の
関数を使って検証する（詳細は各テストのdocstring参照）。
"""

from datetime import date, timedelta

import pytest

from src.datasets.race_schedule import model, transform
from src.managers import race_schedule_dataset_manager as new_race_id


# --- race_idの算出 -----------------------------------------------------------


@pytest.mark.parametrize(
    ("place_id", "race_day", "expected_courses"),
    [
        (0, date(2025, 1, 5), [(6, 1, 1), (7, 1, 1)]),  # 中山・中京 開催日
        (0, date(2025, 1, 7), []),  # 非開催日
        (0, date(2025, 6, 1), [(5, 2, 12), (8, 2, 12)]),  # 京都・札幌 開催日
        (5, date(2025, 1, 5), []),
        (5, date(2025, 1, 7), []),
        (5, date(2025, 6, 1), [(5, 2, 12)]),
        (6, date(2025, 1, 5), [(6, 1, 1)]),
        (6, date(2025, 1, 7), []),
        (6, date(2025, 6, 1), []),
    ],
)
def test_get_daily_id_returns_expected(place_id, race_day, expected_courses):
    result = new_race_id.get_daily_id(place_id, race_day)

    expected = []
    for course, times, days in expected_courses:
        expected.extend(transform.build_race_ids_for_day(race_day.year, course, times, days))

    assert result == expected


@pytest.mark.parametrize("place_id", [0, 6])
@pytest.mark.parametrize("race_day", [date(2025, 1, 13), date(2025, 4, 1)])
def test_get_past_weekly_id_returns_expected(place_id, race_day):
    result = new_race_id.get_past_weekly_id(place_id, race_day)

    # 7日前から当日までの7日分のget_daily_idを連結した結果になる
    expected = []
    for i in range(7):
        expected.extend(new_race_id.get_daily_id(place_id, race_day - timedelta(days=(7 - i))))

    assert result == expected


@pytest.mark.parametrize(
    ("place_id", "race_day", "expected_len", "expected_first", "expected_last"),
    [
        (0, date(2025, 6, 1), 1464, "202506010101", "202508021212"),
        (0, date(2025, 1, 1), 0, None, None),
        (5, date(2025, 6, 1), 240, "202505010101", "202505021212"),
        (5, date(2025, 1, 1), 0, None, None),
    ],
)
def test_get_past_year_id_returns_expected(place_id, race_day, expected_len, expected_first, expected_last):
    result = new_race_id.get_past_year_id(place_id, race_day)

    assert len(result) == expected_len
    if expected_len:
        assert result[0] == expected_first
        assert result[-1] == expected_last
        assert len(set(result)) == len(result)


@pytest.mark.parametrize("place_id", [1, 5, 6])
@pytest.mark.parametrize("year", [2024, 2025])
def test_get_year_id_all_returns_expected(place_id, year):
    result = new_race_id.get_year_id_all(place_id, year)

    # カレンダーに依存せず、開催回・開催日目の全組み合わせから生成される
    expected = []
    for times in range(1, model.MAX_TIMES_PER_YEAR + 1):
        for days in range(1, model.MAX_DAYS_PER_TIMES + 1):
            expected.extend(transform.build_race_ids_for_day(year, place_id, times, days))

    assert result == expected


@pytest.mark.parametrize(
    ("race_day", "expected"),
    [
        (date(2025, 1, 13), [6, 7]),
        (date(2025, 1, 1), [6, 8]),
    ],
)
def test_get_past_weekly_place_id_returns_expected(race_day, expected):
    assert new_race_id.get_past_weekly_place_id(race_day) == expected


@pytest.mark.parametrize(
    ("race_day", "expected"),
    [
        (date(2025, 1, 5), [6, 7]),
        (date(2025, 1, 7), []),
    ],
)
def test_get_daily_place_id_returns_expected(race_day, expected):
    assert new_race_id.get_daily_place_id(race_day) == expected


def test_get_place_id_list_from_race_id_list_returns_expected():
    race_id_list = new_race_id.get_year_id_all(6, 2025)

    assert new_race_id.get_place_id_list_from_race_id_list(race_id_list) == [6]


# --- カレンダー行ごとの組み立て -----------------------------------------------


def test_get_year_id_calendar_returns_expected():
    """カレンダーの各行(開催回・開催日ごと)のtimes/daysからrace_idを生成するため、
    同じ開催場で複数回・複数日開催される場合でも重複が発生しないことを確認する。
    """
    year = 2025
    place_id = 6  # 中山（年間で複数回・複数日開催される）

    result = new_race_id.get_year_id_calendar(place_id, year)

    # 重複は発生しない
    assert len(set(result)) == len(result)

    # カレンダーの各行から計算したrace_idと一致することを確認
    calendar = new_race_id.get_race_calendar(year)
    race_data = transform.filter_calendar_by_course(calendar, place_id).reset_index(drop=True)
    expected = []
    for i in range(len(race_data.index)):
        expected.extend(
            transform.build_race_ids_for_day(
                year,
                race_data.at[i, "course"],
                race_data.at[i, "times"],
                race_data.at[i, "days"],
            )
        )
    assert result == expected


def test_get_next_weekly_id_returns_expected():
    """当日から6日後までの7日分のget_daily_idを連結した結果になることを確認する。"""
    place_id = 0
    race_day = date(2025, 1, 1)

    result = new_race_id.get_next_weekly_id(place_id, race_day)

    expected = []
    for offset in range(7):
        expected.extend(new_race_id.get_daily_id(place_id, race_day + timedelta(days=offset)))

    assert result == expected
