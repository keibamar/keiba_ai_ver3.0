"""src/managers/race_schedule_dataset_manager.py の出力が
旧 libs/get_race_id.py と一致することを確認するテスト（オフライン）。

get_year_id_calendar と get_next_weekly_id は、旧実装に存在するバグを
新実装で修正しているため、意図的に旧実装とは異なる結果になる
（詳細は各テストのdocstring参照）。
"""

from datetime import date, timedelta

import pytest

import get_race_id as old_race_id
from src.config import paths
from src.datasets.race_schedule import transform
from src.managers import race_schedule_dataset_manager as new_race_id


@pytest.fixture(autouse=True)
def use_ver3_calendar_path(monkeypatch):
    """旧実装のCALENDAR_PATHを、新実装と同じdata/race_schedule/に向ける"""
    monkeypatch.setattr(old_race_id, "CALENDAR_PATH", paths.RACE_SCHEDULE_DATA_PATH + "/")


# --- 旧実装と完全に一致するはずの関数 ---------------------------------------


@pytest.mark.parametrize("place_id", [0, 5, 6])
@pytest.mark.parametrize(
    "race_day",
    [
        date(2025, 1, 5),  # 中山・中京 開催日
        date(2025, 1, 7),  # 非開催日
        date(2025, 6, 1),
    ],
)
def test_get_daily_id_matches_old(place_id, race_day):
    assert new_race_id.get_daily_id(place_id, race_day) == old_race_id.get_daily_id(place_id, race_day)


@pytest.mark.parametrize("place_id", [0, 6])
@pytest.mark.parametrize("race_day", [date(2025, 1, 13), date(2025, 4, 1)])
def test_get_past_weekly_id_matches_old(place_id, race_day):
    assert new_race_id.get_past_weekly_id(place_id, race_day) == old_race_id.get_past_weekly_id(
        place_id, race_day
    )


@pytest.mark.parametrize("place_id", [0, 5])
@pytest.mark.parametrize("race_day", [date(2025, 6, 1), date(2025, 1, 1)])
def test_get_past_year_id_matches_old(place_id, race_day):
    assert new_race_id.get_past_year_id(place_id, race_day) == old_race_id.get_past_year_id(
        place_id, race_day
    )


@pytest.mark.parametrize("place_id", [1, 5, 6])
@pytest.mark.parametrize("year", [2024, 2025])
def test_get_year_id_all_matches_old(place_id, year):
    assert new_race_id.get_year_id_all(place_id, year) == old_race_id.get_year_id_all(place_id, year)


@pytest.mark.parametrize("race_day", [date(2025, 1, 13), date(2025, 1, 1)])
def test_get_past_weekly_place_id_matches_old(race_day):
    assert new_race_id.get_past_weekly_place_id(race_day) == old_race_id.get_past_weekly_place_id(race_day)


@pytest.mark.parametrize("race_day", [date(2025, 1, 5), date(2025, 1, 7)])
def test_get_daily_place_id_matches_old(race_day):
    assert new_race_id.get_daily_place_id(race_day) == old_race_id.get_daily_place_id(race_day)


def test_get_place_id_list_from_race_id_list_matches_old():
    race_id_list = new_race_id.get_year_id_all(6, 2025)
    assert new_race_id.get_place_id_list_from_race_id_list(
        race_id_list
    ) == old_race_id.get_place_id_list_from_race_id_list(race_id_list)


# --- 旧実装のバグを修正した関数（意図的に旧実装とは異なる結果になる） ---------


def test_get_year_id_calendar_fixes_row0_duplication_bug():
    """旧実装は、同じ開催場で複数の開催回・開催日がある場合、
    レースカレンダーの1行目(times, days)のデータだけを使って
    race_idを重複生成してしまうバグがある。
    新実装は、各行ごとのtimes/daysを使って正しいrace_idを生成する。
    """
    year = 2025
    place_id = 6  # 中山（年間で複数回・複数日開催される）

    old_result = old_race_id.get_year_id_calendar(place_id, year)
    new_result = new_race_id.get_year_id_calendar(place_id, year)

    # 件数（行数 x 12レース）は変わらない
    assert len(old_result) == len(new_result)

    # 旧実装は1行目のtimes/daysのrace_id(12件)を重複生成するだけなので、
    # 重複を除くと1開催日分(12件)になってしまう
    assert len(set(old_result)) == 12

    # 新実装は各行ごとに異なるrace_idを生成するため、重複は発生しない
    assert len(set(new_result)) == len(new_result)

    # 新実装の出力が、カレンダーの各行から計算したrace_idと一致することを確認
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
    assert new_result == expected


def test_get_next_weekly_id_fixes_cumulative_offset_bug():
    """旧実装は race_day += timedelta(days=i) を毎回累積してしまい、
    本来0,1,2,3,4,5,6日後を取得すべきところ0,1,3,6,10,15,21日後を
    取得してしまうバグがある。新実装は正しく0〜6日後を取得する。
    """
    place_id = 0
    race_day = date(2025, 1, 1)

    old_result = old_race_id.get_next_weekly_id(place_id, race_day)
    new_result = new_race_id.get_next_weekly_id(place_id, race_day)

    # 新実装 = 当日から6日後までの7日分のget_daily_idを連結した結果
    expected_new = []
    for offset in range(7):
        expected_new.extend(new_race_id.get_daily_id(place_id, race_day + timedelta(days=offset)))
    assert new_result == expected_new

    # 旧実装 = 0,1,3,6,10,15,21日後のget_daily_idを連結した結果（バグ）
    expected_old = []
    cumulative_day = race_day
    for i in range(7):
        cumulative_day = cumulative_day + timedelta(days=i)
        expected_old.extend(old_race_id.get_daily_id(place_id, cumulative_day))
    assert old_result == expected_old

    # 上記の通り、旧実装と新実装は意図的に異なる結果になる
    assert old_result != new_result
