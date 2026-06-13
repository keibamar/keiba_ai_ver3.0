"""race_schedule（race_calendar）データセットのManager層

CSVの読み込みと、race_id/place_idの算出を担う。
変換ロジック自体は src/datasets/race_schedule/transform.py に切り出している。

旧 libs/get_race_id.py からの移植にあたり、以下2点は不具合修正を行っている:
- get_year_id_calendar: 同一開催場で複数の開催回・開催日がある場合に、
  1行目のデータのみを使って重複生成していたバグを修正し、各行の
  times/days を使ってrace_idを生成するようにした。
- get_next_weekly_id: 日付オフセットが累積してしまうバグ(0,1,3,6,10,15,21日後)
  を修正し、正しく0〜6日後を取得するようにした。
"""

import os
from datetime import date, timedelta

from src.config import paths
from src.datasets.race_schedule import transform
from src.utils.file_utils import read_csv_or_empty


def get_race_calendar(year):
    """指定年のrace_calendarを読み込む

    Args:
        year (int): 開催年

    Returns:
        pd.DataFrame: race_calendar（存在しない場合は空のDataFrame）
    """
    path = os.path.join(paths.RACE_SCHEDULE_DATA_PATH, f"{year}_race_calendar.csv")
    return read_csv_or_empty(path, dtype=str)


def get_year_id_all(place_id, year=date.today().year):
    """開催コースと年を指定して、1年間のrace_idを取得

    Args:
        place_id (int): 開催コースid
        year (int): 年（初期値：今年）

    Returns:
        list: 指定した、開催コース、年のstr型のrace_idを返す
    """
    return transform.build_all_race_ids(place_id, year)


def get_year_id_calendar(place_id, year=date.today().year):
    """開催コースと年を指定して、1年間のrace_idをレースカレンダーから取得

    Args:
        place_id (int): 開催コースid
        year (int): 年（初期値：今年）

    Returns:
        list: 指定した、開催コース、年のstr型のrace_idを返す
    """
    race_calendar = get_race_calendar(year)
    if race_calendar.empty:
        return []

    race_data = transform.filter_calendar_by_course(race_calendar, place_id).reset_index(drop=True)

    race_id_list = []
    for i in range(len(race_data.index)):
        race_id_list.extend(
            transform.build_race_ids_for_day(
                year,
                race_data.at[i, "course"],
                race_data.at[i, "times"],
                race_data.at[i, "days"],
            )
        )
    return race_id_list


def get_past_year_id(place_id=0, race_day=date.today()):
    """指定した日にちまでの、その年のrace_idを取得

    Args:
        place_id (int): 開催コースid (初期値0=全コース)
        race_day (date): 日（初期値：今日）

    Returns:
        list: 指定した日にちまでの、その年のstr型のrace_idを返す
    """
    race_calendar = get_race_calendar(race_day.year)
    if race_calendar.empty:
        return []

    place_ids = range(1, 11) if place_id == 0 else [place_id]

    race_id_list = []
    for m in range(1, race_day.month + 1):
        last_day = 31 if m < race_day.month else race_day.day
        for d in range(1, last_day + 1):
            daily_calendar = transform.filter_calendar_by_date(race_calendar, m, d)
            for pid in place_ids:
                race_data = transform.filter_calendar_by_course(daily_calendar, pid).reset_index(drop=True)
                if not race_data.empty:
                    race_id_list.extend(
                        transform.build_race_ids_for_day(
                            race_day.year,
                            race_data.at[0, "course"],
                            race_data.at[0, "times"],
                            race_data.at[0, "days"],
                        )
                    )
    return race_id_list


def get_past_weekly_id(place_id=0, base_day=date.today()):
    """指定した日にちから、1週間前のrace_idを取得

    Args:
        place_id (int): 開催コースid (初期値0=全コース)
        base_day (date): 日（初期値：今日）

    Returns:
        list: 指定した日にちから、1週間前までのstr型のrace_idを返す
    """
    race_id_list = []
    for i in range(7):
        race_day = base_day - timedelta(days=(7 - i))
        race_id_list.extend(get_daily_id(place_id, race_day))
    return race_id_list


def get_next_weekly_id(place_id=0, race_day=date.today()):
    """指定した日にちから、次の1週間のrace_idを取得

    Args:
        place_id (int): 開催コースid (初期値0=全コース)
        race_day (date): 日（初期値：今日）

    Returns:
        list: 指定した日にちから、次の1週間のstr型のrace_idを返す
    """
    race_id_list = []
    for i in range(7):
        target_day = race_day + timedelta(days=i)
        race_id_list.extend(get_daily_id(place_id, target_day))
    return race_id_list


def get_daily_id(place_id=0, race_day=date.today()):
    """指定した日にちのrace_idを取得

    Args:
        place_id (int): 開催コースid (初期値0=全コース)
        race_day (date): 日（初期値：今日）

    Returns:
        list: 指定した日にちのstr型のrace_idを返す
    """
    race_calendar = get_race_calendar(race_day.year)
    if race_calendar.empty:
        return []

    daily_calendar = transform.filter_calendar_by_date(race_calendar, race_day.month, race_day.day)

    place_ids = range(1, 11) if place_id == 0 else [place_id]

    race_id_list = []
    for pid in place_ids:
        race_data = transform.filter_calendar_by_course(daily_calendar, pid).reset_index(drop=True)
        if not race_data.empty:
            race_id_list.extend(
                transform.build_race_ids_for_day(
                    race_day.year,
                    race_data.at[0, "course"],
                    race_data.at[0, "times"],
                    race_data.at[0, "days"],
                )
            )
    return race_id_list


def get_place_id_list_from_race_id_list(race_id_list):
    """race_id_listからplace_id_listを抽出する

    Args:
        race_id_list (list): race_idのリスト

    Returns:
        list: 重複のないplace_idのリスト（昇順）
    """
    return transform.extract_unique_place_ids(race_id_list)


def get_past_weekly_place_id(race_day=date.today()):
    """指定日から1週間前までの開催場のplace_idを取得

    Args:
        race_day (date): 日（初期値：今日）

    Returns:
        list: 開催場のplace_id
    """
    race_id_list = get_past_weekly_id(base_day=race_day)
    return get_place_id_list_from_race_id_list(race_id_list)


def get_daily_place_id(race_day=date.today()):
    """指定日の開催場のplace_idを取得

    Args:
        race_day (date): 日（初期値：今日）

    Returns:
        list: 開催場のplace_id
    """
    race_id_list = get_daily_id(race_day=race_day)
    return get_place_id_list_from_race_id_list(race_id_list)
