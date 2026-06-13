"""JRA公式サイトから開催カレンダー(race_calendar)を取得するスクレイパー

旧 src/Datasets/make_calender.py は、どこからも呼び出されておらず、
出力形式([day, course, event, day_num])も race_calendar CSV の列
(month, day, course, times, days)と一致していなかったため、
race_calendar形式で新規実装する。

JRA公式の開催カレンダーページ(/keiba/calendarYYYY/MM.html)はJavaScriptで
JSON(/keiba/common/calendar/json/YYYYMM.json)を読み込んで描画している。
本モジュールはそのJSONを直接取得し、"4回東京"のような開催回・開催場の
表記から course(開催場id)・times(開催回)を求め、days(開催日目)は
(course, times)ごとに日付の早い順に1から連番を振って算出する。
"""

import os
import re

import pandas as pd
import requests

from src.config import paths
from src.config.constants import NAME_LIST
from src.datasets.race_schedule import model

JSON_URL_TEMPLATE = "https://www.jra.go.jp/keiba/common/calendar/json/{year}{month:02d}.json"

# 「4回東京」のような開催回・開催場の表記
MEETING_NAME_PATTERN = re.compile(r"(\d+)回(.+)")

# NAME_LIST(0始まり) -> course(place_id, 1始まり)
PLACE_ID_BY_NAME = {name: index + 1 for index, name in enumerate(NAME_LIST)}


def fetch_month_json(year, month):
    """指定年月のJRA開催カレンダーJSONを取得する

    Args:
        year (int): 年
        month (int): 月(1〜12)

    Returns:
        list: JRA開催カレンダーJSON（[{"month": "...", "data": [...]}, ...]）
    """
    url = JSON_URL_TEMPLATE.format(year=year, month=month)
    response = requests.get(url)
    response.raise_for_status()
    return response.json()


def parse_meeting_name(name):
    """「4回東京」のような表記から (times, course) を抽出する

    NAME_LISTに含まれない開催場名（地方競馬等）の場合はNoneを返す。

    Args:
        name (str): 開催回・開催場の表記

    Returns:
        tuple[int, int] | tuple[None, None]: (times, course)
    """
    matched = MEETING_NAME_PATTERN.match(name)
    if not matched:
        return None, None

    times_str, place_name = matched.groups()
    course = PLACE_ID_BY_NAME.get(place_name)
    if course is None:
        return None, None

    return int(times_str), course


def extract_race_entries(month_json):
    """JRA開催カレンダーJSONから (month, day, course, times) のリストを抽出する

    Args:
        month_json (list): fetch_month_jsonの戻り値

    Returns:
        list[tuple[int, int, int, int]]: (month, day, course, times) のリスト
    """
    entries = []
    for month_data in month_json:
        month = int(month_data["month"])
        for day_data in month_data.get("data", []):
            day = int(day_data["date"])
            for info in day_data.get("info", []):
                for race in info.get("race", []):
                    times, course = parse_meeting_name(race.get("name", ""))
                    if times is not None:
                        entries.append((month, day, course, times))
    return entries


def assign_days(entries):
    """(month, day, course, times) のリストに、(course, times)ごとの開催日目(days)を割り振る

    Args:
        entries (list[tuple[int, int, int, int]]): (month, day, course, times) のリスト

    Returns:
        list[dict]: model.CALENDAR_COLUMNS の各列を持つ辞書のリスト
    """
    rows = []
    days_counter = {}
    for month, day, course, times in sorted(entries, key=lambda entry: (entry[0], entry[1])):
        key = (course, times)
        days_counter[key] = days_counter.get(key, 0) + 1
        rows.append(
            {
                "month": month,
                "day": day,
                "course": course,
                "times": times,
                "days": days_counter[key],
            }
        )
    return rows


def get_jra_calendar(year):
    """指定年のJRA開催カレンダーを、race_calendar形式のDataFrameで取得する

    Args:
        year (int): 年

    Returns:
        pd.DataFrame: 列 = model.CALENDAR_COLUMNS（month, day, course, times, days）
    """
    entries = []
    for month in range(1, 13):
        month_json = fetch_month_json(year, month)
        entries.extend(extract_race_entries(month_json))

    rows = assign_days(entries)
    calendar_df = pd.DataFrame(rows, columns=model.CALENDAR_COLUMNS)
    return calendar_df.astype(str)


def save_race_calendar(year, calendar_df):
    """race_calendar形式のDataFrameをCSVとして保存する

    Args:
        year (int): 年
        calendar_df (pd.DataFrame): get_jra_calendarの戻り値

    Returns:
        str: 保存先のCSVパス
    """
    os.makedirs(paths.RACE_SCHEDULE_DATA_PATH, exist_ok=True)
    path = os.path.join(paths.RACE_SCHEDULE_DATA_PATH, f"{year}_race_calendar.csv")
    calendar_df.to_csv(path, index=False)
    return path
