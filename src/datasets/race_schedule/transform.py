"""race_schedule（race_calendar）データセットの変換ロジック（純粋関数のみ・I/Oなし）"""

from src.datasets.race_schedule import model


def build_race_id(year, course, times, days, race_no):
    """year, course, times, days, race_no から race_id文字列を組み立てる

    Args:
        year (int): 開催年
        course (int|str): 開催場id
        times (int|str): 開催回
        days (int|str): 開催日目
        race_no (int|str): レース番号

    Returns:
        str: race_id
    """
    return (
        str(year)
        + str(course).zfill(model.RACE_ID_COURSE_DIGITS)
        + str(times).zfill(model.RACE_ID_TIMES_DIGITS)
        + str(days).zfill(model.RACE_ID_DAYS_DIGITS)
        + str(race_no).zfill(model.RACE_ID_RACE_NO_DIGITS)
    )


def build_race_ids_for_day(year, course, times, days):
    """1開催日(course, times, days)分のrace_id(1R〜12R)のリストを作成する

    Returns:
        list[str]: race_idのリスト（12件）
    """
    return [
        build_race_id(year, course, times, days, race_no)
        for race_no in range(1, model.RACES_PER_DAY + 1)
    ]


def build_all_race_ids(place_id, year):
    """開催コースと年を指定して、1年間のrace_idを全組み合わせで作成する

    times: 1〜MAX_TIMES_PER_YEAR, days: 1〜MAX_DAYS_PER_TIMES の総当りで生成する
    （レースカレンダーには依存しない）。

    Args:
        place_id (int): 開催コースid
        year (int): 開催年

    Returns:
        list[str]: race_idのリスト
    """
    race_id_list = []
    for times in range(1, model.MAX_TIMES_PER_YEAR + 1):
        for days in range(1, model.MAX_DAYS_PER_TIMES + 1):
            race_id_list.extend(build_race_ids_for_day(year, place_id, times, days))
    return race_id_list


def extract_place_id(race_id):
    """race_idから開催場id(place_id)を抽出する"""
    return int(str(race_id)[4:6])


def extract_unique_place_ids(race_id_list):
    """race_id_listから重複のないplace_idリスト(昇順)を抽出する"""
    return sorted({extract_place_id(race_id) for race_id in race_id_list})


def filter_calendar_by_date(calendar_df, month, day):
    """race_calendarをmonth, dayで絞り込む"""
    filtered = calendar_df[calendar_df["month"] == str(month)]
    filtered = filtered[filtered["day"] == str(day)]
    return filtered


def filter_calendar_by_course(calendar_df, course):
    """race_calendarをcourse(開催場id)で絞り込む"""
    return calendar_df[calendar_df["course"] == str(course)]
