"""AI予想成績（的中率・回収率）の集計

Oracleの予想（race_card_dataset_manager.get_race_cards のrank列、rank=1を本命とする）と
確定配当（race_info_dataset_manager.get_race_return_csv_for_race）を照合し、
1レース単位で的中・回収を計算する（calc_race_hit_returns）。

複数レースまとめた集計（年別・開催別・コース別等）は、本モジュールの計算結果を
永続化した src.managers.ai_performance_dataset_manager 側のaggregate/filter_by_*/
group_breakdownで行う（data/race_card/の全件スキャンを避けて高速化するため）。
本モジュールが残しているのは、その永続化データセットの構築
（update_ai_performance_dataset）と、Homeページの「今週/先週の結果」「開催中の競馬場」用の
1レース単位の処理のみ。

過去の予想データ（出馬表+score/rank）は date/RaceCards/ から data/race_card/ に
移行済み（2024-10-20〜、161日分）。一方、配当の的中判定に使う
race_info_dataset_manager.get_race_return_csv_for_race（1レース単位の分割保存）は
現時点では限られた範囲しか蓄積されていない。本モジュールは「予想と確定配当の両方が
存在するレース」のみを対象に集計する。
"""

import math
import os
import re
from datetime import date, datetime, timedelta

from src.config import paths
from src.managers import (
    race_card_dataset_manager,
    race_info_dataset_manager,
    race_result_dataset_manager,
    race_schedule_dataset_manager,
)

BET_TYPES = ("win", "place", "trio_box")


def parse_race_id(race_id):
    """race_id（YYYYCCTTDDNN形式）を年・開催場・開催回・開催日目・レース番号に分解する"""
    race_id = str(race_id)
    return {
        "year": int(race_id[0:4]),
        "place_id": int(race_id[4:6]),
        "times": int(race_id[6:8]),
        "days": int(race_id[8:10]),
        "race_num": int(race_id[10:12]),
    }


def list_predicted_races():
    """data/race_card/ 配下に保存されている予想済みの (race_day, race_id) を全て列挙する

    Returns:
        list[tuple[date, str]]: (race_day, race_id) のリスト（日付昇順）
    """
    pairs = []
    if not os.path.isdir(paths.RACE_CARD_DATA_PATH):
        return pairs

    for date_str in sorted(os.listdir(paths.RACE_CARD_DATA_PATH)):
        date_dir = os.path.join(paths.RACE_CARD_DATA_PATH, date_str)
        if not os.path.isdir(date_dir):
            continue
        try:
            race_day = datetime.strptime(date_str, "%Y%m%d").date()
        except ValueError:
            continue
        for filename in sorted(os.listdir(date_dir)):
            if filename.endswith(".csv"):
                pairs.append((race_day, filename[:-4]))
    return pairs


def calc_race_hit_returns(race_day, race_id, box_num=5):
    """1レース分の的中・回収額を、予想rank1位の馬を基準に計算する

    Args:
        race_day (date): レース開催日
        race_id (str): race_id
        box_num (int): 三連複BOXの頭数（初期値5、上位5位までを軸とする）
    Returns:
        dict | None: {"win": (hit, return), "place": (hit, return), "trio_box": (hit, return)}
            （hitは0/1、returnは100円あたりの配当額）。予想または確定配当が
            存在しない場合はNoneを返す。
    """
    pred_df = race_card_dataset_manager.get_race_cards(race_day, race_id)
    if "rank" not in pred_df.columns:
        return None

    returns_df = race_info_dataset_manager.get_race_return_csv_for_race(race_id)
    if returns_df.empty:
        return None

    result = {bet_type: (0, 0.0) for bet_type in BET_TYPES}

    win_df = returns_df[returns_df["式別"] == "単勝"].reset_index(drop=True)
    for i in range(len(win_df)):
        num = int(win_df.at[i, "馬番"])
        if pred_df.at[num - 1, "rank"] == 1:
            result["win"] = (1, float(win_df.at[i, "配当"]))

    place_df = returns_df[returns_df["式別"] == "複勝"].reset_index(drop=True)
    for i in range(len(place_df)):
        num = int(place_df.at[i, "馬番"])
        if pred_df.at[num - 1, "rank"] == 1:
            payout = place_df.at[i, "配当"]
            if isinstance(payout, str):
                payout = re.sub(r"\D", "", payout)
            result["place"] = (1, float(payout))

    trio_df = returns_df[returns_df["式別"] == "三連複"].reset_index(drop=True)
    for i in range(len(trio_df)):
        num_list = re.findall(r"\d+", trio_df.at[i, "馬番"])
        ranks = [
            pred_df[pred_df["馬番"] == int(num)].reset_index(drop=True).at[0, "rank"] for num in num_list
        ]
        if all(rank <= box_num for rank in ranks):
            payout = trio_df.at[i, "配当"]
            if isinstance(payout, str):
                payout = re.sub(r"\D", "", payout)
            result["trio_box"] = (1, float(payout) / math.comb(box_num, 3))

    return result


def get_current_meetings(today=None):
    """今日時点で開催期間中の開催回（開催場×times）を列挙する

    race_schedule_dataset_manager.get_race_calendar(year) のmonth/day/course/times行を
    (course, times) ごとにグルーピングし、その開催の初日〜最終日の範囲に今日が
    含まれるものを返す（旧ver2.0 web/src/generators/performance/make_ai_index.py の
    get_current_tracksと同じ考え方）。

    Args:
        today (date): 基準日（初期値: 今日）
    Returns:
        list[dict]: [{"place_id", "times", "first_day", "last_day"}, ...]（place_id昇順）
    """
    today = today or date.today()
    calendar = race_schedule_dataset_manager.get_race_calendar(today.year)
    if calendar.empty:
        return []

    meeting_days = {}
    for _, row in calendar.iterrows():
        key = (int(row["course"]), int(row["times"]))
        try:
            day_date = date(today.year, int(row["month"]), int(row["day"]))
        except ValueError:
            continue
        meeting_days.setdefault(key, []).append(day_date)

    current_meetings = []
    for (place_id, times), days in meeting_days.items():
        first_day, last_day = min(days), max(days)
        if first_day <= today <= last_day:
            current_meetings.append(
                {"place_id": place_id, "times": times, "first_day": first_day, "last_day": last_day}
            )

    current_meetings.sort(key=lambda m: m["place_id"])
    return current_meetings


def get_today_main_races_with_course(today=None):
    """今日のメインレース（11R）を、レース名・発走時刻・コース情報付きで返す

    今日のレースはまだ確定結果も予想データも無いため、calc_race_hit_returns
    （的中判定）は使えない。代わりに出走馬一覧ページ（shutuba.html、
    netkeiba_scraper.scrape_race_card）を当日スクレイピングし、コース詳細データへの
    リンクに使うrace_type/course_lenだけを取得する（出走馬一覧自体は使わない）。
    取得に失敗したレースはrace_type/course_lenをNoneにして残す
    （呼び出し側でコースへのリンクなしの表示に切り替えられるようにする）。

    Args:
        today (date): 基準日（初期値: 今日）。

    Returns:
        list[dict]: [{"race_id", "place_id", "race_name", "race_time",
            "race_type", "course_len"}, ...]（発走時刻昇順）。
    """
    from src.logic.scraping import netkeiba_scraper

    today = today or date.today()
    time_id_df = race_card_dataset_manager.get_race_time_id_list_df(today)
    if time_id_df.empty:
        return []

    main_df = time_id_df[
        time_id_df["race_id"].apply(lambda rid: parse_race_id(rid)["race_num"] == 11)
    ].sort_values("race_time")

    races = []
    for _, row in main_df.iterrows():
        race_id = str(row["race_id"])
        race_type, course_len = None, None
        try:
            _, race_info_df, _ = netkeiba_scraper.scrape_race_card(race_id)
            if not race_info_df.empty:
                race_type = race_info_df.iloc[0]["race_type"]
                course_len = race_info_df.iloc[0]["course_len"]
        except Exception:
            pass

        races.append(
            {
                "race_id": race_id,
                "place_id": parse_race_id(race_id)["place_id"],
                "race_name": row["race_name"],
                "race_time": row["race_time"],
                "race_type": race_type,
                "course_len": course_len,
                "race_day": today,
            }
        )
    return races


def get_week_main_races_with_course(today=None):
    """「今週のメインレース」とみなす週末（土曜・日曜）のメインレース（11R）を、
    コース情報付きで返す

    current_schedule_weekend_endが返す週末（土・日）のメインレースをまとめて返す
    （get_today_main_races_with_courseを土日それぞれに適用する）。まだ出走馬一覧
    （shutuba）が公開されていない日は空リストになる（呼び出し側は無視してよい）。

    Args:
        today (date): 基準日（初期値: 今日）。

    Returns:
        list[dict]: get_today_main_races_with_courseと同じ形式の辞書のリスト
            （race_day・race_time昇順）。
    """
    today = today or date.today()
    sunday = current_schedule_weekend_end(today)
    saturday = sunday - timedelta(days=1)

    races = []
    for day in (saturday, sunday):
        races.extend(get_today_main_races_with_course(day))
    return races


def current_results_weekend_end(today=None):
    """結果が反映済みとみなせる、最も直近の週末（日曜日）を返す

    開催結果・確定配当の取得・反映には数日かかるため、週末（土日）が終わった直後
    （月〜火）はまだその週末の結果が反映済みとはみなさず、1つ前の週末を指す。
    その週末の3日後（水曜日）になった時点で初めて1週間分更新される
    （Homeページの「先週の結果」のデータ範囲として使う）。

    Args:
        today (date): 基準日（初期値: 今日）。
    Returns:
        date: 結果が反映済みとみなせる週末の日曜日。
    """
    today = today or date.today()
    most_recent_sunday = today - timedelta(days=(today.weekday() + 1) % 7)
    if today >= most_recent_sunday + timedelta(days=3):
        return most_recent_sunday
    return most_recent_sunday - timedelta(days=7)


def current_schedule_weekend_end(today=None):
    """「今週のメインレース」とみなす週末（日曜日）を返す

    出馬表（出走馬一覧）は開催の数日前にならないと公開されないため、今週
    （月曜始まりのtodayが属する週）の水曜日になるまでは、今週本来の週末
    （まだ出馬表が無いことが多い）ではなく1つ前の週末（直近に終わった週末）を
    「今週のメインレース」として指す。今週の水曜日になった時点で本来の週末に
    切り替わる（current_results_weekend_endを7日先の日付に適用するのと同じ計算）。

    Args:
        today (date): 基準日（初期値: 今日）。
    Returns:
        date: 「今週のメインレース」とみなす週末の日曜日。
    """
    today = today or date.today()
    return current_results_weekend_end(today + timedelta(days=7))


def _race_detail_summary(race_day, race_id):
    """1レース分の詳細サマリー（勝ち馬・AI本命馬・本命馬の着順・単勝/複勝の的中結果）を返す

    Args:
        race_day (date): レース開催日
        race_id (str): race_id
    Returns:
        dict | None: 予想（rank列）または確定結果が無ければNone。
            {"winner_name", "pick_name", "pick_finish", "win_hit", "win_payout",
            "place_hit", "place_payout"}
    """
    pred_df = race_card_dataset_manager.get_race_cards(race_day, race_id)
    if "rank" not in pred_df.columns:
        return None

    result_df = race_result_dataset_manager.get_race_id_result(race_id)
    if result_df.empty:
        return None

    winner_row = result_df[result_df["着順"].astype(str) == "1"]
    winner_name = winner_row.iloc[0]["馬名"] if not winner_row.empty else None

    pick_rows = pred_df[pred_df["rank"] == 1]
    if pick_rows.empty:
        return None
    pick_row = pick_rows.iloc[0]
    pick_name = pick_row["馬名"]
    pick_num = int(pick_row["馬番"])

    pick_result = result_df[result_df["馬番"].astype(str) == str(pick_num)]
    pick_finish = pick_result.iloc[0]["着順"] if not pick_result.empty else None

    win_hit, win_payout = False, None
    place_hit, place_payout = False, None
    returns_df = race_info_dataset_manager.get_race_return_csv_for_race(race_id)
    if not returns_df.empty:
        win_df = returns_df[returns_df["式別"] == "単勝"]
        for _, row in win_df.iterrows():
            if int(row["馬番"]) == pick_num:
                win_hit = True
                win_payout = float(row["配当"])

        place_df = returns_df[returns_df["式別"] == "複勝"]
        for _, row in place_df.iterrows():
            if int(row["馬番"]) == pick_num:
                place_hit = True
                payout = row["配当"]
                if isinstance(payout, str):
                    payout = re.sub(r"\D", "", payout)
                place_payout = float(payout)

    return {
        "winner_name": winner_name,
        "pick_name": pick_name,
        "pick_finish": pick_finish,
        "win_hit": win_hit,
        "win_payout": win_payout,
        "place_hit": place_hit,
        "place_payout": place_payout,
    }


def get_weekend_main_race_details(weekend_end_day):
    """指定した週末（土日）のメインレース（11R）の詳細サマリーを日付順で返す

    current_results_weekend_endが返す日曜日（または7日前を渡せば前の週末）を
    受け取り、その土曜・日曜のメインレースについて、勝ち馬・AI本命馬・本命馬の
    着順・単勝/複勝の的中結果（的中時は配当そのものを返す。的中率ではない）を返す。

    Args:
        weekend_end_day (date): 週末の日曜日。
    Returns:
        list[dict]: [{"race_day", "place_id", "race_name", ...（_race_detail_summary
            の戻り値）}, ...]（日付昇順）。予想・確定結果が無いレースは含まない。
    """
    races = []
    for race_day in (weekend_end_day - timedelta(days=1), weekend_end_day):
        time_id_df = race_card_dataset_manager.get_race_time_id_list_df(race_day)
        if time_id_df.empty:
            continue
        main_df = time_id_df[
            time_id_df["race_id"].apply(lambda rid: parse_race_id(rid)["race_num"] == 11)
        ].sort_values("race_time")

        for _, row in main_df.iterrows():
            race_id = str(row["race_id"])
            detail = _race_detail_summary(race_day, race_id)
            if detail is None:
                continue
            detail.update(
                {
                    "race_day": race_day,
                    "place_id": parse_race_id(race_id)["place_id"],
                    "race_name": row["race_name"],
                }
            )
            races.append(detail)
    return races
