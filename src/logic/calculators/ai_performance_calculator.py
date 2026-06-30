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

import pandas as pd

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


def _is_scratched_finish(finish):
    """着順の文字列が除外・取消（出走しなかった）を表すかどうかを返す

    race_prediction_engine.get_past_race_info_dataと同じ判定基準
    （着順に"除"または"取"を含む）を使う。
    """
    finish_str = str(finish)
    return "除" in finish_str or "取" in finish_str


def calc_race_hit_returns(race_day, race_id, box_num=5):
    """1レース分の的中・回収額を、予想rank1位の馬を基準に計算する

    Args:
        race_day (date): レース開催日
        race_id (str): race_id
        box_num (int): 三連複BOXの頭数（初期値5、上位5位までを軸とする）
    Returns:
        dict | None: {"win": (hit, return), "place": (hit, return), "trio_box": (hit, return)}
            （hitは0/1、returnは100円あたりの配当額）。予想または確定配当が
            存在しない場合、またはAI本命馬（rank=1）が除外・取消で出走しなかった
            場合（的中/不的中ではなく払い戻しのため、的中率・回収率の対象から
            除外したい）はNoneを返す。
    """
    pred_df = race_card_dataset_manager.get_race_cards(race_day, race_id)
    if "rank" not in pred_df.columns or "score" not in pred_df.columns:
        return None

    returns_df = race_info_dataset_manager.get_race_return_csv_for_race(race_id)
    if returns_df.empty:
        return None

    # rank列の値（1等）でAI本命馬を探すと、スコアが同値の馬が複数いる場合に
    # 「本命馬」が1頭に決まらず、的中判定がゆるくなって的中率・回収率が
    # 実態より高く出てしまう。score降順で並べ、常にちょうど1頭を本命馬として
    # 確定させる（三連複BOXも同様にscore降順の上位box_num頭で固定する）。
    sorted_df = pred_df.sort_values("score", ascending=False).reset_index(drop=True)
    sorted_nums = sorted_df["馬番"].astype(int).tolist()
    pick_num = sorted_nums[0] if sorted_nums else None
    box_nums = set(sorted_nums[:box_num])

    if pick_num is not None:
        result_df = race_result_dataset_manager.get_race_id_result(race_id)
        pick_result = result_df[result_df["馬番"].astype(str) == str(pick_num)] if not result_df.empty else result_df
        if not pick_result.empty and _is_scratched_finish(pick_result.iloc[0]["着順"]):
            return None

    result = {bet_type: (0, 0.0) for bet_type in BET_TYPES}

    win_df = returns_df[returns_df["式別"] == "単勝"].reset_index(drop=True)
    for i in range(len(win_df)):
        num = int(win_df.at[i, "馬番"])
        if num == pick_num:
            result["win"] = (1, float(win_df.at[i, "配当"]))

    place_df = returns_df[returns_df["式別"] == "複勝"].reset_index(drop=True)
    for i in range(len(place_df)):
        num = int(place_df.at[i, "馬番"])
        if num == pick_num:
            payout = place_df.at[i, "配当"]
            if isinstance(payout, str):
                payout = re.sub(r"\D", "", payout)
            result["place"] = (1, float(payout))

    trio_df = returns_df[returns_df["式別"] == "三連複"].reset_index(drop=True)
    for i in range(len(trio_df)):
        num_list = re.findall(r"\d+", trio_df.at[i, "馬番"])
        if all(int(num) in box_nums for num in num_list):
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


def get_meeting_info(place_id, race_day):
    """指定した開催場・日付が、開催の第何回・何日目にあたるかを返す

    race_schedule_dataset_manager.get_race_calendar(year)の"days"列（開催開始からの
    レース日数、get_current_meeting_summariesと同じ列）をそのまま使う。該当する
    開催日程が見つからない場合（出馬表のみ先行公開されている等）はNoneを返す。

    Args:
        place_id (int): 開催場のplace_id
        race_day (date): 対象日
    Returns:
        dict | None: {"times": 第何回, "day_number": 何日目}
    """
    calendar = race_schedule_dataset_manager.get_race_calendar(race_day.year)
    if calendar.empty:
        return None
    match = calendar[
        (calendar["course"].astype(int) == place_id)
        & (calendar["month"].astype(int) == race_day.month)
        & (calendar["day"].astype(int) == race_day.day)
    ]
    if match.empty:
        return None
    return {"times": int(match.iloc[0]["times"]), "day_number": int(match.iloc[0]["days"])}


def get_today_main_races_with_course(today=None):
    """今日のメインレース（11R）を、レース名・発走時刻・コース情報付きで返す

    今日のレースはまだ確定結果も予想データも無いため、calc_race_hit_returns
    （的中判定）は使えない。コース詳細データへのリンクに使うrace_type/course_lenは、
    まずレースカード作成時に保存済みのレース情報（race_card_dataset_manager.
    get_race_info_csv）を見る。まだ保存されていない場合のみ出走馬一覧ページ
    （shutuba.html、netkeiba_scraper.scrape_race_card）を当日スクレイピングする
    （shutuba.htmlはレース終了後しばらくするとnetkeiba側で参照できなくなるため、
    レース終了から数日経った「今週のメインレース」表示でライブスクレイピングのみに
    依存するとコース情報が取得できなくなる不具合があった）。取得に失敗したレースは
    race_type/course_lenをNoneにして残す（呼び出し側でコースへのリンクなしの表示に
    切り替えられるようにする）。

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
        # grade（G1/G2/G3）はtime_id_df保存時点の値をまず使い、後段の取得で
        # 上書きできればその値を使う（race_time_id_listが古い保存形式（grade列が
        # 無い時期のもの）の場合のフォールバックとして、保存済みの値を先に見ておく）。
        grade = row.get("grade")
        grade = grade if pd.notna(grade) else None

        cached_info_df = race_card_dataset_manager.get_race_info_csv(race_id)
        if not cached_info_df.empty:
            race_type = cached_info_df.iloc[0]["race_type"]
            course_len = cached_info_df.iloc[0]["course_len"]
            cached_grade = cached_info_df.iloc[0].get("grade")
            if pd.notna(cached_grade):
                grade = cached_grade
        else:
            try:
                _, race_info_df, _ = netkeiba_scraper.scrape_race_card(race_id)
                if not race_info_df.empty:
                    race_type = race_info_df.iloc[0]["race_type"]
                    course_len = race_info_df.iloc[0]["course_len"]
                    grade = race_info_df.iloc[0].get("grade", grade)
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
                "grade": grade,
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


def current_meeting_reference_day(today=None):
    """「開催場」表示（開催中の競馬場の成績・今週の開催情報）に使う基準日を返す

    開催（土日）が終わった直後はまだ今週の開催回が実施されていないため、今週の
    水曜日までは直近に終わった週末（日曜日）を基準日とし、木曜日になった時点で
    今週の週末（日曜日。まだ開催前でもスケジュール自体は確定しているため使える）
    に切り替える（結果の確定には数日かかるcurrent_results_weekend_endとは異なり、
    開催そのものは終わった時点で確定しているため、追加の確認待ち期間は設けない）。

    Args:
        today (date): 基準日（初期値: 今日）。
    Returns:
        date: 開催場表示に使う基準日。
    """
    today = today or date.today()
    monday = today - timedelta(days=today.weekday())
    thursday = monday + timedelta(days=3)
    if today >= thursday:
        return monday + timedelta(days=6)
    return monday - timedelta(days=1)


def get_current_meeting_summaries(today=None):
    """「今週の開催」（開催場・第○回・土曜/日曜それぞれの○日目）の一覧を返す

    current_meeting_reference_dayが返す週末（水曜までは先週までの週末、木曜から
    今週の週末）について、開催場・開催回と、その週末の土曜・日曜それぞれが開催の
    何日目にあたるかをまとめて返す（Homeページ最上部の「今週の開催」用）。
    土曜・日曜のうち、実際にその開催回のレース日にあたる日のみdaysに含める
    （日曜しか開催が無い場合などは1日分のみになる）。

    Args:
        today (date): 基準日（初期値: 今日）。
    Returns:
        list[dict]: [{"place_id", "times", "days": [{"day_date", "day_number"}, ...]}, ...]
            （place_id昇順、daysは日付昇順）。
    """
    today = today or date.today()
    sunday = current_meeting_reference_day(today)
    saturday = sunday - timedelta(days=1)

    meetings_by_key = {}
    for day in (saturday, sunday):
        for meeting in get_current_meetings(day):
            meetings_by_key[(meeting["place_id"], meeting["times"])] = meeting
    if not meetings_by_key:
        return []

    calendar = race_schedule_dataset_manager.get_race_calendar(sunday.year)
    summaries = []
    for place_id, times in sorted(meetings_by_key):
        days = []
        for day in (saturday, sunday):
            match = calendar[
                (calendar["course"].astype(int) == place_id)
                & (calendar["times"].astype(int) == times)
                & (calendar["month"].astype(int) == day.month)
                & (calendar["day"].astype(int) == day.day)
            ]
            if not match.empty:
                days.append({"day_date": day, "day_number": int(match.iloc[0]["days"])})
        summaries.append({"place_id": place_id, "times": times, "days": days})
    return summaries


def _race_detail_summary(race_day, race_id):
    """1レース分の詳細サマリー（コース・馬場・クラス・勝ち馬・AI本命馬・本命馬の人気/着順・
    単勝/複勝の的中結果）を返す

    Args:
        race_day (date): レース開催日
        race_id (str): race_id
    Returns:
        dict | None: 予想（rank列）が無ければNone。確定結果（race_result）側の
            データが取得できないレースでも、配当（race_return）からの的中判定は
            可能な場合があるため、その場合は"race_type"等をNoneにした上で返す
            （この開催に予想・配当データが存在するのに詳細ページから抜け落ちる
            ことを避けるため。data/race_result/側に一部抜けがあるのが既知の問題）。
            {"race_type", "course_len", "ground_state", "class",
            "winner_name", "pick_name", "pick_pop", "pick_finish",
            "win_hit", "win_payout", "place_hit", "place_payout"}
    """
    pred_df = race_card_dataset_manager.get_race_cards(race_day, race_id)
    if "rank" not in pred_df.columns or "score" not in pred_df.columns or pred_df.empty:
        return None

    # rank==1の馬を探すと、スコアが同値の馬が複数いる場合に本命馬が1頭に
    # 決まらない（calc_race_hit_returnsと同じ理由）ため、score降順で確定させる
    pick_row = pred_df.sort_values("score", ascending=False).iloc[0]
    pick_name = pick_row["馬名"]
    pick_num = int(pick_row["馬番"])

    result_df = race_result_dataset_manager.get_race_id_result(race_id)
    if result_df.empty:
        result_info = {}
        winner_name = None
        pick_finish = None
        pick_pop = None
    else:
        result_info = result_df.iloc[0]
        winner_row = result_df[result_df["着順"].astype(str) == "1"]
        winner_name = winner_row.iloc[0]["馬名"] if not winner_row.empty else None
        pick_result = result_df[result_df["馬番"].astype(str) == str(pick_num)]
        pick_finish = pick_result.iloc[0]["着順"] if not pick_result.empty else None
        pick_pop = pick_result.iloc[0]["人気"] if not pick_result.empty else None

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
        "race_type": result_info.get("race_type"),
        "course_len": result_info.get("course_len"),
        "ground_state": result_info.get("ground_state"),
        "class": result_info.get("class"),
        "winner_name": winner_name,
        "pick_name": pick_name,
        "pick_pop": pick_pop,
        "pick_finish": pick_finish,
        # 除外・取消（出走しなかった）の場合、単勝/複勝は的中/不的中ではなく
        # 払い戻しとなるため、表示側で「-」にできるようフラグを立てる
        "pick_scratched": _is_scratched_finish(pick_finish) if pick_finish is not None else False,
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
            grade = row.get("grade")
            detail.update(
                {
                    "race_day": race_day,
                    "place_id": parse_race_id(race_id)["place_id"],
                    "race_name": row["race_name"],
                    "grade": grade if pd.notna(grade) else None,
                }
            )
            races.append(detail)
    return races


def get_meeting_race_details(race_day_ids):
    """指定したレース（開催日・race_idのペア一覧）の詳細サマリーを、開催日ごとにまとめて返す

    開催別成績ページ（開催のTOTALの下に各開催日のレース詳細を並べる）用。
    呼び出し側（ai_performance_report_generator）が、永続化済みのai_performance
    データセット（ai_performance_dataset_manager.filter_by_meeting等）からその開催の
    実際のrace_day・race_idを渡す想定（race_calendar側の「開催日目」とdata/race_result/
    側の実際の日番号がズレているケースがあり、calendarから独自に日付・race_idを
    再構築すると一部のレースが取得できなくなるため、永続化データセット側を正とする）。

    日付は新しい順（直近の開催日が先頭）、各日のレースはレース番号の昇順で返す。
    予想（rank列）または確定結果が無いレースは含まない（_race_detail_summary参照）。

    Args:
        race_day_ids (list[tuple[date, str]]): (race_day, race_id) のペア一覧。
    Returns:
        list[dict]: [{"race_day": date, "races": [detail, ...]}, ...]（日付降順）。
            detail は _race_detail_summary の戻り値に "race_id"・"race_name" を加えたもの。
    """
    races_by_day = {}
    for race_day, race_id in race_day_ids:
        races_by_day.setdefault(race_day, []).append(str(race_id))

    result = []
    for race_day in sorted(races_by_day, reverse=True):
        race_id_list = sorted(races_by_day[race_day])
        time_id_df = race_card_dataset_manager.get_race_time_id_list_df(race_day)
        race_names = (
            dict(zip(time_id_df["race_id"].astype(str), time_id_df["race_name"]))
            if not time_id_df.empty
            else {}
        )

        races = []
        for race_id in race_id_list:
            try:
                detail = _race_detail_summary(race_day, race_id)
            except Exception:
                # data/race_result/ 側に一部重複行が混入しているレースがあり、
                # get_race_id_resultが例外を投げることがある（既知のデータ品質問題）。
                # 1件の異常データで開催全体の表示が止まらないよう、そのレースだけ読み飛ばす。
                continue
            if detail is None:
                continue
            detail["race_id"] = race_id
            detail["race_name"] = race_names.get(str(race_id), "")
            races.append(detail)

        if races:
            result.append({"race_day": race_day, "races": races})

    return result
