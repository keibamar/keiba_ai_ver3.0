"""public_html/ への出力を担うManager層（Forge: HTMLFactory）

レースページ・日次インデックスのHTML書き出し・存在確認・日付ディレクトリ一覧を提供する。
旧 web/site/races/ の新しい置き場所。

また、旧 web/src/generators/date_index.py の add_race_day を移植し、
public_html/assets/js/raceDays.js（カレンダーが参照するwindow.racedays）への
日付追加も提供する。
"""

import json
import os
import re
from datetime import date, datetime

import pandas as pd

from src.config import paths
from src.config.constants import NAME_LIST, PLACE_LIST

DAY_DIR_PATTERN = re.compile(r"^\d{8}$")

RACE_DAYS_JS_PATH = os.path.join(paths.PUBLIC_HTML_ASSETS_PATH, "js", "raceDays.js")
RACE_MEETINGS_JS_PATH = os.path.join(paths.PUBLIC_HTML_ASSETS_PATH, "js", "raceMeetings.js")


def get_race_page_dir(date_str):
    """public_html/races/{date_str}/ のパスを返す"""
    return os.path.join(paths.PUBLIC_HTML_RACES_PATH, date_str)


def race_page_exists(date_str, filename):
    """public_html/races/{date_str}/{filename} が存在するか判定する"""
    return os.path.isfile(os.path.join(get_race_page_dir(date_str), filename))


def save_race_page_html(date_str, filename, html_content):
    """public_html/races/{date_str}/{filename} にHTMLを保存する"""
    out_dir = get_race_page_dir(date_str)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, filename), "w", encoding="utf-8") as f:
        f.write(html_content)


def save_daily_index_html(date_str, html_content):
    """public_html/races/{date_str}/index.html にHTMLを保存する"""
    save_race_page_html(date_str, "index.html", html_content)


def save_home_html(html_content):
    """public_html/index.html にHTMLを保存する"""
    os.makedirs(paths.PUBLIC_HTML_PATH, exist_ok=True)
    with open(os.path.join(paths.PUBLIC_HTML_PATH, "index.html"), "w", encoding="utf-8") as f:
        f.write(html_content)


def save_races_calendar_html(html_content):
    """public_html/races/index.html にHTMLを保存する（開催日カレンダーページ）"""
    os.makedirs(paths.PUBLIC_HTML_RACES_PATH, exist_ok=True)
    with open(os.path.join(paths.PUBLIC_HTML_RACES_PATH, "index.html"), "w", encoding="utf-8") as f:
        f.write(html_content)


def save_privacy_policy_html(html_content):
    """public_html/privacy.html にHTMLを保存する"""
    os.makedirs(paths.PUBLIC_HTML_PATH, exist_ok=True)
    with open(os.path.join(paths.PUBLIC_HTML_PATH, "privacy.html"), "w", encoding="utf-8") as f:
        f.write(html_content)


def save_terms_html(html_content):
    """public_html/terms.html にHTMLを保存する"""
    os.makedirs(paths.PUBLIC_HTML_PATH, exist_ok=True)
    with open(os.path.join(paths.PUBLIC_HTML_PATH, "terms.html"), "w", encoding="utf-8") as f:
        f.write(html_content)


def save_about_html(html_content):
    """public_html/about.html にHTMLを保存する"""
    os.makedirs(paths.PUBLIC_HTML_PATH, exist_ok=True)
    with open(os.path.join(paths.PUBLIC_HTML_PATH, "about.html"), "w", encoding="utf-8") as f:
        f.write(html_content)


def save_404_html(html_content):
    """public_html/404.html にHTMLを保存する"""
    os.makedirs(paths.PUBLIC_HTML_PATH, exist_ok=True)
    with open(os.path.join(paths.PUBLIC_HTML_PATH, "404.html"), "w", encoding="utf-8") as f:
        f.write(html_content)


def save_course_index_html(html_content):
    """public_html/courses/index.html にHTMLを保存する（開催場一覧ページ）"""
    os.makedirs(paths.PUBLIC_HTML_COURSES_PATH, exist_ok=True)
    with open(os.path.join(paths.PUBLIC_HTML_COURSES_PATH, "index.html"), "w", encoding="utf-8") as f:
        f.write(html_content)


def save_track_index_html(place_id, html_content):
    """public_html/courses/{place}/index.html にHTMLを保存する（場別コース一覧ページ）"""
    out_dir = os.path.join(paths.PUBLIC_HTML_COURSES_PATH, PLACE_LIST[place_id - 1])
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "index.html"), "w", encoding="utf-8") as f:
        f.write(html_content)


def save_course_detail_html(place_id, race_type, course_len, html_content):
    """public_html/courses/{place}/{race_type}-{course_len}.html にHTMLを保存する"""
    out_dir = os.path.join(paths.PUBLIC_HTML_COURSES_PATH, PLACE_LIST[place_id - 1])
    os.makedirs(out_dir, exist_ok=True)
    filename = f"{race_type}-{course_len}.html"
    with open(os.path.join(out_dir, filename), "w", encoding="utf-8") as f:
        f.write(html_content)


def save_ai_performance_index_html(html_content):
    """public_html/performance/index.html にHTMLを保存する"""
    os.makedirs(paths.PUBLIC_HTML_PERFORMANCE_PATH, exist_ok=True)
    with open(os.path.join(paths.PUBLIC_HTML_PERFORMANCE_PATH, "index.html"), "w", encoding="utf-8") as f:
        f.write(html_content)


def save_ai_annual_performance_html(year, html_content):
    """public_html/performance/annual/{year}.html にHTMLを保存する"""
    out_dir = os.path.join(paths.PUBLIC_HTML_PERFORMANCE_PATH, "annual")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{year}.html"), "w", encoding="utf-8") as f:
        f.write(html_content)


def save_ai_meeting_performance_html(year, place_id, times, html_content):
    """public_html/performance/meeting/{year}/{place}-{times}th.html にHTMLを保存する"""
    out_dir = os.path.join(paths.PUBLIC_HTML_PERFORMANCE_PATH, "meeting", str(year))
    os.makedirs(out_dir, exist_ok=True)
    filename = f"{PLACE_LIST[place_id - 1]}-{times}th.html"
    with open(os.path.join(out_dir, filename), "w", encoding="utf-8") as f:
        f.write(html_content)


def save_ai_course_performance_html(place_id, race_type, course_len, html_content):
    """public_html/performance/course/{place}/{race_type}-{course_len}.html にHTMLを保存する"""
    out_dir = os.path.join(paths.PUBLIC_HTML_PERFORMANCE_PATH, "course", PLACE_LIST[place_id - 1])
    os.makedirs(out_dir, exist_ok=True)
    filename = f"{race_type}-{course_len}.html"
    with open(os.path.join(out_dir, filename), "w", encoding="utf-8") as f:
        f.write(html_content)


def save_ai_course_performance_index_html(place_id, html_content):
    """public_html/performance/course/{place}/index.html にHTMLを保存する"""
    out_dir = os.path.join(paths.PUBLIC_HTML_PERFORMANCE_PATH, "course", PLACE_LIST[place_id - 1])
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "index.html"), "w", encoding="utf-8") as f:
        f.write(html_content)


def list_race_day_dirs():
    """public_html/races/ 配下の8桁日付ディレクトリ名一覧を昇順で返す"""
    if not os.path.isdir(paths.PUBLIC_HTML_RACES_PATH):
        return []
    return sorted(
        name
        for name in os.listdir(paths.PUBLIC_HTML_RACES_PATH)
        if DAY_DIR_PATTERN.match(name) and os.path.isdir(os.path.join(paths.PUBLIC_HTML_RACES_PATH, name))
    )


def add_race_day(race_day):
    """public_html/assets/js/raceDays.js の window.racedays に日付を追加する

    旧 web/src/generators/date_index.py の add_race_day を移植したもの。

    Args:
        race_day (date): 追加する日付
    """
    new_day = race_day.strftime("%Y%m%d")

    if not os.path.exists(RACE_DAYS_JS_PATH):
        os.makedirs(os.path.dirname(RACE_DAYS_JS_PATH), exist_ok=True)
        with open(RACE_DAYS_JS_PATH, "w", encoding="utf-8") as f:
            f.write(f'window.racedays = [\n  "{new_day}"\n];\n')
        return

    with open(RACE_DAYS_JS_PATH, "r", encoding="utf-8") as f:
        content = f.read()

    if "window.racedays" not in content:
        raise ValueError("raceDays.js に window.racedays 配列が見つかりません")

    # すでに存在する場合は追加しない
    if new_day in content:
        return

    # 最後の "]" の直前に追加
    lines = content.strip().splitlines()
    new_content = []
    added = False
    for line in lines:
        if line.strip().startswith("]") and not added:
            # 前の要素の末尾に , がなければ追加
            if not new_content[-1].strip().endswith(","):
                new_content[-1] = new_content[-1] + ","
            new_content.append(f'  "{new_day}"')
            new_content.append("];")
            added = True
        else:
            new_content.append(line)

    with open(RACE_DAYS_JS_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(new_content) + "\n")


def regenerate_race_days_js(min_year=None):
    """public_html/assets/js/raceDays.js をpublic_html/races/配下の実在ディレクトリから再生成する

    add_race_dayはwindow.racedaysに日付を追記していくだけのため、ページ自体が
    削除された日付（実データが無い日）や、去年以前の日付が残り続け、カレンダーに
    実体のないリンクが表示されることがあった（実際に発見・修正した不整合）。
    list_race_day_dirs（実在するディレクトリ一覧）から作り直すことで、
    実体の無いリンクを出さないようにする。

    Args:
        min_year (int | None): この年以降の日付だけを含める（初期値: 今年）。
            「去年以前の日付はカレンダーから見られなくてよい」という方針のため、
            デフォルトでは今年より前の日付を除外する。
    """
    min_year = min_year if min_year is not None else date.today().year
    day_strs = [d for d in list_race_day_dirs() if int(d[:4]) >= min_year]

    os.makedirs(os.path.dirname(RACE_DAYS_JS_PATH), exist_ok=True)
    lines = ",\n".join(f'  "{d}"' for d in day_strs)
    with open(RACE_DAYS_JS_PATH, "w", encoding="utf-8") as f:
        f.write(f"window.racedays = [\n{lines}\n];\n")


def regenerate_race_meetings_js(min_year=None):
    """public_html/assets/js/raceMeetings.js を再生成する

    レースカレンダーページ（races/index.html）の月表示カレンダーで、日付だけでなく
    「その日どこで開催があり、メインレース（11R）は何か」を一覧できるようにするための
    データ。regenerate_race_days_jsと同じ対象日（list_race_day_dirs、min_year以降）に
    ついて、race_card_dataset_manager.get_race_time_id_list_dfから11Rのレース名を、
    race_id（YYYYCCTTDDNN形式）自体から開催回・開催日目を取り出す（カレンダーCSVを
    別途引く必要が無く、当日スクレイピングも不要なため過去日も含めて軽量に作れる）。
    出馬表ページ（races/{day_str}/{place_key}R11.html）が生成済みならrace_card_url
    （public_html直下からの相対パス）も含め、レース名から出馬表ページへ直接
    リンクできるようにする（未生成ならnullにし、呼び出し側でリンクなし表示にする）。
    calendar.jsはwindow.raceMeetingsが無ければ何も表示しないため、サイドバーの
    小さなカレンダー（このデータを読み込まないページ）は今までと同じ見た目のままになる。

    Args:
        min_year (int | None): regenerate_race_days_js参照。
    """
    from src.config.constants import PLACE_LIST
    from src.managers import race_card_dataset_manager

    min_year = min_year if min_year is not None else date.today().year
    day_strs = [d for d in list_race_day_dirs() if int(d[:4]) >= min_year]

    meetings_by_day = {}
    for day_str in day_strs:
        race_day = datetime.strptime(day_str, "%Y%m%d").date()
        time_id_df = race_card_dataset_manager.get_race_time_id_list_df(race_day)
        if time_id_df.empty:
            continue
        main_df = time_id_df[
            time_id_df["race_id"].apply(lambda rid: int(str(rid)[10:12]) == 11)
        ].sort_values("race_time")
        if main_df.empty:
            continue
        meetings = []
        for _, row in main_df.iterrows():
            race_id = str(row["race_id"])
            place_id = int(race_id[4:6])
            place_key = PLACE_LIST[place_id - 1]
            race_card_file = f"{place_key}R11.html"
            race_card_url = (
                f"races/{day_str}/{race_card_file}" if race_page_exists(day_str, race_card_file) else None
            )
            grade = row.get("grade")
            meetings.append(
                {
                    "place_name": NAME_LIST[place_id - 1],
                    "race_name": row["race_name"],
                    "times": int(race_id[6:8]),
                    "day_number": int(race_id[8:10]),
                    "race_card_url": race_card_url,
                    "grade": grade if pd.notna(grade) else None,
                }
            )
        meetings_by_day[day_str] = meetings

    os.makedirs(os.path.dirname(RACE_MEETINGS_JS_PATH), exist_ok=True)
    with open(RACE_MEETINGS_JS_PATH, "w", encoding="utf-8") as f:
        f.write(f"window.raceMeetings = {json.dumps(meetings_by_day, ensure_ascii=False, indent=2)};\n")
