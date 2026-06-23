"""public_html/ への出力を担うManager層（Forge: HTMLFactory）

レースページ・日次インデックスのHTML書き出し・存在確認・日付ディレクトリ一覧を提供する。
旧 web/site/races/ の新しい置き場所。

また、旧 web/src/generators/date_index.py の add_race_day を移植し、
public_html/assets/js/raceDays.js（カレンダーが参照するwindow.racedays）への
日付追加も提供する。
"""

import os
import re
from datetime import date

from src.config import paths
from src.config.constants import PLACE_LIST

DAY_DIR_PATTERN = re.compile(r"^\d{8}$")

RACE_DAYS_JS_PATH = os.path.join(paths.PUBLIC_HTML_ASSETS_PATH, "js", "raceDays.js")


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
