"""public_html/ への出力を担うManager層（Forge: HTMLFactory）

レースページ・日次インデックスのHTML書き出し・存在確認・日付ディレクトリ一覧を提供する。
旧 web/site/races/ の新しい置き場所。
"""

import os
import re

from src.config import paths

DAY_DIR_PATTERN = re.compile(r"^\d{8}$")


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


def list_race_day_dirs():
    """public_html/races/ 配下の8桁日付ディレクトリ名一覧を昇順で返す"""
    if not os.path.isdir(paths.PUBLIC_HTML_RACES_PATH):
        return []
    return sorted(
        name
        for name in os.listdir(paths.PUBLIC_HTML_RACES_PATH)
        if DAY_DIR_PATTERN.match(name) and os.path.isdir(os.path.join(paths.PUBLIC_HTML_RACES_PATH, name))
    )
