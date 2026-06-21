"""検索ボックス（site_nav_html.search_box_html）用の静的データファイル生成

PLACE_LIST/NAME_LIST/COURSE_LISTSから全競馬場・全コースのページ一覧
（表示ラベル, サイト直下からの相対パス）を作り、public_html/assets/js/に
静的JSデータファイルとして書き出す。JS側（page-search.js）はこのファイルが
定義する PAGE_SEARCH_INDEX をそのまま部分一致検索の対象にする。

PLACE_LIST/COURSE_LISTSはコードの構成情報であり、競走結果データに依存しないため、
他のページ生成（make_all_course_pages等）とは独立して呼べる。
"""

import json
import os

from src.config import paths
from src.config.constants import NAME_LIST, PLACE_LIST
from src.config.lists import COURSE_LISTS

PAGE_SEARCH_INDEX_JS_PATH = os.path.join(paths.PUBLIC_HTML_ASSETS_PATH, "js", "page-search-index.js")


def build_page_search_index():
    """全競馬場・全コースの(表示ラベル, 相対パス)一覧を返す"""
    entries = []
    for i in range(len(PLACE_LIST)):
        place_name = NAME_LIST[i]
        place_dir = PLACE_LIST[i]
        entries.append({"label": f"{place_name} コース詳細データ", "path": f"courses/{place_dir}/index.html"})
        entries.append({"label": f"{place_name} AI成績", "path": f"performance/course/{place_dir}/index.html"})
        for race_type, course_len in COURSE_LISTS[i]:
            course_label = f"{place_name} {race_type}{course_len}m"
            entries.append({"label": f"{course_label} コース詳細", "path": f"courses/{place_dir}/{race_type}-{course_len}.html"})
            entries.append({"label": f"{course_label} AI成績", "path": f"performance/course/{place_dir}/{race_type}-{course_len}.html"})
    return entries


def make_page_search_index_js():
    """検索インデックスをpublic_html/assets/js/page-search-index.jsに書き出す"""
    entries = build_page_search_index()
    os.makedirs(os.path.dirname(PAGE_SEARCH_INDEX_JS_PATH), exist_ok=True)
    with open(PAGE_SEARCH_INDEX_JS_PATH, "w", encoding="utf-8") as f:
        f.write(f"const PAGE_SEARCH_INDEX = {json.dumps(entries, ensure_ascii=False, indent=2)};\n")
