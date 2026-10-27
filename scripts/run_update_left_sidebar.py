"""左サイドバーを breadcrumb 解析版に更新する一括スクリプト

run_add_left_sidebar.py で挿入した基本版サイドバーを、
各ページの <p class="breadcrumb"> を解析して breadcrumb_items を復元した
階層対応版に差し替える。

レースカード・コース詳細・コース会場インデックスは sibling 情報（同日の他会場、
同会場の他コース）も復元して右サイドバーと同等の表示にする。

Usage:
    python scripts/run_update_left_sidebar.py
"""

import os
import re
import sys
import warnings

warnings.simplefilter("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import paths
from src.config.constants import NAME_LIST, PLACE_LIST
from src.config.lists import COURSE_LISTS
from src.logic.html_generator.race_type_badge_html import course_label_html
from src.logic.html_generator.race_page_generator import _race_card_breadcrumb_items
from src.logic.html_generator.site_nav_html import left_sidebar_html
from src.managers import race_card_dataset_manager

PUBLIC_HTML = paths.PUBLIC_HTML_PATH
INSERT_BEFORE = '<aside class="page-calendar-tab">'

# 既存の <aside class="page-left-sidebar">...</aside> ブロックを一括マッチ
SIDEBAR_RE = re.compile(
    r'<aside\s+class="page-left-sidebar"[^>]*>.*?</aside>\s*\n?',
    re.DOTALL,
)

# ページ種別の正規表現
_RACE_CARD_RE = re.compile(r'^races/(\d{8})/(\d{2}_\w+)R(\d+)\.html$')
_COURSE_DETAIL_RE = re.compile(r'^courses/(\d{2}_\w+)/([^/]+)\.html$')

_HTML_ENTITY = {
    '&rsaquo;': '›', '&amp;': '&', '&lt;': '<',
    '&gt;': '>', '&nbsp;': ' ', '&quot;': '"',
}


def _strip_html(text: str) -> str:
    clean = re.sub(r'<[^>]+>', '', text)
    for ent, ch in _HTML_ENTITY.items():
        clean = clean.replace(ent, ch)
    return clean.strip()


def _base_path(rel: str) -> str:
    return '../' * (len(rel.split('/')) - 1)


def _parse_breadcrumb(html_content: str, depth: int) -> list | None:
    """<p class="breadcrumb"> を解析して breadcrumb_items を返す（HOME 除外）。

    Returns [(label, path_or_None), ...] or None
    """
    m = re.search(r'<p class="breadcrumb">(.*?)</p>', html_content, re.DOTALL)
    if not m:
        return None

    bc = m.group(1)
    base_prefix = '../' * depth  # e.g. '../../'

    items = []
    pat = re.compile(
        r'<a\s[^>]*href="([^"]*)"[^>]*>(.*?)</a>'
        r'|<span\s+class="breadcrumb-current">(.*?)</span>',
        re.DOTALL,
    )
    for hit in pat.finditer(bc):
        if hit.group(1) is not None:
            href = hit.group(1)
            label = _strip_html(hit.group(2))

            # HOMEスキップ
            stripped = href[len(base_prefix):] if href.startswith(base_prefix) else href
            if stripped in ('', './'):
                continue

            items.append((label, stripped))
        else:
            label = _strip_html(hit.group(3))
            items.append((label, None))

    return items or None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ページ種別ごとの breadcrumb_items 構築
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def _venue_siblings_list() -> list:
    """全競馬場を (名前, courses/{key}/index.html) のリストで返す。"""
    return [(NAME_LIST[i], f"courses/{PLACE_LIST[i]}/index.html") for i in range(len(PLACE_LIST))]


def _course_detail_bc_items(rel: str) -> list | None:
    """courses/{venue}/{surface}-{distance}.html → sibling 付き breadcrumb_items"""
    m = _COURSE_DETAIL_RE.match(rel)
    if not m:
        return None
    venue_dir = m.group(1)   # e.g. '05_tokyo'
    filename_noext = m.group(2)  # e.g. '芝-1600'

    if '-' not in filename_noext:
        return None  # index.html や予期外ファイル

    try:
        place_idx = PLACE_LIST.index(venue_dir)
    except ValueError:
        return None
    place_id = place_idx + 1
    place_name = NAME_LIST[place_idx]

    race_type_cur, course_len_cur = filename_noext.split('-', 1)
    current_label = course_label_html(race_type_cur, course_len_cur)

    # この会場の全コース一覧（siblings）
    course_siblings = []
    for rt, cl in COURSE_LISTS[place_idx]:
        is_cur = (rt == race_type_cur and cl == course_len_cur)
        label = course_label_html(rt, cl)
        path = None if is_cur else f"courses/{venue_dir}/{rt}-{cl}.html"
        course_siblings.append((label, path))

    venue_siblings = _venue_siblings_list()

    return [
        ("コース詳細データ", "courses/index.html"),
        (place_name, f"courses/{venue_dir}/index.html", venue_siblings),
        (current_label, None, course_siblings),
    ]


def _course_venue_index_bc_items(rel: str) -> list | None:
    """courses/{venue}/index.html → sibling 付き breadcrumb_items（全コース展開）"""
    m = _COURSE_DETAIL_RE.match(rel)
    if not m or m.group(2) != 'index':
        return None
    venue_dir = m.group(1)

    try:
        place_idx = PLACE_LIST.index(venue_dir)
    except ValueError:
        return None
    place_name = NAME_LIST[place_idx]

    # 会場 siblings: 現在の会場のみ path=None
    venue_siblings = []
    for i, (pname, pkey) in enumerate(zip(NAME_LIST, PLACE_LIST)):
        if i == place_idx:
            venue_siblings.append((pname, None))
        else:
            venue_siblings.append((pname, f"courses/{pkey}/index.html"))

    # コース siblings: 全てリンク（会場インデックスなので current なし）
    course_siblings = [
        (course_label_html(rt, cl), f"courses/{venue_dir}/{rt}-{cl}.html")
        for rt, cl in COURSE_LISTS[place_idx]
    ]

    return [
        ("コース詳細データ", "courses/index.html"),
        (place_name, None, venue_siblings),
        (f"{place_name}のコース", None, course_siblings),
    ]


# race_id の DB 問い合わせをキャッシュ（日付単位）
_race_df_cache: dict = {}


def _get_race_df(date_str: str):
    """指定日の race_id DataFrame をキャッシュ付きで取得する。"""
    if date_str not in _race_df_cache:
        from datetime import datetime as _dt
        try:
            race_day = _dt.strptime(date_str, '%Y%m%d')
            _race_df_cache[date_str] = race_card_dataset_manager.get_race_time_id_list_df(race_day)
        except Exception:
            _race_df_cache[date_str] = None
    return _race_df_cache[date_str]


def _race_card_bc_items(rel: str, content: str) -> list | None:
    """races/{date}/{venue}R{n}.html → sibling 付き breadcrumb_items

    race_id の形式: YYYY{place_id:02d}{kai:02d}{nichi:02d}{race_num:02d}
    [4:6] = place_id が PLACE_LIST の 1 始まりインデックスに対応する。
    """
    m = _RACE_CARD_RE.match(rel)
    if not m:
        return None
    date_str = m.group(1)   # '20261025'
    venue_key = m.group(2)  # '05_tokyo'
    race_num = int(m.group(3))

    try:
        place_idx = PLACE_LIST.index(venue_key)
    except ValueError:
        return None
    place_id = place_idx + 1

    date_display = f"{date_str[:4]}/{date_str[4:6]}/{date_str[6:]}"

    # DB から当日の race_id を取得し、venue と race_num が一致するものを探す
    df_info = _get_race_df(date_str)
    if df_info is None or df_info.empty:
        return None

    ids = df_info['race_id'].astype(str)
    # race_id[4:6] が place_id、[-2:] が race_num
    matching = [rid for rid in ids if int(rid[4:6]) == place_id and int(rid[-2:]) == race_num]
    if not matching:
        return None
    target_id = int(matching[0])

    # レース名をパンくずから取得
    bc_m = re.search(r'<span class="breadcrumb-current">(.*?)</span>', content, re.DOTALL)
    race_name = ''
    if bc_m:
        current_text = re.sub(r'<[^>]+>', '', bc_m.group(1)).strip()
        race_name = re.sub(r'^\d+R\s*', '', current_text)

    try:
        return _race_card_breadcrumb_items(date_str, date_display, place_id, target_id, race_name)
    except Exception:
        return None


def update_file(file_path: str) -> bool:
    """サイドバーを sibling 対応版に更新する。変更があれば True を返す。"""
    with open(file_path, encoding='utf-8') as f:
        content = f.read()

    if INSERT_BEFORE not in content:
        return False  # site_nav_html 非使用ページ

    rel = os.path.relpath(file_path, PUBLIC_HTML).replace('\\', '/')
    depth = len(rel.split('/')) - 1
    bp = _base_path(rel)

    # ページ種別ごとに breadcrumb_items を決定
    is_trend_or_diary = rel.startswith('trend/') or rel.startswith('diary/')

    if is_trend_or_diary:
        # trend/diary は current_path のみ → 月別アコーディオンを維持
        breadcrumb_items = None
    elif _RACE_CARD_RE.match(rel):
        # レースカード: 同日の全会場 + 同会場の全レース
        breadcrumb_items = _race_card_bc_items(rel, content)
    else:
        cm = _COURSE_DETAIL_RE.match(rel)
        if cm:
            fname = cm.group(2)
            if fname == 'index':
                # 会場インデックス: 全会場 + 当該会場の全コース
                breadcrumb_items = _course_venue_index_bc_items(rel)
            elif '-' in fname:
                # コース詳細: 全会場 + 当該コースの siblings
                breadcrumb_items = _course_detail_bc_items(rel)
            else:
                breadcrumb_items = _parse_breadcrumb(content, depth)
        else:
            breadcrumb_items = _parse_breadcrumb(content, depth)

    current_path = rel

    sidebar = left_sidebar_html(
        base_path=bp,
        current_path=current_path,
        breadcrumb_items=breadcrumb_items,
    )

    # 既存サイドバーを削除して新版を挿入
    stripped = SIDEBAR_RE.sub('', content)
    new_content = stripped.replace(INSERT_BEFORE, sidebar + '\n' + INSERT_BEFORE, 1)

    if new_content == content:
        return False

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    return True


def main():
    updated = skipped = errors = 0
    print(f'対象: {PUBLIC_HTML}')

    for root, dirs, files in os.walk(PUBLIC_HTML):
        dirs[:] = sorted(d for d in dirs if not d.startswith('.'))
        for fname in sorted(files):
            if not fname.endswith('.html'):
                continue
            fp = os.path.join(root, fname)
            try:
                if update_file(fp):
                    updated += 1
                    if updated <= 5 or updated % 500 == 0:
                        print(f'  [{updated}] {os.path.relpath(fp, PUBLIC_HTML)}')
                else:
                    skipped += 1
            except Exception as e:
                errors += 1
                print(f'  エラー: {os.path.relpath(fp, PUBLIC_HTML)} → {e}')

    print(f'\n=== 完了 ===  更新: {updated}件  スキップ: {skipped}件  エラー: {errors}件')


if __name__ == '__main__':
    main()
