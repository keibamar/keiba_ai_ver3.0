"""全HTMLページに左サイドバーを挿入する一括注入スクリプト

site_nav_html.left_sidebar_html() で生成した HTML を
<aside class="page-calendar-tab"> の直前に挿入する。
すでに page-left-sidebar を持つファイルはスキップ。

Usage:
    python scripts/run_add_left_sidebar.py
"""

import os
import sys
import warnings

warnings.simplefilter("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import paths
from src.logic.html_generator.site_nav_html import left_sidebar_html

PUBLIC_HTML = paths.PUBLIC_HTML_PATH
INSERT_BEFORE = '<aside class="page-calendar-tab">'
SKIP_MARKER = 'class="page-left-sidebar"'


def _base_path(rel_path: str) -> str:
    """public_html 相対パスから base_path（"../../" 形式）を計算する。"""
    depth = len(rel_path.replace("\\", "/").split("/")) - 1
    return "../" * depth


def inject_sidebar(file_path: str) -> bool:
    """ファイルに左サイドバーを挿入する。変更した場合は True を返す。"""
    with open(file_path, encoding="utf-8") as f:
        content = f.read()

    if SKIP_MARKER in content:
        return False  # 挿入済み

    if INSERT_BEFORE not in content:
        return False  # site_nav_html 非使用ページ（privacy等）

    rel = os.path.relpath(file_path, PUBLIC_HTML).replace("\\", "/")
    bp = _base_path(rel)
    current_path = rel  # current_path でセクション強調（trend/diary は月別展開も自動）

    sidebar = left_sidebar_html(base_path=bp, current_path=current_path)
    new_content = content.replace(INSERT_BEFORE, sidebar + "\n" + INSERT_BEFORE, 1)

    with open(file_path, "w", encoding="utf-8") as f:
        f.write(new_content)
    return True


def main():
    updated = 0
    skipped = 0
    errors = 0
    print(f"対象ディレクトリ: {PUBLIC_HTML}")

    for root, dirs, files in os.walk(PUBLIC_HTML):
        dirs[:] = sorted(d for d in dirs if not d.startswith("."))
        for fname in sorted(files):
            if not fname.endswith(".html"):
                continue
            fpath = os.path.join(root, fname)
            try:
                if inject_sidebar(fpath):
                    updated += 1
                    if updated <= 5 or updated % 500 == 0:
                        rel = os.path.relpath(fpath, PUBLIC_HTML)
                        print(f"  [{updated}] {rel}")
                else:
                    skipped += 1
            except Exception as e:
                errors += 1
                rel = os.path.relpath(fpath, PUBLIC_HTML)
                print(f"  エラー: {rel} → {e}")

    print(f"\n=== 完了 ===  更新: {updated}件  スキップ: {skipped}件  エラー: {errors}件")


if __name__ == "__main__":
    main()
