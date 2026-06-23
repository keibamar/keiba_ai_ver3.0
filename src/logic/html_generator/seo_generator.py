"""robots.txt・sitemap.xml（Forge: HTMLFactory）の生成

検索エンジンへのインデックス登録のために必要な2ファイルを生成する。
sitemap.xmlは、ページ生成ロジックを個別に再現するのではなく、
public_html配下に実際に生成済みの.htmlファイルを列挙することで作る
（生成ロジックと内容がズレる心配がなく、確実に実態と一致する）。
"""

import os
from datetime import date

from src.config import paths
from src.logic.html_generator.site_nav_html import SITE_URL


def robots_txt_content():
    """robots.txtの内容を返す（全クローラーに全ページの取得を許可し、sitemapを案内する）"""
    return f"""User-agent: *
Allow: /

Sitemap: {SITE_URL}/sitemap.xml
"""


def _list_html_urls():
    """public_html配下の全.htmlファイルを、サイトのURL一覧として返す（昇順）

    index.htmlは、ディレクトリ自体（末尾"/"）のURLにする
    （https://mar-keiba.com/ と https://mar-keiba.com/index.html の重複URLを
    sitemap上で増やさないため、site_nav_html._home_hrefと同じ方針）。
    """
    urls = []
    for root, _dirs, files in os.walk(paths.PUBLIC_HTML_PATH):
        for filename in sorted(files):
            if not filename.endswith(".html"):
                continue
            rel_dir = os.path.relpath(root, paths.PUBLIC_HTML_PATH).replace("\\", "/")
            rel_dir = "" if rel_dir == "." else f"{rel_dir}/"
            if filename == "index.html":
                urls.append(f"{SITE_URL}/{rel_dir}")
            else:
                urls.append(f"{SITE_URL}/{rel_dir}{filename}")
    return sorted(set(urls))


def sitemap_xml_content():
    """sitemap.xmlの内容を返す"""
    today = date.today().strftime("%Y-%m-%d")
    url_entries = "\n".join(
        f"  <url>\n    <loc>{url}</loc>\n    <lastmod>{today}</lastmod>\n  </url>"
        for url in _list_html_urls()
    )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
        f"{url_entries}\n"
        "</urlset>\n"
    )


def make_robots_txt():
    """public_html/robots.txt を生成する"""
    os.makedirs(paths.PUBLIC_HTML_PATH, exist_ok=True)
    with open(os.path.join(paths.PUBLIC_HTML_PATH, "robots.txt"), "w", encoding="utf-8") as f:
        f.write(robots_txt_content())


def make_sitemap_xml():
    """public_html/sitemap.xml を生成する"""
    os.makedirs(paths.PUBLIC_HTML_PATH, exist_ok=True)
    with open(os.path.join(paths.PUBLIC_HTML_PATH, "sitemap.xml"), "w", encoding="utf-8") as f:
        f.write(sitemap_xml_content())
