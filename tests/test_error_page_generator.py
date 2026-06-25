"""src/logic/html_generator/error_page_generator.py のテスト（オフライン）。

404ページ（public_html/404.html）が、他ページと同じヘッダー（検索ボックス含む）・
フッター・右側タブを含む形で生成されることを確認する。
"""

import pytest

from src.config import paths
from src.logic.html_generator import error_page_generator as error_page


@pytest.fixture
def new_roots(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "PUBLIC_HTML_PATH", str(tmp_path / "public_html"))
    return tmp_path


def test_make_404_page_generates_html(new_roots):
    error_page.make_404_page()

    out_file = new_roots / "public_html" / "404.html"
    assert out_file.exists()
    html_content = out_file.read_text(encoding="utf-8")

    print(f"\n--- make_404_page() ---\n{html_content[:500]}")

    # サイト共通ヘッダー（検索ボックス含む）・フッター・右側タブ（他ページと統一）
    assert '<meta name="viewport" content="width=device-width, initial-scale=1">' in html_content
    assert "pagead2.googlesyndication.com" in html_content
    assert 'rel="icon"' in html_content
    assert "googletagmanager.com/gtag/js?id=G-DNC949064T" in html_content
    assert '<nav class="site-nav">' in html_content
    assert '<div class="page-search"' in html_content
    assert '<aside class="page-calendar-tab">' in html_content
    assert "<footer>" in html_content

    # 検索エンジンにインデックスさせない
    assert '<meta name="robots" content="noindex">' in html_content

    # 見出し・案内リンク
    assert "<h1>404 - ページが見つかりません</h1>" in html_content
    assert '<li><a href="./">HOME</a></li>' in html_content
    assert '<li><a href="races/index.html">レースカレンダー</a></li>' in html_content
    assert '<li><a href="courses/index.html">コース詳細データ</a></li>' in html_content
    assert '<li><a href="performance/index.html">AI成績</a></li>' in html_content
    assert '<a href="./">&larr; HOMEへ戻る</a>' in html_content
