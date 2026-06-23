"""src/logic/html_generator/legal_pages_generator.py のテスト（オフライン）。

プライバシーポリシー（public_html/privacy.html）・利用規約（public_html/terms.html）が、
他ページと同じヘッダー・フッター・右側タブを含む形で生成されることを確認する。
"""

import pytest

from src.config import paths
from src.logic.html_generator import legal_pages_generator as legal


@pytest.fixture
def new_roots(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "PUBLIC_HTML_PATH", str(tmp_path / "public_html"))
    return tmp_path


def test_make_privacy_policy_page_generates_html(new_roots):
    legal.make_privacy_policy_page()

    out_file = new_roots / "public_html" / "privacy.html"
    assert out_file.exists()
    html_content = out_file.read_text(encoding="utf-8")

    print(f"\n--- make_privacy_policy_page() ---\n{html_content[:500]}")

    # サイト共通ヘッダー・フッター・右側タブ（他ページと統一）
    assert '<meta name="viewport" content="width=device-width, initial-scale=1">' in html_content
    assert '<nav class="site-nav">' in html_content
    assert '<aside class="page-calendar-tab">' in html_content
    assert "<footer>" in html_content
    assert '<a href="terms.html">利用規約はこちら</a>' in html_content
    assert '<a href="index.html">&larr; HOMEへ戻る</a>' in html_content

    # 広告配信・アクセス解析・免責事項についての記載
    assert "<h1>プライバシーポリシー</h1>" in html_content
    assert "Cookie" in html_content
    assert "広告の配信について" in html_content
    assert "免責事項" in html_content


def test_make_terms_page_generates_html(new_roots):
    legal.make_terms_page()

    out_file = new_roots / "public_html" / "terms.html"
    assert out_file.exists()
    html_content = out_file.read_text(encoding="utf-8")

    print(f"\n--- make_terms_page() ---\n{html_content[:500]}")

    assert '<meta name="viewport" content="width=device-width, initial-scale=1">' in html_content
    assert '<nav class="site-nav">' in html_content
    assert '<aside class="page-calendar-tab">' in html_content
    assert "<footer>" in html_content
    assert '<a href="privacy.html">プライバシーポリシーはこちら</a>' in html_content

    assert "<h1>利用規約</h1>" in html_content
    # データの出典（netkeiba等の第三者サイトから収集している旨）についての記載
    assert "データの出典について" in html_content
    assert "的中や回収を保証するものではありません" in html_content
