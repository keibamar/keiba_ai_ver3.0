"""src/logic/html_generator/home_generator.py のテスト（オフライン）。

Homeページ（public_html/index.html）が、レースカレンダー・AI成績・コース詳細データへの
リンクを含む形で生成されることを確認する。
"""

import pytest

from src.config import paths
from src.logic.html_generator import home_generator as h


@pytest.fixture
def new_roots(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "PUBLIC_HTML_PATH", str(tmp_path / "public_html"))
    return tmp_path


def test_make_home_page_generates_index_html(new_roots):
    h.make_home_page()

    out_file = new_roots / "public_html" / "index.html"
    assert out_file.exists()
    html_content = out_file.read_text(encoding="utf-8")

    print(f"\n--- make_home_page() ---")
    print(f"  出力先: {out_file}")
    print(html_content)

    assert "<h1>MAR(まーる）|競馬AIデータサイト</h1>" in html_content
    assert '<a href="races/index.html">レースカレンダー</a>' in html_content
    assert '<a href="performance/index.html">AI成績</a>' in html_content
    assert '<a href="courses/index.html">コース詳細データ</a>' in html_content
