"""src/logic/html_generator/site_nav_html.py のテスト（オフライン）。"""

from src.logic.html_generator import site_nav_html as n


def test_site_nav_html_default_base_path():
    html = n.site_nav_html()

    assert '<nav class="site-nav">' in html
    assert '<a href="index.html">HOME</a>' in html
    assert '<a href="races/index.html">レースカレンダー</a>' in html
    assert '<a href="performance/index.html">AI成績</a>' in html
    assert '<a href="courses/index.html">コース詳細データ</a>' in html


def test_site_nav_html_with_nested_base_path():
    html = n.site_nav_html(base_path="../../")

    assert '<a href="../../index.html">HOME</a>' in html
    assert '<a href="../../performance/index.html">AI成績</a>' in html
