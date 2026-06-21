"""src/logic/html_generator/site_nav_html.py のテスト（オフライン）。"""

from src.logic.html_generator import site_nav_html as n


def test_site_nav_html_default_base_path():
    html = n.site_nav_html()

    assert '<nav class="site-nav">' in html
    assert '<a href="index.html">HOME</a>' in html
    assert '<a href="races/index.html">レースカレンダー</a>' in html
    assert '<a href="performance/index.html">AI成績</a>' in html
    assert '<a href="courses/index.html">コース詳細データ</a>' in html
    # 検索ボックスとそれを動かすスクリプトタグが、ナビゲーションに常に付随する
    assert '<div class="page-search" data-base-path="">' in html
    assert '<input type="text" id="page-search-input"' in html
    assert '<script src="assets/js/page-search-index.js"></script>' in html
    assert '<script src="assets/js/page-search.js"></script>' in html


def test_site_nav_html_with_nested_base_path():
    html = n.site_nav_html(base_path="../../")

    assert '<a href="../../index.html">HOME</a>' in html
    assert '<a href="../../performance/index.html">AI成績</a>' in html
    assert '<div class="page-search" data-base-path="../../">' in html
    assert '<script src="../../assets/js/page-search-index.js"></script>' in html
    assert '<script src="../../assets/js/page-search.js"></script>' in html


def test_search_box_html_renders_input_and_results_container():
    html = n.search_box_html(base_path="../")

    print(f"\n--- search_box_html ---\n{html}")

    assert '<div class="page-search" data-base-path="../">' in html
    assert 'id="page-search-input"' in html
    assert '<ul id="page-search-results" class="page-search-results" hidden></ul>' in html


def test_breadcrumb_html_includes_home_and_intermediate_links():
    html = n.breadcrumb_html(
        [("コース詳細データ", "courses/index.html"), ("東京", "courses/05_tokyo/index.html"), ("芝1400m", None)],
        base_path="../../",
    )

    print(f"\n--- breadcrumb_html ---\n{html}")

    assert '<p class="breadcrumb">' in html
    assert '<a href="../../index.html">HOME</a>' in html
    assert '<a href="../../courses/index.html">コース詳細データ</a>' in html
    assert '<a href="../../courses/05_tokyo/index.html">東京</a>' in html
    # 最後の要素（現在地）はリンクを張らず強調表示する
    assert '<span class="breadcrumb-current">芝1400m</span>' in html
    assert '<a href="../../芝1400m">' not in html
    assert "&rsaquo;" in html


def test_breadcrumb_html_default_base_path():
    html = n.breadcrumb_html([("AI成績", "performance/index.html"), ("2026年", None)])
    assert '<a href="performance/index.html">AI成績</a>' in html
    assert '<span class="breadcrumb-current">2026年</span>' in html


def test_sidebar_html_renders_single_section_and_highlights_current():
    html = n.sidebar_html(
        [("東京のコース", [("芝1400m", "芝-1400.html"), ("芝1600m", "芝-1600.html")], "芝1400m")],
    )

    print(f"\n--- sidebar_html(1セクション) ---\n{html}")

    assert '<aside class="page-sidebar">' in html
    assert "<h3>東京のコース</h3>" in html
    # 現在地はリンクにせず強調表示する
    assert '<li><span class="page-sidebar-current">芝1400m</span></li>' in html
    assert '<li><a href="芝-1600.html">芝1600m</a></li>' in html
    assert '<a href="芝-1400.html">' not in html
    # up_linkを指定していないので「上に戻る」リンクは出ない
    assert "page-sidebar-up" not in html


def test_sidebar_html_renders_multiple_sections_for_hierarchy():
    html = n.sidebar_html(
        [
            ("競馬場", [("東京", "../05_tokyo/index.html"), ("中山", "../06_nakayama/index.html")], "東京"),
            ("東京のコース", [("芝1400m", "芝-1400.html"), ("芝1600m", "芝-1600.html")], "芝1400m"),
        ],
        up_link=("東京のコース一覧", "index.html"),
    )

    print(f"\n--- sidebar_html(2セクション+up_link) ---\n{html}")

    # 上の階層（競馬場一覧）に戻るリンクが最上部にある
    assert html.index("page-sidebar-up") < html.index("<h3>競馬場</h3>")
    assert '<p class="page-sidebar-up"><a href="index.html">&uarr; 東京のコース一覧</a></p>' in html
    # 競馬場セクション：現在地（東京）は強調、他競馬場（中山）はリンク
    assert "<h3>競馬場</h3>" in html
    assert '<li><span class="page-sidebar-current">東京</span></li>' in html
    assert '<li><a href="../06_nakayama/index.html">中山</a></li>' in html
    # コースセクション：現在地（芝1400m）は強調、他コース（芝1600m）はリンク
    assert "<h3>東京のコース</h3>" in html
    assert '<li><span class="page-sidebar-current">芝1400m</span></li>' in html
    assert '<li><a href="芝-1600.html">芝1600m</a></li>' in html


def test_sidebar_html_without_current_label_links_everything():
    html = n.sidebar_html(
        [("他の競馬場", [("東京", "../05_tokyo/index.html"), ("中山", "../06_nakayama/index.html")], None)],
    )

    assert '<a href="../05_tokyo/index.html">東京</a>' in html
    assert '<a href="../06_nakayama/index.html">中山</a>' in html
    assert "page-sidebar-current" not in html
