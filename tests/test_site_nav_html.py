"""src/logic/html_generator/site_nav_html.py のテスト（オフライン）。"""

import json
from datetime import date

from src.logic.html_generator import site_nav_html as n


def test_site_nav_html_default_base_path():
    html = n.site_nav_html()

    assert '<nav class="site-nav">' in html
    # HOMEへのリンクはindex.htmlを明示せず、ディレクトリ自体（"./"）を指す
    # （/index.htmlと/の重複URLによるSEO上の評価分散を避けるため）
    assert '<a href="./">HOME</a>' in html
    assert '<a href="races/index.html">レースカレンダー</a>' in html
    assert '<a href="performance/index.html">AI成績</a>' in html
    assert '<a href="courses/index.html">コース詳細データ</a>' in html
    # 検索ボックスとそれを動かすスクリプトタグが、ナビゲーションに常に付随する
    assert '<div class="page-search" data-base-path="">' in html
    assert '<input type="text" id="page-search-input"' in html
    assert '<script src="assets/js/page-search-index.js"></script>' in html
    assert '<script src="assets/js/page-search.js"></script>' in html
    # 右側の小さなカレンダータブ（矢印で前後月へ移動可能）が常に付随する
    assert '<aside class="page-calendar-tab">' in html
    assert '<button id="prevMonth">&larr;</button>' in html
    # スマホ幅ではポップアップ表示にするための開閉ボタン・背景・スクリプト
    assert 'class="page-calendar-tab-toggle"' in html
    assert 'class="page-calendar-tab-backdrop"' in html
    assert 'class="page-calendar-tab-close"' in html
    assert '<script src="assets/js/sidebar-toggle.js"></script>' in html


def test_page_calendar_tab_html_includes_close_button():
    html = n.page_calendar_tab_html(base_path="")

    assert 'class="page-calendar-tab-close"' in html


def test_site_nav_html_top_nav_has_no_duplicate_links_with_sidebar():
    html = n.site_nav_html()

    # 主要セクション（HOME・レースカレンダー等）へのリンクは右側タブに一本化し、
    # 画面上部のnav本体にはブランド表示・検索ボックスのみを置く
    # （サイドバーとの二重表示を避ける）
    nav_only = html[: html.index("</nav>")]
    # HOMEはブランド表示（ロゴ）自体が常にリンク先として持つため対象外にする
    for label, path in n.NAV_LINKS:
        if label == "HOME":
            continue
        assert f'href="{path}"' not in nav_only
    assert '<div class="page-search"' in nav_only


def test_nav_links_order_is_calendar_courses_performance():
    assert [label for label, _ in n.NAV_LINKS] == ["HOME", "レースカレンダー", "コース詳細データ", "AI成績"]


def test_location_tree_always_shows_icons_for_each_section():
    html = n.page_calendar_tab_html(base_path="")

    assert '<span class="nav-icon">📅</span>' in html
    assert '<span class="nav-icon">🏟️</span>' in html
    assert '<span class="nav-icon">📊</span>' in html


def test_location_tree_highlights_active_section_with_color_class():
    html = n.page_calendar_tab_html(base_path="", breadcrumb_items=[("AI成績", None)])

    assert 'class="nav-color-performance"' in html
    # 他の2つ（現在地ではない）には差し色クラスが付かない
    assert "nav-color-calendar" not in html
    assert "nav-color-courses" not in html


def test_site_nav_html_with_nested_base_path():
    html = n.site_nav_html(base_path="../../")

    assert '<a href="../../">HOME</a>' in html
    assert '<a href="../../performance/index.html">AI成績</a>' in html
    assert '<div class="page-search" data-base-path="../../">' in html
    assert '<script src="../../assets/js/page-search-index.js"></script>' in html
    assert '<script src="../../assets/js/page-search.js"></script>' in html
    assert 'window.CALENDAR_BASE_PATH = "../../";' in html
    assert '<script src="../../assets/js/calendar.js"></script>' in html


def test_site_nav_html_highlights_current_path_in_calendar_tab():
    html = n.site_nav_html(current_path="performance/index.html")

    # 上部のnav本体は常に全項目リンクのまま、カレンダータブ側の現在地表示だけ強調する
    tab_html = html[html.index('<aside class="page-calendar-tab">') :]
    assert '<span class="page-calendar-tab-current">AI成績</span>' in tab_html
    assert '<a href="performance/index.html">AI成績</a>' not in tab_html
    assert '<a href="./">HOME</a>' in tab_html


def test_page_calendar_tab_html_uses_given_base_path_and_calendar_widget():
    html = n.page_calendar_tab_html(base_path="../")

    assert '<aside class="page-calendar-tab">' in html
    assert '<div class="page-calendar-tab-calendar">' in html
    # 大きな月表示カレンダー（calendar.js）の仕組みをそのまま使う
    assert 'window.CALENDAR_BASE_PATH = "../";' in html
    assert '<script src="../assets/js/raceDays.js"></script>' in html
    assert '<script src="../assets/js/calendar.js"></script>' in html
    assert '<table id="calendar"></table>' in html
    assert '<button id="prevMonth">&larr;</button>' in html
    assert '<button id="nextMonth">&rarr;</button>' in html
    # current_path未指定なら何も強調しない
    assert "page-calendar-tab-current" not in html


def test_page_calendar_tab_html_without_current_path_links_everything():
    html = n.page_calendar_tab_html(base_path="")

    assert '<a href="./">HOME</a>' in html
    assert '<a href="races/index.html">レースカレンダー</a>' in html
    assert '<a href="performance/index.html">AI成績</a>' in html
    assert '<a href="courses/index.html">コース詳細データ</a>' in html


def test_page_calendar_tab_html_with_breadcrumb_items_shows_full_hierarchy():
    html = n.page_calendar_tab_html(
        base_path="../../",
        breadcrumb_items=[
            ("コース詳細データ", "courses/index.html"),
            ("東京", "courses/05_tokyo/index.html"),
            ("芝1800m", None),
        ],
    )

    print(f"\n--- page_calendar_tab_html(breadcrumb_items) ---\n{html}")

    # HOMEは常にリンクとして表示され、レースカレンダー/AI成績/コース詳細データは
    # HOMEの1段下に常時表示される（NAV_LINKSはどのページからも消えない）
    assert '<a href="../../">HOME</a>' in html
    assert '<a href="../../races/index.html">レースカレンダー</a>' in html
    assert '<a href="../../performance/index.html">AI成績</a>' in html
    # ページが属する大分類（コース詳細データ）の下に、ページ固有の階層
    # （競馬場・コース）がさらに入れ子で続き、各階層に直接遷移できる
    assert '<a href="../../courses/index.html">コース詳細データ</a>' in html
    assert '<a href="../../courses/05_tokyo/index.html">東京</a>' in html
    # 最後の要素（現在地）はリンクを張らず強調表示する
    assert '<span class="page-calendar-tab-current">芝1800m</span>' in html
    assert '<a href="../../芝1800m">' not in html
    # 階層が深くなるごとに入れ子（page-calendar-tab-sublevel）になっている
    assert html.count("page-calendar-tab-sublevel") == 3


def test_page_calendar_tab_html_with_siblings_shows_all_items_at_that_level():
    html = n.page_calendar_tab_html(
        base_path="../../",
        breadcrumb_items=[
            ("コース詳細データ", "courses/index.html"),
            (
                "東京",
                "courses/05_tokyo/index.html",
                [
                    ("函館", "courses/02_hakodate/index.html"),
                    ("東京", "courses/05_tokyo/index.html"),
                    ("阪神", "courses/09_hanshin/index.html"),
                ],
            ),
            (
                "芝1800m",
                None,
                [
                    ("芝1600m", "courses/05_tokyo/芝-1600.html"),
                    ("芝1800m", None),
                    ("ダート1400m", "courses/05_tokyo/ダート-1400.html"),
                ],
            ),
        ],
    )

    print(f"\n--- page_calendar_tab_html(siblings) ---\n{html}")

    # 競馬場の階層では、東京だけでなく兄弟（函館・阪神）もリンクとして並ぶ
    assert '<a href="../../courses/02_hakodate/index.html">函館</a>' in html
    assert '<a href="../../courses/09_hanshin/index.html">阪神</a>' in html
    # 一致する兄弟（東京）だけがさらに下の階層（コース）へ展開される
    assert '<a href="../../courses/05_tokyo/index.html">東京</a>' in html
    # コースの階層でも、現在地（芝1800m）以外の兄弟（芝1600m・ダート1400m）が並ぶ
    assert '<a href="../../courses/05_tokyo/芝-1600.html">芝1600m</a>' in html
    assert '<a href="../../courses/05_tokyo/ダート-1400.html">ダート1400m</a>' in html
    assert '<span class="page-calendar-tab-current">芝1800m</span>' in html
    # 展開していない兄弟（函館・阪神）はさらに下の階層を持たない
    hakodate_li = html[html.index("函館") - 30 : html.index("函館") + 10]
    assert "page-calendar-tab-sublevel" not in hakodate_li


def test_page_calendar_tab_html_with_empty_breadcrumb_items_highlights_home():
    html = n.page_calendar_tab_html(base_path="", breadcrumb_items=[])

    assert '<span class="page-calendar-tab-current">HOME</span>' in html
    assert '<a href="./">HOME</a>' not in html
    # HOMEが現在地でも、他の3項目は常にリンクとして表示される
    assert '<a href="races/index.html">レースカレンダー</a>' in html
    assert '<a href="performance/index.html">AI成績</a>' in html
    assert '<a href="courses/index.html">コース詳細データ</a>' in html


def test_site_nav_html_passes_breadcrumb_items_to_calendar_tab():
    html = n.site_nav_html(
        base_path="../",
        breadcrumb_items=[("コース詳細データ", None)],
    )

    tab_html = html[html.index('<aside class="page-calendar-tab">') :]
    assert '<span class="page-calendar-tab-current">コース詳細データ</span>' in tab_html
    # 現在地の大分類（コース詳細データ）はアイコン+専用の差し色クラスが付く
    assert 'class="nav-color-courses"' in tab_html
    assert '<span class="nav-icon">🏟️</span>' in tab_html


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
    assert '<a href="../../">HOME</a>' in html
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


def test_breadcrumb_html_includes_breadcrumb_list_json_ld():
    html = n.breadcrumb_html(
        [("コース詳細データ", "courses/index.html"), ("東京", "courses/05_tokyo/index.html"), ("芝1400m", None)],
        base_path="../../",
    )

    assert '<script type="application/ld+json">' in html
    assert '"@type": "BreadcrumbList"' in html


def test_breadcrumb_json_ld_html_omits_item_for_current_page():
    html = n.breadcrumb_json_ld_html(
        [("コース詳細データ", "courses/index.html"), ("東京", "courses/05_tokyo/index.html"), ("芝1400m", None)],
    )
    payload = json.loads(html.removeprefix('<script type="application/ld+json">').removesuffix("</script>"))

    assert payload["@context"] == "https://schema.org"
    assert payload["@type"] == "BreadcrumbList"
    items = payload["itemListElement"]
    assert [item["name"] for item in items] == ["HOME", "コース詳細データ", "東京", "芝1400m"]
    assert items[0]["item"] == f"{n.SITE_URL}/"
    assert items[1]["item"] == f"{n.SITE_URL}/courses/index.html"
    assert items[2]["item"] == f"{n.SITE_URL}/courses/05_tokyo/index.html"
    # 現在地（最後の要素）はitemを持たない
    assert "item" not in items[3]
    assert [item["position"] for item in items] == [1, 2, 3, 4]


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


def test_site_footer_html_renders_footer():
    html = n.site_footer_html()

    print(f"\n--- site_footer_html() ---\n{html}")

    # フッターでも「MAR」という名前を伝える
    assert '<p class="site-footer-brand">&copy; MAR(まーる) 競馬AIデータサイト</p>' in html
    # 予想・データの利用上の注意（免責事項）
    assert '<p class="site-footer-disclaimer">' in html
    assert "的中や回収を保証するものではありません" in html
    # データ最終更新日時（ページ生成日をそのまま使う）
    today_str = date.today().strftime("%Y/%m/%d")
    assert f'<p class="site-footer-updated">データ最終更新: {today_str}</p>' in html
    # プライバシーポリシー・利用規約へのリンク（広告掲載に必要）
    assert '<a href="privacy.html">プライバシーポリシー</a>' in html
    assert '<a href="terms.html">利用規約</a>' in html


def test_site_footer_html_uses_base_path_for_legal_links():
    html = n.site_footer_html(base_path="../../")

    assert '<a href="../../privacy.html">プライバシーポリシー</a>' in html
    assert '<a href="../../terms.html">利用規約</a>' in html


def test_site_footer_html_includes_a8_program_recommendation_when_configured(monkeypatch):
    from src.logic.html_generator import affiliate_html

    monkeypatch.setattr(
        affiliate_html, "A8_PROGRAM_CANDIDATES",
        [{"name": "お名前.com", "url": "https://px.a8.net/svt/ejp?a8mat=TEST", "note": "テスト用"}],
    )

    html = n.site_footer_html()

    assert "お名前.com" in html
    assert "https://px.a8.net/svt/ejp?a8mat=TEST" in html


def test_site_footer_html_omits_a8_program_recommendation_when_not_configured(monkeypatch):
    from src.logic.html_generator import affiliate_html

    monkeypatch.setattr(affiliate_html, "A8_PROGRAM_CANDIDATES", [{"name": "お名前.com", "url": None}])

    html = n.site_footer_html()

    assert "a8-program-recommendation" not in html


def test_site_brand_html_links_to_home_and_shows_mar():
    html = n.site_brand_html(base_path="../")

    print(f"\n--- site_brand_html(base_path='../') ---\n{html}")

    assert html == (
        '<a class="site-brand" href="../">'
        '<span class="site-brand-name">MAR</span>'
        '<span class="site-brand-sub">(まーる) 競馬AIデータサイト</span>'
        "</a>"
    )


def test_site_brand_html_uses_dot_slash_at_root():
    # base_pathが空（HOME自身等）の場合は、href=""（自分自身）ではなく"./"
    # （このディレクトリ自体）を使う
    html = n.site_brand_html(base_path="")

    assert html.startswith('<a class="site-brand" href="./">')


def test_site_nav_html_includes_site_brand():
    html = n.site_nav_html(base_path="../")

    assert '<a class="site-brand" href="../">' in html
    assert '<span class="site-brand-name">MAR</span>' in html


def test_site_nav_html_show_calendar_false_omits_small_calendar():
    html = n.site_nav_html(base_path="../", current_path="races/index.html", show_calendar=False)

    print(f"\n--- site_nav_html(show_calendar=False) ---\n{html}")

    # 小カレンダー（page-calendar-tab-calendar）は省略されるが、階層表示は残る
    assert "page-calendar-tab-calendar" not in html
    assert "calendar-widget" not in html
    assert '<aside class="page-calendar-tab">' in html
    assert "page-calendar-tab-location" in html


def test_site_nav_html_show_calendar_true_by_default_includes_calendar():
    html = n.site_nav_html(base_path="../")

    assert "page-calendar-tab-calendar" in html
    assert "calendar-widget" in html
    assert "page-sidebar-current" not in html


def test_adsense_script_html_includes_client_id():
    html = n.adsense_script_html()

    print(f"\n--- adsense_script_html() ---\n{html}")

    assert html == (
        '<script async src="https://pagead2.googlesyndication.com/pagead/js/adsbygoogle.js'
        f'?client={n.ADSENSE_CLIENT_ID}" crossorigin="anonymous"></script>'
    )
    assert n.ADSENSE_CLIENT_ID == "ca-pub-5124016618612171"


def test_ga4_script_html_includes_measurement_id():
    html = n.ga4_script_html()

    print(f"\n--- ga4_script_html() ---\n{html}")

    assert n.GA4_MEASUREMENT_ID == "G-DNC949064T"
    assert f"https://www.googletagmanager.com/gtag/js?id={n.GA4_MEASUREMENT_ID}" in html
    assert f"gtag('config', '{n.GA4_MEASUREMENT_ID}');" in html
    assert "window.dataLayer = window.dataLayer || [];" in html


def test_meta_tags_html_includes_description_and_ogp():
    html = n.meta_tags_html(
        title="東京 芝1400m コース詳細｜MAR",
        description="東京競馬場 芝1400mのコース傾向・血統・枠番データ分析。",
        url=f"{n.SITE_URL}/courses/05_tokyo/芝-1400.html",
    )

    print(f"\n--- meta_tags_html() ---\n{html}")

    assert '<meta name="description" content="東京競馬場 芝1400mのコース傾向・血統・枠番データ分析。">' in html
    assert '<meta property="og:title" content="東京 芝1400m コース詳細｜MAR">' in html
    assert '<meta property="og:url" content="https://mar-keiba.com/courses/05_tokyo/芝-1400.html">' in html
    assert '<meta property="og:type" content="website">' in html
    assert f'<meta property="og:image" content="{n.OG_IMAGE_URL}">' in html
    assert '<meta name="twitter:card" content="summary_large_image">' in html
    assert f'<meta name="twitter:image" content="{n.OG_IMAGE_URL}">' in html


def test_sns_links_html_includes_x_and_instagram():
    html = n.sns_links_html()

    print(f"\n--- sns_links_html() ---\n{html}")

    assert '<a href="https://x.com/keiba_mar" target="_blank" rel="noopener noreferrer">' in html
    assert '<a href="https://www.instagram.com/__mar_gram__/" target="_blank" rel="noopener noreferrer">' in html


def test_page_calendar_tab_html_includes_sns_links():
    html = n.page_calendar_tab_html(base_path="")

    assert "page-calendar-tab-sns" in html
    assert "x.com/keiba_mar" in html
    assert "instagram.com/__mar_gram__" in html


def test_ad_unit_html_includes_client_and_slot():
    html = n.ad_unit_html("1234567890")

    assert '<div class="ad-unit">' in html
    assert 'data-ad-client="ca-pub-5124016618612171"' in html
    assert 'data-ad-slot="1234567890"' in html
    assert 'data-full-width-responsive="true"' in html
    assert "adsbygoogle = window.adsbygoogle || []" in html


def test_ad_unit_html_uses_custom_css_class():
    html = n.ad_unit_html("1234567890", css_class="ad-unit ad-unit--sidebar")

    assert '<div class="ad-unit ad-unit--sidebar">' in html


def test_sidebar_ad_unit_html_uses_sidebar_slot_and_class():
    html = n.sidebar_ad_unit_html()

    assert '<div class="ad-unit ad-unit--sidebar">' in html
    assert f'data-ad-slot="{n.AD_SLOT_SIDEBAR}"' in html


def test_page_calendar_tab_html_includes_sidebar_ad_unit():
    html = n.page_calendar_tab_html(base_path="")

    assert "ad-unit--sidebar" in html
