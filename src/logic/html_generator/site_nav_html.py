"""サイト共通ナビゲーション（Forge: HTMLFactory）のHTML断片

Home・AI成績・コース詳細データの各ページ群を横断して同じナビゲーションを表示し、
ページごとの「&larr; ◯◯へ戻る」リンクだけに頼らずどのページからでも主要セクションに
アクセスできるようにする。daily_index_generator.calendar_widget_html(base_path=...)と
同じ「埋め込み先ページの階層に応じた相対パスを引数で受け取る」パターンを踏襲する。

また、site_nav_htmlはサイト内の全ページから必ず1回呼ばれるため、ここに
「常に右側に小さく表示するカレンダー（矢印で前後の月へ移動可能）＋現在地（階層）表示」
のタブ（page_calendar_tab_html）も統合し、ページ生成側の個別対応なしに行き渡らせる。
画面上部のnav本体には主要セクションへのリンクを並べず（右側タブに一本化し、
画面TOPとサイドバーでの二重表示を避ける）、検索ボックスのみを置く。大きな月表示
カレンダー自体を持つレースカレンダーページ（races_calendar_template）だけは、
右側タブの小カレンダーを二重表示しないようshow_calendar=Falseを渡す。
"""

import json
import os
from datetime import date

from src.config import paths
from src.logic.html_generator import affiliate_html, daily_index_generator

# サイト名（HOMEのタイトル・各ページのヘッダー・フッターで共通して使う）。
# 「MAR」という名前自体をどのページからでも常に伝えられるよう、site_brand_html
# （ヘッダー）・site_footer_html（フッター）の両方でこの名前を使う。
SITE_NAME = "MAR"
SITE_NAME_READING = "まーる"
SITE_TAGLINE = "競馬AIデータサイト"
SITE_TITLE = f"{SITE_NAME}({SITE_NAME_READING})|{SITE_TAGLINE}"

# 本番のドメイン。canonicalタグ・OGP（og:url）・sitemap.xml（seo_generator）で
# 絶対URLを組み立てる際の基準として共通で使う。
SITE_URL = "https://mar-keiba.com"


# SNS共有時に表示するOGP画像。全ページ共通の固定画像（ページごとの個別画像は持たない）。
OG_IMAGE_URL = f"{SITE_URL}/assets/og-image.png"
OG_IMAGE_WIDTH = 1200
OG_IMAGE_HEIGHT = 630


def meta_tags_html(title, description, url, og_type="website"):
    """検索結果・SNSシェア時の表示用メタタグ（description・OGP・Twitter Card）を返す

    Args:
        title (str): ページタイトル（<title>と揃える）。
        description (str): ページの内容を要約した説明文（120字程度を目安にする）。
        url (str): このページの絶対URL（SITE_URLを基準に組み立てる。例:
            f"{SITE_URL}/courses/05_tokyo/芝-1400.html"）。
        og_type (str): og:type（通常は"website"のままでよい）。
    """
    return f"""<meta name="description" content="{description}">
  <meta property="og:title" content="{title}">
  <meta property="og:description" content="{description}">
  <meta property="og:url" content="{url}">
  <meta property="og:type" content="{og_type}">
  <meta property="og:site_name" content="{SITE_NAME}({SITE_NAME_READING})">
  <meta property="og:locale" content="ja_JP">
  <meta property="og:image" content="{OG_IMAGE_URL}">
  <meta property="og:image:width" content="{OG_IMAGE_WIDTH}">
  <meta property="og:image:height" content="{OG_IMAGE_HEIGHT}">
  <meta name="twitter:card" content="summary_large_image">
  <meta name="twitter:title" content="{title}">
  <meta name="twitter:description" content="{description}">
  <meta name="twitter:image" content="{OG_IMAGE_URL}">"""

# Google AdSenseの所有権確認・広告配信用スクリプト。Googleの推奨に従い、
# 全ページの<head>のできるだけ上部に置く（審査前の所有権確認に必要なため、
# 審査が完了する前から全ページに入れておく）。
ADSENSE_CLIENT_ID = "ca-pub-5124016618612171"


def adsense_script_html():
    """全ページの<head>に置くAdSenseの所有権確認・広告配信用スクリプトタグを返す"""
    return (
        f'<script async src="https://pagead2.googlesyndication.com/pagead/js/adsbygoogle.js'
        f'?client={ADSENSE_CLIENT_ID}" crossorigin="anonymous"></script>'
    )


# 広告ユニット（手動配置・本文中/サイドバー）のスロットID。
# AdSense審査通過後、ダッシュボードで実際の広告ユニットを作成し、
# 下記のプレースホルダー値を実際のdata-ad-slot値に差し替える
# （審査完了・本物のスロットIDに差し替えるまでは広告は表示されない）。
#
# 配置方針（ユーザーに広告が多いと感じさせないための上限）:
#   - 本文中: ページ内の「意味のある区切り」1箇所につき1つまで、1ページ上限3〜4個
#   - サイドバー: PCのみ1つ（モバイルでは右サイドバー自体が縦に流れるため非表示にする）
#   - プライバシーポリシー・利用規約等の低コンテンツページには置かない
AD_SLOT_IN_CONTENT_1 = "0000000001"
AD_SLOT_IN_CONTENT_2 = "0000000002"
AD_SLOT_IN_CONTENT_3 = "0000000003"
AD_SLOT_SIDEBAR = "0000000004"


def ad_unit_html(slot, css_class="ad-unit"):
    """本文中に配置するレスポンシブ広告ユニットのHTMLを返す

    呼び出し側は、本当に意味のある区切り（テーブルの後、セクションの切れ目等）
    でのみ使い、1ページあたりの広告数を抑える（site_nav_html.pyのコメント参照）。
    """
    return f"""<div class="{css_class}">
  <ins class="adsbygoogle"
       style="display:block"
       data-ad-client="{ADSENSE_CLIENT_ID}"
       data-ad-slot="{slot}"
       data-ad-format="auto"
       data-full-width-responsive="true"></ins>
  <script>(adsbygoogle = window.adsbygoogle || []).push({{}});</script>
</div>"""


def sidebar_ad_unit_html():
    """右サイドバー用の広告ユニット（PCのみ表示）を返す"""
    return ad_unit_html(AD_SLOT_SIDEBAR, css_class="ad-unit ad-unit--sidebar")


# Google Analytics 4（GA4）の測定ID。アクセス解析・広告効果測定のため全ページに入れる。
GA4_MEASUREMENT_ID = "G-DNC949064T"


def ga4_script_html():
    """全ページの<head>に置くGoogle Analytics（GA4）のトラッキングタグを返す"""
    return (
        f'<script async src="https://www.googletagmanager.com/gtag/js?id={GA4_MEASUREMENT_ID}"></script>\n'
        "  <script>\n"
        "    window.dataLayer = window.dataLayer || [];\n"
        "    function gtag(){dataLayer.push(arguments);}\n"
        "    gtag('js', new Date());\n"
        f"    gtag('config', '{GA4_MEASUREMENT_ID}');\n"
        "  </script>"
    )

NAV_LINKS = [
    ("HOME", "index.html"),
    ("レースカレンダー", "races/index.html"),
    ("コース詳細データ", "courses/index.html"),
    ("AI成績", "performance/index.html"),
    ("傾向分析日記", "trend/index.html"),
]

# レースカレンダー/コース詳細データ/AI成績を、サイドバー上で色とアイコンで区別する。
# ベースのボルドー（var(--mar-primary)）の雰囲気は変えず、差し色として使うだけ。
NAV_ICONS = {
    "レースカレンダー": "📅",
    "コース詳細データ": "🏟️",
    "AI成績": "📊",
    "傾向分析日記": "📓",
}
NAV_COLOR_CLASSES = {
    "レースカレンダー": "nav-color-calendar",
    "コース詳細データ": "nav-color-courses",
    "AI成績": "nav-color-performance",
    "傾向分析日記": "nav-color-trend",
}


def site_nav_html(base_path="", current_path=None, breadcrumb_items=None, show_calendar=True):
    """サイト共通ナビゲーションのHTML断片を返す

    競馬場・コースのページ検索ボックス（search_box_html）、それを動かすための
    スクリプトタグ、右側の小さなカレンダータブ（page_calendar_tab_html）も併せて
    返す。主要セクションへのリンクは右側タブに一本化しているため、画面上部のnav
    本体には検索ボックスのみを置く。site_nav_htmlはサイト内の全ページで必ず1回
    呼ばれるため、ここにまとめることでページ生成側の個別対応なしに全ページへ
    行き渡らせる（ヘッダー・フッター・右側タブの統一はsite_footer_htmlと合わせて
    この2つの共通部品だけで実現する）。

    Args:
        base_path (str): 埋め込み先ページからpublic_html直下までの相対パス
            （Home直下なら""、1階層下なら"../"、2階層下なら"../../"等）。
        current_path (str | None): NAV_LINKSのpathのうち、現在表示中のページに
            対応するもの（例: "courses/index.html"）。breadcrumb_items未指定時の
            フォールバックとして使う（一致するものを右側タブの現在地表示で強調する）。
        breadcrumb_items (list[tuple[str, str | None]] | None): breadcrumb_htmlと
            同じ形式の階層パス（例: [("コース詳細データ", "courses/index.html"),
            ("東京", "courses/05_tokyo/index.html"), ("芝1800m", None)]）。指定すると
            右側タブの現在地表示がNAV_LINKSの大分類だけでなく、その下の階層
            （競馬場・コース等）まで含めて表示し、各階層へ直接遷移できるようになる。
        show_calendar (bool): Falseにすると右側タブの小カレンダーを省略し、
            階層表示のみにする（レースカレンダーページ自身は大きな月表示
            カレンダーを別途持つため、右側タブで同じカレンダーを二重に
            表示しないようにする用途）。
    """
    return f"""<nav class="site-nav">
  {site_brand_html(base_path)}
  {search_box_html(base_path)}
  <button type="button" class="page-calendar-tab-toggle" aria-label="メニューを開く">&#9776; メニュー</button>
</nav>
<script src="{base_path}assets/js/page-search-index.js"></script>
<script src="{base_path}assets/js/page-search.js"></script>
<div class="page-calendar-tab-backdrop"></div>
{page_calendar_tab_html(base_path, current_path, breadcrumb_items, show_calendar=show_calendar)}
<script src="{base_path}assets/js/calendar-tab-height.js"></script>
<script src="{base_path}assets/js/sidebar-toggle.js"></script>"""


def _home_href(base_path):
    """HOMEへのリンクのhref値を返す（index.htmlを明示しない、ディレクトリ参照）

    https://mar-keiba.com/ と https://mar-keiba.com/index.html はサーバーの
    仕様上同じファイルを指す（index.htmlはディレクトリ要求時の既定ファイルとして
    必須なので削除はできない）が、サイト内のリンクが常に/index.html付きの形を
    指すと、検索エンジンに重複URLとして扱われやすくなる。サイト内リンクは
    常にディレクトリ自体（末尾"/"）を指すようにし、base_pathが空の場合
    （HOME自身やprivacy.html等ルート直下の他ページ）は"./"（このディレクトリ
    自体）を使う。
    """
    return base_path or "./"


def site_brand_html(base_path=""):
    """ヘッダー左側に常時表示するサイト名（ブランド表示）のHTML断片を返す

    「MAR」という名前自体をどのページからでも常に伝えられるよう、site_nav_html
    （ヘッダー）に組み込む。HOMEへのリンクも兼ねる。
    """
    return (
        f'<a class="site-brand" href="{_home_href(base_path)}">'
        f'<span class="site-brand-name">{SITE_NAME}</span>'
        f'<span class="site-brand-sub">({SITE_NAME_READING}) {SITE_TAGLINE}</span>'
        f"</a>"
    )


def site_footer_html(base_path=""):
    """サイト共通フッターのHTML断片を返す

    site_nav_htmlと対になる、全ページ共通のフッター。ページ生成側で個別に
    フッターを書く・書き忘れるのを防ぐため、ここに集約する。「MAR」という
    名前に加え、予想・データの利用上の注意（免責事項）と、データ最終更新日時
    （= このページの生成日。週次更新等でデータ取得後にページを再生成する運用のため、
    生成日をそのまま「データ最終更新」の目安として使う）も表示する。広告掲載に
    必要なプライバシーポリシー・利用規約（legal_pages_generator）へのリンクも
    どのページからでも辿れるようにここに置く。

    A8.net提携プログラムの紹介（affiliate_html.a8_program_recommendation_html）も
    ここに含める。本文中・サイドバーは既に広告等で手狭なため、フッターという
    控えめな位置に1箇所だけ置く（提携プログラムが無い間は空文字列のため、
    何も表示されない）。

    Args:
        base_path (str): 埋め込み先ページからpublic_html直下までの相対パス。
    """
    updated_at = date.today().strftime("%Y/%m/%d")
    return f"""<footer>
    <p class="site-footer-brand">&copy; {SITE_NAME}({SITE_NAME_READING}) {SITE_TAGLINE}</p>
    <p class="site-footer-disclaimer">本サイトの予想・データは参考情報です。的中や回収を保証するものではありません。実際の購入は自己責任でお願いします。</p>
    <p class="site-footer-updated">データ最終更新: {updated_at}</p>
    {affiliate_html.a8_program_recommendation_html()}
    <p class="site-footer-links">
      <a href="{base_path}about.html">このサイトについて</a>
      <a href="{base_path}privacy.html">プライバシーポリシー</a>
      <a href="{base_path}terms.html">利用規約</a>
    </p>
  </footer>"""


def page_calendar_tab_html(base_path="", current_path=None, breadcrumb_items=None, show_calendar=True):
    """ページ右側に常時表示する、小さなカレンダー＋現在地（階層）表示のタブを返す

    カレンダー部分はdaily_index_generator.calendar_widget_htmlをそのまま再利用する
    （矢印での前後月移動・本日のレースへのリンクの仕組みは変えず、CSS側で縮小表示する）。
    現在地表示は、HOME・レースカレンダー・AI成績・コース詳細データの4項目を
    どのページからも常にリンクとして表示しつつ（NAV_LINKS）、breadcrumb_itemsを
    指定すればそのページが属する大分類の下にさらにページ固有の階層
    （競馬場・コース等）を入れ子で表示し、どの階層からも直接遷移できるようにする。

    Args:
        base_path (str): 埋め込み先ページからpublic_html直下までの相対パス。
        current_path (str | None): NAV_LINKSのpathのうち、現在地として強調するもの
            （breadcrumb_items未指定時のみ使う）。
        breadcrumb_items (list[tuple[str, str | None]] | None): breadcrumb_htmlと
            同じ形式の階層パス（先頭の要素のラベルがNAV_LINKSの大分類のいずれかと
            一致する想定。例: [("コース詳細データ", "courses/index.html"),
            ("東京", "courses/05_tokyo/index.html"), ("芝1800m", None)]）。
        show_calendar (bool): Falseにすると小カレンダー部分を省略する
            （site_nav_html参照）。
    """
    calendar_html = (
        f"""<div class="page-calendar-tab-calendar">
    {daily_index_generator.calendar_widget_html(base_path=base_path)}
  </div>"""
        if show_calendar
        else ""
    )
    return f"""<aside class="page-calendar-tab">
  <button type="button" class="page-calendar-tab-close" aria-label="閉じる">&times;</button>
  {calendar_html}
  <ul class="page-calendar-tab-location">
    {_location_tree_html(base_path, current_path, breadcrumb_items)}
  </ul>
  {sns_links_html()}
  {sidebar_ad_unit_html()}
</aside>"""


# SNSアカウント。Xは予想・結果の投稿とサイト更新のお知らせに使う運用のため、
# サイトとの結びつきが強く常時表示する。Instagramは中の人の趣味の投稿で更新頻度も
# 低いが、ユーザーの希望でこちらも表示する。
SNS_LINKS = [
    ("X (Twitter)", "https://x.com/keiba_mar", "𝕏"),
    ("Instagram", "https://www.instagram.com/__mar_gram__/", "📷"),
]


def sns_links_html():
    """右側タブの下部に表示するSNSリンク一覧を返す"""
    items = "".join(
        f'<li><a href="{url}" target="_blank" rel="noopener noreferrer">'
        f'<span class="nav-icon">{icon}</span> {label}</a></li>\n'
        for label, url, icon in SNS_LINKS
    )
    return f'<ul class="page-calendar-tab-sns">\n{items}</ul>'


def _crumb_item_html(label, path, base_path="", current_class="page-calendar-tab-current"):
    """1つの階層項目を、現在地ならspan、そうでなければリンクのHTMLにする"""
    if path is None:
        return f'<span class="{current_class}">{label}</span>'
    return f'<a href="{base_path}{path}">{label}</a>'


def _nested_crumbs_html(items, base_path="", current_class="page-calendar-tab-current"):
    """階層パスを、先頭から順に入れ子の&lt;ul&gt;で階層を表す&lt;li&gt;に変換する

    itemsの各要素は (表示名, 相対パス|None) または (表示名, 相対パス|None, 兄弟一覧) の
    どちらでもよい。3つ目の要素（兄弟一覧、[(表示名, 相対パス|None), ...]）を指定すると、
    その階層に存在する全項目（例: 全競馬場・あるコースの全距離）を並べて表示し、
    その中で現在の項目（1番目・2番目の要素と一致するもの）だけが、続く階層
    （items[1:]）をさらに入れ子で表示する（他の兄弟はリンクのみで展開しない）。

    例えば[("コース詳細データ", "courses/index.html"),
    ("東京", "courses/05_tokyo/index.html", [全競馬場のリスト]),
    ("芝1800m", None, [このコースの全距離のリスト])]なら、コース詳細データ→
    （全競馬場のうち東京のみ）→（東京の全距離のうち芝1800mのみ現在地）という
    形で、各階層の兄弟項目も含めて1段ずつインデントが深くなるリストになる。
    """
    if not items:
        return ""
    item = items[0]
    if len(item) == 3:
        label, path, siblings = item
    else:
        label, path = item
        siblings = [(label, path)]
    rest_html = _nested_crumbs_html(items[1:], base_path, current_class)

    rows = ""
    for sib_label, sib_path in siblings:
        crumb = _crumb_item_html(sib_label, sib_path, base_path, current_class)
        if rest_html and (sib_label, sib_path) == (label, path):
            rows += f'<li>{crumb}\n<ul class="page-calendar-tab-sublevel">\n{rest_html}</ul></li>\n'
        else:
            rows += f"<li>{crumb}</li>\n"
    return rows


def _get_trend_nav_links(base_path: str, n: int = 8) -> list:
    """public_html/trend/ から最新n件の（ラベル, パス）リストを返す"""
    trend_dir = os.path.join(paths.PUBLIC_HTML_PATH, "trend")
    try:
        files = [
            f for f in os.listdir(trend_dir)
            if f.endswith(".html") and f not in ("index.html",)
        ]
    except FileNotFoundError:
        return []
    links = []
    for fname in sorted(files, reverse=True)[:n]:
        is_weekly = fname.startswith("weekly_")
        date_key = fname.replace("weekly_", "").replace(".html", "")
        if len(date_key) != 8:
            continue
        try:
            y, m, d_num = int(date_key[:4]), int(date_key[4:6]), int(date_key[6:8])
            weekday = date(y, m, d_num).weekday()
            wd = ["月", "火", "水", "木", "金", "土", "日"][weekday]
            label = f"{m}/{d_num} 週次振り返り" if is_weekly else f"{m}/{d_num}（{wd}）"
        except (ValueError, IndexError):
            continue
        links.append((label, f"trend/{fname}"))
    return links


def _location_tree_html(base_path="", current_path=None, breadcrumb_items=None):
    """HOMEを根に、NAV_LINKSの3項目を1段下にネストした、常時表示用の階層ツリーを返す

    HOME・レースカレンダー・コース詳細データ・AI成績はどのページからも常にリンク
    として表示する。breadcrumb_itemsが指定され、その先頭ラベルがNAV_LINKSの
    いずれかと一致する場合は、その項目の下にさらにページ固有の階層を入れ子で続ける。
    各大分類にはアイコン（NAV_ICONS）を常に付け、現在その大分類の中にいる場合は
    専用の差し色（NAV_COLOR_CLASSES、CSS側で定義）でその枝全体（入れ子の下の階層も
    含む）を強調し、今どのページ系列にいるかを色とアイコンの両方で分かるようにする。
    """
    home_is_current = breadcrumb_items == [] or (breadcrumb_items is None and current_path == "index.html")
    home_crumb = (
        '<span class="page-calendar-tab-current">HOME</span>'
        if home_is_current
        else f'<a href="{_home_href(base_path)}">HOME</a>'
    )

    active_label = breadcrumb_items[0][0] if breadcrumb_items else None
    sub_rows = ""
    for label, path in NAV_LINKS[1:]:
        icon = NAV_ICONS.get(label, "")
        icon_html = f'<span class="nav-icon">{icon}</span> ' if icon else ""

        if breadcrumb_items is not None and label == active_label:
            head_path = breadcrumb_items[0][1]
            crumb = _crumb_item_html(label, head_path, base_path)
            nested_html = _nested_crumbs_html(breadcrumb_items[1:], base_path)
            is_active = True
        elif label == "傾向分析日記":
            is_current = breadcrumb_items is None and current_path == path
            crumb = _crumb_item_html(label, None if is_current else path, base_path)
            is_active = is_current
            trend_links = _get_trend_nav_links(base_path)
            if trend_links:
                inner = "".join(
                    f"<li>{_crumb_item_html(tl, tp, base_path)}</li>\n"
                    for tl, tp in trend_links
                )
                nested_html = inner
            else:
                nested_html = ""
        else:
            is_current = breadcrumb_items is None and current_path == path
            crumb = _crumb_item_html(label, None if is_current else path, base_path)
            nested_html = ""
            is_active = is_current

        color_class = NAV_COLOR_CLASSES.get(label, "") if is_active else ""
        li_class = f' class="{color_class}"' if color_class else ""
        if nested_html:
            sub_rows += f'<li{li_class}>{icon_html}{crumb}\n<ul class="page-calendar-tab-sublevel">\n{nested_html}</ul></li>\n'
        else:
            sub_rows += f"<li{li_class}>{icon_html}{crumb}</li>\n"

    return f'<li>{home_crumb}\n<ul class="page-calendar-tab-sublevel">\n{sub_rows}</ul></li>\n'


def _hierarchy_crumbs(items, base_path="", current_class="breadcrumb-current"):
    """[(表示名, 相対パス|None), ...]を、先頭にHOMEを補ったHTML断片のリストに変換する

    最後の要素のパスがNoneなら現在地として強調表示する。itemsが空の場合は
    HOME自体が現在地（そのページがHOMEそのもの）とみなす。breadcrumb_htmlが
    横並びのパンくずを作る際にこのロジックを使う。
    """
    simple_items = [(label, path) for label, path, *_ in items]
    home_path = None if not simple_items else ("" if base_path else "./")
    all_items = [("HOME", home_path)] + simple_items
    return [_crumb_item_html(label, path, base_path, current_class) for label, path in all_items]


def search_box_html(base_path=""):
    """競馬場・コースのページへすぐ移動できる検索ボックスのHTML断片を返す

    入力値での絞り込み・結果一覧の表示・クリック/Enterでの遷移は
    assets/js/page-search.js（PAGE_SEARCH_INDEXを参照）が行う。

    Args:
        base_path (str): 埋め込み先ページからpublic_html直下までの相対パス。
            検索結果のパスはサイト直下からの相対パスで保持しているため、
            実際に遷移する際にこのbase_pathを前置する。
    """
    return f"""<div class="page-search" data-base-path="{base_path}">
    <input type="text" id="page-search-input" class="page-search-input"
      placeholder="競馬場・コースを検索（例: 東京 芝1400）" autocomplete="off">
    <ul id="page-search-results" class="page-search-results" hidden></ul>
  </div>"""


def breadcrumb_html(items, base_path=""):
    """現在地の階層を示すブレッドクラム（HOME &gt; ... &gt; 現在地）を返す

    site_nav_html（常時同じ4リンク）とは異なり、ページごとに階層が変わる
    パスを表示する。先頭の"HOME"は常に固定で追加する。検索結果でもパンくずが
    表示されるよう、schema.org BreadcrumbListのJSON-LD（breadcrumb_json_ld）も
    併せて返す（呼び出し側の変更なしに全ページへ行き渡らせるため）。

    Args:
        items (list[tuple[str, str | None]]): [(表示名, 相対パス), ...]。
            最後の要素はリンクを張らず現在地として強調表示したいので、
            パスをNoneにする（例: [("コース詳細データ", "courses/index.html"),
            ("東京", "courses/05_tokyo/index.html"), ("芝1400m", None)]）。
        base_path (str): 埋め込み先ページからpublic_html直下までの相対パス。
    """
    crumbs = _hierarchy_crumbs(items, base_path, current_class="breadcrumb-current")
    visible = f'<p class="breadcrumb">{" &rsaquo; ".join(crumbs)}</p>'
    return f"{visible}\n  {breadcrumb_json_ld_html(items)}"


def breadcrumb_json_ld_html(items):
    """breadcrumb_htmlと同じ階層からschema.org BreadcrumbListのJSON-LDを返す

    検索結果でのパンくず表示（リッチリザルト）向け。現在地（最後の要素、
    パスがNoneのもの）はitemプロパティを省略する（Googleのガイドラインで
    最後の要素のitemは省略可とされているため、このページの絶対URLを
    別途受け取る必要がない）。

    Args:
        items (list[tuple[str, str | None]]): breadcrumb_htmlと同じ形式。
    """
    simple_items = [(label, path) for label, path, *_ in items]
    all_items = [("HOME", "")] + simple_items
    list_items = []
    for position, (label, path) in enumerate(all_items, start=1):
        entry = {"@type": "ListItem", "position": position, "name": label}
        if path is not None:
            entry["item"] = f"{SITE_URL}/{path}"
        list_items.append(entry)
    payload = {
        "@context": "https://schema.org",
        "@type": "BreadcrumbList",
        "itemListElement": list_items,
    }
    return f'<script type="application/ld+json">{json.dumps(payload, ensure_ascii=False)}</script>'


def sidebar_html(sections, up_link=None):
    """ページの階層に応じて切り替わる右サイドバーを返す

    複数セクション（例: 「競馬場」一覧 + 現在の競馬場の「コース」一覧）を並べることで、
    1か所から複数の階層へ行き来できるようにする。さらに先頭に「ひとつ上の階層へ」の
    リンク（up_link）を置くことで、常に上の階層に戻れるようにする。

    Args:
        sections (list[tuple[str, list[tuple[str, str]], str | None]]):
            [(見出し, [(表示名, 相対パス), ...], 現在地の表示名 | None), ...]。
            現在地の表示名と一致する項目はリンクにせず強調表示する。
        up_link (tuple[str, str] | None): (表示名, 相対パス)。指定すると
            サイドバー最上部に「&uarr; 表示名」のリンクを追加する。
    """
    up_html = ""
    if up_link is not None:
        up_label, up_path = up_link
        up_html = f'<p class="page-sidebar-up"><a href="{up_path}">&uarr; {up_label}</a></p>\n  '

    sections_html = ""
    for title, links, current_label in sections:
        items = ""
        for label, path in links:
            if label == current_label:
                items += f'<li><span class="page-sidebar-current">{label}</span></li>\n'
            else:
                items += f'<li><a href="{path}">{label}</a></li>\n'
        sections_html += f"""<h3>{title}</h3>
  <ul>
    {items}
  </ul>
  """

    return f"""<aside class="page-sidebar">
  {up_html}{sections_html}</aside>"""
