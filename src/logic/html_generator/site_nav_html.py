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

from datetime import date

from src.logic.html_generator import daily_index_generator

# サイト名（HOMEのタイトル・各ページのヘッダー・フッターで共通して使う）。
# 「MAR」という名前自体をどのページからでも常に伝えられるよう、site_brand_html
# （ヘッダー）・site_footer_html（フッター）の両方でこの名前を使う。
SITE_NAME = "MAR"
SITE_NAME_READING = "まーる"
SITE_TAGLINE = "競馬AIデータサイト"
# HOMEの<title>・<h1>で使ってきた既存の表記（半角(・全角）・半角|の組み合わせ）を
# そのまま保つ。新規のヘッダー（site_brand_html）・フッターはSITE_NAME/SITE_TAGLINEを
# 使って半角に統一した表記にする。
SITE_TITLE = f"{SITE_NAME}({SITE_NAME_READING}）|{SITE_TAGLINE}"

NAV_LINKS = [
    ("HOME", "index.html"),
    ("レースカレンダー", "races/index.html"),
    ("コース詳細データ", "courses/index.html"),
    ("AI成績", "performance/index.html"),
]

# レースカレンダー/コース詳細データ/AI成績を、サイドバー上で色とアイコンで区別する。
# ベースのボルドー（var(--mar-primary)）の雰囲気は変えず、差し色として使うだけ。
NAV_ICONS = {
    "レースカレンダー": "📅",
    "コース詳細データ": "🏟️",
    "AI成績": "📊",
}
NAV_COLOR_CLASSES = {
    "レースカレンダー": "nav-color-calendar",
    "コース詳細データ": "nav-color-courses",
    "AI成績": "nav-color-performance",
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
</nav>
<script src="{base_path}assets/js/page-search-index.js"></script>
<script src="{base_path}assets/js/page-search.js"></script>
{page_calendar_tab_html(base_path, current_path, breadcrumb_items, show_calendar=show_calendar)}
<script src="{base_path}assets/js/calendar-tab-height.js"></script>"""


def site_brand_html(base_path=""):
    """ヘッダー左側に常時表示するサイト名（ブランド表示）のHTML断片を返す

    「MAR」という名前自体をどのページからでも常に伝えられるよう、site_nav_html
    （ヘッダー）に組み込む。HOMEへのリンクも兼ねる。
    """
    return (
        f'<a class="site-brand" href="{base_path}index.html">'
        f'<span class="site-brand-name">{SITE_NAME}</span>'
        f'<span class="site-brand-sub">({SITE_NAME_READING}) {SITE_TAGLINE}</span>'
        f"</a>"
    )


def site_footer_html():
    """サイト共通フッターのHTML断片を返す

    site_nav_htmlと対になる、全ページ共通のフッター。ページ生成側で個別に
    フッターを書く・書き忘れるのを防ぐため、ここに集約する。「MAR」という
    名前に加え、予想・データの利用上の注意（免責事項）と、データ最終更新日時
    （= このページの生成日。週次更新等でデータ取得後にページを再生成する運用のため、
    生成日をそのまま「データ最終更新」の目安として使う）も表示する。
    """
    updated_at = date.today().strftime("%Y/%m/%d")
    return f"""<footer>
    <p class="site-footer-brand">&copy; {SITE_NAME}({SITE_NAME_READING}) {SITE_TAGLINE}</p>
    <p class="site-footer-disclaimer">本サイトの予想・データは参考情報です。的中や回収を保証するものではありません。実際の購入は自己責任でお願いします。</p>
    <p class="site-footer-updated">データ最終更新: {updated_at}</p>
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
  {calendar_html}
  <ul class="page-calendar-tab-location">
    {_location_tree_html(base_path, current_path, breadcrumb_items)}
  </ul>
</aside>"""


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
    home_crumb = _crumb_item_html("HOME", None if home_is_current else "index.html", base_path)

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
    all_items = [("HOME", None if not simple_items else "index.html")] + simple_items
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
    パスを表示する。先頭の"HOME"は常に固定で追加する。

    Args:
        items (list[tuple[str, str | None]]): [(表示名, 相対パス), ...]。
            最後の要素はリンクを張らず現在地として強調表示したいので、
            パスをNoneにする（例: [("コース詳細データ", "courses/index.html"),
            ("東京", "courses/05_tokyo/index.html"), ("芝1400m", None)]）。
        base_path (str): 埋め込み先ページからpublic_html直下までの相対パス。
    """
    crumbs = _hierarchy_crumbs(items, base_path, current_class="breadcrumb-current")
    return f'<p class="breadcrumb">{" &rsaquo; ".join(crumbs)}</p>'


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
