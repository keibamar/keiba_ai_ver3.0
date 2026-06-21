"""サイト共通ナビゲーション（Forge: HTMLFactory）のHTML断片

Home・AI成績・コース詳細データの各ページ群を横断して同じナビゲーションを表示し、
ページごとの「&larr; ◯◯へ戻る」リンクだけに頼らずどのページからでも主要セクションに
アクセスできるようにする。daily_index_generator.calendar_widget_html(base_path=...)と
同じ「埋め込み先ページの階層に応じた相対パスを引数で受け取る」パターンを踏襲する。
"""

NAV_LINKS = [
    ("HOME", "index.html"),
    ("レースカレンダー", "races/index.html"),
    ("AI成績", "performance/index.html"),
    ("コース詳細データ", "courses/index.html"),
]


def site_nav_html(base_path=""):
    """サイト共通ナビゲーションのHTML断片を返す

    競馬場・コースのページ検索ボックス（search_box_html）と、それを動かすための
    スクリプトタグも併せて返す。site_nav_htmlはどのページでも必ず1回呼ばれるため、
    ここにまとめることで検索ボックスをページ生成側の個別対応なしに全ページへ
    行き渡らせる。

    Args:
        base_path (str): 埋め込み先ページからpublic_html直下までの相対パス
            （Home直下なら""、1階層下なら"../"、2階層下なら"../../"等）。
    """
    links = "\n  ".join(f'<a href="{base_path}{path}">{label}</a>' for label, path in NAV_LINKS)
    return f"""<nav class="site-nav">
  {links}
  {search_box_html(base_path)}
</nav>
<script src="{base_path}assets/js/page-search-index.js"></script>
<script src="{base_path}assets/js/page-search.js"></script>"""


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
    all_items = [("HOME", "index.html")] + list(items)
    crumbs = []
    for label, path in all_items:
        if path is None:
            crumbs.append(f'<span class="breadcrumb-current">{label}</span>')
        else:
            crumbs.append(f'<a href="{base_path}{path}">{label}</a>')
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
