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

    Args:
        base_path (str): 埋め込み先ページからpublic_html直下までの相対パス
            （Home直下なら""、1階層下なら"../"、2階層下なら"../../"等）。
    """
    links = "\n  ".join(f'<a href="{base_path}{path}">{label}</a>' for label, path in NAV_LINKS)
    return f"""<nav class="site-nav">
  {links}
</nav>"""
