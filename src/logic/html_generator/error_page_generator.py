"""404エラーページ（Forge: HTMLFactory）のHTML生成

他ページと同じsite_nav_html/site_footer_htmlを使い、ヘッダー（検索ボックス含む）・
フッター・右側タブの統一を保つ。検索エンジンにインデックスさせないよう
noindexを付ける（存在しないURLのページが検索結果に出ないようにするため）。

public_html/.htaccess（ErrorDocument 404 /404.html）と組み合わせて、
ConoHa（Apache）側で実際に404発生時にこのページが表示されるようにする。
"""

from src.logic.html_generator.site_nav_html import (
    SITE_NAME,
    SITE_URL,
    adsense_script_html,
    ga4_script_html,
    meta_tags_html,
    site_footer_html,
    site_nav_html,
)
from src.managers import html_manager

NOT_FOUND_DESCRIPTION = "お探しのページが見つかりませんでした。URLが変更または削除された可能性があります。"


def not_found_template():
    """404ページ（public_html/404.html）のHTMLを返す"""
    return f"""
<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="robots" content="noindex">
  {meta_tags_html(
      f"ページが見つかりません｜{SITE_NAME}",
      NOT_FOUND_DESCRIPTION,
      f"{SITE_URL}/404.html",
  )}
  {adsense_script_html()}
  {ga4_script_html()}
  <title>ページが見つかりません｜{SITE_NAME}</title>
  <link rel="stylesheet" href="assets/css/styles.css">
  <link rel="icon" type="image/svg+xml" href="assets/favicon.svg">
</head>
<body>
  {site_nav_html(base_path="")}
  <h1>404 - ページが見つかりません</h1>
  <p>{NOT_FOUND_DESCRIPTION}</p>

  <h2>よく見られているページ</h2>
  <ul>
    <li><a href="./">HOME</a></li>
    <li><a href="races/index.html">レースカレンダー</a></li>
    <li><a href="courses/index.html">コース詳細データ</a></li>
    <li><a href="performance/index.html">AI成績</a></li>
  </ul>

  <p><a href="./">&larr; HOMEへ戻る</a></p>
  {site_footer_html()}
</body>
</html>
"""


def make_404_page():
    """public_html/404.html を生成する"""
    html_manager.save_404_html(not_found_template())
