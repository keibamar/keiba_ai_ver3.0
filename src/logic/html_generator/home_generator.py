"""Homeページ（Forge: HTMLFactory）のHTML生成

旧 web/src/generators/global_index.py の移植。旧実装はJinja2テンプレートを
使っていたが、新実装は他のhtml_generator（daily_index_generator等）と同様に
Pythonのf-stringで直接HTMLを組み立てる。

レースカレンダーは public_html/races/index.html（daily_index_generator.
make_races_calendar_page）に独立したページとして置き、Homeはそこへの
リンクのみを持つ（旧 web/site/index.html と同じ構成）。
"""

from src.managers import html_manager

SITE_TITLE = "MAR(まーる）|競馬AIデータサイト"


def home_template():
    return f"""
<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="UTF-8">
  <title>{SITE_TITLE}</title>
  <link rel="stylesheet" href="assets/css/styles.css">
  <style>
    body {{
      font-family: sans-serif;
      margin: 0;
      padding: 0;
      background: #f8f8f8;
    }}
    main {{
      max-width: 1100px;
      margin: 20px auto;
      padding: 20px;
      background: white;
      border-radius: 8px;
    }}
    footer {{
      text-align: center;
      padding: 20px;
      color: #666;
      font-size: 14px;
    }}
  </style>
</head>
<body>
  <main>
    <h1>{SITE_TITLE}</h1>
    <p>このサイトでは、競馬AIの成績、レースカレンダー、コース別データを閲覧できます。</p>

    <h2>メニュー</h2>
    <ul>
      <li><a href="races/index.html">レースカレンダー</a></li>
      <li><a href="performance/index.html">AI成績</a></li>
      <li><a href="courses/index.html">コース詳細データ</a></li>
    </ul>
  </main>
  <footer>
    &copy; 競馬AIデータシステム
  </footer>
</body>
</html>
"""


def make_home_page():
    """public_html/index.html を生成する"""
    html_manager.save_home_html(home_template())
