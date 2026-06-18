"""AI成績データページ（Forge: HTMLFactory）のHTML生成

src.logic.calculators.ai_performance_calculator で計算した的中率・回収率を、
年別・開催別・コース別のページとして表示する。

旧ver2.0の web/site/performance/ 相当だが、的中率・回収率の集計ロジック自体は
ver2.0でも未実装だったため、ページ構成のみを参考にした新規実装。
"""

from datetime import date

from src.config.constants import NAME_LIST, PLACE_LIST
from src.logic.calculators import ai_performance_calculator as calc
from src.managers import html_manager

BET_TYPE_LABELS = {"win": "単勝", "place": "複勝", "trio_box": "三連複(5頭BOX)"}


def _performance_table_html(performance):
    rows = "".join(
        f"<tr><td>{BET_TYPE_LABELS[bet_type]}</td>"
        f"<td>{performance[bet_type]['hit_rate']:.1f}%</td>"
        f"<td>{performance[bet_type]['return_rate']:.1f}%</td>"
        f"<td>{performance[bet_type]['n']}</td></tr>\n"
        for bet_type in BET_TYPE_LABELS
    )
    return f"""<table>
    <thead><tr><th>式別</th><th>的中率</th><th>回収率</th><th>対象レース数</th></tr></thead>
    <tbody>
      {rows}
    </tbody>
  </table>"""


def make_ai_performance_index_page(year=None):
    """AI成績トップページ（public_html/performance/index.html）を生成する"""
    year = year or date.today().year
    place_rows = "".join(
        f'<li><a href="course/{PLACE_LIST[i]}/index.html">{NAME_LIST[i]}</a></li>\n'
        for i in range(len(PLACE_LIST))
    )
    html = f"""
<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="UTF-8">
  <title>AI予想成績</title>
  <link rel="stylesheet" href="../assets/css/styles.css">
</head>
<body>
  <h1>AI予想成績</h1>

  <h2>年間成績</h2>
  <p><a href="annual/{year}.html">{year}年の成績を見る</a></p>

  <h2>コース別成績</h2>
  <ul>
    {place_rows}
  </ul>

  <p><a href="../index.html">&larr; HOMEへ戻る</a></p>
</body>
</html>
"""
    html_manager.save_ai_performance_index_html(html)


def make_annual_performance_page(year):
    """年間成績ページ（public_html/performance/annual/{year}.html）を生成する"""
    pairs = calc.filter_by_year(calc.list_predicted_races(), year)
    performance = calc.aggregate_ai_performance(pairs)

    html = f"""
<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="UTF-8">
  <title>{year}年 AI予想成績</title>
  <link rel="stylesheet" href="../../assets/css/styles.css">
</head>
<body>
  <h1>{year}年 AI予想成績</h1>
  {_performance_table_html(performance)}
  <p><a href="../index.html">&larr; AI成績トップへ</a></p>
</body>
</html>
"""
    html_manager.save_ai_annual_performance_html(year, html)


def make_meeting_performance_page(year, place_id, times):
    """開催別成績ページ（public_html/performance/meeting/{year}/{place}-{times}th.html）を生成する"""
    place_name = NAME_LIST[place_id - 1]
    pairs = calc.filter_by_meeting(calc.list_predicted_races(), year, place_id, times)
    performance = calc.aggregate_ai_performance(pairs)

    html = f"""
<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="UTF-8">
  <title>{year}年 {place_name}{times}回 AI予想成績</title>
  <link rel="stylesheet" href="../../../assets/css/styles.css">
</head>
<body>
  <h1>{year}年 {place_name}{times}回 AI予想成績</h1>
  {_performance_table_html(performance)}
  <p><a href="../../index.html">&larr; AI成績トップへ</a></p>
</body>
</html>
"""
    html_manager.save_ai_meeting_performance_html(year, place_id, times, html)


def make_course_performance_page(place_id, race_type, course_len):
    """コース別成績ページ（public_html/performance/course/{place}/{race_type}-{course_len}.html）を生成する"""
    place_name = NAME_LIST[place_id - 1]
    pairs = calc.filter_by_course(calc.list_predicted_races(), place_id, race_type, course_len)
    performance = calc.aggregate_ai_performance(pairs)

    html = f"""
<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="UTF-8">
  <title>{place_name} {race_type}{course_len}m AI予想成績</title>
  <link rel="stylesheet" href="../../../assets/css/styles.css">
</head>
<body>
  <h1>{place_name} {race_type}{course_len}m AI予想成績</h1>
  {_performance_table_html(performance)}
  <p><a href="../../../courses/{PLACE_LIST[place_id - 1]}/{race_type}-{course_len}.html">&larr; コース詳細データへ</a></p>
  <p><a href="../../index.html">&larr; AI成績トップへ</a></p>
</body>
</html>
"""
    html_manager.save_ai_course_performance_html(place_id, race_type, course_len, html)
