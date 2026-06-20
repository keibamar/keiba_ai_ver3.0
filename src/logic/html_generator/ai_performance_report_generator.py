"""AI成績データページ（Forge: HTMLFactory）のHTML生成

src.logic.calculators.ai_performance_calculator で計算した的中率・回収率を、
年別・開催別・コース別のページとして表示する。

旧ver2.0の web/site/performance/ 相当だが、的中率・回収率の集計ロジック自体は
ver2.0でも未実装だったため、ページ構成のみを参考にした新規実装。
"""

from src.config.constants import NAME_LIST, PLACE_LIST
from src.config.lists import COURSE_LISTS
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


def make_ai_performance_index_page():
    """AI成績トップページ（public_html/performance/index.html）を生成する

    過去の年間成績全てにアクセスできるよう、予想データが存在する年を
    （ai_performance_calculator.get_predicted_years）新しい順に列挙する。
    """
    years = calc.get_predicted_years()
    if years:
        year_rows = "".join(
            f'<li><a href="annual/{year}.html">{year}年</a></li>\n' for year in reversed(years)
        )
        annual_section = f"<ul>\n    {year_rows}\n  </ul>"
    else:
        annual_section = "<p>予想データがまだありません。</p>"

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
  {annual_section}

  <h2>コース別成績</h2>
  <ul>
    {place_rows}
  </ul>

  <p><a href="../index.html">&larr; HOMEへ戻る</a></p>
</body>
</html>
"""
    html_manager.save_ai_performance_index_html(html)


def make_all_annual_performance_pages():
    """予想データが存在する年について、年間成績ページを一括生成する"""
    for year in calc.get_predicted_years():
        make_annual_performance_page(year)


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


def make_course_performance_index_page(place_id):
    """場別のコース成績一覧ページ（public_html/performance/course/{place}/index.html）を生成する

    performance/index.html の「コース別成績」リンク先として必要なページ
    （make_ai_performance_index_pageからリンクされるが、これまで生成関数が無く404になっていた）。
    """
    place_name = NAME_LIST[place_id - 1]
    course_list = COURSE_LISTS[place_id - 1]
    rows = "".join(
        f'<li><a href="{race_type}-{course_len}.html">{race_type}{course_len}m</a></li>\n'
        for race_type, course_len in course_list
    )
    html = f"""
<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="UTF-8">
  <title>{place_name} AI予想成績</title>
  <link rel="stylesheet" href="../../../assets/css/styles.css">
</head>
<body>
  <h1>{place_name} AI予想成績</h1>
  <ul>
    {rows}
  </ul>
  <p><a href="../../index.html">&larr; AI成績トップへ</a></p>
</body>
</html>
"""
    html_manager.save_ai_course_performance_index_html(place_id, html)


def make_course_performance_page(place_id, race_type, course_len, all_pairs=None, race_conditions=None):
    """コース別成績ページ（public_html/performance/course/{place}/{race_type}-{course_len}.html）を生成する

    Args:
        all_pairs (list | None): calc.list_predicted_races()の戻り値を事前に渡すと、
            毎回のdata/race_card/走査を省略できる（多数のコースを一括生成する場合に高速化）。
        race_conditions (dict | None): calc.get_race_conditions(all_pairs)の戻り値を
            事前に渡すと、get_race_info_csvの再呼び出しを省略できる。
    """
    place_name = NAME_LIST[place_id - 1]
    all_pairs = all_pairs if all_pairs is not None else calc.list_predicted_races()
    pairs = calc.filter_by_course(all_pairs, place_id, race_type, course_len, race_conditions=race_conditions)
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


def make_all_course_performance_pages():
    """全開催場・全コースのAI成績ページを一括生成する

    filter_by_courseが多数回呼ばれるため、list_predicted_races/get_race_conditionsを
    1回だけ計算して全コースで再利用する（個別に呼ぶより大幅に高速）。
    """
    all_pairs = calc.list_predicted_races()
    race_conditions = calc.get_race_conditions(all_pairs)

    for place_id in range(1, len(PLACE_LIST) + 1):
        make_course_performance_index_page(place_id)
        for race_type, course_len in COURSE_LISTS[place_id - 1]:
            make_course_performance_page(
                place_id, race_type, course_len, all_pairs=all_pairs, race_conditions=race_conditions
            )
