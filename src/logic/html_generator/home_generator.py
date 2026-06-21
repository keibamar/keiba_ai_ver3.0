"""Homeページ（Forge: HTMLFactory）のHTML生成

旧 web/src/generators/global_index.py の移植だが、他のhtml_generator（daily_index_generator等）
と同様にPythonのf-stringで直接HTMLを組み立てる。

レースカレンダー・AI成績サマリー・先週の結果・コース詳細データへの導線をカード形式で
1ページに集約する。的中率・回収率は永続化済みの src.managers.ai_performance_dataset_manager
（data/ai_performance/ai_performance.csv）をpandasで集計して表示するため、
data/race_card/ の全件スキャンを避けて高速に生成できる
（先週の結果カードのみ1レース単位の処理のため ai_performance_calculator を直接使う）。
"""

from datetime import date, timedelta

from src.config.constants import NAME_LIST
from src.logic.calculators import ai_performance_calculator as calc
from src.logic.html_generator import daily_index_generator
from src.logic.html_generator.rate_gauge_html import hit_rate_gauge_html, return_rate_gauge_html
from src.logic.html_generator.site_nav_html import site_nav_html
from src.managers import ai_performance_dataset_manager as dataset_manager
from src.managers import html_manager

SITE_TITLE = "MAR(まーる）|競馬AIデータサイト"
BET_TYPE_LABELS = {"win": "単勝", "place": "複勝", "trio_box": "三連複(5頭BOX)"}


def _summary_stats_html(performance):
    win = performance["win"]
    return f"""
<div class="summary-stats">
  <div class="summary-stat">
    <span class="value">{win['hit_rate']:.1f}%</span>
    <span class="label">単勝的中率（全期間）</span>
  </div>
  <div class="summary-stat">
    <span class="value">{win['return_rate']:.1f}%</span>
    <span class="label">単勝回収率（全期間）</span>
  </div>
  <div class="summary-stat">
    <span class="value">{win['n']}</span>
    <span class="label">対象レース数</span>
  </div>
</div>
"""


def _weekly_trend_html(trend):
    rows = ""
    for week in trend:
        return_rate = week["performance"]["win"]["return_rate"]
        rows += f"""<div class="bar-row">
      <span class="bar-label">{week['week_start'].strftime('%m/%d')}〜</span>
      {return_rate_gauge_html(return_rate)}
    </div>\n"""
    return f'<div class="bar-chart">\n{rows}</div>'


def _current_meetings_html(meetings, df):
    if not meetings:
        return "<p>現在開催中の競馬場はありません。</p>"

    rows = ""
    for meeting in meetings:
        place_name = NAME_LIST[meeting["place_id"] - 1]
        meeting_df = dataset_manager.filter_by_meeting(
            df, meeting["first_day"].year, meeting["place_id"], meeting["times"]
        )
        performance = dataset_manager.aggregate(meeting_df)
        win = performance["win"]
        rows += (
            f"<tr><td>{place_name}{meeting['times']}回</td>"
            f"<td>{hit_rate_gauge_html(win['hit_rate'])}</td>"
            f"<td>{return_rate_gauge_html(win['return_rate'])}</td><td>{win['n']}</td></tr>\n"
        )

    return f"""<div class="table-wrap">
  <table class="sortable">
    <thead><tr><th>開催</th><th>単勝的中率</th><th>単勝回収率</th><th>対象レース数</th></tr></thead>
    <tbody>
      {rows}
    </tbody>
  </table>
  </div>"""


def _last_week_results_html(main_races):
    if not main_races:
        return "<p>先週のメインレース（11R）データがありません。</p>"

    rows = ""
    for race in main_races:
        place_name = NAME_LIST[race["place_id"] - 1]
        date_str = race["race_day"].strftime("%m/%d")
        result = race["result"]
        if result is None:
            badge = '<span class="hit-badge miss">データなし</span>'
            return_text = "-"
        else:
            hit, payout = result["win"]
            badge = (
                '<span class="hit-badge win">的中</span>' if hit else '<span class="hit-badge miss">不的中</span>'
            )
            return_text = f"{payout:.1f}%"
        rows += (
            f"<tr><td>{date_str}</td><td>{place_name}11R</td><td>{badge}</td><td>{return_text}</td></tr>\n"
        )

    return f"""<div class="table-wrap">
  <table class="sortable">
    <thead><tr><th>日付</th><th>レース</th><th>単勝</th><th>回収率</th></tr></thead>
    <tbody>
      {rows}
    </tbody>
  </table>
  </div>"""


def _weekly_trend(df, num_weeks, end_day):
    """直近num_weeks週について、週ごとの的中率・回収率の推移を返す

    Returns:
        list[dict]: [{"week_start", "week_end", "performance"}, ...]（古い週→新しい週の順）
    """
    trend = []
    for i in range(num_weeks):
        week_end = end_day - timedelta(days=7 * i)
        week_start = week_end - timedelta(days=6)
        week_df = dataset_manager.filter_by_date_range(df, week_start, week_end)
        trend.append(
            {"week_start": week_start, "week_end": week_end, "performance": dataset_manager.aggregate(week_df)}
        )
    trend.reverse()
    return trend


def home_template():
    today = date.today()
    df = dataset_manager.get_ai_performance_dataset()
    overall_performance = dataset_manager.aggregate(df)
    trend = _weekly_trend(df, num_weeks=8, end_day=today)
    current_meetings = calc.get_current_meetings(today)
    last_week_main_races = calc.get_last_week_main_races(today)

    return f"""
<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="UTF-8">
  <title>{SITE_TITLE}</title>
  <link rel="stylesheet" href="assets/css/styles.css">
</head>
<body>
  <main>
    {site_nav_html(base_path="")}
    <h1>{SITE_TITLE}</h1>
    <p>このサイトでは、競馬AIの成績、レースカレンダー、コース別データを閲覧できます。</p>

    <h2>レースカレンダー</h2>
    {daily_index_generator.calendar_widget_html(base_path="")}

    <div class="card-grid">
      <div class="card">
        <h3>AI予想成績</h3>
        {_summary_stats_html(overall_performance)}
        <h4>週別推移（単勝回収率、直近8週）</h4>
        {_weekly_trend_html(trend)}
        <h4>開催中の競馬場の成績</h4>
        {_current_meetings_html(current_meetings, df)}
        <a class="card-link" href="performance/index.html">AI成績の詳細を見る &rarr;</a>
      </div>

      <div class="card">
        <h3>先週の結果（メインレース）</h3>
        {_last_week_results_html(last_week_main_races)}
        <a class="card-link" href="races/index.html">レースカレンダーを見る &rarr;</a>
      </div>
    </div>
  </main>
  <footer>
    &copy; 競馬AIデータシステム
  </footer>
  <script src="assets/js/sortable-table.js"></script>
</body>
</html>
"""


def make_home_page():
    """public_html/index.html を生成する"""
    html_manager.save_home_html(home_template())
