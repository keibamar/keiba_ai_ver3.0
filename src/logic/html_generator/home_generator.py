"""Homeページ（Forge: HTMLFactory）のHTML生成

旧 web/src/generators/global_index.py の移植だが、他のhtml_generator（daily_index_generator等）
と同様にPythonのf-stringで直接HTMLを組み立てる。

レースカレンダー・AI成績サマリー・今週のメインレース・先週の結果・
コース詳細データへの導線をカード形式で1ページに集約する。的中率・回収率は
永続化済みの src.managers.ai_performance_dataset_manager
（data/ai_performance/ai_performance.csv）をpandasで集計して表示するため、
data/race_card/ の全件スキャンを避けて高速に生成できる
（今週のメインレース・先週の結果カードのみ1レース単位の処理のため
ai_performance_calculator を直接使う）。
"""

from datetime import date, timedelta

import jpholiday

from src.config.constants import NAME_LIST, PLACE_LIST
from src.logic.calculators import ai_performance_calculator as calc
from src.logic.html_generator.race_type_badge_html import course_label_html
from src.logic.html_generator.rate_gauge_html import hit_rate_gauge_html, return_rate_gauge_html
from src.logic.html_generator.site_nav_html import SITE_TITLE, adsense_script_html, site_footer_html, site_nav_html
from src.managers import ai_performance_dataset_manager as dataset_manager
from src.managers import html_manager

BET_TYPE_LABELS = {"win": "単勝", "place": "複勝", "trio_box": "三連複(5頭BOX)"}


def _main_sub_cell_html(main, sub):
    """1セルの中で「メイン（大きい文字）＋サブ（小さい文字）」の2行を表示するHTMLを返す

    レース名（メイン）と開催場・レース番号（サブ）、開催場名（メイン）と開催回数
    （サブ）など、主従関係のある2つの情報を1セル内で改行して表示する用途に使う。
    """
    return f'<div class="cell-main-sub"><span class="main">{main}</span><span class="sub">{sub}</span></div>'


def _date_with_weekday_html(day):
    """日付に曜日を付けて返す（土曜は青、日曜・祝日は赤で表示する）"""
    weekday_kanji = "月火水木金土日"[day.weekday()]
    if day.weekday() == 5:
        css_class = "weekday-sat"
    elif day.weekday() == 6 or jpholiday.is_holiday(day):
        css_class = "weekday-sun"
    else:
        css_class = ""
    class_attr = f' class="{css_class}"' if css_class else ""
    return f'{day.strftime("%m/%d")}<span{class_attr}>({weekday_kanji})</span>'


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
    """開催中の競馬場の成績テーブルを返す

    競馬場名はそのコースのコース詳細データ（courses/{place}/index.html）への
    リンクにし、Homeから開催中コースのコース別データへすぐアクセスできるようにする。
    """
    if not meetings:
        return "<p>現在開催中の競馬場はありません。</p>"

    rows = ""
    for meeting in meetings:
        place_id = meeting["place_id"]
        place_name = NAME_LIST[place_id - 1]
        place_link = f'<a href="courses/{PLACE_LIST[place_id - 1]}/index.html">{place_name}</a>'
        meeting_df = dataset_manager.filter_by_meeting(
            df, meeting["first_day"].year, place_id, meeting["times"]
        )
        performance = dataset_manager.aggregate(meeting_df)
        win = performance["win"]
        meeting_cell = _main_sub_cell_html(place_link, f"{meeting['times']}回")
        rows += (
            f"<tr><td>{meeting_cell}</td>"
            f"<td>{hit_rate_gauge_html(win['hit_rate'])}</td>"
            f"<td>{return_rate_gauge_html(win['return_rate'])}</td><td>{win['n']}</td></tr>\n"
        )

    return f"""<div class="table-wrap">
  <table class="sortable">
    <thead><tr><th>開催（コース詳細データへ）</th><th>単勝的中率</th><th>単勝回収率</th><th>対象レース数</th></tr></thead>
    <tbody>
      {rows}
    </tbody>
  </table>
  </div>"""


def _week_main_races_html(races):
    """今週（土・日）のメインレース（11R）を、開催日・コース詳細データへのリンク付きで返す

    get_today_main_races_with_courseと同じ「race_type/course_lenが取得できなければ
    競馬場のコース一覧ページへリンクする」方針だが、土・日の2日分をまとめて表示する
    ため日付列を持つ（_today_main_races_htmlは当日のみのため日付列が無い）。
    """
    if not races:
        return "<p>今週のメインレース（11R）はありません。</p>"

    rows = ""
    for race in races:
        place_name = NAME_LIST[race["place_id"] - 1]
        place_key = PLACE_LIST[race["place_id"] - 1]
        date_str = _date_with_weekday_html(race["race_day"])
        time_str = str(race["race_time"]).zfill(4)
        time_disp = f"{time_str[:2]}:{time_str[2:]}"
        if race["race_type"] and race["course_len"]:
            # 「函館芝1200m」のように競馬場名とコースが続けて並ぶと読みにくいため、
            # 間で改行する（例: 函館 / 芝1200m）。
            course_link = (
                f'<a href="courses/{place_key}/{race["race_type"]}-{race["course_len"]}.html">'
                f'{place_name}<br>{course_label_html(race["race_type"], race["course_len"])}</a>'
            )
        else:
            course_link = f'<a href="courses/{place_key}/index.html">{place_name}</a>'
        race_day_str = race["race_day"].strftime("%Y%m%d")
        race_card_file = f"{place_key}R11.html"
        if html_manager.race_page_exists(race_day_str, race_card_file):
            race_name_html = f'<a href="races/{race_day_str}/{race_card_file}">{race["race_name"]}</a>'
        else:
            race_name_html = race["race_name"]
        race_cell = _main_sub_cell_html(race_name_html, f"{place_name}11R")
        rows += (
            f"<tr><td>{date_str} {time_disp}</td><td>{race_cell}</td>"
            f"<td>{course_link}</td></tr>\n"
        )

    return f"""<div class="table-wrap">
  <table class="sortable">
    <thead><tr><th>日付・発走時刻</th><th>レース</th><th>コース詳細データへ</th></tr></thead>
    <tbody>
      {rows}
    </tbody>
  </table>
  </div>"""


def _bet_result_cell_html(hit, payout):
    """単勝/複勝1つ分の的中結果セルを返す

    的中した場合は的中率ではなく実際の配当そのものを表示する（的中バッジ＋配当額）。
    不的中の場合は配当が無いため、バッジのみ表示する。
    """
    if hit and payout is not None:
        return f'<span class="hit-badge win">的中</span> {payout:.0f}円'
    return '<span class="hit-badge miss">不的中</span>'


def _weekend_results_html(races):
    """週末のメインレース結果（勝ち馬・AI本命馬・本命馬の着順・単勝/複勝の的中結果）を返す

    ai_performance_calculator.get_weekend_main_race_detailsの戻り値をそのまま表示する。
    レース名は、_week_main_races_htmlと同様にその日の出馬表ページ
    （races/{date}/{place}R11.html）が生成済みであればそこへリンクする
    （結果が出た後でも、出走馬・オッズ等の詳細を確認できるようにする）。
    """
    if not races:
        return "<p>対象レースのデータがありません。</p>"

    rows = ""
    for race in races:
        place_name = NAME_LIST[race["place_id"] - 1]
        place_key = PLACE_LIST[race["place_id"] - 1]
        date_str = _date_with_weekday_html(race["race_day"])
        pick_finish = f"{race['pick_finish']}着" if race["pick_finish"] is not None else "-"
        race_day_str = race["race_day"].strftime("%Y%m%d")
        race_card_file = f"{place_key}R11.html"
        if html_manager.race_page_exists(race_day_str, race_card_file):
            race_name_html = f'<a href="races/{race_day_str}/{race_card_file}">{race["race_name"]}</a>'
        else:
            race_name_html = race["race_name"]
        race_cell = _main_sub_cell_html(race_name_html, f"{place_name}11R")
        rows += (
            f"<tr><td>{date_str}</td><td>{race_cell}</td>"
            f"<td>{race['winner_name'] or '-'}</td>"
            f"<td>{race['pick_name'] or '-'}</td>"
            f"<td>{pick_finish}</td>"
            f"<td>{_bet_result_cell_html(race['win_hit'], race['win_payout'])}</td>"
            f"<td>{_bet_result_cell_html(race['place_hit'], race['place_payout'])}</td></tr>\n"
        )

    return f"""<div class="table-wrap">
  <table class="sortable">
    <thead><tr><th>日付</th><th>レース</th><th>勝ち馬</th><th>AI本命</th><th>本命着順</th><th>単勝</th><th>複勝</th></tr></thead>
    <tbody>
      {rows}
    </tbody>
  </table>
  </div>"""


def _weekly_meeting_summary_html(summaries):
    """今週の開催情報（開催場・第○回、土曜/日曜それぞれの○日目）を一覧で返す

    ai_performance_calculator.get_current_meeting_summariesの戻り値（水曜までは
    先週まで、木曜から今週の開催を指す）を表示する。開催場名はコース詳細データ
    （courses/{place}/index.html）へ、土曜・日曜それぞれの日付は、その日の
    出馬表一覧ページ（races/{date_str}/index.html）が生成済みであればそこへリンクし、
    コース別データ・出馬表データのどちらにもすぐアクセスできるようにする。
    """
    if not summaries:
        return "<p>今週開催中の競馬場はありません。</p>"

    items = ""
    for summary in summaries:
        place_name = NAME_LIST[summary["place_id"] - 1]
        place_key = PLACE_LIST[summary["place_id"] - 1]
        course_link = f'<a href="courses/{place_key}/index.html">{place_name}</a>'

        day_links = ""
        for day_info in summary["days"]:
            day_date = day_info["day_date"]
            day_str = day_date.strftime("%Y%m%d")
            weekday_kanji = "月火水木金土日"[day_date.weekday()]
            label = f'{day_date.strftime("%m/%d")}({weekday_kanji}) {day_info["day_number"]}日目'
            if html_manager.race_page_exists(day_str, "index.html"):
                day_links += f'<a href="races/{day_str}/index.html">{label}</a>'
            else:
                day_links += f"<span>{label}</span>"

        items += (
            f'<li class="weekly-meeting-item">'
            f'<span class="main">{course_link} 第{summary["times"]}回</span>'
            f'<span class="sub">{day_links}</span>'
            f"</li>\n"
        )

    return f'<ul class="weekly-meeting-list">\n{items}</ul>'


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
    # 「開催中の競馬場」も「今週のメインレース」「先週の結果」と同じ理屈で、
    # 今週の水曜日までは直近に終わった週末を、木曜日になった時点で今週の週末を指す
    # （詳細はcurrent_meeting_reference_day参照）。
    current_meetings = calc.get_current_meetings(calc.current_meeting_reference_day(today))
    weekly_meetings = calc.get_current_meeting_summaries(today)
    week_main_races = calc.get_week_main_races_with_course(today)

    # 開催結果・確定配当の反映には数日かかるため、週末（土日）が終わった直後
    # （月〜火）はまだその週末を「先週の結果」とみなさず、1つ前の週末を指す
    # （水曜日になった時点で1週間分更新される。詳細はcurrent_results_weekend_end参照）。
    last_week_races = calc.get_weekend_main_race_details(calc.current_results_weekend_end(today))

    return f"""
<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  {adsense_script_html()}
  <title>{SITE_TITLE}</title>
  <link rel="stylesheet" href="assets/css/styles.css">
</head>
<body>
  <main>
    {site_nav_html(base_path="", current_path="index.html")}
    <h1>{SITE_TITLE}</h1>
    <p>このサイトでは、競馬AIの成績、レースカレンダー、コース別データを閲覧できます。</p>

    <h2>今週の開催</h2>
    {_weekly_meeting_summary_html(weekly_meetings)}

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
        <h3>メインレース</h3>
        <h4>今週のメインレース</h4>
        {_week_main_races_html(week_main_races)}
        <h4>先週の結果</h4>
        {_weekend_results_html(last_week_races)}
        <a class="card-link" href="races/index.html">レースカレンダーを見る &rarr;</a>
      </div>
    </div>
  </main>
  {site_footer_html(base_path="")}
  <script src="assets/js/sortable-table.js"></script>
</body>
</html>
"""


def make_home_page():
    """public_html/index.html を生成する"""
    html_manager.save_home_html(home_template())
