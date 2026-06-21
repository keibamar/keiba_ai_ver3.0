"""AI成績データページ（Forge: HTMLFactory）のHTML生成

永続化済みの src.managers.ai_performance_dataset_manager（data/ai_performance/ai_performance.csv）
をpandasで集計し、年別・開催別・競馬場別・コース別のページとして表示する。
data/race_card/ の全件スキャンを伴うライブ計算（旧 ai_performance_calculator.aggregate_ai_performance等）
は使わず、データセットのフィルタ・集計のみで生成するため、競馬場別・コース別の
年度/クラス/芝ダート/馬場別の内訳テーブルを多数並べても実用速度で生成できる。

旧ver2.0の web/site/performance/ 相当だが、的中率・回収率の集計ロジック自体は
ver2.0でも未実装だったため、ページ構成のみを参考にした新規実装。
"""

from datetime import date

from src.config.constants import NAME_LIST, PLACE_LIST
from src.config.lists import COURSE_LISTS
from src.logic.html_generator.rate_gauge_html import hit_rate_gauge_html, return_rate_gauge_html
from src.logic.html_generator.site_nav_html import breadcrumb_html, sidebar_html, site_nav_html
from src.logic.html_generator.sparkline_html import sparkline_svg
from src.managers import ai_performance_dataset_manager as m
from src.managers import html_manager

BET_TYPE_LABELS = {"win": "単勝", "place": "複勝", "trio_box": "三連複(5頭BOX)"}


def _performance_table_html(performance, title=None):
    rows = "".join(
        f"<tr><td>{BET_TYPE_LABELS[bet_type]}</td>"
        f"<td>{hit_rate_gauge_html(performance[bet_type]['hit_rate'])}</td>"
        f"<td>{return_rate_gauge_html(performance[bet_type]['return_rate'])}</td>"
        f"<td>{performance[bet_type]['n']}</td></tr>\n"
        for bet_type in BET_TYPE_LABELS
    )
    heading = f"<h3>{title}</h3>\n  " if title else ""
    return f"""{heading}<div class="table-wrap">
  <table class="sortable">
    <thead><tr><th>式別</th><th>的中率</th><th>回収率</th><th>対象レース数</th></tr></thead>
    <tbody>
      {rows}
    </tbody>
  </table>
  </div>"""


def _breakdown_table_html(breakdown, value_label, title=None):
    if not breakdown:
        body = "<p>対象データがありません。</p>"
    else:
        header_cells = "".join(
            f"<th>{label}的中率</th><th>{label}回収率</th>" for label in BET_TYPE_LABELS.values()
        )
        rows = "".join(
            f"<tr><td>{item['value']}</td>"
            + "".join(
                f"<td>{hit_rate_gauge_html(item['performance'][bt]['hit_rate'])}</td>"
                f"<td>{return_rate_gauge_html(item['performance'][bt]['return_rate'])}</td>"
                for bt in BET_TYPE_LABELS
            )
            + f"<td>{item['performance']['win']['n']}</td></tr>\n"
            for item in breakdown
        )
        body = f"""<div class="table-wrap">
  <table class="sortable">
    <thead><tr><th>{value_label}</th>{header_cells}<th>対象レース数</th></tr></thead>
    <tbody>
      {rows}
    </tbody>
  </table>
  </div>"""

    heading = f"<h3>{title}</h3>\n  " if title else ""
    return f"{heading}{body}"


def _year_trend_html(by_year):
    """年度別成績（新しい順のbreakdownリスト）から、単勝の的中率・回収率の
    年次トレンドをスパークラインで返す。2年未満（折れ線が描けない）の場合は空文字列。
    """
    if len(by_year) < 2:
        return ""
    chronological = list(reversed(by_year))
    years = [item["value"] for item in chronological]
    hit_rates = [item["performance"]["win"]["hit_rate"] for item in chronological]
    return_rates = [item["performance"]["win"]["return_rate"] for item in chronological]
    return f"""<div class="trend-section">
  <p class="trend-label">単勝的中率の推移 {sparkline_svg(hit_rates, years)}</p>
  <p class="trend-label">単勝回収率の推移 {sparkline_svg(return_rates, years)}</p>
</div>"""


def _cross_filter_panel_html(ground_state, class_value, performance):
    return f"""<div class="cross-filter-panel" data-ground-state="{ground_state}" data-class="{class_value}" hidden>
    {_performance_table_html(performance)}
  </div>"""


def _cross_filter_html(df, total_performance, by_ground_state, by_class):
    """馬場状態×クラスを2つのセレクトボックスで選び、該当する成績を表示する
    インタラクティブな絞り込みUIを返す

    section-tabs.jsと同じ「サーバー側で全組み合わせぶん事前計算し、JSは表示/非表示の
    切り替えのみ行う」方針で、全体合計・馬場のみ・クラスのみ・馬場×クラスの全組み合わせの
    成績表をあらかじめ生成して埋め込む（assets/js/cross-filter.jsが選択値に応じて
    一致するパネルだけを表示する）。
    """
    ground_states = [item["value"] for item in by_ground_state]
    classes = [item["value"] for item in by_class]
    cross = m.cross_breakdown(df, "ground_state", "class")

    ground_options = "".join(f'<option value="{gs}">{gs}</option>\n' for gs in ground_states)
    class_options = "".join(f'<option value="{cls}">{cls}</option>\n' for cls in classes)

    panels = [_cross_filter_panel_html("all", "all", total_performance)]
    panels += [_cross_filter_panel_html(item["value"], "all", item["performance"]) for item in by_ground_state]
    panels += [_cross_filter_panel_html("all", item["value"], item["performance"]) for item in by_class]
    panels += [_cross_filter_panel_html(gs, cls, performance) for (gs, cls), performance in cross.items()]

    return f"""<div class="cross-filter">
    <div class="cross-filter-controls">
      <label>馬場状態:
        <select class="cross-filter-ground-state">
          <option value="all">全て</option>
          {ground_options}
        </select>
      </label>
      <label>クラス:
        <select class="cross-filter-class">
          <option value="all">全て</option>
          {class_options}
        </select>
      </label>
    </div>
    {"".join(panels)}
    <p class="cross-filter-empty" hidden>対象データがありません。</p>
  </div>"""


def make_ai_performance_index_page():
    """AI成績トップページ（public_html/performance/index.html）を生成する

    トータル成績・今年の成績を表示し、過去の年間成績全てにアクセスできるよう
    予想データが存在する年を新しい順に列挙する。
    """
    df = m.get_ai_performance_dataset()
    this_year = date.today().year

    total_performance = m.aggregate(df)
    this_year_performance = m.aggregate(m.filter_by_year(df, this_year))

    years = sorted({int(year) for year in df["year"]}) if not df.empty else []
    if years:
        year_rows = "".join(
            f'<li><a href="annual/{year}.html">{year}年</a></li>\n' for year in reversed(years)
        )
        annual_section = f"<ul>\n    {year_rows}\n  </ul>"
        by_year_desc = [
            {"value": year, "performance": m.aggregate(m.filter_by_year(df, year))}
            for year in reversed(years)
        ]
        trend_section = _year_trend_html(by_year_desc)
    else:
        annual_section = "<p>予想データがまだありません。</p>"
        trend_section = ""

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
  {site_nav_html(base_path="../")}
  {breadcrumb_html([("AI成績", None)], base_path="../")}
  <h1>AI予想成績</h1>

  {_performance_table_html(total_performance, title="トータル成績")}
  {_performance_table_html(this_year_performance, title=f"{this_year}年の成績")}

  <h2>年間成績</h2>
  {trend_section}
  {annual_section}

  <h2>コース別成績</h2>
  <ul>
    {place_rows}
  </ul>

  <p><a href="../index.html">&larr; HOMEへ戻る</a></p>
  <script src="../assets/js/sortable-table.js"></script>
</body>
</html>
"""
    html_manager.save_ai_performance_index_html(html)


def make_all_annual_performance_pages():
    """予想データが存在する年について、年間成績ページを一括生成する"""
    df = m.get_ai_performance_dataset()
    if df.empty:
        return
    for year in sorted({int(year) for year in df["year"]}):
        make_annual_performance_page(year, df=df)


def make_annual_performance_page(year, df=None):
    """年間成績ページ（public_html/performance/annual/{year}.html）を生成する"""
    df = df if df is not None else m.get_ai_performance_dataset()
    performance = m.aggregate(m.filter_by_year(df, year))

    years = sorted({int(y) for y in df["year"]}, reverse=True) if not df.empty else []
    sidebar = sidebar_html(
        [("年度", [(f"{y}年", f"{y}.html") for y in years], f"{year}年")],
        up_link=("AI成績トップ", "../index.html"),
    )
    breadcrumb = breadcrumb_html([("AI成績", "performance/index.html"), (f"{year}年", None)], base_path="../../")

    html = f"""
<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="UTF-8">
  <title>{year}年 AI予想成績</title>
  <link rel="stylesheet" href="../../assets/css/styles.css">
</head>
<body>
  {site_nav_html(base_path="../../")}
  {breadcrumb}
  <div class="page-layout">
  <main class="page-content">
  <h1>{year}年 AI予想成績</h1>
  {_performance_table_html(performance)}
  <p><a href="../index.html">&larr; AI成績トップへ</a></p>
  </main>
  {sidebar}
  </div>
  <script src="../../assets/js/sortable-table.js"></script>
</body>
</html>
"""
    html_manager.save_ai_annual_performance_html(year, html)


def make_meeting_performance_page(year, place_id, times, df=None):
    """開催別成績ページ（public_html/performance/meeting/{year}/{place}-{times}th.html）を生成する"""
    df = df if df is not None else m.get_ai_performance_dataset()
    place_name = NAME_LIST[place_id - 1]
    performance = m.aggregate(m.filter_by_meeting(df, year, place_id, times))

    html = f"""
<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="UTF-8">
  <title>{year}年 {place_name}{times}回 AI予想成績</title>
  <link rel="stylesheet" href="../../../assets/css/styles.css">
</head>
<body>
  {site_nav_html(base_path="../../../")}
  {breadcrumb_html([("AI成績", "performance/index.html"), (f"{year}年 {place_name}{times}回", None)], base_path="../../../")}
  <h1>{year}年 {place_name}{times}回 AI予想成績</h1>
  {_performance_table_html(performance)}
  <p><a href="../../index.html">&larr; AI成績トップへ</a></p>
  <script src="../../../assets/js/sortable-table.js"></script>
</body>
</html>
"""
    html_manager.save_ai_meeting_performance_html(year, place_id, times, html)


def make_course_performance_index_page(place_id, df=None):
    """競馬場別成績ページ（public_html/performance/course/{place}/index.html）を生成する

    トータル・今年・年度別・クラス別・芝/ダート別・馬場別の内訳テーブルに加え、
    コース（race_type×距離）別ページへのリンク一覧を表示する。
    """
    df = df if df is not None else m.get_ai_performance_dataset()
    this_year = date.today().year
    place_name = NAME_LIST[place_id - 1]
    course_list = COURSE_LISTS[place_id - 1]

    place_df = m.filter_by_place(df, place_id)
    total_performance = m.aggregate(place_df)
    this_year_performance = m.aggregate(m.filter_by_year(place_df, this_year))
    by_year = sorted(m.group_breakdown(place_df, "year"), key=lambda item: item["value"], reverse=True)
    by_class = m.group_breakdown(place_df, "class")
    by_race_type = m.group_breakdown(place_df, "race_type")
    by_ground_state = m.group_breakdown(place_df, "ground_state")

    course_rows = "".join(
        f'<li><a href="{race_type}-{course_len}.html">{race_type}{course_len}m</a></li>\n'
        for race_type, course_len in course_list
    )
    sidebar = sidebar_html(
        [
            (
                "競馬場",
                [(NAME_LIST[i], f"../{PLACE_LIST[i]}/index.html") for i in range(len(PLACE_LIST))],
                place_name,
            ),
        ],
        up_link=("AI成績トップ", "../../index.html"),
    )
    breadcrumb = breadcrumb_html(
        [("AI成績", "performance/index.html"), (place_name, None)], base_path="../../../",
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
  {site_nav_html(base_path="../../../")}
  {breadcrumb}
  <div class="page-layout">
  <main class="page-content">
  <h1>{place_name} AI予想成績</h1>

  <div class="tabbed-section">
    <div class="section-tabs">
      <button data-target="overview" aria-selected="true">トータル/今年</button>
      <button data-target="breakdown" aria-selected="false">クラス別・芝ダート別・馬場別</button>
      <button data-target="year" aria-selected="false">年度別</button>
      <button data-target="cross" aria-selected="false">馬場×クラス</button>
    </div>

    <div class="section-panel" data-section="overview">
      {_performance_table_html(total_performance, title="トータル成績")}
      {_performance_table_html(this_year_performance, title=f"{this_year}年の成績")}
    </div>

    <div class="section-panel" data-section="breakdown" hidden>
      {_breakdown_table_html(by_class, "クラス", title="クラス別成績")}
      {_breakdown_table_html(by_race_type, "芝/ダート", title="芝/ダート別成績")}
      {_breakdown_table_html(by_ground_state, "馬場状態", title="馬場別成績")}
    </div>

    <div class="section-panel" data-section="year" hidden>
      {_year_trend_html(by_year)}
      {_breakdown_table_html(by_year, "年度", title="年度別成績")}
    </div>

    <div class="section-panel" data-section="cross" hidden>
      {_cross_filter_html(place_df, total_performance, by_ground_state, by_class)}
    </div>
  </div>

  <h2>コース別成績</h2>
  <ul>
    {course_rows}
  </ul>
  <p><a href="../../index.html">&larr; AI成績トップへ</a></p>
  </main>
  {sidebar}
  </div>
  <script src="../../../assets/js/sortable-table.js"></script>
  <script src="../../../assets/js/section-tabs.js"></script>
  <script src="../../../assets/js/cross-filter.js"></script>
</body>
</html>
"""
    html_manager.save_ai_course_performance_index_html(place_id, html)


def make_course_performance_page(place_id, race_type, course_len, df=None):
    """コース別成績ページ（public_html/performance/course/{place}/{race_type}-{course_len}.html）を生成する

    トータル・今年・クラス別・馬場別・年度別の内訳テーブルを表示する。
    """
    df = df if df is not None else m.get_ai_performance_dataset()
    this_year = date.today().year
    place_name = NAME_LIST[place_id - 1]

    course_df = m.filter_by_course(df, place_id, race_type, course_len)
    total_performance = m.aggregate(course_df)
    this_year_performance = m.aggregate(m.filter_by_year(course_df, this_year))
    by_class = m.group_breakdown(course_df, "class")
    by_ground_state = m.group_breakdown(course_df, "ground_state")
    by_year = sorted(m.group_breakdown(course_df, "year"), key=lambda item: item["value"], reverse=True)

    current_label = f"{race_type}{course_len}m"
    sidebar = sidebar_html(
        [
            (
                "競馬場",
                [(NAME_LIST[i], f"../{PLACE_LIST[i]}/index.html") for i in range(len(PLACE_LIST))],
                place_name,
            ),
            (
                f"{place_name}のコース",
                [(f"{rt}{cl}m", f"{rt}-{cl}.html") for rt, cl in COURSE_LISTS[place_id - 1]],
                current_label,
            ),
        ],
        up_link=(f"{place_name}のAI成績", "index.html"),
    )
    breadcrumb = breadcrumb_html(
        [
            ("AI成績", "performance/index.html"),
            (place_name, f"performance/course/{PLACE_LIST[place_id - 1]}/index.html"),
            (current_label, None),
        ],
        base_path="../../../",
    )

    html = f"""
<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="UTF-8">
  <title>{place_name} {race_type}{course_len}m AI予想成績</title>
  <link rel="stylesheet" href="../../../assets/css/styles.css">
</head>
<body>
  {site_nav_html(base_path="../../../")}
  {breadcrumb}
  <div class="page-layout">
  <main class="page-content">
  <h1>{place_name} {race_type}{course_len}m AI予想成績</h1>

  <div class="tabbed-section">
    <div class="section-tabs">
      <button data-target="overview" aria-selected="true">トータル/今年</button>
      <button data-target="breakdown" aria-selected="false">クラス別・馬場別</button>
      <button data-target="year" aria-selected="false">年度別</button>
      <button data-target="cross" aria-selected="false">馬場×クラス</button>
    </div>

    <div class="section-panel" data-section="overview">
      {_performance_table_html(total_performance, title="トータル成績")}
      {_performance_table_html(this_year_performance, title=f"{this_year}年の成績")}
    </div>

    <div class="section-panel" data-section="breakdown" hidden>
      {_breakdown_table_html(by_class, "クラス", title="クラス別成績")}
      {_breakdown_table_html(by_ground_state, "馬場状態", title="馬場別成績")}
    </div>

    <div class="section-panel" data-section="year" hidden>
      {_year_trend_html(by_year)}
      {_breakdown_table_html(by_year, "年度", title="年度別成績")}
    </div>

    <div class="section-panel" data-section="cross" hidden>
      {_cross_filter_html(course_df, total_performance, by_ground_state, by_class)}
    </div>
  </div>

  <p><a href="../../../courses/{PLACE_LIST[place_id - 1]}/{race_type}-{course_len}.html">&larr; コース詳細データへ</a></p>
  <p><a href="../../index.html">&larr; AI成績トップへ</a></p>
  </main>
  {sidebar}
  </div>
  <script src="../../../assets/js/sortable-table.js"></script>
  <script src="../../../assets/js/section-tabs.js"></script>
  <script src="../../../assets/js/cross-filter.js"></script>
</body>
</html>
"""
    html_manager.save_ai_course_performance_html(place_id, race_type, course_len, html)


def make_all_course_performance_pages():
    """全開催場・全コースのAI成績ページを一括生成する

    データセットは1回だけ取得し、全競馬場・全コースで再利用する。
    """
    df = m.get_ai_performance_dataset()

    for place_id in range(1, len(PLACE_LIST) + 1):
        make_course_performance_index_page(place_id, df=df)
        for race_type, course_len in COURSE_LISTS[place_id - 1]:
            make_course_performance_page(place_id, race_type, course_len, df=df)
