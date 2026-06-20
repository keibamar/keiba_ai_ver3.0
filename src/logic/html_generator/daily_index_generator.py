"""日次レース一覧ページ（Forge: HTMLFactory）のHTML生成

旧 web/src/generators/daily_index.py からの移植。HTMLテンプレート・解析ロジックは
変更せず、データ取得のみ新アーキテクチャのManager層に切り替えている。
"""

from datetime import datetime

from src.config.constants import NAME_LIST, PLACE_LIST
from src.managers import html_manager, race_card_dataset_manager, race_schedule_dataset_manager
from src.utils.format_data import format_date


def load_race_info(date_str):
    """date_str から race_time_id_list を読み込み dict を返す"""
    info = {}
    race_day = datetime.strptime(date_str, "%Y%m%d")
    df_info = race_card_dataset_manager.get_race_time_id_list_df(race_day)

    if not df_info.empty:
        for _, row in df_info.iterrows():
            info[str(row["race_id"])] = {
                "race_time": str(row["race_time"]),
                "race_name": str(row["race_name"]),
            }
    else:
        print(f"警告: レース情報ファイルが存在しません: {date_str}")
    return info


def group_place_races(files_info_list, race_info_dict):
    """files_info_list を place_id ごとにグルーピングして辞書を返す"""
    place_races = {}
    for file_info in files_info_list:
        place_id = file_info['place_id']
        place_key = PLACE_LIST[place_id - 1]
        place_name = NAME_LIST[place_id - 1]
        race_num = file_info['race_num']
        race_id = str(file_info["file"])

        race_name = ""
        race_time = ""
        if race_id in race_info_dict:
            race_name = race_info_dict[race_id]["race_name"]
            race_time = race_info_dict[race_id]["race_time"]

        if place_key not in place_races:
            place_races[place_key] = {"display": place_name, "races": []}
        place_races[place_key]["races"].append({
            "race_num": race_num,
            "race_name": race_name,
            "race_time": race_time,
        })
    return place_races


def build_table_rows(place_races, date_str):
    """place_races から HTML の table_rows を作成して返す（place_keys も返す）"""
    place_keys = sorted(place_races.keys())
    max_races = max(len(data["races"]) for data in place_races.values()) if place_races else 0

    table_rows = ""
    for race_num in range(1, max_races + 1):
        row_cells = f"<th>{race_num}R</th>"
        for place_key in place_keys:
            races = place_races[place_key]["races"]
            race_info = next((r for r in races if r["race_num"] == race_num), None)

            if race_info:
                race_name_disp = race_info['race_name'] if race_info['race_name'] else f"{race_num}R"

                race_time_disp = ""
                if race_info["race_time"]:
                    t = race_info["race_time"].zfill(4)
                    race_time_disp = f"発走時刻: {t[:2]}:{t[2:]}"

                race_file = f"{place_key}R{race_num}.html"
                if html_manager.race_page_exists(date_str, race_file):
                    row_cells += (
                        f'<td>'
                        f'<a href="{race_file}">'
                        f'{race_name_disp}</a><br>'
                        f'{race_time_disp}'
                        f'</td>'
                    )
                else:
                    row_cells += (
                        f'<td>'
                        f'{race_name_disp}<br>'
                        f'{race_time_disp}'
                        f'</td>'
                    )
            else:
                row_cells += "<td>-</td>"
        table_rows += f"<tr>{row_cells}</tr>\n"
    return table_rows, place_keys


def make_index_page(date_str, files_info_list):
    date_display = format_date(date_str)

    # --- レース情報読み込みとグルーピング ---
    race_info_dict = load_race_info(date_str)
    if race_info_dict == {}:
        return

    # === 全開催日フォルダの一覧を取得してソート ===
    all_days = html_manager.list_race_day_dirs()

    prev_day = None
    next_day = None
    if date_str in all_days:
        idx = all_days.index(date_str)
        if idx > 0:
            prev_day = all_days[idx - 1]
        if idx < len(all_days) - 1:
            next_day = all_days[idx + 1]

    prev_day_display = format_date(prev_day)
    next_day_display = format_date(next_day)

    # --- レース情報のグルーピング ---
    place_races = group_place_races(files_info_list, race_info_dict)

    # --- テーブル行とplace_keysを作成 ---
    table_rows, place_keys = build_table_rows(place_races, date_str)

    # === ナビゲーションHTML作成 ===
    nav_links = '<div class="nav">'
    nav_links += '<a href="../index.html">開催日一覧に戻る</a><br>'
    nav_links += '<div class="subnav">'

    if prev_day:
        nav_links += f'<a href="../{prev_day}/index.html">← 前の日</a> ({prev_day_display}) '
    else:
        nav_links += '<span class="disabled">← 前の日</span> '

    if next_day:
        nav_links += f'<a href="../{next_day}/index.html">→ 次の日</a> ({next_day_display})'
    else:
        nav_links += '<span class="disabled">→ 次の日</span>'

    nav_links += '</div></div>'

    # HTML生成
    html = daily_index_template(date_display, nav_links, place_races, place_keys, table_rows)

    html_manager.save_daily_index_html(date_str, html)


def daily_index_template(date_display, nav_links, place_races, place_keys, table_rows):
    return f"""
<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="UTF-8">
  <title>{date_display} レース一覧</title>
  <link rel="stylesheet" href="../../assets/css/styles.css">
  <style>
    body {{
      font-family: sans-serif;
      margin: 20px;
    }}
    h1 {{
      border-bottom: 2px solid #555;
      padding-bottom: 5px;
    }}
    table {{
      border-collapse: collapse;
      width: 100%;
      margin-top: 15px;
    }}
    th, td {{
      border: 1px solid #ccc;
      padding: 6px;
      text-align: center;
    }}
    th {{
      background-color: #f2f2f2;
    }}
    td a {{
      display: block;
      font-weight: bold;
      color: #0044cc;
      text-decoration: none;
      margin-bottom: 3px;
    }}
    td a:hover {{
      text-decoration: underline;
    }}
    .nav {{
      margin-bottom: 10px;
    }}
    .nav {{
      margin-bottom: 15px;
    }}
    .nav a {{
      text-decoration: none;
      color: blue;
      font-weight: bold;
    }}
    .subnav {{
      margin-top: 5px;
    }}
    .subnav a {{
      margin-right: 15px;
    }}
    .nav .disabled {{
      color: #aaa;
      margin-right: 10px;
    }}
  </style>
</head>
<body>
  {nav_links}
  <h1>{date_display} レース一覧</h1>
  <table>
    <thead>
      <tr>
        <th>レース</th>
        {''.join(f'<th>{place_races[k]["display"]}競馬場</th>' for k in place_keys)}
      </tr>
    </thead>
    <tbody>
      {table_rows}
    </tbody>
  </table>
</body>
</html>
"""


def make_daily_index_page(race_day):
    """指定した日のレースカードHTMLインデックスページを生成する"""
    day_str = race_day.strftime("%Y%m%d")
    race_id_list = race_schedule_dataset_manager.get_daily_id(0, race_day)

    files_info_list = [
        {"place_id": int(str(race_id)[4:6]), "race_num": int(str(race_id)[-2:]), "file": race_id}
        for race_id in race_id_list
    ]

    make_index_page(day_str, files_info_list)


def calendar_widget_html(base_path=""):
    """カレンダーwidgetのHTML断片を返す（races/index.html・Home両方から再利用する）

    raceDays.js（html_manager.add_race_day が更新する window.racedays）と
    calendar.js を読み込み、JS側でカレンダーを描画する。calendar.jsが生成する
    リンクは window.CALENDAR_BASE_PATH を基準にするため、埋め込み先のページの
    階層に応じた base_path（races/index.htmlなら"../"、Home直下なら""）を渡す。
    スタイルは public_html/assets/css/styles.css の .calendar-widget 系クラスに依存する。
    """
    return f"""
<div class="calendar-widget">
  <div class="calendar-nav">
    <button id="prevMonth">&larr;</button>
    <span id="monthYear"></span>
    <button id="nextMonth">&rarr;</button>
  </div>
  <table id="calendar"></table>
  <div id="todayRace"></div>
</div>
<script>window.CALENDAR_BASE_PATH = "{base_path}";</script>
<script src="{base_path}assets/js/raceDays.js"></script>
<script src="{base_path}assets/js/calendar.js"></script>
"""


def races_calendar_template():
    """開催日カレンダーページ（public_html/races/index.html）のHTMLを返す

    旧 web/site/races/index.html の構成を移植。
    """
    return f"""
<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="UTF-8">
  <title>開催日カレンダー</title>
  <link rel="stylesheet" href="../assets/css/styles.css">
</head>
<body>
  <h1>開催日カレンダー</h1>

  {calendar_widget_html(base_path="../")}

  <p><a href="../index.html">&larr; HOMEへ戻る</a></p>
</body>
</html>
"""


def make_races_calendar_page():
    """開催日カレンダーページ（public_html/races/index.html）を生成する"""
    html_manager.save_races_calendar_html(races_calendar_template())
