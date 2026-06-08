import os
import sys

import pandas as pd

# pycache を生成しない
sys.dont_write_bytecode = True

# プロジェクト内の src と libs を import パスに追加（カレントディレクトリに依存しない）
PROJECT_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  # web/src
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))  # web
LIBS_PATH = os.path.join(PROJECT_ROOT, "libs")  # web\libs ではなく project の libs 配置に合わせる場合は調整
for p in (PROJECT_SRC, LIBS_PATH):
    if p not in sys.path:
        sys.path.insert(0, p)

import name_header
PLACE_LIST = name_header.PLACE_LIST
NAME_LIST = name_header.NAME_LIST

from config.path import RACE_CALENDAR_FOLDER_PATH, RACE_CARDS_PATH, RACE_HTML_PATH
from utils.data_loader import list_files_and_parse
from utils.format_data import format_date

def load_race_info(date_str):
    """date_str から race_time_id_list の CSV を読み込み dict を返す"""
    info = {}
    race_info_path = os.path.join(RACE_CALENDAR_FOLDER_PATH, "race_time_id_list", f"{date_str}.csv")

    if os.path.exists(race_info_path):
        df_info = pd.read_csv(race_info_path)
        for _, row in df_info.iterrows():
            info[str(row["race_id"])] = {
                "race_time": str(row["race_time"]),
                "race_name": str(row["race_name"]),
            }
    else:
        print(f"警告: レース情報ファイルが存在しません: {race_info_path}")
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

def build_table_rows(place_races, output_dir):
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

                race_file = os.path.join(output_dir, f"{place_key}R{race_num}.html")
                if os.path.exists(race_file):
                    row_cells += (
                        f'<td>'
                        f'<a href="{place_key}R{race_num}.html">'
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

def make_index_page(date_str, output_dir, files_info_list):
    date_display = format_date(date_str)

    # --- レース情報読み込みとグルーピング ---
    race_info_dict = load_race_info(date_str)
    if race_info_dict == {}:
        return
    
    #--- 出力ディレクトリ作成 ---
    if not os.path.exists(output_dir):
          os.makedirs(output_dir)

    # === 全開催日フォルダの一覧を取得してソート ===
    all_days = [
        d for d in os.listdir(RACE_HTML_PATH)
        if os.path.isdir(os.path.join(RACE_HTML_PATH, d)) and d.isdigit()
    ]
    all_days = sorted(all_days)

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
    table_rows, place_keys = build_table_rows(place_races, output_dir)

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

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "index.html"), "w", encoding="utf-8") as f:
        f.write(html)

def daily_index_template(date_display, nav_links, place_races, place_keys, table_rows):
    return f"""
<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="UTF-8">
  <title>{date_display} レース一覧</title>
  <link rel="stylesheet" href="../css/styles.css">
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
  files_info_list = list_files_and_parse(RACE_CARDS_PATH + f"{day_str}")

  output_dir = RACE_HTML_PATH + f"{day_str}"

  make_index_page(day_str, output_dir, files_info_list)

if __name__ == "__main__":
  # テスト用実行コード
  # folders = get_subfolders(RACE_CARDS_PATH)
  day_str = "20251018"
  files_info_list = list_files_and_parse(RACE_CARDS_PATH + f"{day_str}")

  output_dir = RACE_HTML_PATH + f"{day_str}"
  if not os.path.exists(output_dir):
            os.makedirs(output_dir)

  make_index_page(day_str, output_dir, files_info_list)
