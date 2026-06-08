import os
import sys

import pandas as pd
from datetime import date

# pycache を生成しない
sys.dont_write_bytecode = True
sys.path.append(r"C:\keiba_ai\keiba_ai_ver2.0\libs")

# プロジェクト内の src と libs を import パスに追加（カレントディレクトリに依存しない）
PROJECT_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))  # web/src
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))  # web
LIBS_PATH = os.path.join(PROJECT_ROOT, "libs")  # web\libs ではなく project の libs 配置に合わせる場合は調整
for p in (PROJECT_SRC, LIBS_PATH):
    if p not in sys.path:
        sys.path.insert(0, p)

from config.path import JS_FOLDER_PATH, RACE_HTML_PATH
from utils.format_data import format_date


def make_date_index(output_dir = RACE_HTML_PATH):
    """
    開催日一覧をカレンダー形式で表示する index.html を作成。
    dates: ["20250922", "20250928", ...] の形式
    """

    html = """<!DOCTYPE html>
<html lang="ja">
<head>
  <meta charset="UTF-8">
  <title>開催日カレンダー</title>
  <style>
    body {
      font-family: sans-serif;
      text-align: center;
    }
    #calendar {
      border-collapse: collapse;
      margin: 20px auto;
      width: 80%;
    }
    #calendar td, #calendar th {
      border: 1px solid #ccc;
      padding: 10px;
      width: 14%;
      height: 80px;
      vertical-align: top;
    }
    #calendar a {
      text-decoration: none;
      color: blue;
      font-weight: bold;
    }
    #calendar td {
      cursor: default;
    }
    #monthYear {
      font-size: 1.5em;
      margin: 0 10px;
    }
    #todayRace {
      margin-top: 20px;
      font-size: 1.2em;
    }
    #todayRace a {
      text-decoration: none;
      color: white;
      background-color: #007bff;
      padding: 10px 20px;
      border-radius: 5px;
      font-weight: bold;
    }
    #todayRace a:hover {
      background-color: #0056b3;
    }
    #todayRace p {
      color: #555;
    }
  </style>
</head>
<body>
  <h1>開催日カレンダー</h1>

  <div>
    <button id="prevMonth">←</button>
    <span id="monthYear"></span>
    <button id="nextMonth">→</button>
  </div>

  <table id="calendar"></table>

  <!-- 🐎 本日のレースリンク -->
  <div id="todayRace"></div>

  <!-- 開催日一覧（Pythonで自動生成される） -->
  <script src="../assets/js/racedays.js"></script>
  <!-- カレンダー描画用（固定で配置） -->
  <script src="../assets/js/calendar.js"></script>
</body>
</html>
"""
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "index.html"), "w", encoding="utf-8") as f:
        f.write(html)
    print(f"✅ index.html を生成しました: {output_dir}")

def add_race_day(race_day):
    """
    racedays.js に日付を追加する関数
    - race_day: 追加する日付
    """
    new_day = race_day.strftime("%Y%m%d")
    js_file = os.path.join(JS_FOLDER_PATH, "racedays.js")
    # ファイルが存在しない場合 → 新規作成
    if not os.path.exists(js_file):
        with open(js_file, "w", encoding="utf-8") as f:
            f.write(f'window.racedays = [\n  "{new_day}"\n];\n')
        print(f"新規作成: {new_day} を追加しました")
        return

    # 既存内容を読み込み
    with open(js_file, "r", encoding="utf-8") as f:
        content = f.read()

    # raceDays の配列を探す
    if "window.racedays" not in content:
        raise ValueError("racedays.js に raceDays 配列が見つかりません")

    # すでに存在する場合は追加しない
    if new_day in content:
        print(f"{new_day} は既に登録済みです")
        return

    # 最後の "]" の直前に追加
    lines = content.strip().splitlines()
    new_content = []
    added = False
    for line in lines:
        if line.strip().startswith("]") and not added:
            # 前の要素の末尾に , がなければ追加
            if not new_content[-1].strip().endswith(","):
                new_content[-1] = new_content[-1] + ","
            new_content.append(f'  "{new_day}"')
            new_content.append("];")
            added = True
        else:
            new_content.append(line)

    # 書き戻し
    with open(js_file, "w", encoding="utf-8") as f:
        f.write("\n".join(new_content) + "\n")

    print(f"{new_day} を追加しました")

if __name__ == "__main__":
    # テスト用実行コード
    race_day = date.today()
    add_race_day(race_day)