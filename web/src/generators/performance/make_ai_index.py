import os
import sys
from datetime import date, datetime

# pycache を生成しない
sys.dont_write_bytecode = True
SRC_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))  # web/src
sys.path.append(SRC_ROOT)

from config.path import TRACK_MAP, PERFORMACE_HTML_PATH
from config.templates import load_template
from load_meetings import load_meetings


TEMPLATE_NAME = "ai_index.html"

# -----------------------------
# 開催中の競馬場を抽出
# -----------------------------
def get_current_tracks(day = date.today()):
    meetings = load_meetings(day.year)
    current_tracks = []

    for (year, course, times), days in meetings.items():

        first_day = days[0]["date"]
        last_day = days[-1]["date"]

        # Normalize types: meetings may contain datetime, while `day` is a date
        if isinstance(first_day, datetime):
            first_day = first_day.date()
        if isinstance(last_day, datetime):
            last_day = last_day.date()

        if first_day <= day <= last_day:
            track_name = TRACK_MAP[course]
            url = f"meeting/{year}/{track_name}-{times}th.html"

            current_tracks.append({
                "track": track_name,
                "times": times,
                "period": f"{first_day.strftime('%Y/%m/%d')}〜{last_day.strftime('%Y/%m/%d')}",
                "url": url
            })

    # 競馬場名でソート
    current_tracks.sort(key=lambda x: x["track"])
    return current_tracks

# -----------------------------
# AIメインページ生成
# -----------------------------
def generate_ai_index(day = date.today()):
    os.makedirs(PERFORMACE_HTML_PATH, exist_ok=True)

    current_tracks = get_current_tracks(day)

    template = load_template(TEMPLATE_NAME)
    
    html = template.render(
        title="AI予想成績",
        current_year = day.year,
        current_tracks=current_tracks,
    )

    with open(os.path.join(PERFORMACE_HTML_PATH, "index.html"), "w", encoding="utf-8") as f:
        f.write(html)

    print("Generated: tracks/index.html")

# -----------------------------
# 実行
# -----------------------------
if __name__ == "__main__":
    generate_ai_index()

