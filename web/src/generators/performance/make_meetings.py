import os
import sys
from datetime import datetime, date

# pycache を生成しない
sys.dont_write_bytecode = True
SRC_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))  # web/src
sys.path.append(SRC_ROOT)

from config.path import TRACK_MAP, PERFORMACE_HTML_PATH
from config.templates import load_template
from load_meetings import load_meetings


TEMPLATE_NAME = "ai_meeting.html"
# -----------------------------
# 開催ページ生成
# -----------------------------
def generate_meeting_pages(day = date.today()):
    os.makedirs(PERFORMACE_HTML_PATH, exist_ok=True)
    meetings = load_meetings( day.year )

    past_meetings = []
    future_meetings = []

    # 開催を分類
    for (year, course, times), days in meetings.items():
        first_day = days[0]["date"]
        last_day = days[-1]["date"]

        if isinstance(first_day, datetime):
            first_day = first_day.date()
        if isinstance(last_day, datetime):
            last_day = last_day.date()
        print(first_day, last_day, day)
        if first_day < day:
            past_meetings.append((course, times, days))
        else:
            future_meetings.append((course, times, days))

    # -----------------------------
    # 過去開催ページを生成
    # -----------------------------
    os.makedirs(os.path.join(PERFORMACE_HTML_PATH,"meeting", str((day.year))), exist_ok=True)
    for course, times, days in past_meetings:
        track_name = TRACK_MAP[course]
        first_day = days[0]["date"]

        filename = f"{track_name}-{times}th.html"

        template = load_template(TEMPLATE_NAME)

        html = template.render(
            title=f"{track_name} {times}回 開催成績",
            track=track_name,
            times=times,
            days=days,
            year=str(day.year),
        )

        with open(os.path.join(PERFORMACE_HTML_PATH, "meeting", str(day.year), filename), "w", encoding="utf-8") as f:
            f.write(html)

        print(f"Generated: {filename}")

# -----------------------------
# 実行
# -----------------------------
if __name__ == "__main__":
    for year in range(2024, date.today().year + 1):
        if year < date.today().year:
            day = date(year, 12, 31)
        else:
            day = date.today()
        generate_meeting_pages(day)
