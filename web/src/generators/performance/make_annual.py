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

TEMPLATE_NAME = "ai_annual.html"

# -----------------------------
# 年間ページ用の開催リンクを生成
# -----------------------------
def get_meeting_links_for_year(day=date.today()):
    meetings = load_meetings(day.year)
    links = []

    for (year, course, times), days in meetings.items():

        first_day = days[0]["date"]
        last_day = days[-1]["date"]
        if isinstance(first_day, datetime):
            first_day = first_day.date()
        # 未来の開催は除外
        if first_day > day:
            continue

        track_name = TRACK_MAP[course]
        filename = f"{track_name}-{times}th.html"
        url = f"../meeting/{year}/{filename}"

        links.append({
            "track": track_name,
            "times": times,
            "url": url,
            "period": f"{first_day.strftime('%Y/%m/%d')}〜{last_day.strftime('%Y/%m/%d')}"
        })

    # 競馬場名 → 開催回でソート
    links.sort(key=lambda x: (x["track"], x["times"]))

    return links


# -----------------------------
# 年間ページ生成
# -----------------------------
def generate_annual_page(year, all_years):
    if year < date.today().year:
        day = date(year, 12, 31)
    else:
        day = date.today()
    meetings = get_meeting_links_for_year(day)

    template = load_template(TEMPLATE_NAME)

    html = template.render(
        title=f"{year}年 AI年間成績",
        year=year,
        meetings=meetings,
        all_years=sorted(all_years, reverse=True)
    )

    out_dir = os.path.join(PERFORMACE_HTML_PATH, "annual")
    os.makedirs(out_dir, exist_ok=True)

    output_path = os.path.join(out_dir, f"{year}.html")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Generated: ai/annual/{year}.html")


# -----------------------------
# 実行
# -----------------------------
if __name__ == "__main__":
    # 過去年のページ
    all_years = {2026, 2025, 2024}
    for y in all_years:
        generate_annual_page(y,  all_years)
