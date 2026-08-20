"""週次傾向振り返りページを生成する

Usage:
    python scripts/generate_weekly_trend.py              # 直近の土曜分
    python scripts/generate_weekly_trend.py 20260816     # 指定の土曜日付
    python scripts/generate_weekly_trend.py 20260816 "来週は東京でBコース替わり"

第2引数に来週末のヒント（天気・コース変更など）を渡すとテキストに反映される。
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from datetime import date, timedelta

from src.logic.analytics import trend_analyzer
from src.logic.text_generator import trend_text_generator
from src.logic.html_generator import trend_page_generator


def _last_saturday(from_date: date) -> date:
    """from_date以前の直近土曜を返す"""
    offset = (from_date.weekday() - 5) % 7
    return from_date - timedelta(days=offset)


def run(sat_date: date, next_weekend_hint: str = "") -> None:
    sun_date = sat_date + timedelta(days=1)
    print(f"[weekly_trend] {sat_date}（土）〜{sun_date}（日）の週次振り返りを生成します...")

    week_stats = trend_analyzer.get_week_stats(sat_date, sun_date)

    # 前週
    prev_sat = sat_date - timedelta(weeks=1)
    prev_sun = prev_sat + timedelta(days=1)
    prev_week_stats = trend_analyzer.get_week_stats(prev_sat, prev_sun)
    if "error" in prev_week_stats.get("sat", {}) and "error" in prev_week_stats.get("sun", {}):
        prev_week_stats = None

    print("  テキスト生成中（Claude API）...")
    comment = trend_text_generator.generate_weekly_comment(week_stats, prev_week_stats, next_weekend_hint)

    print("  HTML生成中...")
    trend_page_generator.make_weekly_trend_page(sat_date, sun_date, week_stats, comment)

    print(f"  完了: public_html/trend/weekly_{sat_date.strftime('%Y%m%d')}.html")


if __name__ == "__main__":
    if len(sys.argv) >= 2:
        ds = sys.argv[1]
        sat = date(int(ds[:4]), int(ds[4:6]), int(ds[6:8]))
    else:
        sat = _last_saturday(date.today())

    hint = sys.argv[2] if len(sys.argv) >= 3 else ""
    run(sat, hint)
