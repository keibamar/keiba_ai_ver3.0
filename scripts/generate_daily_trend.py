"""日次傾向短評ページを生成する

Usage:
    python scripts/generate_daily_trend.py           # 当日分
    python scripts/generate_daily_trend.py 20260816  # 指定日

前日・前週同曜日の比較データも自動取得してテキスト生成する。
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


def run(target_date: date) -> None:
    print(f"[daily_trend] {target_date} の傾向短評を生成します...")

    stats = trend_analyzer.get_day_stats(target_date)
    if "error" in stats:
        print(f"  ERROR: {stats['error']}")
        return

    weekday = target_date.weekday()
    # 前日（日曜なら土曜との比較用）
    prev_day = target_date - timedelta(days=1)
    prev_day_stats = None
    if weekday == 6:  # 日曜 → 前日（土曜）と比較
        prev_day_stats = trend_analyzer.get_day_stats(prev_day)
        if "error" in prev_day_stats:
            prev_day_stats = None

    # 前週同曜日
    prev_week_date = target_date - timedelta(weeks=1)
    prev_week_stats = trend_analyzer.get_day_stats(prev_week_date)
    if "error" in prev_week_stats:
        prev_week_stats = None

    print("  テキスト生成中（Claude API）...")
    comment = trend_text_generator.generate_daily_comment(stats, prev_day_stats, prev_week_stats)

    print("  HTML生成中...")
    trend_page_generator.make_daily_trend_page(target_date, stats, comment)

    print(f"  完了: public_html/trend/{target_date.strftime('%Y%m%d')}.html")


if __name__ == "__main__":
    if len(sys.argv) >= 2:
        ds = sys.argv[1]
        d = date(int(ds[:4]), int(ds[4:6]), int(ds[6:8]))
    else:
        d = date.today()
    run(d)
