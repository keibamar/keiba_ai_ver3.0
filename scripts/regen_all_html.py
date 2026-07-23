"""全HTMLページ一括再生成スクリプト（手動実行用）

出馬表テンプレート変更後など、全HTMLを最新のジェネレーターで再生成したいときに使う。
public_html/races/ 配下の全日付ディレクトリを対象に出馬表を再生成し、
コース別データ・AI成績・Homeも再生成する。

実行例:
  python scripts/regen_all_html.py
"""

import os
import sys
import warnings
from datetime import datetime, date

# Windows cp932環境でのUnicodeEncodeError回避
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf_8"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

warnings.simplefilter("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from src.config import paths  # noqa: E402
from src.logic.html_generator import (  # noqa: E402
    ai_performance_report_generator,
    course_report_generator,
    daily_index_generator,
    home_generator,
    race_page_generator,
)

if __name__ == "__main__":
    races_dir = paths.PUBLIC_HTML_RACES_PATH

    # public_html/races/ 配下の YYYYMMDD ディレクトリを収集
    date_dirs = sorted(
        d for d in os.listdir(races_dir)
        if os.path.isdir(os.path.join(races_dir, d)) and d.isdigit() and len(d) == 8
    )
    print(f"対象日付: {date_dirs[0]} 〜 {date_dirs[-1]}  ({len(date_dirs)}日分)")

    # 1. 出馬表HTML（日付ごと）
    for date_str in date_dirs:
        race_day = datetime.strptime(date_str, "%Y%m%d").date()
        print(f"  出馬表再生成: {date_str}")
        race_page_generator.make_daily_race_card_html(race_day)

    # 2. レース日次インデックス
    print("レースインデックス再生成...")
    for date_str in date_dirs:
        race_day = datetime.strptime(date_str, "%Y%m%d").date()
        daily_index_generator.make_daily_index_page(race_day)
    daily_index_generator.make_races_calendar_page()

    # 3. コース別データページ
    print("コース別データページ再生成...")
    course_report_generator.make_all_course_pages()

    # 4. AI成績ページ
    print("AI成績ページ再生成...")
    ai_performance_report_generator.make_ai_performance_index_page()
    ai_performance_report_generator.make_all_annual_performance_pages()
    ai_performance_report_generator.make_all_course_performance_pages()
    ai_performance_report_generator.make_all_meeting_performance_pages()

    # 5. Homeページ
    print("Homeページ再生成...")
    home_generator.make_home_page()

    print("全HTML再生成完了")
