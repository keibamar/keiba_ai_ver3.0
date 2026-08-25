"""週次データセット更新後のHTML再生成バッチ（bat/MakeHTML/update_weekly_html.bat から呼ばれる）

update_weekly.bat（データセット更新）の直後に実行する想定。コース別データ・AI成績ページの
裏側にあるCSV（race_info/ai_performance）が週次更新で最新化されるため、それを反映した
HTMLを再生成する。また、週次更新で通過位置・上り3Fが確定した前週土日の出馬表HTMLも
再生成する（当日スクレイピング時は速報値のため欠損している場合がある）。
"""

import os
import sys
import warnings
from datetime import date, timedelta

warnings.simplefilter("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.logic.html_generator import (  # noqa: E402
    ai_performance_report_generator,
    course_report_generator,
    legal_pages_generator,
    race_page_generator,
)
from src.managers import ai_performance_dataset_manager  # noqa: E402


def _prev_weekend_days(today: date) -> list[date]:
    """今日を基準に直前の土曜・日曜を返す（水曜実行想定）"""
    # 今週の月曜から -2日 = 前週日曜、-3日 = 前週土曜
    monday = today - timedelta(days=today.weekday())
    prev_sunday = monday - timedelta(days=1)
    prev_saturday = monday - timedelta(days=2)
    return [prev_saturday, prev_sunday]


if __name__ == "__main__":
    # 取りこぼし（payout確定が遅れた等）があれば、ここでもAI成績データセットを最新化しておく
    added = ai_performance_dataset_manager.update_ai_performance_dataset()
    print(f"AI Performance Dataset Updated: {added} rows added")

    course_report_generator.make_all_course_pages()
    ai_performance_report_generator.make_ai_performance_index_page()
    ai_performance_report_generator.make_all_annual_performance_pages()
    ai_performance_report_generator.make_all_course_performance_pages()
    ai_performance_report_generator.make_all_meeting_performance_pages()
    legal_pages_generator.make_about_page()
    legal_pages_generator.make_privacy_policy_page()
    legal_pages_generator.make_terms_page()

    # 前週土日の出馬表HTMLを再生成（週次更新で通過位置・上りが確定するため）
    for race_day in _prev_weekend_days(date.today()):
        print(f"前週出馬表HTML再生成: {race_day}")
        race_page_generator.make_daily_race_card_html(race_day)

    print("Weekly Update Html Done")
