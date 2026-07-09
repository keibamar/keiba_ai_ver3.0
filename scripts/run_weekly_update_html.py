"""週次データセット更新後のHTML再生成バッチ（bat/MakeHTML/update_weekly_html.bat から呼ばれる）

update_weekly.bat（データセット更新）の直後に実行する想定。コース別データ・AI成績ページの
裏側にあるCSV（race_info/ai_performance）が週次更新で最新化されるため、それを反映した
HTMLを再生成する。出馬表・開催日インデックス等（レース当日系）は対象外（make_html_prev_day等の
別バッチが担当）。
"""

import os
import sys
import warnings

warnings.simplefilter("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.logic.html_generator import (  # noqa: E402
    ai_performance_report_generator,
    course_report_generator,
    legal_pages_generator,
)
from src.managers import ai_performance_dataset_manager  # noqa: E402

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
    print("Weekly Update Html Done")
