"""次週分の開催スケジュール作成バッチ（bat/TodayRace/make_time_id_list.bat から呼ばれる）

旧 ver2.0 の make_time_id_list.py の __main__（次の7日分について
[race_time, race_id, race_name] のリストを作成・保存）のver3.0移植。
実体は src.logic.scheduler.race_day_scheduler.update_weekly_time_id_list。
"""

import os
import sys
import warnings
from datetime import date

warnings.simplefilter("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.logic.scheduler import race_day_scheduler  # noqa: E402

if __name__ == "__main__":
    race_day_scheduler.update_weekly_time_id_list(date.today())
    print("Make Time Id List Done")
