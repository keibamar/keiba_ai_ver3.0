"""月次更新バッチ（bat/Datasets/update_monthly.bat から呼ばれる）

旧 ver2.0 の monthly_update.py（race_results/horse_peds/peds_results/
past_performance/race_returnsの月次更新、average_timeの更新）のver3.0移植。

Oracleオフライン学習パイプライン（旧LightGBM_dataset.make_annual_dataset）は
対象外（ver2.0データパスを参照し続ける、README参照）。
"""

import os
import sys
import warnings
from datetime import date

warnings.simplefilter("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.logic.scheduler import (  # noqa: E402
    horse_scheduler,
    race_info_scheduler,
    race_result_scheduler,
    race_returns_scheduler,
)

if __name__ == "__main__":
    day = date.today()

    race_result_scheduler.monthly_update_race_results(day)
    race_returns_scheduler.monthly_update_race_returns(day)

    horse_scheduler.monthly_update_horse_peds(day)
    horse_scheduler.monthly_update_pedsdata(day)
    horse_scheduler.monthly_update_past_performance(day)

    # ver2.0のaverage_time.timedata_update(year)と同一処理（月次専用関数はなく週次関数を共用）
    race_info_scheduler.weekly_update_average_time(day.year)

    print("Monthly Update Done")
