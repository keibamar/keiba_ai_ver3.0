"""週次更新バッチ（bat/Datasets/update_weekly.bat から呼ばれる）

旧 ver2.0 の weekly_update.py（race_results/horse_peds/peds_results/
past_performance/race_returnsの週次更新、race_infoの週次集計）のver3.0移植。
README「### 3-2. 週次更新」に記載の呼び出し列と同一。

Oracleオフライン学習パイプライン（旧LightGBM_dataset.weekly_update_dataset_for_train）
は対象外（ver2.0データパスを参照し続ける、README参照）。
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

    race_result_scheduler.weekly_update_race_results(day)
    race_returns_scheduler.weekly_update_race_returns(day)

    horse_scheduler.weekly_update_horse_peds(day)
    horse_scheduler.weekly_update_pedsdata(day)
    horse_scheduler.weekly_update_past_performance(day)

    race_info_scheduler.weekly_update_horse_name_id_map(day.year)
    race_info_scheduler.weekly_update_average_pops(day.year)
    race_info_scheduler.weekly_update_winners_weight(day.year)
    race_info_scheduler.weekly_update_average_frame_and_horse(day.year)
    race_info_scheduler.weekly_update_winner_time(day.year)
    race_info_scheduler.weekly_update_average_time(day.year)

    print("Weekly Update Done")
