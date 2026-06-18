"""当日のレース予想生成・投稿・結果反映バッチ（bat/TodayRace/post_today_race.bat から呼ばれる）

旧 ver2.0 の src/RacePrediction/post_daily_race.py（post_daily_race_pred）の
ver3.0移植。実体は src.logic.scheduler.race_day_scheduler.post_daily_race_pred。

当日の発走時刻を監視しながら1分間隔でポーリングする長時間実行のライブループ
（最終レースから30分後に終了）。詳細は
specifications/RaceScheduleCenter/SequenceDiagram.pu 参照。
"""

import os
import sys
from datetime import date

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.logic.scheduler import race_day_scheduler  # noqa: E402

if __name__ == "__main__":
    race_day_scheduler.post_daily_race_pred(date.today())
    print("Post Today Race Done")
