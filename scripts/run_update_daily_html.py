"""当日HTML更新バッチ（bat/MakeHTML/update_daily_html.bat から呼ばれる）

旧 ver2.0 の web/src/scripts/update_daily_html.py（make_race_card_html.update_daily_html）
のver3.0移植。実体は src.logic.scheduler.race_day_scheduler.update_daily_html。

post_today_race.bat（race_day_scheduler.post_daily_race_pred）のライブループ内でも
レースごとに同等のHTML再生成が行われるため、本バッチは結果反映漏れがあった場合の
バックアップ的な再実行という位置づけになる。
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
from src.managers import race_schedule_dataset_manager  # noqa: E402

if __name__ == "__main__":
    today = date.today()
    if not race_schedule_dataset_manager.get_daily_id(0, today):
        print(f"{today} は開催なし。スキップします。")
        sys.exit(0)
    race_day_scheduler.update_daily_html(today)
    print("Update Daily Html Done")
