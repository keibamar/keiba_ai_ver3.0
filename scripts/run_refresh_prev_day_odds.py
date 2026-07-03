"""前日夜のオッズ・人気更新バッチ（bat/MakeHTML/refresh_prev_day_odds.bat から呼ばれる）

make_html_prev_day の実行後（夕方）にオッズ・人気が「---.-」/「**」の
プレースホルダーになっている問題を解消するため、前日20時頃に実行する。
AI予想の再計算は行わずオッズ・人気だけを再取得して CSV と HTML を更新する。
"""

import os
import sys
import warnings
from datetime import date, timedelta

warnings.simplefilter("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.logic.scheduler import race_day_scheduler  # noqa: E402

if __name__ == "__main__":
    race_day = date.today() + timedelta(days=1)
    print(f"オッズ・人気更新: {race_day}")
    race_day_scheduler.refresh_prev_day_odds(race_day)
    print("完了:", race_day)
