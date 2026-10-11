"""翌日分の出馬表・予想・HTML事前生成バッチ（bat/MakeHTML/make_html_prev_day.bat から呼ばれる）

旧 ver2.0 の web/src/scripts/make_html_prev_day.py（make_race_card_html.make_html_prev_day）
のver3.0移植。実体は src.logic.scheduler.race_day_scheduler.make_html_prev_day。

実行日の前日（木曜）に run_make_time_id_list.py で翌日分のrace_time_id_listが
保存されている必要がある。
"""

import os
import sys
import warnings
from datetime import date, timedelta

# Prevent UnicodeEncodeError on Windows CP932 consoles (emoji in print statements)
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr.encoding and sys.stderr.encoding.lower() not in ("utf-8", "utf8"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

warnings.simplefilter("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.logic.scheduler import race_day_scheduler  # noqa: E402

if __name__ == "__main__":
    race_day = date.today() + timedelta(days=1)
    race_day_scheduler.make_html_prev_day(race_day)
    print("Make Html Prev Day Done:", race_day)
