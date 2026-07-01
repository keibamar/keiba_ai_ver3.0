"""次週分の開催スケジュール作成・HOMEリセット・週末HTML先行生成バッチ
（bat/MakeHTML/update_weekly_schedule.bat から呼ばれる）

毎週木曜実行想定。実体は src.logic.scheduler.race_day_scheduler の
update_weekly_time_id_list（次の7日分のrace_time_id_list作成・保存とHOMEページの
「今週/先週」リセット）と make_weekend_provisional_html（直近の土日の出馬表HTML先行生成。
枠順抽せん前のためAI予想は空欄）。
"""

import os
import sys
import warnings
from datetime import date

warnings.simplefilter("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.logic.html_generator import home_generator  # noqa: E402
from src.logic.scheduler import race_day_scheduler  # noqa: E402
from src.output import weekly_social_report  # noqa: E402

if __name__ == "__main__":
    today = date.today()
    race_day_scheduler.update_weekly_time_id_list(today)
    race_day_scheduler.make_weekend_provisional_html(today)
    # update_weekly_time_id_list内でもHOMEを再生成しているが、その時点では
    # 週末の出馬表（make_weekend_provisional_html）がまだ存在せず、「今週のメイン
    # レース」からのリンクが付かない。週末分の出馬表ができた後に再度HOMEを
    # 作り直し、リンクが反映された状態にする。
    home_generator.make_home_page()
    weekly_social_report.post_weekend_preview(today)
    print("Weekly Schedule Update Done")
