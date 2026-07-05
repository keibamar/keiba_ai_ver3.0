"""「本日の〜」系コンテンツの日次更新バッチ（bat/MakeHTML/update_today_content.bat から呼ばれる）

HOMEページ（今週/先週のレース）・開催日カレンダーページ（本日の開催）は、生成時点の
date.today()を基準に内容を決めるため、生成し直さない限り日付が変わっても古い内容の
ままになる。既存の自動化は土日・特定の曜日にしか動かないため、平日（特にレースが
無い日）は何日も再生成されない状態だった。この日次バッチを毎日実行することで、
「本日」「今週」を基準にした内容を毎日確実に最新化する。

raceMeetings.js（カレンダーの日付セルに表示する開催場・メインレース・第○回○日目）も
ここで再生成する。週次スケジュール作成（weekly_schedule、木曜）時点で先の日付分は
判明済みだが、レース名の確定・修正等に追従できるよう毎日作り直す。
"""

import os
import sys
import warnings
from datetime import date, datetime, timedelta

warnings.simplefilter("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.logic.html_generator import daily_index_generator, home_generator  # noqa: E402
from src.managers import html_manager  # noqa: E402

if __name__ == "__main__":
    # 深夜0:05の自動実行時、date.today()は既に翌日になっているが
    # ページに表示するコンテンツ（「本日の開催」「今週のメインレース」）は
    # 前日のレース情報を指すべきのため、6時前は前日を基準日とする。
    now = datetime.now()
    target_date = date.today() - timedelta(days=1) if now.hour < 6 else date.today()

    html_manager.regenerate_race_meetings_js()
    daily_index_generator.make_races_calendar_page(target_date)
    home_generator.make_home_page(target_date)
    print("Update Today Content Done")
