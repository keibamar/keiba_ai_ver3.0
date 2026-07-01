"""当日の回収率レポート生成・投稿・最終HTML反映バッチ
（bat/TodayRace/today_race_rerturns.bat から呼ばれる）

旧 ver2.0 の src/RacePrediction/calc_returns.py（回収率計算・投稿）+
web/src/scripts/make_race_card_html.py（結果反映HTML再生成）のver3.0移植。
実体は src.output.return_report.post_daily_race_returns（Herald）と
src.logic.html_generator.race_page_generator.make_daily_race_card_html（Forge）。
高配当的中（単勝/複勝1000円以上）は src.output.highlight_report.post_big_hit_highlights
が画像付きでXに別途投稿する。

post_today_race.bat（当日のライブループ）が全レース終了・最終結果確定まで
完了した後に実行する想定。
"""

import os
import sys
import warnings
from datetime import date

warnings.simplefilter("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.logic.html_generator import race_page_generator  # noqa: E402
from src.managers import ai_performance_dataset_manager  # noqa: E402
from src.output import highlight_report, return_report  # noqa: E402

if __name__ == "__main__":
    race_day = date.today()
    return_report.post_daily_race_returns(race_day)
    race_page_generator.make_daily_race_card_html(race_day)
    # 確定した予想結果をAI成績データセットに反映する（次回のAI成績ページ再生成（水曜）で使われる）
    added = ai_performance_dataset_manager.update_ai_performance_dataset()
    print(f"AI Performance Dataset Updated: {added} rows added")
    highlight_report.post_big_hit_highlights(race_day)
    highlight_report.post_daily_high_payout_summary(race_day)
    print("Today Race Returns Done")
