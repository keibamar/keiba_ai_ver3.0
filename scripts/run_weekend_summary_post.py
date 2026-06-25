"""週末まとめ投稿バッチ（bat/TodayRace/post_weekend_summary.bat から呼ばれる）

毎週火曜実行想定。実体は src.output.weekly_social_report.post_weekend_summary。
土日の予想結果投稿が確定した直後に、週末のAI成績（的中率・回収率）をまとめて
投稿し、AI成績ページへの来訪を促す。
"""

import os
import sys
import warnings
from datetime import date

warnings.simplefilter("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.output import weekly_social_report  # noqa: E402

if __name__ == "__main__":
    weekly_social_report.post_weekend_summary(date.today())
    print("Weekend Summary Post Done")
