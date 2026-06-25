"""週末プレビュー投稿バッチ（bat/TodayRace/post_weekend_preview.bat から呼ばれる）

毎週金曜実行想定。実体は src.output.weekly_social_report.post_weekend_preview。
投稿する話題が少ない金曜に、今週末の注目レースを予告してHOMEへの来訪を促す。
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
    weekly_social_report.post_weekend_preview(date.today())
    print("Weekend Preview Post Done")
