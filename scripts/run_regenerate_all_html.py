"""7/25以降の全レースHTMLを再生成するスクリプト

_lookup_recent_indices のスコア表示バグ修正、スコア補完後に実行する。
全対象日の make_daily_race_card_html を呼び出して HTML を再生成する。

Usage:
    python scripts/run_regenerate_all_html.py
"""

import os
import sys
import warnings
from datetime import datetime, date

warnings.simplefilter("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import paths
from src.logic.html_generator import race_page_generator, daily_index_generator

# 7/25以降の全週末（HTML生成対象日）
TARGET_DATES = [
    date(2026, 7, 25),
    date(2026, 7, 26),
    date(2026, 8, 1),
    date(2026, 8, 2),
    date(2026, 8, 8),
    date(2026, 8, 9),
    date(2026, 8, 15),
    date(2026, 8, 16),
    date(2026, 8, 22),
    date(2026, 8, 23),
    date(2026, 8, 29),
    date(2026, 8, 30),
]


def main():
    total = len(TARGET_DATES)
    for i, race_day in enumerate(TARGET_DATES, 1):
        print(f"\n[{i}/{total}] {race_day} HTML再生成中...")
        try:
            race_page_generator.make_daily_race_card_html(race_day)
            daily_index_generator.make_daily_index_page(race_day)
            print(f"  → 完了")
        except Exception as e:
            print(f"  → エラー: {e}")

    print("\n=== 全日付の再生成完了 ===")


if __name__ == "__main__":
    main()
