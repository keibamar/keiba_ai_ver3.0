"""サイドバーと Homeページ修正スクリプト

- 8/1, 8/2, 8/8, 8/9 のレースカードHTML・デイリーインデックスを再生成
  （raceDays.js のバージョン番号を最新に更新するため）
- Homeページを再生成（先週の結果 + raceDays.js バージョン更新）
X投稿なし。ConoHaアップロードなし。
"""

import os
import sys
import warnings
from datetime import date

warnings.simplefilter("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf_8"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


TARGET_DAYS = [
    date(2026, 8, 1),
    date(2026, 8, 2),
    date(2026, 8, 8),
    date(2026, 8, 9),
]


def step1_regen_race_cards():
    """8/1, 8/2, 8/8, 8/9 のレースカードHTMLを再生成（sidebar の raceDays.js バージョン更新）"""
    print("\n" + "=" * 60)
    print("Step 1: レースカードHTML再生成（サイドバーのraceDays.jsバージョン更新）")
    print("=" * 60)

    from src.logic.html_generator import race_page_generator

    for race_day in TARGET_DAYS:
        print(f"\n  {race_day} レースカード再生成中...")
        try:
            race_page_generator.make_daily_race_card_html(race_day)
            print(f"    完了: {race_day}")
        except Exception as e:
            print(f"    エラー: {e}")

    print("Step 1 完了")


def step2_regen_daily_index():
    """8/1, 8/2, 8/8, 8/9 のデイリーインデックスページを再生成"""
    print("\n" + "=" * 60)
    print("Step 2: デイリーインデックスページ再生成")
    print("=" * 60)

    from src.logic.html_generator import daily_index_generator

    for race_day in TARGET_DAYS:
        print(f"\n  {race_day} デイリーインデックス再生成中...")
        try:
            daily_index_generator.make_daily_index_page(race_day)
            print(f"    完了: {race_day}")
        except Exception as e:
            print(f"    エラー: {e}")

    print("Step 2 完了")


def step3_regen_home():
    """Homeページを再生成（先週の結果 + raceDays.js バージョン更新）"""
    print("\n" + "=" * 60)
    print("Step 3: Homeページ再生成（先週の結果更新）")
    print("=" * 60)

    from src.logic.html_generator import home_generator
    from src.logic.calculators import ai_performance_calculator

    today = date(2026, 8, 14)
    weekend_end = ai_performance_calculator.current_results_weekend_end(today)
    print(f"  先週の結果対象日: {weekend_end} (土={weekend_end.replace(day=weekend_end.day-1)}, 日={weekend_end})")

    try:
        home_generator.make_home_page(today)
        print("  Homeページ再生成完了")
    except Exception as e:
        print(f"  エラー: {e}")
        import traceback
        traceback.print_exc()

    print("Step 3 完了")


if __name__ == "__main__":
    print("=" * 60)
    print("サイドバー・Homeページ修正")
    print(f"実行日: {date.today()}")
    print("=" * 60)

    step1_regen_race_cards()
    step2_regen_daily_index()
    step3_regen_home()

    print("\n" + "=" * 60)
    print("修正完了")
    print("=" * 60)
