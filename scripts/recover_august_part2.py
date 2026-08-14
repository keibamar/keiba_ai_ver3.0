"""8月上旬 第2次復元スクリプト

- 8/2の race_results を取得（8/3基準で再スクレイピング）
- 8/8、8/9、8/10 の race_card を過去日付で生成
- 8/1〜8/10 の HTML を再生成（結果反映）
- AI成績データセット更新
X投稿なし。
"""

import os
import sys
import warnings
from datetime import date, timedelta

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf_8"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

warnings.simplefilter("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

sys.path.append(r"C:\keiba_ai\keiba_ai_ver2.0\libs")
sys.path.append(r"C:\keiba_ai\keiba_ai_ver2.0\src\Datasets")

import numpy as np


def step1_get_aug2_results():
    """8/2の race_results と race_returns を 8/3基準で取得"""
    print("\n" + "=" * 60)
    print("Step 1: 8/2 race_results + race_returns 補完（8/3基準）")
    print("=" * 60)

    from src.logic.scheduler import race_result_scheduler, race_returns_scheduler
    from src.managers import race_result_dataset_manager, race_info_dataset_manager

    day = date(2026, 8, 3)
    for place_id in range(1, 11):
        print(f"  place_id={place_id} race_results 更新中...")
        race_result_scheduler.update_race_results_dataset(place_id, day)
        race_result_dataset_manager.export_race_info_per_race(place_id, 2026)
        race_result_dataset_manager.split_race_results_by_year(place_id, 2026)

        print(f"  place_id={place_id} race_returns 更新中...")
        race_returns_scheduler.update_race_returns_dataset(place_id, day)
        race_info_dataset_manager.split_race_returns_csv(place_id, 2026)

    print("Step 1 完了")


def step2_make_racecards_for_past_days():
    """8/8、8/9 の time_id_list 作成 → race_card スクレイピング → HTML 生成
    8/10 (山の日) は race_calendar になし→スキップ
    """
    print("\n" + "=" * 60)
    print("Step 2: 8/8〜8/9 race_card 生成（time_id_list 作成 → make_html_prev_day）")
    print("=" * 60)

    from src.logic.scheduler import race_day_scheduler
    from src.managers import race_card_dataset_manager

    for race_day in [date(2026, 8, 8), date(2026, 8, 9)]:
        # 既存の time_id_list を確認
        existing = race_card_dataset_manager.get_time_id_list(race_day)
        if existing:
            print(f"  {race_day}: time_id_list 既存（{len(existing)}件）→ そのまま make_html_prev_day へ")
        else:
            print(f"  {race_day}: time_id_list 作成中...")
            time_id_list = race_day_scheduler.make_time_id_list(race_day)
            if not time_id_list:
                print(f"    → race_calendar に該当日なし。スキップ。")
                continue
            race_card_dataset_manager.save_time_id_list(race_day, time_id_list)
            print(f"    → {len(time_id_list)}レース分の time_id_list を保存")

        print(f"  {race_day}: race_card スクレイピング + HTML 生成中...")
        try:
            race_day_scheduler.make_html_prev_day(race_day)
            print(f"    完了: {race_day}")
        except Exception as e:
            print(f"    エラー: {e}")

    print("Step 2 完了")


def step3_regen_html():
    """8/1・8/2 の HTML 再生成（結果反映）
    8/8・8/9 は Step2 の make_html_prev_day 内で生成済みのため対象外
    """
    print("\n" + "=" * 60)
    print("Step 3: 8/1・8/2 HTML 再生成")
    print("=" * 60)

    from src.logic.html_generator import race_page_generator

    for race_day in [date(2026, 8, 1), date(2026, 8, 2)]:
        print(f"\n  {race_day} HTML再生成中...")
        try:
            race_page_generator.make_daily_race_card_html(race_day)
            print(f"    完了: {race_day}")
        except Exception as e:
            print(f"    エラー（スキップ）: {e}")

    print("Step 3 完了")


def step4_update_ai_performance():
    """AI成績データセット更新"""
    print("\n" + "=" * 60)
    print("Step 4: AI成績更新")
    print("=" * 60)

    from src.managers import ai_performance_dataset_manager

    added = ai_performance_dataset_manager.update_ai_performance_dataset()
    print(f"  追加行数: {added}")

    from src.logic.html_generator import ai_performance_report_generator
    ai_performance_report_generator.make_ai_performance_index_page()
    ai_performance_report_generator.make_all_annual_performance_pages()
    ai_performance_report_generator.make_all_meeting_performance_pages()
    print("Step 4 完了")


if __name__ == "__main__":
    print("=" * 60)
    print("8月上旬 第2次復元")
    print(f"実行日: {date.today()}")
    print("=" * 60)

    step1_get_aug2_results()
    step2_make_racecards_for_past_days()
    step3_regen_html()
    step4_update_ai_performance()

    print("\n" + "=" * 60)
    print("第2次復元完了")
    print("=" * 60)
