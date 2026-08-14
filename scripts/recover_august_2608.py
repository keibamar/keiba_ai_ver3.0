"""8/1〜8/13の欠落データ・HTML復元スクリプト

8/1〜8/13の期間、PCがスリープから復帰せず自動スクリプトが未実行だった。
本スクリプトはその期間のデータ・HTMLを順番に復元する。
X（Twitter）への投稿は一切行わない。

実行: python scripts/recover_august_2608.py
"""

import os
import sys
import warnings
from datetime import date, datetime, timedelta

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf_8"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

warnings.simplefilter("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

sys.path.append(r"C:\keiba_ai\keiba_ai_ver2.0\libs")
sys.path.append(r"C:\keiba_ai\keiba_ai_ver2.0\src\Datasets")


def step1_weekly_update():
    """Step1: 週次データ更新（レース結果・払い戻し・馬情報・race_info統計）"""
    print("\n" + "=" * 60)
    print("Step 1: 週次データ更新")
    print("=" * 60)

    from src.logic.scheduler import (
        horse_scheduler,
        race_info_scheduler,
        race_result_scheduler,
        race_returns_scheduler,
    )

    day = date.today()

    print("  race_results 更新中...")
    race_result_scheduler.weekly_update_race_results(day)

    print("  race_returns 更新中...")
    race_returns_scheduler.weekly_update_race_returns(day)

    print("  horse_peds 更新中...")
    horse_scheduler.weekly_update_horse_peds(day)

    print("  peds_results 更新中...")
    horse_scheduler.weekly_update_pedsdata(day)

    print("  past_performance 更新中...")
    horse_scheduler.weekly_update_past_performance(day)

    print("  race_info 統計更新中...")
    race_info_scheduler.weekly_update_horse_name_id_map(day.year)
    race_info_scheduler.weekly_update_average_pops(day.year)
    race_info_scheduler.weekly_update_winners_weight(day.year)
    race_info_scheduler.weekly_update_average_frame_and_horse(day.year)
    race_info_scheduler.weekly_update_winner_time(day.year)
    race_info_scheduler.weekly_update_average_time(day.year)

    print("Step 1 完了")


def step2_regen_html_for_missing_days():
    """Step2: 8/1〜8/13の開催日のHTML再生成（結果反映）"""
    print("\n" + "=" * 60)
    print("Step 2: 過去開催日のHTML再生成")
    print("=" * 60)

    from src.logic.html_generator import race_page_generator
    from src.config import paths

    races_dir = paths.PUBLIC_HTML_RACES_PATH

    # 対象期間（8/1〜8/13）のうち実際にHTMLディレクトリが存在する日 or
    # 土日祝として出走が想定される日
    target_dates = [
        date(2026, 8, 1),   # 土
        date(2026, 8, 2),   # 日
        date(2026, 8, 8),   # 土
        date(2026, 8, 9),   # 日
        date(2026, 8, 10),  # 月・山の日
    ]

    for race_day in target_dates:
        date_str = race_day.strftime("%Y%m%d")
        dir_path = os.path.join(races_dir, date_str)
        print(f"\n  {date_str} ({race_day.strftime('%a')}) ...")

        # ディレクトリがなければ新規作成扱いで実行
        try:
            race_page_generator.make_daily_race_card_html(race_day)
            print(f"    HTML再生成完了: {date_str}")
        except Exception as e:
            print(f"    スキップ (エラーまたはレースなし): {e}")

    print("\nStep 2 完了")


def step3_update_ai_performance():
    """Step3: AI成績データセット更新"""
    print("\n" + "=" * 60)
    print("Step 3: AI成績データセット更新")
    print("=" * 60)

    from src.managers import ai_performance_dataset_manager

    added = ai_performance_dataset_manager.update_ai_performance_dataset()
    print(f"  追加行数: {added}")
    print("Step 3 完了")


def step4_weekly_schedule_update():
    """Step4: 今週分スケジュール更新（X投稿なし）"""
    print("\n" + "=" * 60)
    print("Step 4: 週次スケジュール更新（X投稿スキップ）")
    print("=" * 60)

    from src.logic.html_generator import daily_index_generator, home_generator
    from src.logic.scheduler import race_day_scheduler
    from src.managers import html_manager

    today = date.today()

    print("  time_id_list 更新中...")
    race_day_scheduler.update_weekly_time_id_list(today)

    print("  今週末 provisional HTML 生成中...")
    race_day_scheduler.make_weekend_provisional_html(today)

    print("  raceMeetings.js 再生成中...")
    html_manager.regenerate_race_meetings_js()

    print("  レースカレンダーページ再生成中...")
    daily_index_generator.make_races_calendar_page()

    print("  Homeページ再生成中...")
    home_generator.make_home_page()

    # weekly_social_report.post_weekend_preview(today) ← X投稿: スキップ
    print("  (X投稿: スキップ)")

    print("Step 4 完了")


def step5_update_today_content():
    """Step5: 本日コンテンツ最終更新"""
    print("\n" + "=" * 60)
    print("Step 5: 本日コンテンツ更新")
    print("=" * 60)

    from src.logic.html_generator import daily_index_generator, home_generator
    from src.managers import html_manager

    today = date.today()

    html_manager.regenerate_race_meetings_js()
    daily_index_generator.make_races_calendar_page(today)
    home_generator.make_home_page(today)

    print("Step 5 完了")


def step6_ai_performance_pages():
    """Step6: AI成績ページ再生成"""
    print("\n" + "=" * 60)
    print("Step 6: AI成績ページ再生成")
    print("=" * 60)

    from src.logic.html_generator import ai_performance_report_generator

    ai_performance_report_generator.make_ai_performance_index_page()
    ai_performance_report_generator.make_all_annual_performance_pages()
    ai_performance_report_generator.make_all_course_performance_pages()
    ai_performance_report_generator.make_all_meeting_performance_pages()

    print("Step 6 完了")


if __name__ == "__main__":
    print("=" * 60)
    print("8月上旬欠落データ復元")
    print(f"実行日: {date.today()}")
    print("X投稿: スキップ")
    print("=" * 60)

    step1_weekly_update()
    step2_regen_html_for_missing_days()
    step3_update_ai_performance()
    step4_weekly_schedule_update()
    step5_update_today_content()
    step6_ai_performance_pages()

    print("\n" + "=" * 60)
    print("復元完了")
    print("=" * 60)
