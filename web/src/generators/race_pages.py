"""旧 Forge（HTML生成）モジュールのリダイレクト

実装は src/logic/html_generator/race_page_generator.py に移植済み。
このモジュールは後方互換のための re-export のみを提供する。
"""

import os
import sys
from datetime import date

sys.dont_write_bytecode = True

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.logic.html_generator.race_page_generator import (
    build_html_content,
    build_nav_html,
    build_table_race_cards,
    generate_frame_horse_info,
    generate_payout_table_html,
    generate_peds_result_html,
    generate_pops_info,
    generate_race_info,
    generate_recent_same_condition_html,
    generate_result_table,
    generate_run_time_info,
    generate_weight_info,
    get_race_info,
    get_result_table,
    get_returns_table,
    make_daily_race_card_html,
    make_race_card_html,
    make_up_to_date_race_card_html,
    read_race_csv,
)

if __name__ == "__main__":
    # 今日のhtmlを作成
    make_daily_race_card_html(date.today())
