"""旧 Forge（HTML生成）モジュールのリダイレクト

実装は src/logic/html_generator/horse_report_generator.py に移植済み。
このモジュールは後方互換のための re-export のみを提供する。
"""

import os
import sys

sys.dont_write_bytecode = True

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.logic.html_generator.horse_report_generator import (
    build_horse_report,
    calc_average_norm_passages,
    extract_peds_name,
    get_avg_time,
    get_class_color,
    get_ground_state_color,
    get_horse_id_by_name,
    get_race_type_color,
    get_time_diff_color,
    horse_report_to_html,
    load_horse_id_map,
    load_horse_peds,
    load_past_performance,
    load_peds_results,
    ms_to_time_str,
    normalize_passage,
    peds_results_for_bloodline,
    recent_5_performances,
    safe_value,
    same_course_best_time,
    time_str_to_ms,
    turf_dirt_summary,
)
