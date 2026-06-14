"""旧 Forge（HTML生成）モジュールのリダイレクト

実装は src/logic/html_generator/daily_index_generator.py に移植済み。
このモジュールは後方互換のための re-export のみを提供する。
"""

import os
import sys

sys.dont_write_bytecode = True

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.logic.html_generator.daily_index_generator import (
    build_table_rows,
    daily_index_template,
    group_place_races,
    load_race_info,
    make_daily_index_page,
    make_index_page,
)
