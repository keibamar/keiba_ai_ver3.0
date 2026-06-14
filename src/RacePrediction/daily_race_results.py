import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.logic.scheduler.race_result_scheduler import update_daily_race_results  # noqa: E402,F401
from src.logic.scraping.netkeiba_scraper import scrape_day_race_result  # noqa: E402,F401
from src.managers.race_result_dataset_manager import save_race_result_for_race_id  # noqa: E402,F401

# 旧名の再エクスポート（呼び出し元: post_daily_race.py, web/src/generators/make_race_card_html.py）
get_each_race_results = scrape_day_race_result
save_each_race_result_csv = save_race_result_for_race_id
save_day_race_result_each = update_daily_race_results
