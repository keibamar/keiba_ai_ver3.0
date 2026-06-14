"""src/RacePrediction/daily_race_results.py のテスト（オフライン）。

旧 daily_race_results.py の get_each_race_results / save_each_race_result_csv /
save_day_race_result_each が、それぞれ新構造の関数への再エクスポートであることを確認する。
"""

from src.RacePrediction import daily_race_results
from src.logic.scheduler import race_result_scheduler
from src.logic.scraping import netkeiba_scraper
from src.managers import race_result_dataset_manager


def test_get_each_race_results_is_scrape_day_race_result():
    assert daily_race_results.get_each_race_results is netkeiba_scraper.scrape_day_race_result


def test_save_each_race_result_csv_is_save_race_result_for_race_id():
    assert daily_race_results.save_each_race_result_csv is race_result_dataset_manager.save_race_result_for_race_id


def test_save_day_race_result_each_is_update_daily_race_results():
    assert daily_race_results.save_day_race_result_each is race_result_scheduler.update_daily_race_results
