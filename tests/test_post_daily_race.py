"""src/RacePrediction/post_daily_race.py のテスト（オフライン）。

旧 post_daily_race.py の post_daily_race_pred / post_pred_return / post_race_pred が、
それぞれ新構造の race_day_scheduler の同名関数への再エクスポートであることを確認する。
"""

from src.RacePrediction import post_daily_race
from src.logic.scheduler import race_day_scheduler


def test_post_daily_race_pred_is_race_day_scheduler_post_daily_race_pred():
    assert post_daily_race.post_daily_race_pred is race_day_scheduler.post_daily_race_pred


def test_post_pred_return_is_race_day_scheduler_post_pred_return():
    assert post_daily_race.post_pred_return is race_day_scheduler.post_pred_return


def test_post_race_pred_is_race_day_scheduler_post_race_pred():
    assert post_daily_race.post_race_pred is race_day_scheduler.post_race_pred
