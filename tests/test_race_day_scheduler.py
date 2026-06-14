"""src/logic/scheduler/race_day_scheduler.py のテスト（オフライン）。

post_race_pred / post_pred_return が組み立てるテキストパスと、
prediction_publisher.post_text_data への連携を確認する。

post_daily_race_pred はレース当日の発走時刻監視・1分間隔のポーリング・
最終レース後30分待機など sleep / datetime.now() に依存するライブループであり、
旧実装（src/RacePrediction/post_daily_race.py）も同様にユニットテスト対象外
だったため、本テストでは対象としない。
"""

import os
from datetime import date

from src.config import paths
from src.logic.scheduler import race_day_scheduler


def test_post_race_pred_posts_prediction_text(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "RACE_PREDICTION_TEXT_PATH", str(tmp_path / "race_prediction"))

    posted = []
    monkeypatch.setattr(race_day_scheduler.prediction_publisher, "post_text_data", posted.append)

    race_id = "202405010101"
    race_day = date(2024, 1, 27)

    race_day_scheduler.post_race_pred(race_id, race_day)

    expected = os.path.join(str(tmp_path / "race_prediction"), "20240127", f"{race_id}.txt")
    assert posted == [expected]


def test_post_pred_return_posts_return_text(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "RACE_RETURN_REPORT_TEXT_PATH", str(tmp_path / "race_returns"))

    posted = []
    monkeypatch.setattr(race_day_scheduler.prediction_publisher, "post_text_data", posted.append)

    place_id = 5
    race_day = date(2024, 1, 27)

    race_day_scheduler.post_pred_return(place_id, race_day)

    expected = os.path.join(str(tmp_path / "race_returns"), "20240127", "05_tokyo_pred_score.txt")
    assert posted == [expected]
