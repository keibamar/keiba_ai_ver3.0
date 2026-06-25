"""src/output/weekly_social_report.py のテスト（オフライン）。

post_weekend_preview/post_weekend_summaryはXへの実投稿（prediction_publisher.post_text）
を伴うため、それ自体はmonkeypatchで差し替えてオフラインで検証する。
"""

from datetime import date

import pandas as pd
import pytest

from src.config import paths
from src.output import weekly_social_report as w


@pytest.fixture
def captured_post(monkeypatch):
    posted = []
    monkeypatch.setattr(w.prediction_publisher, "post_text", lambda text: posted.append(text))
    return posted


def test_post_weekend_preview_lists_main_races_and_links_home(monkeypatch, captured_post):
    races = [
        {
            "race_id": "202602010411", "place_id": 2, "race_name": "UHB杯",
            "race_time": "1520", "race_type": "芝", "course_len": 1200,
            "race_day": date(2026, 6, 27),
        },
        {
            "race_id": "202605030711", "place_id": 5, "race_name": "七夕賞",
            "race_time": "1545", "race_type": None, "course_len": None,
            "race_day": date(2026, 6, 28),
        },
    ]
    monkeypatch.setattr(w.calc, "get_week_main_races_with_course", lambda today: races)

    w.post_weekend_preview(date(2026, 6, 26))

    assert len(captured_post) == 1
    text = captured_post[0]
    print(f"\n--- post_weekend_preview ---\n{text}")

    assert "今週末の注目レース" in text
    assert "・土 函館11R UHB杯" in text
    assert "・日 東京11R 七夕賞" in text
    assert "https://mar-keiba.com/" in text
    assert "#MAR競馬予想" in text


def test_post_weekend_preview_skips_when_no_races(monkeypatch, captured_post):
    monkeypatch.setattr(w.calc, "get_week_main_races_with_course", lambda today: [])

    w.post_weekend_preview(date(2026, 6, 26))

    assert captured_post == []


def test_post_weekend_summary_reports_win_and_place_for_recent_saturday_sunday(monkeypatch, captured_post, tmp_path):
    monkeypatch.setattr(w.m, "AI_PERFORMANCE_DATASET_PATH", str(tmp_path / "ai_performance.csv"))
    df = pd.DataFrame(
        {
            "race_day": ["2026-06-20", "2026-06-21"],
            "year": ["2026", "2026"],
            "place_id": ["5", "5"],
            "times": ["3", "3"],
            "win_hit": ["1", "0"],
            "win_return": ["200.0", "0.0"],
            "place_hit": ["1", "1"],
            "place_return": ["150.0", "120.0"],
            "trio_box_hit": ["0", "0"],
            "trio_box_return": ["0.0", "0.0"],
        },
        index=["A", "B"],
    )
    w.m.save_ai_performance_dataset(df)

    # 2026-06-23は火曜日。直近の土日（06-20/06-21）が対象になる
    w.post_weekend_summary(date(2026, 6, 23))

    assert len(captured_post) == 1
    text = captured_post[0]
    print(f"\n--- post_weekend_summary ---\n{text}")

    assert "06/20〜06/21のAI成績" in text
    # win: hit=1/2=50.0%, return=(200+0)/2=100.0
    assert "単勝 的中率50.0% 回収率100.0%" in text
    # place: hit=2/2=100.0%, return=(150+120)/2=135.0
    assert "複勝 的中率100.0% 回収率135.0%" in text
    assert "(対象2件)" in text
    assert "https://mar-keiba.com/performance/" in text


def test_post_weekend_summary_skips_when_no_data(monkeypatch, captured_post, tmp_path):
    monkeypatch.setattr(w.m, "AI_PERFORMANCE_DATASET_PATH", str(tmp_path / "ai_performance.csv"))

    w.post_weekend_summary(date(2026, 6, 23))

    assert captured_post == []
