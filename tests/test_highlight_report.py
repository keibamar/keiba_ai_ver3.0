"""src/output/highlight_report.py のテスト（オフライン、実データ）。

2024-10-20の新潟9R（十日町特別）はAI本命馬ヴァンヴィーヴの単勝が1000円で的中した、
既知の高配当レース（data/ai_performance/ai_performance.csv参照）。
"""

from datetime import date

import pytest

from src.output import highlight_report as hr

KNOWN_BIG_HIT_DAY = date(2024, 10, 20)


def test_find_big_hits_detects_known_high_payout_race():
    hits = hr.find_big_hits(KNOWN_BIG_HIT_DAY)

    print(f"\n--- find_big_hits({KNOWN_BIG_HIT_DAY}) ---")
    for h in hits:
        print(f"  {h}")

    hit = next(h for h in hits if h["race_id"] == "202404040609")
    assert hit["place_id"] == 4
    assert hit["race_num"] == 9
    assert hit["race_name"] == "十日町特別"
    assert hit["bet_type"] == "win"
    assert hit["bet_type_label"] == "単勝"
    assert hit["payout"] == pytest.approx(1000.0)
    assert hit["pick_name"] == "ヴァンヴィーヴ"


def test_find_big_hits_excludes_payouts_below_threshold():
    hits = hr.find_big_hits(KNOWN_BIG_HIT_DAY)

    assert all(h["payout"] >= 1000.0 for h in hits)


def test_find_big_hits_returns_empty_list_when_no_schedule():
    assert hr.find_big_hits(date(2020, 1, 5)) == []


def test_highlight_post_text_links_to_race_card_and_includes_payout():
    hit = {
        "race_id": "202404040609",
        "place_id": 4,
        "race_num": 9,
        "race_name": "十日町特別",
        "bet_type": "win",
        "bet_type_label": "単勝",
        "payout": 1000.0,
        "pick_name": "ヴァンヴィーヴ",
    }

    text = hr._highlight_post_text(KNOWN_BIG_HIT_DAY, hit, "新潟", "04_nigata")

    print(f"\n--- _highlight_post_text ---\n{text}")

    assert "新潟9R 十日町特別" in text
    assert "単勝 1000円" in text
    assert "AI本命: ヴァンヴィーヴ" in text
    assert "https://mar-keiba.com/races/20241020/04_nigataR9.html" in text
    assert "#MAR競馬予想" in text
