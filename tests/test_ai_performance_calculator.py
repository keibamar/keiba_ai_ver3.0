"""src/logic/calculators/ai_performance_calculator.py のテスト（オフライン）。

src.output.return_report の get_win_result/get_place_result/get_trio_box_result と
同じ既知データ（rank1=馬番5, rank2=馬番9, rank3=馬番7、配当は単勝160円・複勝110円・
三連複4220円）を使い、1レース判定・複数レース集計の両方を検証する。
"""

import shutil
from datetime import date

import pandas as pd
import pytest

from src.config import paths
from src.logic.calculators import ai_performance_calculator as ai
from src.managers import race_info_dataset_manager

SAMPLE_DATE_STR = "20241020"
SAMPLE_RACE_DAY = date(2024, 10, 20)
SAMPLE_PLACE = "04_nigata"
SAMPLE_RACE_ID = "202404040601"


@pytest.fixture
def new_roots(tmp_path, monkeypatch):
    """race_card/race_returnsの出力先をtmp_path配下に切り替える。

    race_card（出馬表+score/rank）は実データ（data/race_card/20241020/202404040601.csv）
    をコピーして使う。race_returns（確定配当）は既知の値を仕込む
    （rank1=馬番5, rank2=馬番9, rank3=馬番7）。
    """
    monkeypatch.setattr(paths, "RACE_CARD_DATA_PATH", str(tmp_path / "race_card"))
    monkeypatch.setattr(race_info_dataset_manager, "RACE_RETURNS_DATA_PATH", str(tmp_path / "race_returns"))

    race_card_dir = tmp_path / "race_card" / SAMPLE_DATE_STR
    race_card_dir.mkdir(parents=True)
    shutil.copy(
        f"data/race_card/{SAMPLE_DATE_STR}/{SAMPLE_RACE_ID}.csv",
        race_card_dir / f"{SAMPLE_RACE_ID}.csv",
    )

    returns_dir = tmp_path / "race_returns" / SAMPLE_PLACE / "2024"
    returns_dir.mkdir(parents=True)
    returns_df = pd.DataFrame(
        {
            "式別": ["単勝", "複勝", "複勝", "複勝", "三連複"],
            "馬番": ["5", "5", "9", "7", "5-9-7"],
            "配当": ["160", "110", "150", "180", "4220"],
            "人気": ["1", "1", "3", "4", "12"],
        },
        index=[SAMPLE_RACE_ID] * 5,
    )
    returns_df.index.name = ""
    returns_df.to_csv(returns_dir / f"{SAMPLE_RACE_ID}.csv")

    return tmp_path


def test_calc_race_hit_returns_matches_known_result(new_roots):
    result = ai.calc_race_hit_returns(SAMPLE_RACE_DAY, SAMPLE_RACE_ID)

    print(f"\n--- calc_race_hit_returns({SAMPLE_RACE_DAY}, {SAMPLE_RACE_ID}) ---")
    print(f"  結果: {result}")

    assert result == {
        "win": (1, 160.0),
        "place": (1, 110.0),
        "trio_box": (1, 422.0),
    }


def test_calc_race_hit_returns_returns_none_when_no_rank(new_roots, monkeypatch):
    monkeypatch.setattr(
        ai.race_card_dataset_manager, "get_race_cards", lambda race_day, race_id: pd.DataFrame({"馬番": [1, 2]})
    )

    assert ai.calc_race_hit_returns(SAMPLE_RACE_DAY, SAMPLE_RACE_ID) is None


def test_calc_race_hit_returns_returns_none_when_no_returns_data(new_roots, monkeypatch):
    monkeypatch.setattr(
        ai.race_info_dataset_manager, "get_race_return_csv_for_race", lambda race_id: pd.DataFrame()
    )

    assert ai.calc_race_hit_returns(SAMPLE_RACE_DAY, SAMPLE_RACE_ID) is None


def test_parse_race_id_returns_expected():
    assert ai.parse_race_id(SAMPLE_RACE_ID) == {
        "year": 2024,
        "place_id": 4,
        "times": 4,
        "days": 6,
        "race_num": 1,
    }


def test_get_current_meetings_returns_real_meetings_for_date():
    meetings = ai.get_current_meetings(date(2026, 6, 19))

    print(f"\n--- get_current_meetings(2026-06-19) ---")
    print(f"  結果: {meetings}")

    assert all(m["first_day"] <= date(2026, 6, 19) <= m["last_day"] for m in meetings)
    place_ids = [m["place_id"] for m in meetings]
    assert place_ids == sorted(place_ids)


def test_get_current_meetings_returns_empty_for_offseason_date():
    # JRAは年末年始は基本的に開催がない
    meetings = ai.get_current_meetings(date(2026, 1, 1))
    assert meetings == []


def test_get_last_week_main_races_filters_race_num_11():
    # 2026-06-07が含まれる週には202605030111(5R...11)/202609030111/202605030211/202609030211 がある
    main_races = ai.get_last_week_main_races(date(2026, 6, 7))

    print(f"\n--- get_last_week_main_races(2026-06-07) ---")
    print(f"  結果: {main_races}")

    assert len(main_races) == 4
    assert all(ai.parse_race_id(r["race_id"])["race_num"] == 11 for r in main_races)
    assert main_races == sorted(main_races, key=lambda r: r["race_day"])


def test_list_predicted_races_returns_real_dates(new_roots):
    # new_rootsで20241020分のみ用意しているため、その1日・1レースのみ列挙される
    pairs = ai.list_predicted_races()

    print(f"\n--- list_predicted_races() ---")
    print(f"  結果: {pairs}")

    assert pairs == [(SAMPLE_RACE_DAY, SAMPLE_RACE_ID)]
