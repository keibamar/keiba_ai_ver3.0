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


def test_aggregate_ai_performance_matches_known_result(new_roots):
    result = ai.aggregate_ai_performance([(SAMPLE_RACE_DAY, SAMPLE_RACE_ID)])

    print(f"\n--- aggregate_ai_performance([1レース]) ---")
    print(f"  結果: {result}")

    assert result["win"] == {"hit_rate": 100.0, "return_rate": 160.0, "n": 1}
    assert result["place"] == {"hit_rate": 100.0, "return_rate": 110.0, "n": 1}
    assert result["trio_box"] == {"hit_rate": 100.0, "return_rate": 422.0, "n": 1}


def test_aggregate_ai_performance_averages_over_multiple_races(new_roots):
    # 同じレースを2件まとめて渡しても、件数(n)が増え、的中率・回収率は平均として変わらないことを確認
    pairs = [(SAMPLE_RACE_DAY, SAMPLE_RACE_ID), (SAMPLE_RACE_DAY, SAMPLE_RACE_ID)]
    result = ai.aggregate_ai_performance(pairs)

    assert result["win"]["n"] == 2
    assert result["win"]["hit_rate"] == 100.0
    assert result["win"]["return_rate"] == 160.0


def test_aggregate_ai_performance_skips_races_without_data(new_roots):
    # 予想・配当データが存在しないrace_idは集計対象外（n に含まれない）
    pairs = [(SAMPLE_RACE_DAY, SAMPLE_RACE_ID), (SAMPLE_RACE_DAY, "999999999999")]
    result = ai.aggregate_ai_performance(pairs)

    assert result["win"]["n"] == 1


def test_aggregate_ai_performance_returns_zeros_when_no_valid_races():
    result = ai.aggregate_ai_performance([])

    assert result == {
        "win": {"hit_rate": 0.0, "return_rate": 0.0, "n": 0},
        "place": {"hit_rate": 0.0, "return_rate": 0.0, "n": 0},
        "trio_box": {"hit_rate": 0.0, "return_rate": 0.0, "n": 0},
    }


def test_parse_race_id_returns_expected():
    assert ai.parse_race_id(SAMPLE_RACE_ID) == {
        "year": 2024,
        "place_id": 4,
        "times": 4,
        "days": 6,
        "race_num": 1,
    }


def test_filter_by_year_and_meeting():
    pairs = [(SAMPLE_RACE_DAY, SAMPLE_RACE_ID), (date(2025, 1, 1), "202506010101")]

    assert ai.filter_by_year(pairs, 2024) == [(SAMPLE_RACE_DAY, SAMPLE_RACE_ID)]
    assert ai.filter_by_year(pairs, 2026) == []
    assert ai.filter_by_meeting(pairs, 2024, 4, 4) == [(SAMPLE_RACE_DAY, SAMPLE_RACE_ID)]
    assert ai.filter_by_meeting(pairs, 2024, 5, 4) == []


def test_filter_by_course_uses_saved_race_info(new_roots, monkeypatch):
    monkeypatch.setattr(
        ai.race_card_dataset_manager, "get_race_info_csv",
        lambda race_id: pd.DataFrame([{"race_type": "ダート", "course_len": "1200"}]),
    )
    pairs = [(SAMPLE_RACE_DAY, SAMPLE_RACE_ID)]

    assert ai.filter_by_course(pairs, 4, "ダート", "1200") == pairs
    assert ai.filter_by_course(pairs, 4, "芝", "1200") == []
    assert ai.filter_by_course(pairs, 5, "ダート", "1200") == []


def test_get_race_conditions_returns_dict_per_race_id(new_roots, monkeypatch):
    monkeypatch.setattr(
        ai.race_card_dataset_manager, "get_race_info_csv",
        lambda race_id: pd.DataFrame([{"race_type": "ダート", "course_len": "1200"}]),
    )
    pairs = [(SAMPLE_RACE_DAY, SAMPLE_RACE_ID)]

    conditions = ai.get_race_conditions(pairs)

    assert conditions == {SAMPLE_RACE_ID: ("ダート", "1200")}


def test_get_race_conditions_returns_none_when_missing(new_roots, monkeypatch):
    monkeypatch.setattr(ai.race_card_dataset_manager, "get_race_info_csv", lambda race_id: pd.DataFrame())
    pairs = [(SAMPLE_RACE_DAY, SAMPLE_RACE_ID)]

    assert ai.get_race_conditions(pairs) == {SAMPLE_RACE_ID: None}


def test_filter_by_course_uses_precomputed_race_conditions():
    pairs = [(SAMPLE_RACE_DAY, SAMPLE_RACE_ID), (SAMPLE_RACE_DAY, "202405010101")]
    race_conditions = {SAMPLE_RACE_ID: ("ダート", "1200"), "202405010101": ("芝", "1400")}

    # race_card_dataset_manager.get_race_info_csvをmonkeypatchしていないので、
    # race_conditionsを渡さないとエラー（実ファイル参照）になるはずの状況で、
    # 渡した場合はそれだけで判定できることを確認する
    result = ai.filter_by_course(pairs, 4, "ダート", "1200", race_conditions=race_conditions)

    assert result == [(SAMPLE_RACE_DAY, SAMPLE_RACE_ID)]


def test_get_predicted_years_returns_sorted_years(monkeypatch):
    monkeypatch.setattr(
        ai, "list_predicted_races",
        lambda: [(date(2025, 1, 1), "A"), (date(2024, 6, 1), "B"), (date(2025, 6, 1), "C")],
    )

    assert ai.get_predicted_years() == [2024, 2025]


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


def test_weekly_trend_groups_pairs_by_week(monkeypatch):
    monkeypatch.setattr(
        ai, "list_predicted_races",
        lambda: [(date(2026, 6, 1), "A"), (date(2026, 6, 7), "B"), (date(2026, 6, 8), "C")],
    )
    # aggregate_ai_performanceをそのまま使うと的中判定で実データI/Oが走るため、
    # 集計対象のpairsをそのまま返すダミーに差し替えてグルーピング結果のみ検証する
    monkeypatch.setattr(ai, "aggregate_ai_performance", lambda pairs: {"pairs": pairs})

    trend = ai.weekly_trend(num_weeks=2, end_day=date(2026, 6, 14))

    print(f"\n--- weekly_trend(num_weeks=2, end_day=2026-06-14) ---")
    for week in trend:
        print(f"  {week['week_start']}~{week['week_end']}: {week['performance']['pairs']}")

    assert [w["week_start"] for w in trend] == [date(2026, 6, 1), date(2026, 6, 8)]
    assert [w["week_end"] for w in trend] == [date(2026, 6, 7), date(2026, 6, 14)]
    # week_start=6/1の週にはA, Bが含まれる（B=6/7は週末日に含まれる）
    assert trend[0]["performance"]["pairs"] == [(date(2026, 6, 1), "A"), (date(2026, 6, 7), "B")]
    assert trend[1]["performance"]["pairs"] == [(date(2026, 6, 8), "C")]


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
