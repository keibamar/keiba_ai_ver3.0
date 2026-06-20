"""src/managers/ai_performance_dataset_manager.py のテスト（オフライン）。

tests/test_ai_performance_calculator.py と同じ既知データ（rank1=馬番5, rank2=馬番9,
rank3=馬番7、配当は単勝160円・複勝110円・三連複4220円）を使い、データセットの
作成（update）・取得（get）・差分更新（既存行のスキップ）・分類列（race_type等、
実データの確定結果から取得）・高速集計（aggregate/filter_by_*/group_breakdown）を検証する。
"""

import shutil
from datetime import date

import pandas as pd
import pytest

from src.config import paths
from src.managers import ai_performance_dataset_manager as m
from src.managers import race_info_dataset_manager

SAMPLE_DATE_STR = "20241020"
SAMPLE_RACE_DAY = date(2024, 10, 20)
SAMPLE_PLACE = "04_nigata"
SAMPLE_RACE_ID = "202404040601"
# data/race_result/04_nigata/2024_race_results.csv の実データより
# （race_type=芝, course_len=2000, ground_state=稍重, class=未勝利）


@pytest.fixture
def new_roots(tmp_path, monkeypatch):
    """race_card/race_returns/ai_performanceの出力先をtmp_path配下に切り替える。

    race_card（出馬表+score/rank）は実データ（data/race_card/20241020/202404040601.csv）
    をコピーして使う。race_returns（確定配当）は既知の値を仕込む
    （rank1=馬番5, rank2=馬番9, rank3=馬番7）。race_resultは実データ
    （data/race_result/04_nigata/2024_race_results.csv）をそのまま参照する
    （race_type等の分類列の取得元）。
    """
    monkeypatch.setattr(paths, "RACE_CARD_DATA_PATH", str(tmp_path / "race_card"))
    monkeypatch.setattr(race_info_dataset_manager, "RACE_RETURNS_DATA_PATH", str(tmp_path / "race_returns"))
    monkeypatch.setattr(paths, "AI_PERFORMANCE_DATA_PATH", str(tmp_path / "ai_performance"))
    monkeypatch.setattr(m, "AI_PERFORMANCE_DATASET_PATH", str(tmp_path / "ai_performance" / "ai_performance.csv"))

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


def test_get_ai_performance_dataset_returns_empty_when_missing(new_roots):
    df = m.get_ai_performance_dataset()
    assert df.empty


def test_update_ai_performance_dataset_creates_new_rows_with_conditions(new_roots):
    added = m.update_ai_performance_dataset()

    print(f"\n--- update_ai_performance_dataset() (初回) ---")
    print(f"  追加件数: {added}")

    assert added == 1

    df = m.get_ai_performance_dataset()
    print(df.to_string())

    assert df.index.tolist() == [SAMPLE_RACE_ID]
    row = df.loc[SAMPLE_RACE_ID]
    assert row["year"] == "2024"
    assert row["place_id"] == "4"
    assert row["times"] == "4"
    assert row["race_type"] == "芝"
    assert row["course_len"] == "2000"
    assert row["ground_state"] == "稍重"
    assert row["class"] == "未勝利"
    assert row["win_hit"] == "1"
    assert row["win_return"] == "160.0"
    assert row["place_hit"] == "1"
    assert row["place_return"] == "110.0"
    assert row["trio_box_hit"] == "1"
    assert row["trio_box_return"] == "422.0"


def test_update_ai_performance_dataset_skips_already_recorded_races(new_roots, monkeypatch):
    m.update_ai_performance_dataset()

    calls = []
    original = m.ai_performance_calculator.calc_race_hit_returns
    monkeypatch.setattr(
        m.ai_performance_calculator, "calc_race_hit_returns",
        lambda race_day, race_id, box_num=5: (calls.append(race_id), original(race_day, race_id, box_num))[1],
    )

    added = m.update_ai_performance_dataset()

    print(f"\n--- update_ai_performance_dataset() (2回目) ---")
    print(f"  追加件数: {added}, calc_race_hit_returns呼び出し: {calls}")

    assert added == 0
    assert calls == []


def test_update_ai_performance_dataset_skips_races_without_data(new_roots, monkeypatch):
    monkeypatch.setattr(
        m.ai_performance_calculator, "list_predicted_races",
        lambda: [(SAMPLE_RACE_DAY, SAMPLE_RACE_ID), (SAMPLE_RACE_DAY, "999999999999")],
    )

    added = m.update_ai_performance_dataset()

    assert added == 1
    df = m.get_ai_performance_dataset()
    assert df.index.tolist() == [SAMPLE_RACE_ID]


def test_update_ai_performance_dataset_leaves_conditions_blank_when_no_race_result(new_roots, monkeypatch):
    # race_resultが存在しないrace_idでも、win/place/trio_box自体は集計される
    monkeypatch.setattr(
        m.race_result_dataset_manager, "get_race_results_csv", lambda place_id, year: pd.DataFrame()
    )

    added = m.update_ai_performance_dataset()

    assert added == 1
    row = m.get_ai_performance_dataset().loc[SAMPLE_RACE_ID]
    assert row["race_type"] == ""
    assert row["win_hit"] == "1"


# --- 高速集計・フィルタ（pandasのみ、既知のDataFrameで検証） -------------------------


SAMPLE_DATASET = pd.DataFrame(
    {
        "race_day": ["2024-10-20", "2024-10-21", "2025-05-01"],
        "year": ["2024", "2024", "2025"],
        "place_id": ["4", "4", "5"],
        "times": ["4", "4", "1"],
        "race_type": ["芝", "ダート", "芝"],
        "course_len": ["2000", "1200", "2000"],
        "ground_state": ["良", "良", "稍重"],
        "class": ["未勝利", "未勝利", "1勝クラス"],
        "win_hit": ["1", "0", "1"],
        "win_return": ["160.0", "0.0", "200.0"],
        "place_hit": ["1", "0", "1"],
        "place_return": ["110.0", "0.0", "150.0"],
        "trio_box_hit": ["1", "0", "0"],
        "trio_box_return": ["422.0", "0.0", "0.0"],
    },
    index=["A", "B", "C"],
)


def test_aggregate_computes_hit_rate_and_return_rate():
    result = m.aggregate(SAMPLE_DATASET)

    print(f"\n--- aggregate(3レース) ---")
    print(result)

    assert result["win"]["n"] == 3
    assert result["win"]["hit_rate"] == pytest.approx(200 / 3)
    assert result["win"]["return_rate"] == pytest.approx(120.0)


def test_aggregate_returns_zeros_for_empty_dataframe():
    result = m.aggregate(SAMPLE_DATASET.iloc[0:0])
    assert result["win"] == {"hit_rate": 0.0, "return_rate": 0.0, "n": 0}


def test_filter_by_place_and_year():
    assert m.filter_by_place(SAMPLE_DATASET, 4).index.tolist() == ["A", "B"]
    assert m.filter_by_year(SAMPLE_DATASET, 2025).index.tolist() == ["C"]


def test_filter_by_course():
    result = m.filter_by_course(SAMPLE_DATASET, 4, "芝", "2000")
    assert result.index.tolist() == ["A"]


def test_group_breakdown_by_class():
    breakdown = m.group_breakdown(SAMPLE_DATASET, "class")

    print(f"\n--- group_breakdown(class) ---")
    print(breakdown)

    values = {b["value"] for b in breakdown}
    assert values == {"未勝利", "1勝クラス"}
    maiden = next(b for b in breakdown if b["value"] == "未勝利")
    assert maiden["performance"]["win"]["n"] == 2


def test_group_breakdown_excludes_blank_values():
    df = SAMPLE_DATASET.copy()
    df.loc["A", "class"] = ""

    breakdown = m.group_breakdown(df, "class")
    values = {b["value"] for b in breakdown}
    assert "" not in values
    assert values == {"未勝利", "1勝クラス"}
