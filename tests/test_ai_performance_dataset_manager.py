"""src/managers/ai_performance_dataset_manager.py のテスト（オフライン）。

tests/test_ai_performance_calculator.py と同じ既知データ（rank1=馬番5, rank2=馬番9,
rank3=馬番7、配当は単勝160円・複勝110円・三連複4220円）を使い、データセットの
作成（update）・取得（get）・差分更新（既存行のスキップ）を検証する。
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


@pytest.fixture
def new_roots(tmp_path, monkeypatch):
    """race_card/race_returns/ai_performanceの出力先をtmp_path配下に切り替える。

    race_card（出馬表+score/rank）は実データ（data/race_card/20241020/202404040601.csv）
    をコピーして使う。race_returns（確定配当）は既知の値を仕込む
    （rank1=馬番5, rank2=馬番9, rank3=馬番7）。
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


def test_update_ai_performance_dataset_creates_new_rows(new_roots):
    added = m.update_ai_performance_dataset()

    print(f"\n--- update_ai_performance_dataset() (初回) ---")
    print(f"  追加件数: {added}")

    assert added == 1

    df = m.get_ai_performance_dataset()
    print(df.to_string())

    assert df.index.tolist() == [SAMPLE_RACE_ID]
    row = df.loc[SAMPLE_RACE_ID]
    assert row["place_id"] == "4"
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
