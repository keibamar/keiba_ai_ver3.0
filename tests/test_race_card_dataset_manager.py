from datetime import date

import pandas as pd
import pytest

from src.config import paths
from src.managers import race_card_dataset_manager

SAMPLE_RACE_DAY = date(2024, 10, 20)
SAMPLE_RACE_ID = 202404040601


@pytest.fixture
def new_roots(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "RACE_CARD_DATA_PATH", str(tmp_path / "race_card"))
    monkeypatch.setattr(paths, "RACE_INFO_DATA_PATH", str(tmp_path / "race_info"))
    monkeypatch.setattr(paths, "RACE_TIME_ID_LIST_PATH", str(tmp_path / "race_time_id_list"))
    return tmp_path


def test_save_and_get_race_cards_roundtrip(new_roots):
    df = pd.DataFrame({"馬名": ["A", "B"], "score": [1.234, -0.5], "rank": [1, 2]})

    race_card_dataset_manager.save_race_cards(df, SAMPLE_RACE_DAY, SAMPLE_RACE_ID)
    result = race_card_dataset_manager.get_race_cards(SAMPLE_RACE_DAY, SAMPLE_RACE_ID)

    assert result.reset_index(drop=True)[["馬名", "score", "rank"]].equals(df)


def test_get_race_cards_empty_when_missing(new_roots):
    result = race_card_dataset_manager.get_race_cards(SAMPLE_RACE_DAY, SAMPLE_RACE_ID)
    assert result.empty


def test_save_and_get_race_info_csv_roundtrip(new_roots):
    df = pd.DataFrame({
        "race_type": ["芝"],
        "course_len": [1800],
        "weather": ["晴"],
        "ground_state": ["良"],
        "class": ["3勝クラス"],
    })

    race_card_dataset_manager.save_race_info_df(df, SAMPLE_RACE_DAY, SAMPLE_RACE_ID)
    result = race_card_dataset_manager.get_race_info_csv(SAMPLE_RACE_ID)

    assert result.iloc[0]["race_type"] == "芝"
    assert result.iloc[0]["course_len"] == "1800"
    assert result.iloc[0]["class"] == "3勝クラス"


def test_get_race_info_csv_empty_when_missing(new_roots):
    result = race_card_dataset_manager.get_race_info_csv(SAMPLE_RACE_ID)
    assert result.empty


def test_save_and_get_time_id_list_roundtrip(new_roots):
    time_id_list = [
        ["1010", "202404040601", "２歳未勝利", None],
        ["1530", "202404040612", "３歳上１勝クラス", None],
    ]

    race_card_dataset_manager.save_time_id_list(SAMPLE_RACE_DAY, time_id_list)
    result = race_card_dataset_manager.get_time_id_list(SAMPLE_RACE_DAY)

    assert result == [[t[0], t[1]] for t in time_id_list]


def test_save_and_get_time_id_list_roundtrip_includes_grade(new_roots):
    time_id_list = [
        ["1545", "202602010611", "函館記念", "G3"],
        ["1010", "202404040601", "２歳未勝利", None],
    ]

    race_card_dataset_manager.save_time_id_list(SAMPLE_RACE_DAY, time_id_list)
    df = race_card_dataset_manager.get_race_time_id_list_df(SAMPLE_RACE_DAY)

    assert df.set_index("race_id").loc["202602010611", "grade"] == "G3"
    assert pd.isna(df.set_index("race_id").loc["202404040601", "grade"])


def test_get_time_id_list_empty_when_missing(new_roots):
    assert race_card_dataset_manager.get_time_id_list(SAMPLE_RACE_DAY) == []


def test_save_time_id_list_skips_empty_list(new_roots):
    race_card_dataset_manager.save_time_id_list(SAMPLE_RACE_DAY, [])
    df = race_card_dataset_manager.get_race_time_id_list_df(SAMPLE_RACE_DAY)
    assert df.empty


def test_get_race_time_id_list_df_matches_existing_sample_file():
    """既存の実データ(data/race_schedule/race_time_id_list/20241020.csv)を
    新 race_card_dataset_manager が正しく読み込めることを確認する。"""
    df = race_card_dataset_manager.get_race_time_id_list_df(SAMPLE_RACE_DAY)

    assert not df.empty
    assert {"race_time", "race_id", "race_name"}.issubset(df.columns)

    time_id_list = race_card_dataset_manager.get_time_id_list(SAMPLE_RACE_DAY)
    assert all(len(item) == 2 for item in time_id_list)
    assert str(SAMPLE_RACE_ID) in [item[1] for item in time_id_list]
