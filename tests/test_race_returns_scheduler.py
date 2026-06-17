"""src/logic/scheduler/race_returns_scheduler.py のテスト（オフライン）。

netkeiba_scraper.scrape_race_returns_dataframe と
race_schedule_dataset_manager のrace_id取得系をモックし、
race_info_dataset_manager（保存・分割）への連携が正しく行われることを確認する。
"""

import os
from datetime import date

import pandas as pd
import pytest

from src.config.constants import PLACE_LIST
from src.datasets.race_info import model
from src.logic.scheduler import race_returns_scheduler
from src.managers import race_info_dataset_manager

SAMPLE_PLACE_ID = 2
SAMPLE_PLACE = PLACE_LIST[SAMPLE_PLACE_ID - 1]


@pytest.fixture
def race_returns_root(tmp_path, monkeypatch):
    """race_info_dataset_manager.RACE_RETURNS_DATA_PATHをtmp_path配下に向ける"""
    race_returns_dir = tmp_path / "race_returns"
    monkeypatch.setattr(race_info_dataset_manager, "RACE_RETURNS_DATA_PATH", str(race_returns_dir))
    return race_returns_dir


def _sample_race_returns_df(race_id):
    return pd.DataFrame(
        [
            ["単勝", "7", "140", "1"],
            ["複勝", "7", "110", "1"],
        ],
        columns=model.RACE_RETURNS_COLUMNS,
        index=[race_id, race_id],
    )


def test_update_race_returns_dataset_creates_new_file(race_returns_root, monkeypatch):
    race_id = "202405010101"
    monkeypatch.setattr(
        race_returns_scheduler.race_schedule_dataset_manager, "get_past_weekly_id",
        lambda place_id, day: [race_id],
    )
    monkeypatch.setattr(
        race_returns_scheduler.netkeiba_scraper, "scrape_race_returns_dataframe",
        lambda race_id_list: _sample_race_returns_df(race_id_list[0]),
    )

    day = date(2024, 1, 28)
    race_returns_scheduler.update_race_returns_dataset(SAMPLE_PLACE_ID, day)

    result = race_info_dataset_manager.get_race_returns_csv(SAMPLE_PLACE_ID, day.year)
    assert list(result.columns) == model.RACE_RETURNS_COLUMNS
    assert result.index.astype(str).tolist() == [race_id, race_id]


def test_update_race_returns_dataset_merges_with_existing(race_returns_root, monkeypatch):
    old_race_id = "202405010102"
    race_info_dataset_manager.save_race_returns_dataset(
        SAMPLE_PLACE_ID, 2024, _sample_race_returns_df(old_race_id)
    )

    # drop_duplicates(keep="first")で同一内容の行が消えないよう、配当内容を変えておく
    new_race_id = "202405010103"
    new_race_returns_df = pd.DataFrame(
        [["単勝", "3", "250", "2"]],
        columns=model.RACE_RETURNS_COLUMNS,
        index=[new_race_id],
    )
    monkeypatch.setattr(
        race_returns_scheduler.race_schedule_dataset_manager, "get_past_weekly_id",
        lambda place_id, day: [new_race_id],
    )
    monkeypatch.setattr(
        race_returns_scheduler.netkeiba_scraper, "scrape_race_returns_dataframe",
        lambda race_id_list: new_race_returns_df,
    )

    race_returns_scheduler.update_race_returns_dataset(SAMPLE_PLACE_ID, date(2024, 1, 29))

    result = race_info_dataset_manager.get_race_returns_csv(SAMPLE_PLACE_ID, 2024)
    assert set(result.index.astype(str)) == {old_race_id, new_race_id}


def test_update_race_returns_dataset_noop_when_scrape_empty(race_returns_root, monkeypatch):
    monkeypatch.setattr(
        race_returns_scheduler.race_schedule_dataset_manager, "get_past_weekly_id",
        lambda place_id, day: [],
    )
    monkeypatch.setattr(
        race_returns_scheduler.netkeiba_scraper, "scrape_race_returns_dataframe",
        lambda race_id_list: pd.DataFrame(columns=model.RACE_RETURNS_COLUMNS),
    )

    race_returns_scheduler.update_race_returns_dataset(SAMPLE_PLACE_ID, date(2024, 1, 28))

    result = race_info_dataset_manager.get_race_returns_csv(SAMPLE_PLACE_ID, 2024)
    assert result.empty


def test_make_yearly_race_returns_dataset_saves(race_returns_root, monkeypatch):
    race_id = "201902010101"
    monkeypatch.setattr(
        race_returns_scheduler.race_schedule_dataset_manager, "get_year_id_all",
        lambda place_id, year: [race_id],
    )
    monkeypatch.setattr(
        race_returns_scheduler.netkeiba_scraper, "scrape_race_returns_dataframe",
        lambda race_id_list: _sample_race_returns_df(race_id_list[0]),
    )

    race_returns_scheduler.make_yearly_race_returns_dataset(SAMPLE_PLACE_ID, 2019)

    result = race_info_dataset_manager.get_race_returns_csv(SAMPLE_PLACE_ID, 2019)
    assert result.index.astype(str).tolist() == [race_id, race_id]


def test_make_up_to_day_dataset_saves(race_returns_root, monkeypatch):
    race_id = "202405010101"
    monkeypatch.setattr(
        race_returns_scheduler.race_schedule_dataset_manager, "get_past_year_id",
        lambda place_id, day: [race_id],
    )
    monkeypatch.setattr(
        race_returns_scheduler.netkeiba_scraper, "scrape_race_returns_dataframe",
        lambda race_id_list: _sample_race_returns_df(race_id_list[0]),
    )

    race_returns_scheduler.make_up_to_day_dataset(SAMPLE_PLACE_ID, date(2024, 1, 28))

    result = race_info_dataset_manager.get_race_returns_csv(SAMPLE_PLACE_ID, 2024)
    assert result.index.astype(str).tolist() == [race_id, race_id]


def test_weekly_update_race_returns_updates_and_splits(race_returns_root, monkeypatch):
    race_id = "202405010101"
    monkeypatch.setattr(
        race_returns_scheduler.race_schedule_dataset_manager, "get_past_weekly_id",
        lambda place_id, day: [race_id],
    )
    monkeypatch.setattr(
        race_returns_scheduler.netkeiba_scraper, "scrape_race_returns_dataframe",
        lambda race_id_list: _sample_race_returns_df(race_id_list[0]),
    )

    day = date(2024, 1, 28)
    race_returns_scheduler.weekly_update_race_returns(day)

    result = race_info_dataset_manager.get_race_returns_csv(SAMPLE_PLACE_ID, day.year)
    split_path = os.path.join(str(race_returns_root), SAMPLE_PLACE, str(day.year), f"{race_id}.csv")

    print(f"\n--- weekly_update_race_returns({day}) ---")
    print(f"  集計CSVの行: {result.index.astype(str).tolist()}")
    print(f"  race_id別分割ファイル: {split_path} (存在: {os.path.isfile(split_path)})")

    assert result.index.astype(str).tolist() == [race_id, race_id]
    assert os.path.isfile(split_path)


def test_monthly_update_race_returns_calls_make_up_to_day_dataset_per_place(monkeypatch):
    calls = []
    monkeypatch.setattr(
        race_returns_scheduler, "make_up_to_day_dataset",
        lambda place_id, day: calls.append((place_id, day)),
    )

    day = date(2024, 1, 28)
    race_returns_scheduler.monthly_update_race_returns(day)

    assert calls == [(place_id, day) for place_id in range(1, len(PLACE_LIST) + 1)]


def test_make_all_race_returns_calls_make_yearly_dataset_per_year_and_place(monkeypatch):
    calls = []
    monkeypatch.setattr(
        race_returns_scheduler, "make_yearly_race_returns_dataset",
        lambda place_id, year: calls.append((place_id, year)),
    )

    race_returns_scheduler.make_all_race_returns(2020)

    expected = [
        (place_id, year)
        for year in range(2019, 2021)
        for place_id in range(1, len(PLACE_LIST) + 1)
    ]
    assert calls == expected
