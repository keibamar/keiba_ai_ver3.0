"""src/logic/scheduler/race_result_scheduler.py の update_daily_race_results のテスト（オフライン）。

race_schedule_dataset_manager.get_daily_id と netkeiba_scraper.scrape_day_race_result を
モックし、race_result_dataset_manager.save_race_result_for_race_id への連携が
正しく行われることを確認する。
"""

from datetime import date

import pandas as pd

from src.config import paths
from src.config.constants import PLACE_LIST
from src.logic.scheduler import race_result_scheduler


def _sample_race_result_df(race_id):
    return pd.DataFrame({"着順": ["1", "2"], "馬番": ["7", "3"]}, index=[race_id, race_id])


def test_update_daily_race_results_saves_per_race_csv(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "RACE_RESULT_DATA_PATH", str(tmp_path / "race_result"))

    race_id = "202405010101"
    monkeypatch.setattr(
        race_result_scheduler.race_schedule_dataset_manager, "get_daily_id",
        lambda race_day: [race_id],
    )
    monkeypatch.setattr(
        race_result_scheduler.netkeiba_scraper, "scrape_day_race_result",
        lambda rid: _sample_race_result_df(rid),
    )

    race_result_scheduler.update_daily_race_results(date(2024, 1, 28))

    out_path = tmp_path / "race_result" / PLACE_LIST[4] / "2024" / f"{race_id}.csv"

    print(f"\n--- update_daily_race_results(2024-01-28) ---")
    print(f"  保存先: {out_path}")
    print(f"  ファイル作成: {out_path.is_file()}")

    assert out_path.is_file()
    result = pd.read_csv(out_path, dtype=str, index_col=0)
    result.index = result.index.astype(str)

    print(result.to_string())

    assert result.equals(_sample_race_result_df(race_id))


def test_update_daily_race_results_skips_empty_result(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(paths, "RACE_RESULT_DATA_PATH", str(tmp_path / "race_result"))

    race_id = "202405010101"
    monkeypatch.setattr(
        race_result_scheduler.race_schedule_dataset_manager, "get_daily_id",
        lambda race_day: [race_id],
    )
    monkeypatch.setattr(
        race_result_scheduler.netkeiba_scraper, "scrape_day_race_result",
        lambda rid: pd.DataFrame(),
    )

    race_result_scheduler.update_daily_race_results(date(2024, 1, 28))

    assert not (tmp_path / "race_result").exists()
    assert f"not_race_result: {race_id}" in capsys.readouterr().out


def test_update_daily_race_results_noop_when_no_races(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "RACE_RESULT_DATA_PATH", str(tmp_path / "race_result"))

    monkeypatch.setattr(
        race_result_scheduler.race_schedule_dataset_manager, "get_daily_id",
        lambda race_day: [],
    )

    race_result_scheduler.update_daily_race_results(date(2024, 1, 28))

    assert not (tmp_path / "race_result").exists()
