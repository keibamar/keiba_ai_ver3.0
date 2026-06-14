import pandas as pd
import pytest

from src.config import paths
from src.config.constants import PLACE_LIST
from src.logic.html_generator import horse_report_generator as h
from src.managers import race_info_dataset_manager

PLACE_ID = 2
SAMPLE_HORSE_ID = "2007100107"
SAMPLE_HORSE_NAME = "マックスドリーム"
SAMPLE_RACE_ID = "202402040101"


@pytest.fixture
def tmp_horse_id_map(tmp_path, monkeypatch):
    """馬名↔horse_id対応表とper-raceのrace_info CSVをtmp_path配下に用意する"""
    race_info_dir = tmp_path / "race_info"
    race_info_dir.mkdir(parents=True)

    horse_id_map_path = race_info_dir / "horse_id_map.csv"
    pd.DataFrame({"馬名": [SAMPLE_HORSE_NAME], "horse_id": [SAMPLE_HORSE_ID]}).to_csv(horse_id_map_path, index=False)
    monkeypatch.setattr(race_info_dataset_manager, "HORSE_ID_MAP_PATH", str(horse_id_map_path))

    place = PLACE_LIST[PLACE_ID - 1]
    year = SAMPLE_RACE_ID[:4]
    out_dir = race_info_dir / place / year
    out_dir.mkdir(parents=True)
    pd.DataFrame({
        "race_type": ["ダート"],
        "course_len": [1700],
        "weather": ["晴"],
        "ground_state": ["良"],
        "class": ["未勝利"],
    }).to_csv(out_dir / f"{SAMPLE_RACE_ID}.csv")
    monkeypatch.setattr(paths, "RACE_INFO_DATA_PATH", str(race_info_dir))

    return tmp_path


def test_load_horse_peds_dict_keys_match_old_format():
    """load_horse_peds の辞書変換アダプタが旧形式と同じ peds_0..peds_61 キーを持つことを確認"""
    peds = h.load_horse_peds(SAMPLE_HORSE_ID)

    assert peds
    assert set(peds.keys()) == {f"peds_{i}" for i in range(62)}
    assert all(isinstance(v, str) for v in peds.values())


def test_load_horse_peds_empty_for_unknown_horse():
    assert h.load_horse_peds("0000000000") == {}


def test_load_horse_id_map_and_get_horse_id_by_name(tmp_horse_id_map):
    map_df = h.load_horse_id_map()

    assert list(map_df.columns) == ["horse_id", "馬名"]
    assert map_df.iloc[0]["horse_id"] == SAMPLE_HORSE_ID
    assert map_df.iloc[0]["馬名"] == SAMPLE_HORSE_NAME
    assert h.get_horse_id_by_name(SAMPLE_HORSE_NAME, map_df) == SAMPLE_HORSE_ID
    assert h.get_horse_id_by_name("存在しない馬", map_df) is None


def test_build_horse_report_unknown_horse_returns_error(tmp_horse_id_map):
    report = h.build_horse_report("存在しない馬", PLACE_ID, SAMPLE_RACE_ID, "20240101")
    assert report == {"error": "horse_id not found for 存在しない馬"}


def test_build_horse_report_structure(tmp_horse_id_map):
    report = h.build_horse_report(SAMPLE_HORSE_NAME, PLACE_ID, SAMPLE_RACE_ID, "20240101")

    assert report is not None
    assert "error" not in report
    assert report["horse_id"] == SAMPLE_HORSE_ID
    assert report["place_id"] == PLACE_ID
    assert report["race_type"] == "ダート"
    assert report["course_len"] == 1700
    assert report["ground_state"] == "良"

    peds_results = report["peds_results"]
    assert isinstance(peds_results, pd.DataFrame)
    assert not peds_results.empty
    assert list(peds_results.columns) == ["クラス", "血統", "1着", "2着", "3着", "着外", "勝率", "複勝率"]

    assert isinstance(report["recent5"], list)
    assert isinstance(report["surface_summary"], dict)


def test_horse_report_to_html_structure(tmp_horse_id_map):
    report = h.build_horse_report(SAMPLE_HORSE_NAME, PLACE_ID, SAMPLE_RACE_ID, "20240101")
    html_str = h.horse_report_to_html(report)

    assert "<div class='horse-report'" in html_str
    assert "血統 (父)" in html_str
    assert "近5走成績" in html_str
    assert "芝/ダートサマリ" in html_str
    assert "<table" in html_str


def test_horse_report_to_html_error_report():
    html_str = h.horse_report_to_html({"error": "horse_id not found for 存在しない馬"})
    assert html_str == "<div class='horse-report error'>horse_id not found for 存在しない馬</div>"
