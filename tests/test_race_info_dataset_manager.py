"""src/datasets/race_info, src/managers/race_info_dataset_manager.py の出力が
旧 src/legacy_datasets/analysis_race_info.py と一致することを確認するテスト（オフライン）。

旧実装はパスとして name_header.DATA_PATH（ver2.0のdata/）を参照するため、
旧実装の比較対象テストでは tmp_path 配下に "RaceResults" フォルダを作って
name_header.DATA_PATH をそこに向ける。新実装は src.managers.race_result_dataset_manager
経由で src.config.paths.RACE_RESULT_DATA_PATH（実データ、読み取り専用）を参照する。

horse_id_map.csv / AveragePops / AverageWeights / AverageFrames に相当する新しい出力先
（data/race_info/horse_id_map.csv, data/race_info/average_pops 等）の比較テストでは、
旧実装・新実装それぞれ別のtmp_path配下に向けて出力内容（DataFrameの内容）を比較する。

旧実装の update_average_pops / update_average_frame_and_horse は、top3用・全期間用の
保存判定に誤って result_top.empty を使っていたが、新実装ではそれぞれの結果の空判定に
修正している（参照: src/managers/race_info_dataset_manager.py のモジュールdocstring）。
"""

import os
import shutil

import pandas as pd
import pytest

from src.config import paths
from src.datasets.race_info import transform
from src.legacy_datasets import analysis_race_info as old_analysis
from src.managers import race_info_dataset_manager as new_race_info

SAMPLE_PLACE_ID = 2
SAMPLE_PLACE = "02_hakodate"


# --- 純粋関数の比較（旧 analysis_race_info.py vs 新 src/datasets/race_info/transform.py） ---


def test_make_empty_record_matches_old():
    old_result = old_analysis.make_empty_record("芝", "1800", "良", "未勝利")
    new_result = transform.make_empty_record("芝", "1800", "良", "未勝利")

    assert old_result == new_result


# --- analyze_* 系（実データ: 02_hakodate 2019〜2021） ----------------------------


@pytest.fixture(scope="module")
def hakodate_root(tmp_path_factory):
    """data/race_result/02_hakodate の2019〜2021年分をRaceResults/配下にコピーしたtmp領域"""
    root = tmp_path_factory.mktemp("hakodate_root")
    dst_dir = root / "RaceResults" / SAMPLE_PLACE
    dst_dir.mkdir(parents=True)
    for year in (2019, 2020, 2021):
        src = os.path.join(paths.RACE_RESULT_DATA_PATH, SAMPLE_PLACE, f"{year}_race_results.csv")
        shutil.copy(src, dst_dir / f"{year}_race_results.csv")
    return root


@pytest.fixture
def old_data_root(hakodate_root, monkeypatch):
    """旧実装のname_header.DATA_PATHをhakodate_root配下に向ける"""
    monkeypatch.setattr(old_analysis.name_header, "DATA_PATH", str(hakodate_root) + "/")
    return hakodate_root


def test_analyze_winner_weights_matches_old(old_data_root):
    old_result = old_analysis.analyze_winner_weights(SAMPLE_PLACE_ID, 2019)
    new_result = new_race_info.analyze_winner_weights(SAMPLE_PLACE_ID, 2019)

    assert not old_result.empty
    assert old_result.equals(new_result)


def test_analyze_winner_weights_multi_years_matches_old(old_data_root):
    old_result = old_analysis.analyze_winner_weights_multi_years(SAMPLE_PLACE_ID, start_year=2019, current_year=2021)
    new_result = new_race_info.analyze_winner_weights_multi_years(SAMPLE_PLACE_ID, start_year=2019, current_year=2021)

    assert not old_result.empty
    assert old_result.equals(new_result)


@pytest.mark.parametrize("top3", [False, True])
def test_analyze_average_pops_matches_old(old_data_root, top3):
    old_result = old_analysis.analyze_average_pops(SAMPLE_PLACE_ID, 2019, top3=top3)
    new_result = new_race_info.analyze_average_pops(SAMPLE_PLACE_ID, 2019, top3=top3)

    assert not old_result.empty
    assert old_result.equals(new_result)


@pytest.mark.parametrize("top3", [False, True])
def test_analyze_average_pop_multi_years_matches_old(old_data_root, top3):
    old_result = old_analysis.analyze_average_pop_multi_years(
        SAMPLE_PLACE_ID, start_year=2019, current_year=2021, top3=top3
    )
    new_result = new_race_info.analyze_average_pop_multi_years(
        SAMPLE_PLACE_ID, start_year=2019, current_year=2021, top3=top3
    )

    assert not old_result.empty
    assert old_result.equals(new_result)


def test_analyze_average_frame_and_horse_matches_old(old_data_root):
    old_result = old_analysis.analyze_average_frame_and_horse(SAMPLE_PLACE_ID, 2019)
    new_result = new_race_info.analyze_average_frame_and_horse(SAMPLE_PLACE_ID, 2019)

    assert not old_result.empty
    assert old_result.equals(new_result)


def test_analyze_frame_and_horse_multi_years_matches_old(old_data_root):
    old_result = old_analysis.analyze_frame_and_horse_multi_years(SAMPLE_PLACE_ID, start_year=2019, current_year=2021)
    new_result = new_race_info.analyze_frame_and_horse_multi_years(SAMPLE_PLACE_ID, start_year=2019, current_year=2021)

    assert not old_result.empty
    assert old_result.equals(new_result)


def test_analyze_average_frame_and_horse_top3_matches_old(old_data_root):
    old_result = old_analysis.analyze_average_frame_and_horse_top3(SAMPLE_PLACE_ID, 2019)
    new_result = new_race_info.analyze_average_frame_and_horse_top3(SAMPLE_PLACE_ID, 2019)

    assert not old_result.empty
    assert old_result.equals(new_result)


def test_analyze_frame_and_horse_top3_multi_years_matches_old(old_data_root):
    old_result = old_analysis.analyze_frame_and_horse_top3_multi_years(
        SAMPLE_PLACE_ID, start_year=2019, current_year=2021
    )
    new_result = new_race_info.analyze_frame_and_horse_top3_multi_years(
        SAMPLE_PLACE_ID, start_year=2019, current_year=2021
    )

    assert not old_result.empty
    assert old_result.equals(new_result)


# --- horse_id_map / update_* 系（書き込み比較） -------------------------------------


@pytest.fixture
def old_and_new_roots(tmp_path, monkeypatch):
    """旧実装(name_header.DATA_PATH)と新実装(race_info_dataset_managerの各パス定数)を
    それぞれ別のtmp_path配下に向ける。race_resultsはRaceResults/02_hakodateに2019〜2021年分をコピーする。
    """
    old_root = tmp_path / "old"
    new_root = tmp_path / "new"

    old_results_dir = old_root / "RaceResults" / SAMPLE_PLACE
    old_results_dir.mkdir(parents=True)
    for year in (2019, 2020, 2021):
        src = os.path.join(paths.RACE_RESULT_DATA_PATH, SAMPLE_PLACE, f"{year}_race_results.csv")
        shutil.copy(src, old_results_dir / f"{year}_race_results.csv")

    for d in ("AveragePops", "AverageWeights", "AverageFrames"):
        (old_root / d / SAMPLE_PLACE).mkdir(parents=True)

    new_race_info_dir = new_root / "race_info"
    new_race_info_dir.mkdir(parents=True)

    monkeypatch.setattr(old_analysis.name_header, "DATA_PATH", str(old_root) + "/")
    monkeypatch.setattr(paths, "RACE_INFO_DATA_PATH", str(new_race_info_dir))
    monkeypatch.setattr(new_race_info, "HORSE_ID_MAP_PATH", str(new_race_info_dir / "horse_id_map.csv"))
    monkeypatch.setattr(new_race_info, "AVERAGE_POPS_DATA_PATH", str(new_race_info_dir / "average_pops"))
    monkeypatch.setattr(new_race_info, "AVERAGE_WEIGHTS_DATA_PATH", str(new_race_info_dir / "average_weights"))
    monkeypatch.setattr(new_race_info, "AVERAGE_FRAMES_DATA_PATH", str(new_race_info_dir / "average_frames"))

    return old_root, new_root


def test_update_horse_name_id_map_from_results_matches_old(old_and_new_roots):
    old_root, new_root = old_and_new_roots

    old_analysis.update_horse_name_id_map_from_results(SAMPLE_PLACE_ID, 2019)
    new_race_info.update_horse_name_id_map_from_results(SAMPLE_PLACE_ID, 2019)

    old_csv = old_root / "horse_id_map.csv"
    new_csv = new_root / "race_info" / "horse_id_map.csv"

    assert old_csv.is_file() and new_csv.is_file()
    old_df = pd.read_csv(old_csv, dtype=str)
    new_df = pd.read_csv(new_csv, dtype=str)
    assert not old_df.empty
    assert old_df.equals(new_df)


def test_get_horse_id_list_matches_old(old_and_new_roots):
    old_root, new_root = old_and_new_roots

    df = pd.DataFrame({"馬名": ["馬A", "馬B"], "horse_id": ["1111111111", " 2222222222 "]})
    df.to_csv(old_root / "horse_id_map.csv", index=False, encoding="utf-8-sig")
    df.to_csv(new_root / "race_info" / "horse_id_map.csv", index=False, encoding="utf-8-sig")

    old_result = old_analysis.get_horse_id_list()
    new_result = new_race_info.get_horse_id_list()

    assert old_result == new_result == ["1111111111", "2222222222"]


def test_add_horse_name_id_map_matches_old(old_and_new_roots):
    old_root, new_root = old_and_new_roots

    df = pd.DataFrame({"馬名": ["既存馬"], "horse_id": ["0000000000"]})
    df.to_csv(old_root / "horse_id_map.csv", index=False, encoding="utf-8-sig")
    df.to_csv(new_root / "race_info" / "horse_id_map.csv", index=False, encoding="utf-8-sig")

    old_analysis.add_horse_name_id_map("1234567890", "新馬A")
    new_race_info.add_horse_name_id_map("1234567890", "新馬A")

    old_df = pd.read_csv(old_root / "horse_id_map.csv", dtype=str)
    new_df = pd.read_csv(new_root / "race_info" / "horse_id_map.csv", dtype=str)

    assert len(old_df) == 2
    assert old_df.equals(new_df)


def test_add_horse_name_id_map_skips_duplicate_id(old_and_new_roots):
    old_root, new_root = old_and_new_roots

    df = pd.DataFrame({"馬名": ["既存馬"], "horse_id": ["0000000000"]})
    df.to_csv(old_root / "horse_id_map.csv", index=False, encoding="utf-8-sig")
    df.to_csv(new_root / "race_info" / "horse_id_map.csv", index=False, encoding="utf-8-sig")

    # horse_idが既存と重複（馬名は別）-> 追加されない
    old_analysis.add_horse_name_id_map("0000000000", "別名の馬")
    new_race_info.add_horse_name_id_map("0000000000", "別名の馬")

    old_df = pd.read_csv(old_root / "horse_id_map.csv", dtype=str)
    new_df = pd.read_csv(new_root / "race_info" / "horse_id_map.csv", dtype=str)

    assert len(old_df) == 1
    assert old_df.equals(new_df)


def test_update_horse_name_id_map_matches_old(old_and_new_roots):
    old_root, new_root = old_and_new_roots

    df = pd.DataFrame({"馬名": ["既存馬"], "horse_id": ["0000000000"]})
    df.to_csv(old_root / "horse_id_map.csv", index=False, encoding="utf-8-sig")
    df.to_csv(new_root / "race_info" / "horse_id_map.csv", index=False, encoding="utf-8-sig")

    race_card_df = pd.DataFrame({
        "馬名": ["新馬B", "新馬C", ""],
        "horse_id": ["1111111111", "2222222222", "nan"],
    })

    old_analysis.update_horse_name_id_map(race_card_df.copy())
    new_race_info.update_horse_name_id_map(race_card_df.copy())

    old_df = pd.read_csv(old_root / "horse_id_map.csv", dtype=str)
    new_df = pd.read_csv(new_root / "race_info" / "horse_id_map.csv", dtype=str)

    assert len(old_df) == 3
    assert old_df.equals(new_df)


def test_update_winners_weight_matches_old(old_and_new_roots):
    old_root, new_root = old_and_new_roots

    old_analysis.update_winners_weight(SAMPLE_PLACE_ID, 2019)
    new_race_info.update_winners_weight(SAMPLE_PLACE_ID, 2019)

    for filename in ("2019_winner_weight.csv", "total_winner_weight.csv"):
        old_csv = old_root / "AverageWeights" / SAMPLE_PLACE / filename
        new_csv = new_root / "race_info" / "average_weights" / SAMPLE_PLACE / filename
        assert old_csv.is_file() and new_csv.is_file()
        assert pd.read_csv(old_csv, dtype=str).equals(pd.read_csv(new_csv, dtype=str))


def test_update_average_pops_matches_old(old_and_new_roots):
    old_root, new_root = old_and_new_roots

    old_analysis.update_average_pops(SAMPLE_PLACE_ID, 2019)
    new_race_info.update_average_pops(SAMPLE_PLACE_ID, 2019)

    for filename in (
        "2019_average_pops.csv",
        "2019_average_pops_top3.csv",
        "total_average_pops.csv",
        "total_average_pops_top3.csv",
    ):
        old_csv = old_root / "AveragePops" / SAMPLE_PLACE / filename
        new_csv = new_root / "race_info" / "average_pops" / SAMPLE_PLACE / filename
        assert old_csv.is_file() and new_csv.is_file()
        assert pd.read_csv(old_csv, dtype=str).equals(pd.read_csv(new_csv, dtype=str))


def test_update_average_frame_and_horse_matches_old(old_and_new_roots):
    old_root, new_root = old_and_new_roots

    old_analysis.update_average_frame_and_horse(SAMPLE_PLACE_ID, 2019)
    new_race_info.update_average_frame_and_horse(SAMPLE_PLACE_ID, 2019)

    for filename in (
        "2019_average_frames.csv",
        "2019_average_frames_top3.csv",
        "total_average_frames.csv",
        "total_average_frames_top3.csv",
    ):
        old_csv = old_root / "AverageFrames" / SAMPLE_PLACE / filename
        new_csv = new_root / "race_info" / "average_frames" / SAMPLE_PLACE / filename
        assert old_csv.is_file() and new_csv.is_file()
        assert pd.read_csv(old_csv, dtype=str).equals(pd.read_csv(new_csv, dtype=str))
