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
from src.legacy_datasets import analysis_race_time as old_time
from src.legacy_datasets import average_time as old_avg_time
from src.legacy_datasets import race_returns as old_returns
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


# --- 勝ち馬の上り/通過（analysis_race_time.py） -------------------------------------


def test_analyze_winners_matches_old(old_data_root):
    old_result = old_time.analyze_winners(SAMPLE_PLACE_ID, 2019)
    new_result = new_race_info.analyze_winners(SAMPLE_PLACE_ID, 2019)

    assert not old_result.empty
    assert old_result.equals(new_result)


def test_analyze_winners_multi_years_matches_old(old_data_root):
    old_result = old_time.analyze_winners_multi_years(SAMPLE_PLACE_ID, start_year=2019, year=2021)
    new_result = new_race_info.analyze_winners_multi_years(SAMPLE_PLACE_ID, start_year=2019, current_year=2021)

    assert not old_result.empty
    assert old_result.equals(new_result)


# --- 配当結果（race_returns.py） ---------------------------------------------------


def _sample_race_return_tables():
    """scrape_df(url)が返すテーブル群を模したサンプルデータ（フォーマット前）"""
    table1 = pd.DataFrame([
        ["単勝", "12", "400円", "3"],
        ["複勝", "12 5 7", "130円 140円 110円", "2 3 1"],
        ["枠連", "3 - 6", "1,020円", "5"],
        ["馬連", "5 - 12", "850円", "3"],
    ], columns=[0, 1, 2, 3])
    table2 = pd.DataFrame([
        ["ワイド", "5 - 7 5 - 12 7 - 12", "250円 360円 450円", "4 9 14"],
        ["馬単", "12 → 5", "1,820円", "9"],
        ["三連複", "5 - 7 - 12", "1,660円", "12"],
        ["三連単", "12 → 5 → 7", "6,620円", "29"],
    ], columns=[0, 1, 2, 3])
    return [pd.DataFrame(), table1, table2]


def test_extract_race_return_table_matches_old():
    tables = _sample_race_return_tables()

    old_result = old_returns.extratc_race_return_table(tables)
    new_result = transform.extract_race_return_table(tables)

    assert not new_result.empty
    assert old_result.equals(new_result)


def test_format_race_return_dataframe_matches_old():
    extracted = transform.extract_race_return_table(_sample_race_return_tables())

    old_result = old_returns.format_race_return_dataframe(extracted.copy())
    new_result = transform.format_race_return_dataframe(extracted.copy())

    assert not new_result.empty
    assert old_result.equals(new_result)


@pytest.mark.parametrize("bet_type,expected", [("単勝", True), ("三連複", True), ("枠外", False)])
def test_type_check_matches_old(bet_type, expected):
    extracted = transform.extract_race_return_table(_sample_race_return_tables())
    formatted = transform.format_race_return_dataframe(extracted)

    old_result = old_returns.type_check(formatted, bet_type)
    new_result = transform.type_check(formatted, bet_type)

    assert old_result == new_result == expected


@pytest.mark.parametrize("bet_type", ["単勝", "複勝", "枠連", "馬連", "ワイド", "馬単"])
def test_format_type_returns_dataframe_matches_old(bet_type):
    extracted = transform.extract_race_return_table(_sample_race_return_tables())
    formatted = transform.format_race_return_dataframe(extracted)

    old_result = old_returns.format_type_returns_dataframe(formatted.copy(), bet_type)
    new_result = transform.format_type_returns_dataframe(formatted.copy(), bet_type)

    assert not new_result.empty
    assert old_result.equals(new_result)


@pytest.mark.parametrize("bet_type,expected_combo", [("三連複", "5-7-12"), ("三連単", "12→5→7")])
def test_format_type_returns_dataframe_fixes_sanren_kanji(bet_type, expected_combo):
    """旧実装は"3連複"/"3連単"（アラビア数字）で式別を判定しており、
    BETTING_TYPE_LISTの漢数字表記("三連複"/"三連単")とは一致しないため常に空を返す。
    新実装では漢数字表記に統一し、正しく解析する。
    """
    extracted = transform.extract_race_return_table(_sample_race_return_tables())
    formatted = transform.format_race_return_dataframe(extracted)

    old_result = old_returns.format_type_returns_dataframe(formatted.copy(), bet_type)
    new_result = transform.format_type_returns_dataframe(formatted.copy(), bet_type)

    assert old_result.empty
    assert not new_result.empty
    assert new_result.iloc[0]["馬番"] == expected_combo


# --- 配当結果（race_returns.py、書き込み比較） ----------------------------------------


def test_get_race_returns_csv_matches_old(old_and_new_roots):
    old_root, new_root = old_and_new_roots

    src = os.path.join(paths.DATA_PATH, "RaceReturns", SAMPLE_PLACE, "2019_race_returns.csv")
    (new_root / "race_info" / "race_returns" / SAMPLE_PLACE).mkdir(parents=True, exist_ok=True)
    shutil.copy(src, old_root / "RaceReturns" / SAMPLE_PLACE / "2019_race_returns.csv")
    shutil.copy(src, new_root / "race_info" / "race_returns" / SAMPLE_PLACE / "2019_race_returns.csv")

    old_result = old_returns.get_race_returns_csv(SAMPLE_PLACE_ID, 2019)
    new_result = new_race_info.get_race_returns_csv(SAMPLE_PLACE_ID, 2019)

    assert not new_result.empty
    assert old_result.equals(new_result)


def test_save_race_returns_dataset_matches_old(old_and_new_roots):
    old_root, new_root = old_and_new_roots

    extracted = transform.extract_race_return_table(_sample_race_return_tables())
    formatted = transform.format_race_return_dataframe(extracted)
    df = pd.concat([
        transform.format_type_returns_dataframe(formatted.copy(), bet_type)
        for bet_type in ["単勝", "複勝", "枠連", "馬連", "ワイド", "馬単", "三連複", "三連単"]
    ])
    df.index = ["202999999999"] * len(df)

    old_returns.save_race_returns_dataset(SAMPLE_PLACE_ID, 2099, df.copy())
    new_race_info.save_race_returns_dataset(SAMPLE_PLACE_ID, 2099, df.copy())

    for ext in ("csv", "pickle"):
        old_path = old_root / "RaceReturns" / SAMPLE_PLACE / f"2099_race_returns.{ext}"
        new_path = new_root / "race_info" / "race_returns" / SAMPLE_PLACE / f"2099_race_returns.{ext}"
        assert old_path.is_file() and new_path.is_file()

    old_csv = pd.read_csv(old_root / "RaceReturns" / SAMPLE_PLACE / "2099_race_returns.csv", dtype=str, index_col=0)
    new_csv = pd.read_csv(new_root / "race_info" / "race_returns" / SAMPLE_PLACE / "2099_race_returns.csv", dtype=str, index_col=0)
    assert not new_csv.empty
    assert old_csv.equals(new_csv)


def test_split_race_returns_csv_matches_old(old_and_new_roots):
    old_root, new_root = old_and_new_roots

    src = os.path.join(paths.DATA_PATH, "RaceReturns", SAMPLE_PLACE, "2019_race_returns.csv")
    (new_root / "race_info" / "race_returns" / SAMPLE_PLACE).mkdir(parents=True, exist_ok=True)
    shutil.copy(src, old_root / "RaceReturns" / SAMPLE_PLACE / "2019_race_returns.csv")
    shutil.copy(src, new_root / "race_info" / "race_returns" / SAMPLE_PLACE / "2019_race_returns.csv")

    old_returns.split_race_returns_csv(SAMPLE_PLACE_ID, 2019)
    new_race_info.split_race_returns_csv(SAMPLE_PLACE_ID, 2019)

    old_dir = old_root / "RaceReturns" / SAMPLE_PLACE / "2019"
    new_dir = new_root / "race_info" / "race_returns" / SAMPLE_PLACE / "2019"
    assert old_dir.is_dir() and new_dir.is_dir()

    old_files = sorted(p.name for p in old_dir.iterdir())
    new_files = sorted(p.name for p in new_dir.iterdir())
    assert old_files == new_files
    assert len(old_files) > 100

    for filename in old_files:
        old_df = pd.read_csv(old_dir / filename, dtype=str, index_col=0)
        new_df = pd.read_csv(new_dir / filename, dtype=str, index_col=0)
        assert old_df.equals(new_df)


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

    for d in ("AveragePops", "AverageWeights", "AverageFrames", "AverageTimes", "RaceReturns"):
        (old_root / d / SAMPLE_PLACE).mkdir(parents=True)

    new_race_info_dir = new_root / "race_info"
    new_race_info_dir.mkdir(parents=True)

    monkeypatch.setattr(old_analysis.name_header, "DATA_PATH", str(old_root) + "/")
    monkeypatch.setattr(paths, "RACE_INFO_DATA_PATH", str(new_race_info_dir))
    monkeypatch.setattr(new_race_info, "HORSE_ID_MAP_PATH", str(new_race_info_dir / "horse_id_map.csv"))
    monkeypatch.setattr(new_race_info, "AVERAGE_POPS_DATA_PATH", str(new_race_info_dir / "average_pops"))
    monkeypatch.setattr(new_race_info, "AVERAGE_WEIGHTS_DATA_PATH", str(new_race_info_dir / "average_weights"))
    monkeypatch.setattr(new_race_info, "AVERAGE_FRAMES_DATA_PATH", str(new_race_info_dir / "average_frames"))
    monkeypatch.setattr(new_race_info, "AVERAGE_TIMES_DATA_PATH", str(new_race_info_dir / "average_times"))
    monkeypatch.setattr(new_race_info, "RACE_RETURNS_DATA_PATH", str(new_race_info_dir / "race_returns"))

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


# --- update_winner_time / update_average_time（AverageTimes、書き込み比較） -----------


def test_update_winner_time_matches_old(old_and_new_roots):
    old_root, new_root = old_and_new_roots

    out_dir = old_root / "AverageTimes" / SAMPLE_PLACE

    # 旧 winners_time_update(place_id, year) はpalce_id/yearの引数を無視して
    # 全開催場・全年度（2019〜今年）を処理しようとし、対象データが無い開催場で
    # analyze_winners_multi_years が dict を返してクラッシュするため、ここでは
    # 新 update_winner_time(place_id, year) と同等の処理（analyze_winners /
    # analyze_winners_multi_years を1開催場・1年分のみ実行）を直接呼び出して比較する。
    result = old_time.analyze_winners(SAMPLE_PLACE_ID, 2019)
    result.to_csv(out_dir / "2019_winner_time.csv")
    total_df = old_time.analyze_winners_multi_years(SAMPLE_PLACE_ID, start_year=2019, year=2019)
    total_df.to_csv(out_dir / "total_winner_time.csv")

    new_race_info.update_winner_time(SAMPLE_PLACE_ID, 2019)

    for filename in ("2019_winner_time.csv", "total_winner_time.csv"):
        old_csv = out_dir / filename
        new_csv = new_root / "race_info" / "average_times" / SAMPLE_PLACE / filename
        assert old_csv.is_file() and new_csv.is_file()
        assert pd.read_csv(old_csv, dtype=str).equals(pd.read_csv(new_csv, dtype=str))


def test_update_annual_average_time_matches_old(old_and_new_roots):
    old_root, new_root = old_and_new_roots

    old_avg_time.make_annual_average_time_datasets(SAMPLE_PLACE_ID, 2019)
    new_race_info.update_annual_average_time(SAMPLE_PLACE_ID, 2019)

    old_csv = old_root / "AverageTimes" / SAMPLE_PLACE / "2019_avg_time.csv"
    new_csv = new_root / "race_info" / "average_times" / SAMPLE_PLACE / "2019_avg_time.csv"
    assert old_csv.is_file() and new_csv.is_file()

    old_df = pd.read_csv(old_csv, dtype=str)
    new_df = pd.read_csv(new_csv, dtype=str)
    # "不"->"不良" ラベル修正（test_make_average_time_datasets_matches_old と同様）
    old_df["ground_state"] = old_df["ground_state"].replace("不", "不良")
    assert old_df.equals(new_df)


def test_update_total_average_time_matches_old(old_and_new_roots):
    old_root, new_root = old_and_new_roots

    old_avg_time.total_average_datas(SAMPLE_PLACE_ID, 2021)
    new_race_info.update_total_average_time(SAMPLE_PLACE_ID, 2021)

    old_csv = old_root / "AverageTimes" / SAMPLE_PLACE / "total_avg_time.csv"
    new_csv = new_root / "race_info" / "average_times" / SAMPLE_PLACE / "total_avg_time.csv"
    assert old_csv.is_file() and new_csv.is_file()

    old_df = pd.read_csv(old_csv, dtype=str)
    new_df = pd.read_csv(new_csv, dtype=str)
    old_df["ground_state"] = old_df["ground_state"].replace("不", "不良")
    assert old_df.equals(new_df)


# --- Forge: average_pops/average_weights/average_frames/average_times getter（スモークテスト） ---
#
# 2019〜2026年分のバックフィル済みの実データ（data/race_info/average_pops 等、
# place_id=2 = 02_hakodate）を読み込み、Forgeが必要とする列が存在することを確認する。


@pytest.mark.parametrize("top3", [False, True])
def test_get_total_average_pops_csv_smoke(top3):
    df = new_race_info.get_total_average_pops_csv(SAMPLE_PLACE_ID, top3=top3)
    assert not df.empty
    assert "avg_pop" in df.columns


@pytest.mark.parametrize("top3", [False, True])
def test_get_annual_average_pops_csv_smoke(top3):
    df = new_race_info.get_annual_average_pops_csv(SAMPLE_PLACE_ID, 2019, top3=top3)
    assert not df.empty
    assert "avg_pop" in df.columns


def test_get_total_winner_weight_csv_smoke():
    df = new_race_info.get_total_winner_weight_csv(SAMPLE_PLACE_ID)
    assert not df.empty
    assert "馬体重" in df.columns


def test_get_annual_winner_weight_csv_smoke():
    df = new_race_info.get_annual_winner_weight_csv(SAMPLE_PLACE_ID, 2019)
    assert not df.empty
    assert "馬体重" in df.columns


@pytest.mark.parametrize("top3", [False, True])
def test_get_total_average_frames_csv_smoke(top3):
    df = new_race_info.get_total_average_frames_csv(SAMPLE_PLACE_ID, top3=top3)
    assert not df.empty
    assert "race_type" in df.columns


@pytest.mark.parametrize("top3", [False, True])
def test_get_annual_average_frames_csv_smoke(top3):
    df = new_race_info.get_annual_average_frames_csv(SAMPLE_PLACE_ID, 2019, top3=top3)
    assert not df.empty
    assert "race_type" in df.columns


def test_get_total_winner_time_csv_smoke():
    df = new_race_info.get_total_winner_time_csv(SAMPLE_PLACE_ID)
    assert not df.empty
    assert "race_type" in df.columns


def test_get_annual_winner_time_csv_smoke():
    df = new_race_info.get_annual_winner_time_csv(SAMPLE_PLACE_ID, 2019)
    assert not df.empty
    assert "race_type" in df.columns


# --- Forge: get_race_return_csv_for_race（スモークテスト） ---------------------------------


def test_get_race_return_csv_for_race_smoke(old_and_new_roots):
    old_root, new_root = old_and_new_roots

    src = os.path.join(paths.DATA_PATH, "RaceReturns", SAMPLE_PLACE, "2019_race_returns.csv")
    race_returns_dir = new_root / "race_info" / "race_returns" / SAMPLE_PLACE
    race_returns_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, race_returns_dir / "2019_race_returns.csv")

    new_race_info.split_race_returns_csv(SAMPLE_PLACE_ID, 2019)

    split_dir = race_returns_dir / "2019"
    race_id = sorted(p.stem for p in split_dir.iterdir())[0]

    result = new_race_info.get_race_return_csv_for_race(race_id)
    assert not result.empty
    assert result.index.name == "race_id"


def test_get_race_return_csv_for_race_empty_when_missing(old_and_new_roots):
    result = new_race_info.get_race_return_csv_for_race(SAMPLE_PLACE_ID * 10**10)
    assert result.empty
