"""src/datasets/race_info, src/managers/race_info_dataset_manager.py の出力が
旧 src/legacy_datasets/analysis_race_info.py と一致することを確認するテスト（オフライン）。

make_empty_record・analyze_*系・analyze_winners系（フェーズ4a）、horse_id_map/update_*系
（フェーズ4b）は新実装単体のアサーションに置き換え済み。新実装は
src.managers.race_result_dataset_manager経由で src.config.paths.RACE_RESULT_DATA_PATH
（実データ、読み取り専用）を参照するため、フィクスチャなしで実データ
（data/race_result/02_hakodate/{2019,2020,2021}_race_results.csv）に対して直接実行できる。
horse_id_map.csv / AveragePops / AverageWeights / AverageFrames / AverageTimes に相当する
新しい出力先（data/race_info/horse_id_map.csv, data/race_info/average_pops 等）は、
`new_race_info_root`フィクスチャでtmp_path配下に向けて出力内容を検証する。

race_returns系（フェーズ4c、未対応）は旧実装比較が残っている。旧実装はパスとして
name_header.DATA_PATH（ver2.0のdata/）を参照するため、旧実装の比較対象テストでは
tmp_path 配下に "RaceResults" フォルダを作って name_header.DATA_PATH をそこに向け、
新実装・旧実装それぞれ別のtmp_path配下に出力したRaceReturns関連ファイルの内容
（DataFrameの内容）を比較する（`old_and_new_roots`フィクスチャ）。
"""

import os
import shutil

import pandas as pd
import pytest

from src.config import paths
from src.datasets.race_info import transform
from src.legacy_datasets import analysis_race_info as old_analysis
from src.legacy_datasets import race_returns as old_returns
from src.managers import race_info_dataset_manager as new_race_info

SAMPLE_PLACE_ID = 2
SAMPLE_PLACE = "02_hakodate"


# --- 純粋関数の比較（旧 analysis_race_info.py vs 新 src/datasets/race_info/transform.py） ---


def test_make_empty_record():
    result = transform.make_empty_record("芝", "1800", "良", "未勝利")

    assert result == {
        "race_type": "芝",
        "course_len": 1800,
        "ground_state": "良",
        "class": "未勝利",
        "avg_frame": None,
        "avg_horse": None,
        "total_top3": 0,
        "frame_1_top3": 0,
        "frame_2_top3": 0,
        "frame_3_top3": 0,
        "frame_4_top3": 0,
        "frame_5_top3": 0,
        "frame_6_top3": 0,
        "frame_7_top3": 0,
        "frame_8_top3": 0,
        "horse_1_top3": 0,
        "horse_2_top3": 0,
        "horse_3_top3": 0,
        "horse_4_top3": 0,
        "horse_5_top3": 0,
        "horse_6_top3": 0,
        "horse_7_top3": 0,
        "horse_8_top3": 0,
        "horse_9_top3": 0,
        "horse_10_top3": 0,
        "horse_11_top3": 0,
        "horse_12_top3": 0,
        "horse_13_top3": 0,
        "horse_14_top3": 0,
        "horse_15_top3": 0,
        "horse_16_top3": 0,
        "horse_17_top3": 0,
        "horse_18_top3": 0,
    }


# --- analyze_* 系（実データ: 02_hakodate 2019〜2021） ----------------------------


def test_analyze_winner_weights_returns_real_data():
    result = new_race_info.analyze_winner_weights(SAMPLE_PLACE_ID, 2019)

    assert result.shape == (280, 5)
    assert result.columns.tolist() == ["race_type", "course_len", "ground_state", "class", "馬体重"]
    first = result.iloc[0]
    assert first["race_type"] == "芝"
    assert first["course_len"] == 1000
    assert first["ground_state"] == "全"
    assert first["class"] == "all"
    assert first["馬体重"] == 444.0


def test_analyze_winner_weights_multi_years_returns_real_data():
    result = new_race_info.analyze_winner_weights_multi_years(SAMPLE_PLACE_ID, start_year=2019, current_year=2021)

    assert result.shape == (490, 5)
    assert result.columns.tolist() == ["race_type", "course_len", "ground_state", "class", "馬体重"]
    assert result.iloc[0]["馬体重"] == 428.0


@pytest.mark.parametrize("top3,expected_avg_pop", [(False, 9.0), (True, 5.5)])
def test_analyze_average_pops_returns_real_data(top3, expected_avg_pop):
    result = new_race_info.analyze_average_pops(SAMPLE_PLACE_ID, 2019, top3=top3)

    assert result.shape == (280, 23)
    assert result.columns[:5].tolist() == ["race_type", "course_len", "ground_state", "class", "avg_pop"]
    assert result.iloc[0]["avg_pop"] == expected_avg_pop


@pytest.mark.parametrize("top3,expected_avg_pop", [(False, 5.32), (True, 6.02)])
def test_analyze_average_pop_multi_years_returns_real_data(top3, expected_avg_pop):
    result = new_race_info.analyze_average_pop_multi_years(
        SAMPLE_PLACE_ID, start_year=2019, current_year=2021, top3=top3
    )

    assert result.shape == (490, 23)
    assert result.iloc[0]["avg_pop"] == expected_avg_pop


def test_analyze_average_frame_and_horse_returns_real_data():
    result = new_race_info.analyze_average_frame_and_horse(SAMPLE_PLACE_ID, 2019)

    assert result.shape == (280, 33)
    assert result.columns[:7].tolist() == [
        "race_type",
        "course_len",
        "ground_state",
        "class",
        "avg_frame",
        "avg_horse",
        "total_winners",
    ]
    first = result.iloc[0]
    assert first["avg_frame"] == 6.0
    assert first["avg_horse"] == 8.0
    assert first["total_winners"] == 1


def test_analyze_frame_and_horse_multi_years_returns_real_data():
    result = new_race_info.analyze_frame_and_horse_multi_years(SAMPLE_PLACE_ID, start_year=2019, current_year=2021)

    assert result.shape == (490, 33)
    first = result.iloc[0]
    assert first["avg_frame"] == 3.5
    assert first["avg_horse"] == 4.5
    assert first["total_winners"] == 2


def test_analyze_average_frame_and_horse_top3_returns_real_data():
    result = new_race_info.analyze_average_frame_and_horse_top3(SAMPLE_PLACE_ID, 2019)

    assert result.shape == (280, 33)
    assert result.columns[:7].tolist() == [
        "race_type",
        "course_len",
        "ground_state",
        "class",
        "avg_frame",
        "avg_horse",
        "total_top3",
    ]
    first = result.iloc[0]
    assert first["avg_frame"] == 5.33
    assert first["total_top3"] == 3


def test_analyze_frame_and_horse_top3_multi_years_returns_real_data():
    result = new_race_info.analyze_frame_and_horse_top3_multi_years(
        SAMPLE_PLACE_ID, start_year=2019, current_year=2021
    )

    assert result.shape == (490, 33)
    first = result.iloc[0]
    assert first["avg_frame"] == 4.83
    assert first["avg_horse"] == 6.0
    assert first["total_top3"] == 6


# --- 勝ち馬の上り/通過（analysis_race_time.py） -------------------------------------


def test_analyze_winners_returns_real_data():
    result = new_race_info.analyze_winners(SAMPLE_PLACE_ID, 2019)

    assert result.shape == (280, 9)
    assert result.columns.tolist() == [
        "race_type",
        "course_len",
        "ground_state",
        "class",
        "上り",
        "通過1",
        "通過2",
        "通過3",
        "通過4",
    ]
    first = result.iloc[0]
    assert first["race_type"] == "芝"
    assert first["上り"] == 34.3
    assert first["通過1"] == 4.0
    assert first["通過2"] == 4.0
    assert pd.isna(first["通過3"])


def test_analyze_winners_multi_years_returns_real_data():
    result = new_race_info.analyze_winners_multi_years(SAMPLE_PLACE_ID, start_year=2019, current_year=2021)

    assert result.shape == (490, 9)
    first = result.iloc[0]
    assert first["上り"] == 33.9
    assert first["通過1"] == 4.4
    assert first["通過2"] == 3.6


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


@pytest.fixture
def new_race_info_root(tmp_path, monkeypatch):
    """新実装(race_info_dataset_managerの各パス定数)をtmp_path配下に向ける。
    race_resultsは実データ(paths.RACE_RESULT_DATA_PATH)をそのまま参照する。"""
    new_root = tmp_path / "race_info"
    new_root.mkdir(parents=True)
    monkeypatch.setattr(paths, "RACE_INFO_DATA_PATH", str(new_root))
    monkeypatch.setattr(new_race_info, "HORSE_ID_MAP_PATH", str(new_root / "horse_id_map.csv"))
    monkeypatch.setattr(new_race_info, "AVERAGE_POPS_DATA_PATH", str(new_root / "average_pops"))
    monkeypatch.setattr(new_race_info, "AVERAGE_WEIGHTS_DATA_PATH", str(new_root / "average_weights"))
    monkeypatch.setattr(new_race_info, "AVERAGE_FRAMES_DATA_PATH", str(new_root / "average_frames"))
    monkeypatch.setattr(new_race_info, "AVERAGE_TIMES_DATA_PATH", str(new_root / "average_times"))
    return new_root


def test_update_horse_name_id_map_from_results_returns_real_data(new_race_info_root):
    new_race_info.update_horse_name_id_map_from_results(SAMPLE_PLACE_ID, 2019)

    csv_path = new_race_info_root / "horse_id_map.csv"
    assert csv_path.is_file()

    df = pd.read_csv(csv_path, dtype=str)
    assert df.shape == (981, 2)
    assert df.columns.tolist() == ["馬名", "horse_id"]
    assert df.iloc[0].tolist() == ["ユキノアイオロス", "2008100889"]
    assert df.iloc[1].tolist() == ["ケージーキンカメ", "2011101513"]
    assert df.iloc[2].tolist() == ["カゼノコ", "2011102128"]


def test_get_horse_id_list_returns_expected(new_race_info_root):
    df = pd.DataFrame({"馬名": ["馬A", "馬B"], "horse_id": ["1111111111", " 2222222222 "]})
    df.to_csv(new_race_info_root / "horse_id_map.csv", index=False, encoding="utf-8-sig")

    result = new_race_info.get_horse_id_list()

    assert result == ["1111111111", "2222222222"]


def test_add_horse_name_id_map_adds_new_entry(new_race_info_root):
    df = pd.DataFrame({"馬名": ["既存馬"], "horse_id": ["0000000000"]})
    df.to_csv(new_race_info_root / "horse_id_map.csv", index=False, encoding="utf-8-sig")

    new_race_info.add_horse_name_id_map("1234567890", "新馬A")

    result = pd.read_csv(new_race_info_root / "horse_id_map.csv", dtype=str)
    assert result.columns.tolist() == ["馬名", "horse_id"]
    assert result.values.tolist() == [["既存馬", "0000000000"], ["新馬A", "1234567890"]]


def test_add_horse_name_id_map_skips_duplicate_id(new_race_info_root):
    df = pd.DataFrame({"馬名": ["既存馬"], "horse_id": ["0000000000"]})
    df.to_csv(new_race_info_root / "horse_id_map.csv", index=False, encoding="utf-8-sig")

    # horse_idが既存と重複（馬名は別）-> 追加されない
    new_race_info.add_horse_name_id_map("0000000000", "別名の馬")

    result = pd.read_csv(new_race_info_root / "horse_id_map.csv", dtype=str)
    assert result.values.tolist() == [["既存馬", "0000000000"]]


def test_update_horse_name_id_map_adds_valid_entries(new_race_info_root):
    df = pd.DataFrame({"馬名": ["既存馬"], "horse_id": ["0000000000"]})
    df.to_csv(new_race_info_root / "horse_id_map.csv", index=False, encoding="utf-8-sig")

    race_card_df = pd.DataFrame({
        "馬名": ["新馬B", "新馬C", ""],
        "horse_id": ["1111111111", "2222222222", "nan"],
    })

    new_race_info.update_horse_name_id_map(race_card_df.copy())

    result = pd.read_csv(new_race_info_root / "horse_id_map.csv", dtype=str)
    assert result.values.tolist() == [
        ["既存馬", "0000000000"],
        ["新馬B", "1111111111"],
        ["新馬C", "2222222222"],
    ]


def test_update_winners_weight_writes_expected_files(new_race_info_root):
    new_race_info.update_winners_weight(SAMPLE_PLACE_ID, 2019)

    out_dir = new_race_info_root / "average_weights" / SAMPLE_PLACE
    columns = ["Unnamed: 0", "race_type", "course_len", "ground_state", "class", "馬体重"]

    df_2019 = pd.read_csv(out_dir / "2019_winner_weight.csv", dtype=str)
    assert df_2019.shape == (280, 6)
    assert df_2019.columns.tolist() == columns
    assert df_2019.iloc[0].tolist() == ["0", "芝", "1000", "全", "all", "444.0"]

    df_total = pd.read_csv(out_dir / "total_winner_weight.csv", dtype=str)
    assert df_total.shape == (490, 6)
    assert df_total.columns.tolist() == columns
    assert df_total.iloc[0].tolist() == ["0", "芝", "1000", "全", "all", "444.0"]


def test_update_average_pops_writes_expected_files(new_race_info_root):
    new_race_info.update_average_pops(SAMPLE_PLACE_ID, 2019)

    out_dir = new_race_info_root / "average_pops" / SAMPLE_PLACE
    head_columns = ["Unnamed: 0", "race_type", "course_len", "ground_state", "class", "avg_pop"]

    df = pd.read_csv(out_dir / "2019_average_pops.csv", dtype=str)
    assert df.shape == (280, 24)
    assert df.columns[:6].tolist() == head_columns
    assert df.iloc[0, :6].tolist() == ["0", "芝", "1000", "全", "all", "9.0"]

    df_top3 = pd.read_csv(out_dir / "2019_average_pops_top3.csv", dtype=str)
    assert df_top3.shape == (280, 24)
    assert df_top3.iloc[0, :6].tolist() == ["0", "芝", "1000", "全", "all", "5.5"]

    df_total = pd.read_csv(out_dir / "total_average_pops.csv", dtype=str)
    assert df_total.shape == (490, 24)
    assert df_total.iloc[0, :6].tolist() == ["0", "芝", "1000", "全", "all", "9.0"]

    df_total_top3 = pd.read_csv(out_dir / "total_average_pops_top3.csv", dtype=str)
    assert df_total_top3.shape == (490, 24)
    assert df_total_top3.iloc[0, :6].tolist() == ["0", "芝", "1000", "全", "all", "5.5"]


def test_update_average_frame_and_horse_writes_expected_files(new_race_info_root):
    new_race_info.update_average_frame_and_horse(SAMPLE_PLACE_ID, 2019)

    out_dir = new_race_info_root / "average_frames" / SAMPLE_PLACE
    head_columns = ["Unnamed: 0", "race_type", "course_len", "ground_state", "class", "avg_frame", "avg_horse"]

    df = pd.read_csv(out_dir / "2019_average_frames.csv", dtype=str)
    assert df.shape == (280, 34)
    assert df.columns[:8].tolist() == head_columns + ["total_winners"]
    assert df.iloc[0, :8].tolist() == ["0", "芝", "1000", "全", "all", "6.0", "8.0", "1"]

    df_top3 = pd.read_csv(out_dir / "2019_average_frames_top3.csv", dtype=str)
    assert df_top3.shape == (280, 34)
    assert df_top3.columns[:8].tolist() == head_columns + ["total_top3"]
    assert df_top3.iloc[0, :8].tolist() == ["0", "芝", "1000", "全", "all", "5.33", "7.33", "3"]

    df_total = pd.read_csv(out_dir / "total_average_frames.csv", dtype=str)
    assert df_total.shape == (490, 34)
    assert df_total.iloc[0, :8].tolist() == ["0", "芝", "1000", "全", "all", "6.0", "8.0", "1"]

    df_total_top3 = pd.read_csv(out_dir / "total_average_frames_top3.csv", dtype=str)
    assert df_total_top3.shape == (490, 34)
    assert df_total_top3.iloc[0, :8].tolist() == ["0", "芝", "1000", "全", "all", "5.33", "7.33", "3"]


# --- update_winner_time / update_average_time（AverageTimes） -----------------------


def test_update_winner_time_writes_expected_files(new_race_info_root):
    new_race_info.update_winner_time(SAMPLE_PLACE_ID, 2019)

    out_dir = new_race_info_root / "average_times" / SAMPLE_PLACE
    columns = [
        "Unnamed: 0", "race_type", "course_len", "ground_state", "class",
        "上り", "通過1", "通過2", "通過3", "通過4",
    ]

    df_2019 = pd.read_csv(out_dir / "2019_winner_time.csv", dtype=str)
    assert df_2019.shape == (280, 10)
    assert df_2019.columns.tolist() == columns
    assert df_2019.iloc[0, :8].tolist() == ["0", "芝", "1000", "全", "all", "34.3", "4.0", "4.0"]
    assert pd.isna(df_2019.iloc[0]["通過3"])

    df_total = pd.read_csv(out_dir / "total_winner_time.csv", dtype=str)
    assert df_total.shape == (490, 10)
    assert df_total.columns.tolist() == columns
    assert df_total.iloc[0, :8].tolist() == ["0", "芝", "1000", "全", "all", "34.3", "4.0", "4.0"]


def test_update_annual_average_time_writes_expected_file(new_race_info_root):
    new_race_info.update_annual_average_time(SAMPLE_PLACE_ID, 2019)

    csv_path = new_race_info_root / "average_times" / SAMPLE_PLACE / "2019_avg_time.csv"
    assert csv_path.is_file()

    df = pd.read_csv(csv_path, dtype=str)
    assert df.shape == (280, 6)
    assert df.columns.tolist() == ["Unnamed: 0", "race_type", "course_len", "ground_state", "class", "avg_time"]
    assert df.iloc[0].tolist() == ["0", "芝", "1000", "全", "all", "58100"]
    assert df.iloc[1].tolist() == ["1", "芝", "1000", "良", "all", "58100"]


def test_update_total_average_time_writes_expected_file(new_race_info_root):
    new_race_info.update_total_average_time(SAMPLE_PLACE_ID, 2021)

    csv_path = new_race_info_root / "average_times" / SAMPLE_PLACE / "total_avg_time.csv"
    assert csv_path.is_file()

    df = pd.read_csv(csv_path, dtype=str)
    assert df.shape == (280, 6)
    assert df.columns.tolist() == ["Unnamed: 0", "race_type", "course_len", "ground_state", "class", "avg_time"]
    assert df.iloc[0].tolist() == ["0", "芝", "1000", "全", "all", "57850"]
    assert df.iloc[1].tolist() == ["1", "芝", "1000", "良", "all", "57850"]


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


# --- save_race_return_for_race_id ------------------------------------------------


def test_save_race_return_for_race_id_writes_per_race_csv(tmp_path, monkeypatch):
    monkeypatch.setattr(new_race_info, "RACE_RETURNS_DATA_PATH", str(tmp_path / "race_returns"))

    race_id = "202405010101"
    df = pd.DataFrame(
        [["単勝", "7", "140", "1"], ["複勝", "7", "110", "1"]],
        columns=["式別", "馬番", "配当", "人気"],
        index=[race_id, race_id],
    )

    new_race_info.save_race_return_for_race_id(race_id, df)

    out_path = tmp_path / "race_returns" / "05_tokyo" / "2024" / f"{race_id}.csv"
    assert out_path.is_file()

    result = new_race_info.get_race_return_csv_for_race(race_id)
    result.index = result.index.astype(str)
    assert result.equals(df.rename_axis("race_id"))


def test_save_race_return_for_race_id_noop_for_empty_dataframe(tmp_path, monkeypatch):
    monkeypatch.setattr(new_race_info, "RACE_RETURNS_DATA_PATH", str(tmp_path / "race_returns"))

    new_race_info.save_race_return_for_race_id("202405010101", pd.DataFrame())

    assert not (tmp_path / "race_returns").exists()
