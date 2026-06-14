"""src/datasets/horse, src/managers/peds_results_dataset_manager.py の出力が
旧 src/legacy_datasets/peds_results.py と一致することを確認するテスト（オフライン）。

旧実装はパスとして name_header.DATA_PATH（ver2.0のdata/）を参照するため、
旧実装の比較対象テストでは tmp_path 配下に "RaceResults"/"PedsResults"/"HorsePeds" フォルダを
作って name_header.DATA_PATH をそこに向ける。新実装は src.config.paths の
RACE_RESULT_DATA_PATH / peds_results_dataset_manager.PEDS_RESULTS_DATA_PATH を同様に
tmp_path配下に向ける。
"""

import os
import shutil
from datetime import date

import pandas as pd
import pytest

from src.config import paths
from src.config.constants import PLACE_LIST
from src.datasets.horse import transform
from src.legacy_datasets import peds_results as old_peds_results
from src.managers import peds_results_dataset_manager as new_peds_results
from src.managers import race_result_dataset_manager as new_race_result

PLACE_ID = 2
YEAR = 2019
SAMPLE_HORSE_ID = "2007100107"


# --- 純粋関数の比較（旧 peds_results.py vs 新 src/datasets/horse/transform.py） ---


def test_calc_peds_placed_rate_matches_old():
    df = pd.DataFrame({"着順": ["1", "1", "2", "3", "4", "5", "中止"]})

    old_result = old_peds_results.calc_peds_placed_rate(df.copy())
    new_result = transform.calc_peds_placed_rate(df.copy())

    assert old_result == new_result


def test_calc_peds_data_matches_old():
    df = pd.DataFrame(
        {
            "着順": ["1", "2", "3", "4", "1", "2"],
            "course_len": ["1800", "1800", "1800", "2000", "1800", "1800"],
            "class": ["未勝利", "未勝利", "未勝利", "未勝利", "1勝クラス", "未勝利"],
        }
    )

    old_result = old_peds_results.calc_peds_data(df.copy(), "1800", "未勝利")
    new_result = transform.calc_peds_data(df.copy(), "1800", "未勝利")

    assert old_result.equals(new_result)


def test_get_race_type_data_matches_old():
    df = pd.DataFrame(
        {
            "race_type": ["芝", "ダート", "芝"],
            "ground_state": ["良", "良", "稍重"],
            "着順": ["1", "2", "3"],
        }
    )

    old_result = old_peds_results.get_race_type_data(df.copy(), "芝", "良")
    new_result = transform.get_race_type_data(df.copy(), "芝", "良")

    assert old_result.equals(new_result)


def test_output_results_matches_old():
    df = pd.DataFrame(
        {
            "peds_0": ["父A", "父A", "父A", "父B", "父B", "父B"],
            "着順": [1, 2, 4, 1, 3, 5],
        }
    )

    old_result = old_peds_results.output_results(df.copy())
    new_result = transform.output_results(df.copy())

    assert old_result.equals(new_result)


def test_output_results_empty_returns_empty_dataframe():
    """新実装は空入力に対して空のDataFrameを返す（aggregate_peds_resultsで.emptyを
    呼べるようにするための仕様。旧実装はNoneを返すが、aggregate_peds_results側で
    新実装に合わせて修正済み）"""
    result = transform.output_results(pd.DataFrame())
    assert isinstance(result, pd.DataFrame)
    assert result.empty


# --- CRUD（実データを使った読み込み）の比較 -----------------------------------


@pytest.fixture
def old_data_root(tmp_path, monkeypatch):
    """旧実装のname_header.DATA_PATHをtmp_path配下に向ける"""
    monkeypatch.setattr(old_peds_results.name_header, "DATA_PATH", str(tmp_path) + "/")
    return tmp_path


def _copy_peds_csv(old_data_root, place_id, year, suffix="peds"):
    place = PLACE_LIST[place_id - 1]
    src_path = os.path.join(paths.HORSE_DATA_PATH, "peds_results", place, f"{year}_{suffix}.csv")
    dst_dir = os.path.join(old_data_root, "PedsResults", place)
    os.makedirs(dst_dir, exist_ok=True)
    dst_path = os.path.join(dst_dir, f"{year}_{suffix}.csv")
    shutil.copy(src_path, dst_path)
    return dst_path


def test_get_peds_dataset_csv_matches_old(old_data_root):
    _copy_peds_csv(old_data_root, PLACE_ID, YEAR, "peds")

    old_df = old_peds_results.get_peds_dataset_csv(PLACE_ID, YEAR)
    new_df = new_peds_results.get_peds_dataset_csv(PLACE_ID, YEAR)

    assert not old_df.empty
    assert old_df.equals(new_df)


def test_get_peds_data_dataset_csv_matches_old(old_data_root):
    _copy_peds_csv(old_data_root, PLACE_ID, YEAR, "peds_data")

    old_df = old_peds_results.get_peds_data_dataset_csv(PLACE_ID, YEAR)
    new_df = new_peds_results.get_peds_data_dataset_csv(PLACE_ID, YEAR)

    assert not old_df.empty
    assert old_df.equals(new_df)


def test_get_peds_dataset_csv_returns_empty_for_missing_file(old_data_root):
    old_df = old_peds_results.get_peds_dataset_csv(PLACE_ID, 1999)
    new_df = new_peds_results.get_peds_dataset_csv(PLACE_ID, 1999)

    assert old_df.empty and new_df.empty


def test_get_peds_dataset_from_horse_id_list_matches_old(old_data_root):
    src_path = os.path.join(paths.HORSE_DATA_PATH, "horse_peds", f"{SAMPLE_HORSE_ID}.csv")
    dst_dir = os.path.join(old_data_root, "HorsePeds")
    os.makedirs(dst_dir, exist_ok=True)
    shutil.copy(src_path, os.path.join(dst_dir, f"{SAMPLE_HORSE_ID}.csv"))

    old_result = old_peds_results.get_peds_dataset_from_horse_id_list([SAMPLE_HORSE_ID])
    new_result = new_peds_results.get_peds_dataset_from_horse_id_list([SAMPLE_HORSE_ID])

    assert not old_result.empty
    assert old_result.equals(new_result)


# --- 書き込み系の比較（tmp_path配下にそれぞれ別ディレクトリで出力して比較） -----


@pytest.fixture
def old_and_new_roots(tmp_path, monkeypatch):
    """旧実装(name_header.DATA_PATH)と新実装(peds_results_dataset_manager.PEDS_RESULTS_DATA_PATH)を
    それぞれ別のtmp_path配下に向ける
    """
    old_root = tmp_path / "old"
    new_root = tmp_path / "new"
    place = PLACE_LIST[PLACE_ID - 1]
    (old_root / "PedsResults" / place).mkdir(parents=True)
    (new_root / "horse" / "peds_results" / place).mkdir(parents=True)

    monkeypatch.setattr(old_peds_results.name_header, "DATA_PATH", str(old_root) + "/")
    monkeypatch.setattr(new_peds_results, "PEDS_RESULTS_DATA_PATH", str(new_root / "horse" / "peds_results"))

    return old_root, new_root


def _sample_peds_dataset_df():
    return pd.DataFrame(
        {f"peds_{i}": [f"値A{i}", f"値B{i}"] for i in range(62)},
        index=[1111111111, 2222222222],
    )


def test_save_peds_dataset_matches_old(old_and_new_roots):
    old_root, new_root = old_and_new_roots
    place = PLACE_LIST[PLACE_ID - 1]

    old_peds_results.save_peds_dataset(_sample_peds_dataset_df(), PLACE_ID, YEAR)
    new_peds_results.save_peds_dataset(_sample_peds_dataset_df(), PLACE_ID, YEAR)

    old_csv = old_root / "PedsResults" / place / f"{YEAR}_peds.csv"
    new_csv = new_root / "horse" / "peds_results" / place / f"{YEAR}_peds.csv"

    assert old_csv.is_file() and new_csv.is_file()
    assert pd.read_csv(old_csv, dtype=str).equals(pd.read_csv(new_csv, dtype=str))


# --- merge_pedsdata_with_race_results（実データの1レース分で比較） -------------


@pytest.fixture
def merge_roots(tmp_path, monkeypatch):
    place = PLACE_LIST[PLACE_ID - 1]

    # 実データから1レース分のrace_resultsを抽出
    df_course_full = new_race_result.get_race_results_csv(PLACE_ID, YEAR)
    race_id = df_course_full.index[0]
    df_one_race = df_course_full[df_course_full.index == race_id]

    old_root = tmp_path / "old"
    new_root = tmp_path / "new"

    old_race_dir = old_root / "RaceResults" / place
    old_peds_dir = old_root / "PedsResults" / place
    old_race_dir.mkdir(parents=True)
    old_peds_dir.mkdir(parents=True)
    df_one_race.to_csv(old_race_dir / f"{YEAR}_race_results.csv")
    shutil.copy(
        os.path.join(paths.HORSE_DATA_PATH, "peds_results", place, f"{YEAR}_peds.csv"),
        old_peds_dir / f"{YEAR}_peds.csv",
    )

    new_race_dir = new_root / "race_result" / place
    new_peds_dir = new_root / "horse" / "peds_results" / place
    new_race_dir.mkdir(parents=True)
    new_peds_dir.mkdir(parents=True)
    df_one_race.to_csv(new_race_dir / f"{YEAR}_race_results.csv")
    shutil.copy(
        os.path.join(paths.HORSE_DATA_PATH, "peds_results", place, f"{YEAR}_peds.csv"),
        new_peds_dir / f"{YEAR}_peds.csv",
    )

    monkeypatch.setattr(old_peds_results.name_header, "DATA_PATH", str(old_root) + "/")
    monkeypatch.setattr(paths, "RACE_RESULT_DATA_PATH", str(new_root / "race_result"))
    monkeypatch.setattr(new_peds_results, "PEDS_RESULTS_DATA_PATH", str(new_root / "horse" / "peds_results"))

    return old_root, new_root


def test_merge_pedsdata_with_race_results_matches_old(merge_roots):
    old_root, new_root = merge_roots
    place = PLACE_LIST[PLACE_ID - 1]

    old_peds_results.merge_pedsdata_with_race_results(PLACE_ID, YEAR)
    new_peds_results.merge_pedsdata_with_race_results(PLACE_ID, YEAR)

    old_csv = old_root / "PedsResults" / place / f"{YEAR}_peds_data.csv"
    new_csv = new_root / "horse" / "peds_results" / place / f"{YEAR}_peds_data.csv"

    assert old_csv.is_file() and new_csv.is_file()
    assert pd.read_csv(old_csv, dtype=str).equals(pd.read_csv(new_csv, dtype=str))


# --- peds_index（実データの2019_peds_dataを使用） -------------------------------


def test_peds_index_matches_old(old_data_root):
    target_year = 2020  # range(2019, target_year) == [2019]
    _copy_peds_csv(old_data_root, PLACE_ID, 2019, "peds_data")

    df_2019 = new_peds_results.get_peds_data_dataset_csv(PLACE_ID, 2019)
    row = df_2019.iloc[0]
    father = row["peds_0"]
    mother_father = row["peds_4"]
    course_info = [PLACE_ID, row["race_type"], row["course_len"], row["ground_state"], row["class"]]

    old_result = old_peds_results.peds_index(father, mother_father, course_info, target_year)
    new_result = new_peds_results.peds_index(father, mother_father, course_info, target_year)

    assert old_result.reset_index(drop=True).equals(new_result.reset_index(drop=True))


# --- aggregate_peds_results / aggregate_total_peds_results（実データで比較） ---


@pytest.fixture
def aggregate_roots(tmp_path, monkeypatch):
    place = PLACE_LIST[PLACE_ID - 1]
    src = os.path.join(paths.HORSE_DATA_PATH, "peds_results", place, f"{YEAR}_peds_data.csv")

    old_root = tmp_path / "old"
    new_root = tmp_path / "new"
    old_dir = old_root / "PedsResults" / place
    new_dir = new_root / "horse" / "peds_results" / place
    old_dir.mkdir(parents=True)
    new_dir.mkdir(parents=True)
    shutil.copy(src, old_dir / f"{YEAR}_peds_data.csv")
    shutil.copy(src, new_dir / f"{YEAR}_peds_data.csv")

    monkeypatch.setattr(old_peds_results.name_header, "DATA_PATH", str(old_root) + "/")
    monkeypatch.setattr(new_peds_results, "PEDS_RESULTS_DATA_PATH", str(new_root / "horse" / "peds_results"))

    return old_root, new_root


def test_aggregate_peds_results_matches_old(aggregate_roots):
    old_root, new_root = aggregate_roots
    place = PLACE_LIST[PLACE_ID - 1]

    old_peds_results.aggregate_peds_results(PLACE_ID, YEAR)
    new_peds_results.aggregate_peds_results(PLACE_ID, YEAR)

    old_dir = old_root / "PedsResults" / place / str(YEAR)
    new_dir = new_root / "horse" / "peds_results" / place / str(YEAR)

    old_files = sorted(p.name for p in old_dir.iterdir())
    new_files = sorted(p.name for p in new_dir.iterdir())
    assert old_files and old_files == new_files

    for name in old_files:
        old_df = pd.read_csv(old_dir / name)
        new_df = pd.read_csv(new_dir / name)
        assert old_df.equals(new_df), name


def test_aggregate_total_peds_results_matches_old(aggregate_roots):
    old_root, new_root = aggregate_roots
    place = PLACE_LIST[PLACE_ID - 1]

    old_peds_results.aggregate_peds_results(PLACE_ID, YEAR)
    new_peds_results.aggregate_peds_results(PLACE_ID, YEAR)

    old_peds_results.aggregate_total_peds_results(place_id=PLACE_ID, start_year=YEAR, end_year=YEAR)
    new_peds_results.aggregate_total_peds_results(place_id=PLACE_ID, start_year=YEAR, end_year=YEAR)

    old_dir = old_root / "PedsResults" / place / "Total"
    new_dir = new_root / "horse" / "peds_results" / place / "Total"

    old_files = sorted(p.name for p in old_dir.iterdir())
    new_files = sorted(p.name for p in new_dir.iterdir())
    assert old_files and old_files == new_files

    for name in old_files:
        old_df = pd.read_csv(old_dir / name)
        new_df = pd.read_csv(new_dir / name)
        assert old_df.equals(new_df), name


# --- update_peds_dataset（依存先をモックして比較） -------------------------------


def test_update_peds_dataset_matches_old(tmp_path, monkeypatch):
    place = PLACE_LIST[PLACE_ID - 1]
    day = date(2024, 6, 10)

    old_root = tmp_path / "old"
    new_root = tmp_path / "new"
    (old_root / "PedsResults" / place).mkdir(parents=True)
    (old_root / "HorsePeds").mkdir(parents=True)
    (new_root / "horse" / "peds_results" / place).mkdir(parents=True)

    shutil.copy(
        os.path.join(paths.HORSE_DATA_PATH, "horse_peds", f"{SAMPLE_HORSE_ID}.csv"),
        old_root / "HorsePeds" / f"{SAMPLE_HORSE_ID}.csv",
    )

    monkeypatch.setattr(old_peds_results.name_header, "DATA_PATH", str(old_root) + "/")
    monkeypatch.setattr(new_peds_results, "PEDS_RESULTS_DATA_PATH", str(new_root / "horse" / "peds_results"))

    monkeypatch.setattr(old_peds_results.get_race_id, "get_past_weekly_id", lambda *a, **k: ["dummy"])
    monkeypatch.setattr(
        old_peds_results.past_performance, "get_horse_id_list_from_race_id_list", lambda *a, **k: [SAMPLE_HORSE_ID]
    )
    monkeypatch.setattr(new_peds_results.race_schedule_dataset_manager, "get_past_weekly_id", lambda *a, **k: ["dummy"])
    monkeypatch.setattr(
        new_peds_results.past_performance_dataset_manager,
        "get_horse_id_list_from_race_id_list",
        lambda *a, **k: [SAMPLE_HORSE_ID],
    )

    old_peds_results.update_peds_dataset(PLACE_ID, day)
    new_peds_results.update_peds_dataset(PLACE_ID, day)

    old_csv = old_root / "PedsResults" / place / f"{day.year}_peds.csv"
    new_csv = new_root / "horse" / "peds_results" / place / f"{day.year}_peds.csv"

    assert old_csv.is_file() and new_csv.is_file()
    assert pd.read_csv(old_csv, dtype=str).equals(pd.read_csv(new_csv, dtype=str))


# --- Forge: get_total_peds_results_csv / get_annual_peds_results_csv（スモークテスト） -------


def test_get_total_peds_results_csv_smoke():
    df = new_peds_results.get_total_peds_results_csv(PLACE_ID, "ダート", 1000, "良")
    assert not df.empty


def test_get_annual_peds_results_csv_smoke():
    df = new_peds_results.get_annual_peds_results_csv(PLACE_ID, YEAR, "ダート", 1000, "良")
    assert not df.empty


def test_get_total_peds_results_csv_empty_when_missing():
    df = new_peds_results.get_total_peds_results_csv(PLACE_ID, "存在しない", 9999, "良")
    assert df.empty
