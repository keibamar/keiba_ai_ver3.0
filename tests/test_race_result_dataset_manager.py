"""src/datasets/race_result, src/managers/race_result_dataset_manager.py の出力が
旧 src/legacy_datasets/race_results.py と一致することを確認するテスト（オフライン）。

旧実装はパスとして name_header.DATA_PATH（ver2.0のdata/）を参照するため、
旧実装の比較対象テストでは tmp_path 配下に "RaceResults"/"RaceInfo" フォルダを
作って name_header.DATA_PATH をそこに向ける。新実装は src.config.paths の
RACE_RESULT_DATA_PATH / RACE_INFO_DATA_PATH を同様に tmp_path 配下に向ける。
"""

import os
import shutil

import numpy as np
import pandas as pd
import pytest

from src.config import paths
from src.config.constants import PLACE_LIST
from src.datasets.race_result import transform
from src.legacy_datasets import race_results as old_race_results
from src.managers import race_result_dataset_manager as new_race_result


# --- 純粋関数の比較（旧 race_results.py vs 新 src/datasets/race_result/transform.py） ---


def test_clean_columns_matches_old():
    df_old = pd.DataFrame([[1, 2, 3]], columns=["a", "a", "b"])
    df_new = df_old.copy()

    result_old = old_race_results.clean_columns(df_old)
    result_new = transform.clean_columns(df_new)

    assert result_old.columns.tolist() == result_new.columns.tolist() == ["a_1", "a_2", "b"]


def test_format_race_results_dataframe_matches_old():
    df_old = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})
    df_new = df_old.copy()

    result_old = old_race_results.format_race_results_dataframe(df_old)
    result_new = transform.format_race_results_dataframe(df_new)

    assert result_old.equals(result_new)
    assert len(result_old) == 2


@pytest.mark.parametrize("val", ["1400m", "1400.0", "1400", "", "abc", np.nan])
def test_to_int_matches_old(val):
    old_result = old_race_results._to_int(val)
    new_result = transform.to_int(val)

    if pd.isna(old_result) and pd.isna(new_result):
        return
    assert old_result == new_result


@pytest.mark.parametrize(
    "result",
    [
        pd.Series({"開催": "5回東京2日", "course_len": "ダート1400m", "ground_state": "稍", "class": "未勝利"}),
        pd.Series({"開催": "1回中山1日", "course_len": "芝2000m", "ground_state": "良", "class": "オープン"}),
        pd.Series({"開催": "1回中山1日", "course_len": "障害3000m", "ground_state": "不", "class": "オープン"}),
        pd.Series({"開催": "1回門別1日", "course_len": "ダート1000m", "ground_state": "良", "class": "オープン"}),
    ],
)
def test_get_course_info_matches_old(result):
    assert old_race_results.get_course_info(result) == transform.get_course_info(result)


# --- CRUD（実データを使った読み込み）の比較 -----------------------------------


@pytest.fixture
def old_data_root(tmp_path, monkeypatch):
    """旧実装のname_header.DATA_PATHをtmp_path配下に向ける"""
    monkeypatch.setattr(old_race_results.name_header, "DATA_PATH", str(tmp_path) + "/")
    return tmp_path


def _copy_into_old_race_results(old_data_root, place_id, year):
    """data/race_result の既存csvを、旧実装が読むRaceResults/配下にコピーする"""
    src_path = os.path.join(paths.RACE_RESULT_DATA_PATH, PLACE_LIST[place_id - 1], f"{year}_race_results.csv")
    dst_dir = os.path.join(old_data_root, "RaceResults", PLACE_LIST[place_id - 1])
    os.makedirs(dst_dir, exist_ok=True)
    dst_path = os.path.join(dst_dir, f"{year}_race_results.csv")
    shutil.copy(src_path, dst_path)
    return dst_path


def test_get_race_results_csv_matches_old(old_data_root):
    place_id, year = 5, 2024
    _copy_into_old_race_results(old_data_root, place_id, year)

    old_df = old_race_results.get_race_results_csv(place_id, year)
    new_df = new_race_result.get_race_results_csv(place_id, year)

    assert not old_df.empty
    assert old_df.equals(new_df)


def test_get_race_results_csv_returns_empty_for_missing_file(old_data_root):
    place_id, year = 5, 1999

    old_df = old_race_results.get_race_results_csv(place_id, year)
    new_df = new_race_result.get_race_results_csv(place_id, year)

    assert old_df.empty and new_df.empty


def test_get_race_id_result_matches_old(old_data_root):
    place_id, year = 5, 2024
    race_id = "202405010101"
    _copy_into_old_race_results(old_data_root, place_id, year)

    old_df = old_race_results.get_race_id_result(race_id)
    new_df = new_race_result.get_race_id_result(race_id)

    assert not old_df.empty
    assert old_df.equals(new_df)


# --- 書き込み系の比較（tmp_path配下にそれぞれ別ディレクトリで出力して比較） -----

SAMPLE_RACE_ID = "202405019901"
SAMPLE_PLACE_ID = 5
SAMPLE_YEAR = 2024


def _sample_race_results_df():
    return pd.DataFrame(
        {
            "着順": ["1", "2"],
            "馬名": ["テストホースA", "テストホースB"],
            "course_len": ["1400m", "1400m"],
            "weather": ["晴", "晴"],
            "race_type": ["ダート", "ダート"],
            "ground_state": ["良", "良"],
            "class": ["未勝利", "未勝利"],
        },
        index=[SAMPLE_RACE_ID, SAMPLE_RACE_ID],
    )


@pytest.fixture
def old_and_new_roots(tmp_path, monkeypatch):
    """旧実装(name_header.DATA_PATH)と新実装(paths.RACE_RESULT_DATA_PATH/RACE_INFO_DATA_PATH)を
    それぞれ別のtmp_path配下に向ける

    旧実装のsave_race_results_dataset等はRaceResults/{place}/ディレクトリが
    既に存在することを前提としている（実環境ではdata/race_result/{place}/が
    既に存在する）ため、ここで事前に作成しておく。
    """
    old_root = tmp_path / "old"
    new_root = tmp_path / "new"
    place = PLACE_LIST[SAMPLE_PLACE_ID - 1]
    (old_root / "RaceResults" / place).mkdir(parents=True)
    new_root.mkdir()

    monkeypatch.setattr(old_race_results.name_header, "DATA_PATH", str(old_root) + "/")
    monkeypatch.setattr(paths, "RACE_RESULT_DATA_PATH", str(new_root / "race_result"))
    monkeypatch.setattr(paths, "RACE_INFO_DATA_PATH", str(new_root / "race_info"))

    return old_root, new_root


def test_save_race_results_dataset_matches_old(old_and_new_roots):
    old_root, new_root = old_and_new_roots

    old_race_results.save_race_results_dataset(SAMPLE_PLACE_ID, SAMPLE_YEAR, _sample_race_results_df())
    new_race_result.save_race_results_dataset(SAMPLE_PLACE_ID, SAMPLE_YEAR, _sample_race_results_df())

    place = PLACE_LIST[SAMPLE_PLACE_ID - 1]
    old_csv = old_root / "RaceResults" / place / f"{SAMPLE_YEAR}_race_results.csv"
    new_csv = new_root / "race_result" / place / f"{SAMPLE_YEAR}_race_results.csv"

    assert old_csv.is_file()
    assert new_csv.is_file()
    assert pd.read_csv(old_csv, dtype=str).equals(pd.read_csv(new_csv, dtype=str))

    assert (old_root / "RaceResults" / place / f"{SAMPLE_YEAR}_race_results.pickle").is_file()
    assert (new_root / "race_result" / place / f"{SAMPLE_YEAR}_race_results.pickle").is_file()


def test_split_race_results_by_year_matches_old(old_and_new_roots):
    old_root, new_root = old_and_new_roots

    old_race_results.save_race_results_dataset(SAMPLE_PLACE_ID, SAMPLE_YEAR, _sample_race_results_df())
    new_race_result.save_race_results_dataset(SAMPLE_PLACE_ID, SAMPLE_YEAR, _sample_race_results_df())

    old_race_results.split_race_results_by_year(SAMPLE_PLACE_ID, SAMPLE_YEAR)
    new_race_result.split_race_results_by_year(SAMPLE_PLACE_ID, SAMPLE_YEAR)

    place = PLACE_LIST[SAMPLE_PLACE_ID - 1]
    old_split = old_root / "RaceResults" / place / str(SAMPLE_YEAR) / f"{SAMPLE_RACE_ID}.csv"
    new_split = new_root / "race_result" / place / str(SAMPLE_YEAR) / f"{SAMPLE_RACE_ID}.csv"

    assert old_split.is_file()
    assert new_split.is_file()
    assert pd.read_csv(old_split, dtype=str).equals(pd.read_csv(new_split, dtype=str))


def test_convert_course_len_csv_matches_old(tmp_path):
    df = pd.DataFrame({"course_len": ["1400m", "2000.0", "1800"], "other": ["a", "b", "c"]})

    old_path = tmp_path / "old.csv"
    new_path = tmp_path / "new.csv"
    df.to_csv(old_path, index=False)
    df.to_csv(new_path, index=False)

    old_ok = old_race_results.convert_course_len_csv(str(old_path))
    new_ok = new_race_result.convert_course_len_csv(str(new_path))

    assert old_ok is True and new_ok is True
    assert pd.read_csv(old_path, dtype=str).equals(pd.read_csv(new_path, dtype=str))


def test_save_race_result_for_race_id_writes_per_race_csv(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "RACE_RESULT_DATA_PATH", str(tmp_path / "race_result"))

    race_id = "202405010101"
    df = pd.DataFrame({"着順": ["1", "2"], "馬番": ["7", "3"]}, index=[race_id, race_id])

    new_race_result.save_race_result_for_race_id(race_id, df)

    out_path = tmp_path / "race_result" / PLACE_LIST[4] / "2024" / f"{race_id}.csv"
    assert out_path.is_file()
    result = pd.read_csv(out_path, dtype=str, index_col=0)
    result.index = result.index.astype(str)
    assert result.equals(df)


def test_save_race_result_for_race_id_noop_for_empty_dataframe(tmp_path, monkeypatch):
    monkeypatch.setattr(paths, "RACE_RESULT_DATA_PATH", str(tmp_path / "race_result"))

    new_race_result.save_race_result_for_race_id("202405010101", pd.DataFrame())

    assert not (tmp_path / "race_result").exists()


def test_export_race_info_per_race_matches_old(old_and_new_roots):
    old_root, new_root = old_and_new_roots

    old_race_results.save_race_results_dataset(SAMPLE_PLACE_ID, SAMPLE_YEAR, _sample_race_results_df())
    new_race_result.save_race_results_dataset(SAMPLE_PLACE_ID, SAMPLE_YEAR, _sample_race_results_df())

    old_race_results.export_race_info_per_race(SAMPLE_PLACE_ID, SAMPLE_YEAR)
    new_race_result.export_race_info_per_race(SAMPLE_PLACE_ID, SAMPLE_YEAR)

    place = PLACE_LIST[SAMPLE_PLACE_ID - 1]
    old_info = old_root / "RaceInfo" / place / str(SAMPLE_YEAR) / f"{SAMPLE_RACE_ID}.csv"
    new_info = new_root / "race_info" / place / str(SAMPLE_YEAR) / f"{SAMPLE_RACE_ID}.csv"

    assert old_info.is_file()
    assert new_info.is_file()
    assert pd.read_csv(old_info, dtype=str).equals(pd.read_csv(new_info, dtype=str))
