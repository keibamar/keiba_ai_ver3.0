"""src/datasets/horse, src/managers/horse_peds_dataset_manager.py の出力が
旧 src/legacy_datasets/horse_peds.py と一致することを確認するテスト（オフライン）。

旧実装はパスとして name_header.DATA_PATH（ver2.0のdata/）を参照するため、
旧実装の比較対象テストでは tmp_path 配下に "HorsePeds" フォルダを作って
name_header.DATA_PATH をそこに向ける。新実装は src.config.paths の
HORSE_DATA_PATH を同様に tmp_path 配下に向ける。
"""

import os
import shutil

import pandas as pd
import pytest

from src.config import paths
from src.datasets.horse import transform
from src.legacy_datasets import horse_peds as old_horse_peds
from src.managers import horse_peds_dataset_manager as new_horse_peds

SAMPLE_HORSE_ID = "2007100107"


# --- 純粋関数の比較（旧 horse_peds.py vs 新 src/datasets/horse/transform.py） ---


@pytest.mark.parametrize(
    "s",
    [
        "ディープインパクト",
        "Sunday Silence(USA)",
        "Kingmambo",
        "  テスト　",
        "ノーザンダンサー・ジュニア",
        "123",
        "Wind In Her Hair(IRE)",
    ],
)
def test_normalize_horse_name_strings_matches_old(s):
    assert old_horse_peds.normalize_horse_name_strings(s) == transform.normalize_horse_name_strings(s)


def test_delete_invalid_strings_matches_old():
    series_old = pd.Series(["ディープインパクト 2002年生", "Sunday Silence 1986年生", "Wind In Her Hair"])
    series_new = series_old.copy()

    result_old = old_horse_peds.delete_invalid_strings(series_old)
    result_new = transform.delete_invalid_strings(series_new)

    assert result_old.equals(result_new)


# --- CRUD（実データを使った読み込み）の比較 -----------------------------------


@pytest.fixture
def old_data_root(tmp_path, monkeypatch):
    """旧実装のname_header.DATA_PATHをtmp_path配下に向ける"""
    monkeypatch.setattr(old_horse_peds.name_header, "DATA_PATH", str(tmp_path) + "/")
    return tmp_path


def _copy_into_old_horse_peds(old_data_root, horse_id):
    """data/horse/horse_peds の既存csvを、旧実装が読むHorsePeds/配下にコピーする"""
    src_path = os.path.join(paths.HORSE_DATA_PATH, "horse_peds", f"{horse_id}.csv")
    dst_dir = os.path.join(old_data_root, "HorsePeds")
    os.makedirs(dst_dir, exist_ok=True)
    dst_path = os.path.join(dst_dir, f"{horse_id}.csv")
    shutil.copy(src_path, dst_path)
    return dst_path


def test_get_horse_peds_csv_matches_old(old_data_root):
    _copy_into_old_horse_peds(old_data_root, SAMPLE_HORSE_ID)

    old_df = old_horse_peds.get_horse_peds_csv(SAMPLE_HORSE_ID)
    new_df = new_horse_peds.get_horse_peds_csv(SAMPLE_HORSE_ID)

    assert not old_df.empty
    assert old_df.equals(new_df)


def test_get_horse_peds_csv_returns_empty_for_missing_file(old_data_root):
    old_df = old_horse_peds.get_horse_peds_csv("9999999999")
    new_df = new_horse_peds.get_horse_peds_csv("9999999999")

    assert old_df.empty and new_df.empty


def test_get_peds_info_matches_old(old_data_root):
    _copy_into_old_horse_peds(old_data_root, SAMPLE_HORSE_ID)

    old_result = old_horse_peds.get_peds_info(SAMPLE_HORSE_ID)
    new_result = new_horse_peds.get_peds_info(SAMPLE_HORSE_ID)

    assert old_result == new_result
    assert not pd.isna(old_result[0])


# --- 書き込み系の比較（tmp_path配下にそれぞれ別ディレクトリで出力して比較） -----

SAMPLE_NEW_HORSE_ID = "9999999999"


def _sample_peds_df():
    return pd.DataFrame(
        [["父名", "母名", "父父名", "父母名", "母父名"]],
        columns=[f"peds_{i}" for i in range(5)],
        index=[SAMPLE_NEW_HORSE_ID],
    )


@pytest.fixture
def old_and_new_roots(tmp_path, monkeypatch):
    """旧実装(name_header.DATA_PATH)と新実装(paths.HORSE_DATA_PATH)を
    それぞれ別のtmp_path配下に向ける
    """
    old_root = tmp_path / "old"
    new_root = tmp_path / "new"
    (old_root / "HorsePeds").mkdir(parents=True)
    (new_root / "horse").mkdir(parents=True)

    monkeypatch.setattr(old_horse_peds.name_header, "DATA_PATH", str(old_root) + "/")
    monkeypatch.setattr(paths, "HORSE_DATA_PATH", str(new_root / "horse"))
    monkeypatch.setattr(new_horse_peds, "HORSE_PEDS_DATA_PATH", str(new_root / "horse" / "horse_peds"))

    return old_root, new_root


def test_save_horse_peds_dataset_matches_old(old_and_new_roots):
    old_root, new_root = old_and_new_roots

    old_horse_peds.save_horse_peds_dataset(SAMPLE_NEW_HORSE_ID, _sample_peds_df())
    new_horse_peds.save_horse_peds_dataset(SAMPLE_NEW_HORSE_ID, _sample_peds_df())

    old_csv = old_root / "HorsePeds" / f"{SAMPLE_NEW_HORSE_ID}.csv"
    new_csv = new_root / "horse" / "horse_peds" / f"{SAMPLE_NEW_HORSE_ID}.csv"

    assert old_csv.is_file()
    assert new_csv.is_file()
    assert pd.read_csv(old_csv, dtype=str).equals(pd.read_csv(new_csv, dtype=str))


def test_is_horse_peds_dataset_matches_old(old_and_new_roots):
    assert old_horse_peds.is_horse_peds_dataset(SAMPLE_NEW_HORSE_ID) is False
    assert new_horse_peds.is_horse_peds_dataset(SAMPLE_NEW_HORSE_ID) is False

    old_horse_peds.save_horse_peds_dataset(SAMPLE_NEW_HORSE_ID, _sample_peds_df())
    new_horse_peds.save_horse_peds_dataset(SAMPLE_NEW_HORSE_ID, _sample_peds_df())

    assert old_horse_peds.is_horse_peds_dataset(SAMPLE_NEW_HORSE_ID) is True
    assert new_horse_peds.is_horse_peds_dataset(SAMPLE_NEW_HORSE_ID) is True


def test_make_horse_peds_datasets_from_horse_id_list_skips_existing(old_and_new_roots):
    """既にデータが存在する場合は何もしない（スクレイピングしない）ことを確認"""
    old_root, new_root = old_and_new_roots

    old_horse_peds.save_horse_peds_dataset(SAMPLE_NEW_HORSE_ID, _sample_peds_df())
    new_horse_peds.save_horse_peds_dataset(SAMPLE_NEW_HORSE_ID, _sample_peds_df())

    # 既存データがあるのでスクレイピングは行われず、例外も発生しない
    old_horse_peds.make_horse_peds_datasets_from_horse_id_list([SAMPLE_NEW_HORSE_ID])
    new_horse_peds.make_horse_peds_datasets_from_horse_id_list([SAMPLE_NEW_HORSE_ID])

    old_csv = old_root / "HorsePeds" / f"{SAMPLE_NEW_HORSE_ID}.csv"
    new_csv = new_root / "horse" / "horse_peds" / f"{SAMPLE_NEW_HORSE_ID}.csv"
    assert pd.read_csv(old_csv, dtype=str).equals(pd.read_csv(new_csv, dtype=str))
