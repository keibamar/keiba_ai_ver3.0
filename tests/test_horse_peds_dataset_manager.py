"""src/datasets/horse, src/managers/horse_peds_dataset_manager.py のテスト（オフライン）。

src/datasets/horse/transform.py の純粋関数と、
src/managers/horse_peds_dataset_manager.py のCRUD系関数を、新実装単体で検証する。
実データの参照系テストは data/horse/horse_peds/2007100107.csv（確定済みのhorse_id）を用いる。
"""

import pandas as pd
import pytest

from src.config import paths
from src.datasets.horse import transform
from src.managers import horse_peds_dataset_manager as new_horse_peds

SAMPLE_HORSE_ID = "2007100107"


# --- 純粋関数（src/datasets/horse/transform.py） ---


@pytest.mark.parametrize(
    "s, expected",
    [
        ("ディープインパクト", "ディープインパクト"),
        ("Sunday Silence(USA)", "Sunday Silence"),
        ("Kingmambo", "Kingmambo"),
        ("  テスト　", "テスト"),
        ("ノーザンダンサー・ジュニア", "ノーザンダンサー・ジュニア"),
        ("123", "123"),
        ("Wind In Her Hair(IRE)", "Wind In Her Hair"),
    ],
)
def test_normalize_horse_name_strings(s, expected):
    assert transform.normalize_horse_name_strings(s) == expected


def test_delete_invalid_strings():
    series = pd.Series(["ディープインパクト 2002年生", "Sunday Silence 1986年生", "Wind In Her Hair"])

    result = transform.delete_invalid_strings(series)

    assert result.tolist() == ["ディープインパクト", "Sunday Silence", "Wind In Her Hair"]


# --- CRUD（実データを使った読み込み） -----------------------------------


def test_get_horse_peds_csv_returns_real_data():
    df = new_horse_peds.get_horse_peds_csv(SAMPLE_HORSE_ID)

    print(f"\n--- get_horse_peds_csv(horse_id={SAMPLE_HORSE_ID}) ---")
    print(f"  shape: {df.shape}")
    print(f"  先頭6行（直近3代の血統）: {df[SAMPLE_HORSE_ID].iloc[:6].tolist()}")

    assert df.shape == (62, 1)
    assert df.columns.tolist() == [SAMPLE_HORSE_ID]
    assert df[SAMPLE_HORSE_ID].iloc[:6].tolist() == [
        "アドマイヤマックス",
        "ドリーミングレイナ",
        "サンデーサイレンス",
        "ダイナシュート",
        "パラダイスクリーク",
        "エルレイナ",
    ]


def test_get_horse_peds_csv_returns_empty_for_missing_file():
    df = new_horse_peds.get_horse_peds_csv("9999999999")

    assert df.empty


def test_get_horse_peds_dataset_returns_existing_csv():
    csv_result = new_horse_peds.get_horse_peds_csv(SAMPLE_HORSE_ID)
    dataset_result = new_horse_peds.get_horse_peds_dataset(SAMPLE_HORSE_ID)

    assert not csv_result.empty
    assert csv_result.equals(dataset_result)


def test_get_peds_info_returns_expected():
    result = new_horse_peds.get_peds_info(SAMPLE_HORSE_ID)

    assert result == ["アドマイヤマックス", "パラダイスクリーク"]


# --- 書き込み系（tmp_path配下に出力して検証） -----------------------------

SAMPLE_NEW_HORSE_ID = "9999999999"


def _sample_peds_df():
    return pd.DataFrame(
        [["父名", "母名", "父父名", "父母名", "母父名"]],
        columns=[f"peds_{i}" for i in range(5)],
        index=[SAMPLE_NEW_HORSE_ID],
    )


@pytest.fixture
def new_data_root(tmp_path, monkeypatch):
    """新実装(paths.HORSE_DATA_PATH/HORSE_PEDS_DATA_PATH)をtmp_path配下に向ける"""
    new_root = tmp_path / "horse"
    monkeypatch.setattr(paths, "HORSE_DATA_PATH", str(new_root))
    monkeypatch.setattr(new_horse_peds, "HORSE_PEDS_DATA_PATH", str(new_root / "horse_peds"))
    return new_root


def test_save_horse_peds_dataset_writes_csv_and_pickle(new_data_root):
    new_horse_peds.save_horse_peds_dataset(SAMPLE_NEW_HORSE_ID, _sample_peds_df())

    csv_path = new_data_root / "horse_peds" / f"{SAMPLE_NEW_HORSE_ID}.csv"
    pickle_path = new_data_root / "horse_peds" / f"{SAMPLE_NEW_HORSE_ID}.pickle"

    assert csv_path.is_file()
    assert pickle_path.is_file()

    df = pd.read_csv(csv_path, dtype=str, index_col=0)
    assert df.loc[int(SAMPLE_NEW_HORSE_ID)].tolist() == ["父名", "母名", "父父名", "父母名", "母父名"]


def test_is_horse_peds_dataset(new_data_root):
    assert new_horse_peds.is_horse_peds_dataset(SAMPLE_NEW_HORSE_ID) is False

    new_horse_peds.save_horse_peds_dataset(SAMPLE_NEW_HORSE_ID, _sample_peds_df())

    assert new_horse_peds.is_horse_peds_dataset(SAMPLE_NEW_HORSE_ID) is True


def test_make_horse_peds_datasets_from_horse_id_list_skips_existing(new_data_root):
    """既にデータが存在する場合は何もしない（スクレイピングしない）ことを確認"""
    new_horse_peds.save_horse_peds_dataset(SAMPLE_NEW_HORSE_ID, _sample_peds_df())

    # 既存データがあるのでスクレイピングは行われず、例外も発生しない
    new_horse_peds.make_horse_peds_datasets_from_horse_id_list([SAMPLE_NEW_HORSE_ID])

    csv_path = new_data_root / "horse_peds" / f"{SAMPLE_NEW_HORSE_ID}.csv"
    df = pd.read_csv(csv_path, dtype=str, index_col=0)
    assert df.loc[int(SAMPLE_NEW_HORSE_ID)].tolist() == ["父名", "母名", "父父名", "父母名", "母父名"]
