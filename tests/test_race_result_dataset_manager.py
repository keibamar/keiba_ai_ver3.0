"""src/datasets/race_result, src/managers/race_result_dataset_manager.py のテスト（オフライン）。

src/datasets/race_result/transform.py の純粋関数と、
src/managers/race_result_dataset_manager.py のCRUD・変換系関数を、
新実装単体で検証する。実データの参照系テストは
data/race_result/05_tokyo/2024_race_results.csv（確定済みのrace_id）を用いる。
"""

import numpy as np
import pandas as pd
import pytest

from src.config import paths
from src.config.constants import PLACE_LIST
from src.datasets.race_result import transform
from src.managers import race_result_dataset_manager as new_race_result


# --- 純粋関数（src/datasets/race_result/transform.py） ---


def test_clean_columns_renames_duplicate_columns():
    df = pd.DataFrame([[1, 2, 3]], columns=["a", "a", "b"])

    result = transform.clean_columns(df)

    assert result.columns.tolist() == ["a_1", "a_2", "b"]


def test_format_race_results_dataframe_drops_duplicates():
    df = pd.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})

    result = transform.format_race_results_dataframe(df)

    assert len(result) == 2
    assert result["a"].tolist() == ["1", "2"]


def test_format_race_results_dataframe_keeps_blank_for_missing_values():
    # astype(str)を先にすると欠損値がそのまま文字列"nan"になってCSVに保存されて
    # しまうバグがあったため、空文字列のまま保たれることを確認する
    df = pd.DataFrame({"着差": ["1", np.nan], "通過": [np.nan, "3-3"]})

    result = transform.format_race_results_dataframe(df)

    assert result["着差"].tolist() == ["1", ""]
    assert result["通過"].tolist() == ["", "3-3"]
    assert not (result == "nan").any().any()


@pytest.mark.parametrize(
    "val, expected",
    [
        ("1400m", 1400),
        ("1400.0", 1400),
        ("1400", 1400),
        ("", ""),
        ("abc", "abc"),
        (np.nan, np.nan),
    ],
)
def test_to_int(val, expected):
    result = transform.to_int(val)

    if pd.isna(expected):
        assert pd.isna(result)
    else:
        assert result == expected


@pytest.mark.parametrize(
    "result, expected",
    [
        (
            pd.Series({"開催": "5回東京2日", "course_len": "ダート1400m", "ground_state": "稍", "class": "未勝利"}),
            [5, "ダート", "1400", "稍重", "未勝利"],
        ),
        (
            pd.Series({"開催": "1回中山1日", "course_len": "芝2000m", "ground_state": "良", "class": "オープン"}),
            [6, "芝", "2000", "良", "オープン"],
        ),
        (
            pd.Series({"開催": "1回中山1日", "course_len": "障害3000m", "ground_state": "不", "class": "オープン"}),
            [6, "障害", "3000", "不良", "オープン"],
        ),
        (
            pd.Series({"開催": "1回門別1日", "course_len": "ダート1000m", "ground_state": "良", "class": "オープン"}),
            [-1, " ", " ", " ", " "],
        ),
    ],
)
def test_get_course_info(result, expected):
    assert transform.get_course_info(result) == expected


# --- CRUD（実データを使った読み込み） -----------------------------------

SAMPLE_PLACE_ID = 5
SAMPLE_YEAR = 2024
FIXED_RACE_ID = "202405010101"


def test_get_race_results_csv_returns_real_data():
    df = new_race_result.get_race_results_csv(SAMPLE_PLACE_ID, SAMPLE_YEAR)

    print(f"\n--- get_race_results_csv(place_id={SAMPLE_PLACE_ID}, year={SAMPLE_YEAR}) ---")
    print(f"  shape: {df.shape}")
    print(f"  columns: {df.columns.tolist()}")

    assert not df.empty
    for col in ["着順", "枠番", "馬番", "馬名", "course_len", "race_type", "ground_state", "class"]:
        assert col in df.columns

    df.index = df.index.astype(str)
    race_rows = df.loc[FIXED_RACE_ID:FIXED_RACE_ID]

    print(f"  race_id={FIXED_RACE_ID} の行数: {len(race_rows)}")
    print(race_rows[["着順", "馬番", "馬名"]].to_string())

    assert len(race_rows) == 16


def test_get_race_results_csv_returns_empty_for_missing_file():
    df = new_race_result.get_race_results_csv(SAMPLE_PLACE_ID, 1999)

    assert df.empty


def test_get_race_id_result_falls_back_to_per_race_csv_when_not_in_combined_file(tmp_path, monkeypatch):
    # save_race_result_for_race_id（当日のライブ取得）はper-race CSVにしか保存せず、
    # {year}_race_results.csv（週次バッチ更新）への反映は別。週次更新前でも
    # 出馬表ページに結果が反映されるよう、per-race CSVへフォールバックできることを確認する
    monkeypatch.setattr(paths, "RACE_RESULT_DATA_PATH", str(tmp_path))
    race_id = "202405019999"

    new_race_result.save_race_result_for_race_id(
        race_id, pd.DataFrame({"着順": ["1", "2"], "馬名": ["A", "B"]}, index=[race_id, race_id]),
    )

    df = new_race_result.get_race_id_result(race_id)

    assert df["馬名"].tolist() == ["A", "B"]


def test_get_race_id_result_returns_expected_race():
    df = new_race_result.get_race_id_result(FIXED_RACE_ID)

    assert len(df) == 16
    assert (df["course_len"] == "1400").all()
    assert (df["race_type"] == "ダート").all()
    assert (df["ground_state"] == "良").all()
    assert (df["class"] == "未勝利").all()

    winner = df[df["着順"] == "1"].iloc[0]
    assert winner["馬名"] == "エースアビリティ"
    assert winner["枠番"] == "4"
    assert winner["馬番"] == "7"


# --- 書き込み系（tmp_path配下に出力して検証） -----------------------------

SAMPLE_WRITE_RACE_ID = "202405019901"


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
        index=[SAMPLE_WRITE_RACE_ID, SAMPLE_WRITE_RACE_ID],
    )


@pytest.fixture
def new_data_root(tmp_path, monkeypatch):
    """新実装(paths.RACE_RESULT_DATA_PATH/RACE_INFO_DATA_PATH)をtmp_path配下に向ける"""
    monkeypatch.setattr(paths, "RACE_RESULT_DATA_PATH", str(tmp_path / "race_result"))
    monkeypatch.setattr(paths, "RACE_INFO_DATA_PATH", str(tmp_path / "race_info"))
    return tmp_path


def test_save_race_results_dataset_writes_csv_and_pickle(new_data_root):
    new_race_result.save_race_results_dataset(SAMPLE_PLACE_ID, SAMPLE_YEAR, _sample_race_results_df())

    place = PLACE_LIST[SAMPLE_PLACE_ID - 1]
    csv_path = new_data_root / "race_result" / place / f"{SAMPLE_YEAR}_race_results.csv"
    pickle_path = new_data_root / "race_result" / place / f"{SAMPLE_YEAR}_race_results.pickle"

    assert csv_path.is_file()
    assert pickle_path.is_file()

    df = pd.read_csv(csv_path, dtype=str, index_col=0)
    assert len(df) == 2
    assert df["着順"].tolist() == ["1", "2"]
    assert df["馬名"].tolist() == ["テストホースA", "テストホースB"]


def test_split_race_results_by_year_writes_per_race_csv(new_data_root):
    new_race_result.save_race_results_dataset(SAMPLE_PLACE_ID, SAMPLE_YEAR, _sample_race_results_df())
    new_race_result.split_race_results_by_year(SAMPLE_PLACE_ID, SAMPLE_YEAR)

    place = PLACE_LIST[SAMPLE_PLACE_ID - 1]
    split_path = new_data_root / "race_result" / place / str(SAMPLE_YEAR) / f"{SAMPLE_WRITE_RACE_ID}.csv"

    assert split_path.is_file()
    df = pd.read_csv(split_path, dtype=str, index_col=0)
    assert len(df) == 2
    assert df["馬名"].tolist() == ["テストホースA", "テストホースB"]


def test_convert_course_len_csv_converts_to_int(tmp_path):
    df = pd.DataFrame({"course_len": ["1400m", "2000.0", "1800"], "other": ["a", "b", "c"]})
    csv_path = tmp_path / "course_len.csv"
    df.to_csv(csv_path, index=False)

    ok = new_race_result.convert_course_len_csv(str(csv_path))

    assert ok is True
    result = pd.read_csv(csv_path, dtype=str)
    assert result["course_len"].tolist() == ["1400", "2000", "1800"]


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


def test_export_race_info_per_race_writes_race_info_csv(new_data_root):
    new_race_result.save_race_results_dataset(SAMPLE_PLACE_ID, SAMPLE_YEAR, _sample_race_results_df())
    new_race_result.export_race_info_per_race(SAMPLE_PLACE_ID, SAMPLE_YEAR)

    place = PLACE_LIST[SAMPLE_PLACE_ID - 1]
    info_path = new_data_root / "race_info" / place / str(SAMPLE_YEAR) / f"{SAMPLE_WRITE_RACE_ID}.csv"

    assert info_path.is_file()
    record = pd.read_csv(info_path, index_col=0).iloc[0]
    assert record["race_type"] == "ダート"
    assert record["course_len"] == "1400m"
    assert record["weather"] == "晴"
    assert record["ground_state"] == "良"
    assert record["class"] == "未勝利"
