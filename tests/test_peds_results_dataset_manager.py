"""src/datasets/horse, src/managers/peds_results_dataset_manager.py のテスト（オフライン）。

src/datasets/horse/transform.py の純粋関数と、
src/managers/peds_results_dataset_manager.py のCRUD・集計系関数を、
新実装単体で検証する。実データの参照系テストは
data/horse/peds_results/02_hakodate/{2019_peds.csv,2019_peds_data.csv}、
data/horse/horse_peds/2007100107.csv（確定済みのhorse_id）を用いる。
"""

import os
import shutil
from datetime import date

import pandas as pd
import pytest

from src.config import paths
from src.config.constants import PLACE_LIST
from src.datasets.horse import transform
from src.managers import peds_results_dataset_manager as new_peds_results
from src.managers import race_result_dataset_manager as new_race_result

PLACE_ID = 2
YEAR = 2019
SAMPLE_HORSE_ID = "2007100107"


# --- 純粋関数（src/datasets/horse/transform.py） ---


def test_calc_peds_placed_rate():
    df = pd.DataFrame({"着順": ["1", "1", "2", "3", "4", "5", "中止"]})

    result = transform.calc_peds_placed_rate(df.copy())

    assert result == [2, 1, 1, 3]


def test_calc_peds_data():
    df = pd.DataFrame(
        {
            "着順": ["1", "2", "3", "4", "1", "2"],
            "course_len": ["1800", "1800", "1800", "2000", "1800", "1800"],
            "class": ["未勝利", "未勝利", "未勝利", "未勝利", "1勝クラス", "未勝利"],
        }
    )

    result = transform.calc_peds_data(df.copy(), "1800", "未勝利")

    assert result.columns.tolist() == ["place", "course", "class"]
    assert result.values.tolist() == [[2, 2, 1], [2, 2, 2], [1, 1, 1], [1, 0, 0]]


def test_get_race_type_data():
    df = pd.DataFrame(
        {
            "race_type": ["芝", "ダート", "芝"],
            "ground_state": ["良", "良", "稍重"],
            "着順": ["1", "2", "3"],
        }
    )

    result = transform.get_race_type_data(df.copy(), "芝", "良")

    assert len(result) == 1
    first = result.iloc[0]
    assert first["race_type"] == "芝"
    assert first["ground_state"] == "良"
    assert first["着順"] == "1"


def test_output_results():
    df = pd.DataFrame(
        {
            "peds_0": ["父A", "父A", "父A", "父B", "父B", "父B"],
            "着順": [1, 2, 4, 1, 3, 5],
        }
    )

    result = transform.output_results(df.copy()).reset_index(drop=True)

    assert result["血統"].tolist() == ["父A", "父B"]
    assert result["1着"].tolist() == [1, 1]
    assert result["2着"].tolist() == [1, 0]
    assert result["3着"].tolist() == [0, 1]
    assert result["着外"].tolist() == [1, 1]


def test_output_results_empty_returns_empty_dataframe():
    """新実装は空入力に対して空のDataFrameを返す（aggregate_peds_resultsで.emptyを
    呼べるようにするための仕様）"""
    result = transform.output_results(pd.DataFrame())

    assert isinstance(result, pd.DataFrame)
    assert result.empty


# --- get_peds_dataset_csv / get_peds_data_dataset_csv / get_peds_dataset_from_horse_id_list ---


def test_get_peds_dataset_csv_returns_real_data():
    df = new_peds_results.get_peds_dataset_csv(PLACE_ID, YEAR)

    assert df.shape == (973, 62)
    assert df.columns[:5].tolist() == ["peds_0", "peds_1", "peds_2", "peds_3", "peds_4"]
    assert df.index[:3].tolist() == [2016104740, 2016104405, 2016104680]


def test_get_peds_data_dataset_csv_returns_real_data():
    df = new_peds_results.get_peds_data_dataset_csv(PLACE_ID, YEAR)

    assert df.shape == (1729, 67)
    assert df.columns[:5].tolist() == ["着順", "race_type", "course_len", "ground_state", "class"]


def test_get_peds_dataset_csv_returns_empty_for_missing_file():
    df = new_peds_results.get_peds_dataset_csv(PLACE_ID, 1999)

    assert df.empty


def test_get_peds_dataset_from_horse_id_list_returns_expected():
    result = new_peds_results.get_peds_dataset_from_horse_id_list([SAMPLE_HORSE_ID])

    assert result.shape == (1, 62)
    assert result.index.tolist() == [SAMPLE_HORSE_ID]
    assert result.iloc[0, :6].tolist() == [
        "アドマイヤマックス",
        "ドリーミングレイナ",
        "サンデーサイレンス",
        "ダイナシュート",
        "パラダイスクリーク",
        "エルレイナ",
    ]


# --- save_peds_dataset / update_peds_dataset（tmp_path配下に出力して検証） ---


@pytest.fixture
def new_data_root(tmp_path, monkeypatch):
    """新実装(peds_results_dataset_manager.PEDS_RESULTS_DATA_PATH)をtmp_path配下に向ける"""
    new_root = tmp_path / "peds_results"
    monkeypatch.setattr(new_peds_results, "PEDS_RESULTS_DATA_PATH", str(new_root))
    return new_root


def _sample_peds_dataset_df():
    return pd.DataFrame(
        {f"peds_{i}": [f"値A{i}", f"値B{i}"] for i in range(62)},
        index=[1111111111, 2222222222],
    )


def test_save_peds_dataset_writes_csv_and_pickle(new_data_root):
    place = PLACE_LIST[PLACE_ID - 1]

    new_peds_results.save_peds_dataset(_sample_peds_dataset_df(), PLACE_ID, YEAR)

    csv_path = new_data_root / place / f"{YEAR}_peds.csv"
    pickle_path = new_data_root / place / f"{YEAR}_peds.pickle"

    assert csv_path.is_file()
    assert pickle_path.is_file()

    df = pd.read_csv(csv_path, dtype=str, index_col=0)
    assert df.shape == (2, 62)
    assert df.index.tolist() == [1111111111, 2222222222]


def test_update_peds_dataset_writes_peds_csv(new_data_root, monkeypatch):
    place = PLACE_LIST[PLACE_ID - 1]
    day = date(2024, 6, 10)

    monkeypatch.setattr(new_peds_results.race_schedule_dataset_manager, "get_past_weekly_id", lambda *a, **k: ["dummy"])
    monkeypatch.setattr(
        new_peds_results.past_performance_dataset_manager,
        "get_horse_id_list_from_race_id_list",
        lambda *a, **k: [SAMPLE_HORSE_ID],
    )

    new_peds_results.update_peds_dataset(PLACE_ID, day)

    out_csv = new_data_root / place / f"{day.year}_peds.csv"
    assert out_csv.is_file()

    df = pd.read_csv(out_csv, dtype=str, index_col=0)
    assert df.shape == (1, 62)
    assert df.index.tolist() == [int(SAMPLE_HORSE_ID)]
    assert df.iloc[0, :4].tolist() == [
        "アドマイヤマックス",
        "ドリーミングレイナ",
        "サンデーサイレンス",
        "ダイナシュート",
    ]


# --- merge_pedsdata_with_race_results（実データの1レース分を使用） -------------


@pytest.fixture
def merge_data_root(tmp_path, monkeypatch):
    """新実装(paths.RACE_RESULT_DATA_PATH, peds_results_dataset_manager.PEDS_RESULTS_DATA_PATH)を
    tmp_path配下に向け、実データの1レース分のrace_resultsとpeds.csvを配置する"""
    place = PLACE_LIST[PLACE_ID - 1]

    df_course_full = new_race_result.get_race_results_csv(PLACE_ID, YEAR)
    race_id = df_course_full.index[0]
    df_one_race = df_course_full[df_course_full.index == race_id]

    race_dir = tmp_path / "race_result" / place
    peds_dir = tmp_path / "horse" / "peds_results" / place
    race_dir.mkdir(parents=True)
    peds_dir.mkdir(parents=True)
    df_one_race.to_csv(race_dir / f"{YEAR}_race_results.csv")
    shutil.copy(
        os.path.join(paths.HORSE_DATA_PATH, "peds_results", place, f"{YEAR}_peds.csv"),
        peds_dir / f"{YEAR}_peds.csv",
    )

    monkeypatch.setattr(paths, "RACE_RESULT_DATA_PATH", str(tmp_path / "race_result"))
    monkeypatch.setattr(new_peds_results, "PEDS_RESULTS_DATA_PATH", str(tmp_path / "horse" / "peds_results"))

    return peds_dir


def test_merge_pedsdata_with_race_results_writes_peds_data(merge_data_root):
    new_peds_results.merge_pedsdata_with_race_results(PLACE_ID, YEAR)

    out_csv = merge_data_root / f"{YEAR}_peds_data.csv"
    assert out_csv.is_file()

    df = pd.read_csv(out_csv, dtype=str, index_col=0)
    assert df.shape == (16, 67)
    assert df.columns[:5].tolist() == ["着順", "race_type", "course_len", "ground_state", "class"]


# --- peds_index（実データの2019_peds_dataを使用） -------------------------------


def test_peds_index_returns_expected():
    df_2019 = new_peds_results.get_peds_data_dataset_csv(PLACE_ID, 2019)
    row = df_2019.iloc[0]
    father = row["peds_0"]
    mother_father = row["peds_4"]
    course_info = [PLACE_ID, row["race_type"], row["course_len"], row["ground_state"], row["class"]]

    result = new_peds_results.peds_index(father, mother_father, course_info, 2020)

    assert result.shape == (12, 3)
    assert result.columns.tolist() == ["place", "course", "class"]


# --- aggregate_peds_results / aggregate_total_peds_results（実データを使用） ---


@pytest.fixture
def aggregate_data_root(tmp_path, monkeypatch):
    """新実装(PEDS_RESULTS_DATA_PATH)をtmp_path配下に向け、実データの2019_peds_data.csvを配置する"""
    place = PLACE_LIST[PLACE_ID - 1]
    new_root = tmp_path / "peds_results"
    place_dir = new_root / place
    place_dir.mkdir(parents=True)

    src = os.path.join(paths.HORSE_DATA_PATH, "peds_results", place, f"{YEAR}_peds_data.csv")
    shutil.copy(src, place_dir / f"{YEAR}_peds_data.csv")

    monkeypatch.setattr(new_peds_results, "PEDS_RESULTS_DATA_PATH", str(new_root))

    return place_dir


def test_aggregate_peds_results_writes_per_class_csvs(aggregate_data_root):
    new_peds_results.aggregate_peds_results(PLACE_ID, YEAR)

    year_dir = aggregate_data_root / str(YEAR)
    files = sorted(p.name for p in year_dir.iterdir())

    assert len(files) == 25
    assert "ダート_1000m_all.csv" in files


def test_aggregate_total_peds_results_writes_total_csvs(aggregate_data_root):
    new_peds_results.aggregate_peds_results(PLACE_ID, YEAR)
    new_peds_results.aggregate_total_peds_results(place_id=PLACE_ID, start_year=YEAR, end_year=YEAR)

    total_dir = aggregate_data_root / "Total"
    files = sorted(p.name for p in total_dir.iterdir())

    assert len(files) == 25

    df = pd.read_csv(total_dir / "ダート_1000m_all.csv")
    assert df.shape == (185, 6)
    assert df.columns.tolist() == ["クラス", "血統", "1着", "2着", "3着", "着外"]


# --- get_total_peds_results_csv / get_annual_peds_results_csv（スモークテスト） -------


def test_get_total_peds_results_csv_smoke():
    df = new_peds_results.get_total_peds_results_csv(PLACE_ID, "ダート", 1000, "良")
    assert not df.empty


def test_get_annual_peds_results_csv_smoke():
    df = new_peds_results.get_annual_peds_results_csv(PLACE_ID, YEAR, "ダート", 1000, "良")
    assert not df.empty


def test_get_total_peds_results_csv_empty_when_missing():
    df = new_peds_results.get_total_peds_results_csv(PLACE_ID, "存在しない", 9999, "良")
    assert df.empty
