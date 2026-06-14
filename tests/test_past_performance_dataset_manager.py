"""src/datasets/horse, src/managers/past_performance_dataset_manager.py の出力が
旧 src/legacy_datasets/past_performance.py（race_resultsベースの実装）と
一致することを確認するテスト（オフライン）。

旧実装はパスとして name_header.DATA_PATH（ver2.0のdata/）を参照するため、
旧実装の比較対象テストでは tmp_path 配下に "RaceResults"/"PastPerformance" フォルダを
作って name_header.DATA_PATH をそこに向ける。新実装は src.config.paths の
RACE_RESULT_DATA_PATH / past_performance_dataset_manager.PAST_PERFORMANCE_DATA_PATH を
同様に向ける。
"""

import os
import shutil
from glob import glob

import pandas as pd
import pytest

from src.config import paths
from src.datasets.horse import transform
from src.legacy_datasets import past_performance as old_past_performance
from src.managers import past_performance_dataset_manager as new_past_performance

SAMPLE_HORSE_ID = "2007100107"
SAMPLE_RACE_ID = "201902010101"  # 02_hakodate, 2019年, 16頭立て


# --- 純粋関数の比較（旧 past_performance.py vs 新 src/datasets/horse/transform.py） ---


@pytest.mark.parametrize(
    "text",
    ["３勝クラス", "２勝クラス", "未勝利", "新馬", "ｵｰﾌﾟﾝ", "1勝　クラス", "ｵｰﾌﾟﾝｸﾗｽ"],
)
def test_normalize_class_text_matches_old(text):
    assert old_past_performance.normalize_class_text(text) == transform.normalize_class_text(text)


@pytest.mark.parametrize(
    "text",
    [
        "ムーンライトH(3勝クラス)",
        "ジューンステークス（2勝クラス）",
        "サラブレッドチャレンジカップ",
        "  末尾に空白あり  ",
    ],
)
def test_clean_race_name_matches_old(text):
    assert old_past_performance.clean_race_name(text) == transform.clean_race_name(text)


def test_get_past_race_id_matches_old():
    df = pd.DataFrame({"race_id": ["202301010101", "202301010102"], "着順": ["1", "2"]})

    old_result = old_past_performance.get_past_race_id(df.copy())
    new_result = transform.get_past_race_id(df.copy())

    assert old_result == new_result


def test_reset_horse_result_matches_old():
    df = pd.DataFrame(
        {
            "race_id": ["202301010103", "202301010102", "202301010101"],
            "着順": ["1", "2", "3"],
        }
    )

    old_result = old_past_performance.reset_horse_result(df.copy(), "202301010102")
    new_result = transform.reset_horse_result(df.copy(), "202301010102")

    assert old_result.equals(new_result)


# --- normalize_past_performance_format ----------------------------------------


def test_normalize_past_performance_format_new_format_matches_old():
    """既存のpast_performance（新フォーマット）を読み込み、軽微な整形のみが
    行われるパスを比較する"""
    df = pd.read_csv(
        os.path.join(paths.HORSE_DATA_PATH, "past_performance", f"{SAMPLE_HORSE_ID}.csv"), dtype=str
    )

    old_result = old_past_performance.normalize_past_performance_format(df.copy())
    new_result = transform.normalize_past_performance_format(df.copy())

    assert not old_result.empty
    assert old_result.equals(new_result)


def test_normalize_past_performance_format_old_format_matches_old():
    """旧フォーマットのpast_performanceから新フォーマットへの変換パスを比較する"""
    df = pd.DataFrame(
        [
            {
                "日付": "2023年5月7日",
                "開催": "3中京1",
                "天 気": "晴",
                "R": "8",
                "レース名": "ムーンライトH(3勝クラス)",
                "頭 数": "12",
                "枠 番": "3",
                "馬 番": "5",
                "オ ッ ズ": "4.5",
                "人 気": "2",
                "着 順": "1",
                "騎手": "サンプル騎手",
                "斤 量": "56",
                "距離": "ダ1800",
                "馬 場": "良",
                "タイム": "1:48.5",
                "着差": "-0.2",
                "通過": "3-3-2-1",
                "上り": "35.5",
                "馬体重": "480(+2)",
                "勝ち馬 (2着馬)": "(テスト馬)",
                "ペース": "35.5-36.0",
            }
        ]
    )

    old_result = old_past_performance.normalize_past_performance_format(df.copy())
    new_result = transform.normalize_past_performance_format(df.copy())

    assert not old_result.empty
    assert old_result.reset_index(drop=True).equals(new_result.reset_index(drop=True))


def test_normalize_past_performance_format_empty_input():
    old_result = old_past_performance.normalize_past_performance_format(pd.DataFrame())
    new_result = transform.normalize_past_performance_format(pd.DataFrame())

    assert old_result.empty and new_result.empty


# --- get_horse_id_from_race_id / get_horse_id_list_from_race_id_list -----------


@pytest.fixture
def old_data_root(tmp_path, monkeypatch):
    """旧実装のname_header.DATA_PATHをtmp_path配下に向ける"""
    monkeypatch.setattr(old_past_performance.name_header, "DATA_PATH", str(tmp_path) + "/")
    return tmp_path


def test_get_horse_id_from_race_id_matches_old(old_data_root):
    place = "02_hakodate"
    src = os.path.join(paths.RACE_RESULT_DATA_PATH, place, "2019_race_results.csv")
    dst_dir = old_data_root / "RaceResults" / place
    dst_dir.mkdir(parents=True)
    shutil.copy(src, dst_dir / "2019_race_results.csv")

    old_result = old_past_performance.get_horse_id_from_race_id(SAMPLE_RACE_ID)
    new_result = new_past_performance.get_horse_id_from_race_id(SAMPLE_RACE_ID)

    assert old_result == new_result
    assert len(old_result) == 16


def test_get_horse_id_list_from_race_id_list_matches_old(old_data_root):
    place = "02_hakodate"
    src = os.path.join(paths.RACE_RESULT_DATA_PATH, place, "2019_race_results.csv")
    dst_dir = old_data_root / "RaceResults" / place
    dst_dir.mkdir(parents=True)
    shutil.copy(src, dst_dir / "2019_race_results.csv")

    old_result = old_past_performance.get_horse_id_list_from_race_id_list([SAMPLE_RACE_ID, SAMPLE_RACE_ID])
    new_result = new_past_performance.get_horse_id_list_from_race_id_list([SAMPLE_RACE_ID, SAMPLE_RACE_ID])

    assert old_result == new_result
    assert len(old_result) == 32


# --- get_past_performance_dataset / save_past_performance_dataset (CRUD) ------


@pytest.fixture
def old_and_new_roots(tmp_path, monkeypatch):
    """旧実装(name_header.DATA_PATH)と新実装(past_performance_dataset_manager.PAST_PERFORMANCE_DATA_PATH)を
    それぞれ別のtmp_path配下に向ける
    """
    old_root = tmp_path / "old"
    new_root = tmp_path / "new"
    (old_root / "PastPerformance").mkdir(parents=True)
    (new_root / "past_performance").mkdir(parents=True)

    monkeypatch.setattr(old_past_performance.name_header, "DATA_PATH", str(old_root) + "/")
    monkeypatch.setattr(new_past_performance, "PAST_PERFORMANCE_DATA_PATH", str(new_root / "past_performance"))

    return old_root, new_root


def test_get_past_performance_dataset_matches_old(old_and_new_roots):
    old_root, new_root = old_and_new_roots

    src = os.path.join(paths.HORSE_DATA_PATH, "past_performance", f"{SAMPLE_HORSE_ID}.csv")
    shutil.copy(src, old_root / "PastPerformance" / f"{SAMPLE_HORSE_ID}.csv")
    shutil.copy(src, new_root / "past_performance" / f"{SAMPLE_HORSE_ID}.csv")

    old_df = old_past_performance.get_past_performance_dataset(SAMPLE_HORSE_ID)
    new_df = new_past_performance.get_past_performance_dataset(SAMPLE_HORSE_ID)

    assert not old_df.empty
    assert old_df.equals(new_df)


def test_get_past_performance_dataset_returns_empty_for_missing_file(old_and_new_roots):
    old_df = old_past_performance.get_past_performance_dataset("9999999999")
    new_df = new_past_performance.get_past_performance_dataset("9999999999")

    assert old_df.empty and new_df.empty


def test_save_past_performance_dataset_matches_old(old_and_new_roots):
    old_root, new_root = old_and_new_roots

    sample_df = pd.DataFrame(
        {"race_id": ["202301010101"], "着順": ["1"], "騎手": ["サンプル騎手"]}
    )

    old_past_performance.save_past_performance_dataset("8888888888", sample_df.copy())
    new_past_performance.save_past_performance_dataset("8888888888", sample_df.copy())

    old_csv = old_root / "PastPerformance" / "8888888888.csv"
    new_csv = new_root / "past_performance" / "8888888888.csv"

    assert old_csv.is_file() and new_csv.is_file()
    assert pd.read_csv(old_csv, dtype=str).equals(pd.read_csv(new_csv, dtype=str))


# --- make_past_performance_dataset_from_race_results / update_past_performance ---


@pytest.fixture(scope="module")
def race_results_root(tmp_path_factory):
    """data/race_result の *_race_results.csv をすべて旧形式(RaceResults/{place}/{year}_race_results.csv)
    にコピーしたtmp領域を作成する（モジュール内のテストで共有・読み取り専用）
    """
    root = tmp_path_factory.mktemp("race_results_root")
    for csv_path in glob(os.path.join(paths.RACE_RESULT_DATA_PATH, "*", "*_race_results.csv")):
        place = os.path.basename(os.path.dirname(csv_path))
        dst_dir = root / "RaceResults" / place
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(csv_path, dst_dir / os.path.basename(csv_path))
    return root


def test_make_past_performance_dataset_from_race_results_matches_old(race_results_root, monkeypatch):
    monkeypatch.setattr(old_past_performance.name_header, "DATA_PATH", str(race_results_root) + "/")

    old_result = old_past_performance.make_past_performance_dataset_from_race_results(SAMPLE_HORSE_ID)
    new_result = new_past_performance.make_past_performance_dataset_from_race_results(SAMPLE_HORSE_ID)

    assert not old_result.empty
    assert old_result.equals(new_result)


def test_update_past_performance_matches_old(race_results_root, tmp_path, monkeypatch):
    monkeypatch.setattr(old_past_performance.name_header, "DATA_PATH", str(race_results_root) + "/")

    src = os.path.join(paths.HORSE_DATA_PATH, "past_performance", f"{SAMPLE_HORSE_ID}.csv")

    old_pp_dir = race_results_root / "PastPerformance"
    old_pp_dir.mkdir(exist_ok=True)
    shutil.copy(src, old_pp_dir / f"{SAMPLE_HORSE_ID}.csv")

    new_pp_dir = tmp_path / "past_performance"
    new_pp_dir.mkdir()
    shutil.copy(src, new_pp_dir / f"{SAMPLE_HORSE_ID}.csv")
    monkeypatch.setattr(new_past_performance, "PAST_PERFORMANCE_DATA_PATH", str(new_pp_dir))

    old_result = old_past_performance.update_past_performance(SAMPLE_HORSE_ID)
    new_result = new_past_performance.update_past_performance(SAMPLE_HORSE_ID)

    assert not old_result.empty
    assert old_result.equals(new_result)

    old_csv = old_pp_dir / f"{SAMPLE_HORSE_ID}.csv"
    new_csv = new_pp_dir / f"{SAMPLE_HORSE_ID}.csv"
    assert pd.read_csv(old_csv, dtype=str).equals(pd.read_csv(new_csv, dtype=str))


# --- ensure_past_performance_dataset ---------------------------------------------


def test_ensure_past_performance_dataset_returns_existing(old_and_new_roots):
    old_root, new_root = old_and_new_roots

    src = os.path.join(paths.HORSE_DATA_PATH, "past_performance", f"{SAMPLE_HORSE_ID}.csv")
    shutil.copy(src, new_root / "past_performance" / f"{SAMPLE_HORSE_ID}.csv")

    existing = new_past_performance.get_past_performance_dataset(SAMPLE_HORSE_ID)
    result = new_past_performance.ensure_past_performance_dataset(SAMPLE_HORSE_ID)

    assert not result.empty
    assert result.equals(existing)


def test_ensure_past_performance_dataset_creates_when_missing(old_and_new_roots):
    old_root, new_root = old_and_new_roots

    assert new_past_performance.get_past_performance_dataset(SAMPLE_HORSE_ID).empty

    result = new_past_performance.ensure_past_performance_dataset(SAMPLE_HORSE_ID)

    assert not result.empty
    saved = new_past_performance.get_past_performance_dataset(SAMPLE_HORSE_ID)
    assert saved.equals(result)


# --- get_past_race_info ----------------------------------------------------------


def test_get_past_race_info_matches_old(old_and_new_roots):
    old_root, new_root = old_and_new_roots

    src = os.path.join(paths.HORSE_DATA_PATH, "past_performance", f"{SAMPLE_HORSE_ID}.csv")
    shutil.copy(src, old_root / "PastPerformance" / f"{SAMPLE_HORSE_ID}.csv")
    shutil.copy(src, new_root / "past_performance" / f"{SAMPLE_HORSE_ID}.csv")

    target_df = pd.read_csv(src, dtype=str)
    race_id = target_df.iloc[0]["race_id"]

    old_result = old_past_performance.get_past_race_info(SAMPLE_HORSE_ID, race_id, 5)
    new_result = new_past_performance.get_past_race_info(SAMPLE_HORSE_ID, race_id, 5)

    assert not old_result.empty
    assert old_result.equals(new_result)
