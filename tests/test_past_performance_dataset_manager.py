"""src/datasets/horse, src/managers/past_performance_dataset_manager.py のテスト（オフライン）。

src/datasets/horse/transform.py の純粋関数と、
src/managers/past_performance_dataset_manager.py のCRUD・変換系関数を、
新実装単体で検証する。実データの参照系テストは
data/horse/past_performance/2007100107.csv、data/race_result/02_hakodate/2019_race_results.csv
（確定済みのhorse_id/race_id）を用いる。
"""

import os
import shutil

import pandas as pd
import pytest

from src.config import paths
from src.datasets.horse import model, transform
from src.managers import past_performance_dataset_manager as new_past_performance

SAMPLE_HORSE_ID = "2007100107"
SAMPLE_RACE_ID = "201902010101"  # 02_hakodate, 2019年, 16頭立て


# --- 純粋関数（src/datasets/horse/transform.py） ---


@pytest.mark.parametrize(
    "text, expected",
    [
        ("３勝クラス", "3勝クラス"),
        ("２勝クラス", "2勝クラス"),
        ("未勝利", "未勝利"),
        ("新馬", "新馬"),
        ("ｵｰﾌﾟﾝ", "オープン"),
        ("1勝　クラス", "1勝クラス"),
        ("ｵｰﾌﾟﾝｸﾗｽ", "オープンクラス"),
    ],
)
def test_normalize_class_text(text, expected):
    assert transform.normalize_class_text(text) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("ムーンライトH(3勝クラス)", "ムーンライトH"),
        ("ジューンステークス（2勝クラス）", "ジューンステークス"),
        ("サラブレッドチャレンジカップ", "サラブレッドチャレンジカップ"),
        ("  末尾に空白あり  ", "  末尾に空白あり"),
    ],
)
def test_clean_race_name(text, expected):
    assert transform.clean_race_name(text) == expected


def test_get_past_race_id():
    df = pd.DataFrame({"race_id": ["202301010101", "202301010102"], "着順": ["1", "2"]})

    result = transform.get_past_race_id(df.copy())

    assert result == ["202301010101", "202301010102"]


def test_reset_horse_result():
    df = pd.DataFrame(
        {
            "race_id": ["202301010103", "202301010102", "202301010101"],
            "着順": ["1", "2", "3"],
        }
    )

    result = transform.reset_horse_result(df.copy(), "202301010102")

    assert len(result) == 1
    assert result.iloc[0]["race_id"] == "202301010101"
    assert result.iloc[0]["着順"] == "3"


# --- normalize_past_performance_format ----------------------------------------


def test_normalize_past_performance_format_new_format_returns_real_data():
    """既存のpast_performance（新フォーマット）を読み込み、軽微な整形のみが
    行われることを確認する"""
    df = pd.read_csv(
        os.path.join(paths.HORSE_DATA_PATH, "past_performance", f"{SAMPLE_HORSE_ID}.csv"), dtype=str
    )

    result = transform.normalize_past_performance_format(df.copy())

    assert result.shape == (47, 24)
    assert result.columns.tolist() == model.PAST_PERFORMANCE_COLUMNS

    first = result.iloc[0]
    assert first["race_id"] == "201904010304"
    assert first["日付"] == "2019/05/04"
    assert first["開催"] == "新潟"
    assert first["class"] == "オープン"


def test_normalize_past_performance_format_old_format_converts():
    """旧フォーマットのpast_performanceから新フォーマットへの変換を確認する"""
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

    result = transform.normalize_past_performance_format(df.copy()).reset_index(drop=True)

    assert result.shape == (1, 24)

    first = result.iloc[0]
    assert first["race_id"] == "202307030108"
    assert first["日付"] == "2023/5/7"
    assert first["開催"] == "中京"
    assert first["class"] == "3勝クラス"
    assert first["race_type"] == "ダ"
    assert first["course_len"] == "1800"
    assert first["着差"] == -0.2
    assert first["馬体重"] == "480"
    assert first["レース名"] == "ムーンライトH(3勝クラス)"


def test_normalize_past_performance_format_empty_input():
    result = transform.normalize_past_performance_format(pd.DataFrame())

    assert result.empty


# --- get_horse_id_from_race_id / get_horse_id_list_from_race_id_list -----------


def test_get_horse_id_from_race_id_returns_expected():
    result = new_past_performance.get_horse_id_from_race_id(SAMPLE_RACE_ID)

    assert len(result) == 16
    assert result[0] == "2016104740"


def test_get_horse_id_list_from_race_id_list_returns_expected():
    result = new_past_performance.get_horse_id_list_from_race_id_list([SAMPLE_RACE_ID, SAMPLE_RACE_ID])

    assert len(result) == 32


# --- get_past_performance_dataset / save_past_performance_dataset (CRUD) ------


@pytest.fixture
def new_data_root(tmp_path, monkeypatch):
    """新実装(past_performance_dataset_manager.PAST_PERFORMANCE_DATA_PATH)をtmp_path配下に向ける"""
    new_root = tmp_path / "past_performance"
    monkeypatch.setattr(new_past_performance, "PAST_PERFORMANCE_DATA_PATH", str(new_root))
    return new_root


def test_get_past_performance_dataset_returns_real_data():
    df = new_past_performance.get_past_performance_dataset(SAMPLE_HORSE_ID)

    assert df.shape == (47, 23)

    first = df.iloc[0]
    assert first["日付"] == "2019/05/04"
    assert first["開催"] == "新潟"
    assert first["レース名"] == "障害4歳以上OP"
    assert first["class"] == "オープン"


def test_get_past_performance_dataset_returns_empty_for_missing_file(new_data_root):
    df = new_past_performance.get_past_performance_dataset("9999999999")

    assert df.empty


def test_save_past_performance_dataset_writes_csv_and_pickle(new_data_root):
    sample_df = pd.DataFrame({"race_id": ["202301010101"], "着順": ["1"], "騎手": ["サンプル騎手"]})

    new_past_performance.save_past_performance_dataset("8888888888", sample_df.copy())

    csv_path = new_data_root / "8888888888.csv"
    pickle_path = new_data_root / "8888888888.pickle"

    assert csv_path.is_file()
    assert pickle_path.is_file()

    df = pd.read_csv(csv_path, dtype=str, index_col=0)
    assert df.iloc[0]["race_id"] == "202301010101"
    assert df.iloc[0]["着順"] == "1"
    assert df.iloc[0]["騎手"] == "サンプル騎手"


# --- make_past_performance_dataset_from_race_results / update_past_performance ---


def test_make_past_performance_dataset_from_race_results_returns_real_data():
    result = new_past_performance.make_past_performance_dataset_from_race_results(SAMPLE_HORSE_ID)

    assert result.shape == (3, 24)

    first = result.iloc[0]
    assert first["race_id"] == "201904010304"
    assert first["日付"] == "2019/05/04"
    assert first["開催"] == "新潟"
    assert first["class"] == "オープン"
    assert first["race_type"] == "障害"
    assert first["course_len"] == "3290"


def test_update_past_performance_merges_and_normalizes(new_data_root):
    new_data_root.mkdir(parents=True, exist_ok=True)
    src = os.path.join(paths.HORSE_DATA_PATH, "past_performance", f"{SAMPLE_HORSE_ID}.csv")
    shutil.copy(src, new_data_root / f"{SAMPLE_HORSE_ID}.csv")

    result = new_past_performance.update_past_performance(SAMPLE_HORSE_ID)

    assert result.shape == (47, 24)
    assert result.columns.tolist() == model.PAST_PERFORMANCE_COLUMNS

    first = result.iloc[0]
    assert first["race_id"] == "201904010304"
    assert first["日付"] == "2019/05/04"
    assert first["着順"] == "中"

    saved = pd.read_csv(new_data_root / f"{SAMPLE_HORSE_ID}.csv", dtype=str)
    assert saved.shape == (47, 24)


# --- ensure_past_performance_dataset ---------------------------------------------


def test_ensure_past_performance_dataset_returns_existing(new_data_root):
    new_data_root.mkdir(parents=True, exist_ok=True)
    src = os.path.join(paths.HORSE_DATA_PATH, "past_performance", f"{SAMPLE_HORSE_ID}.csv")
    shutil.copy(src, new_data_root / f"{SAMPLE_HORSE_ID}.csv")

    existing = new_past_performance.get_past_performance_dataset(SAMPLE_HORSE_ID)
    result = new_past_performance.ensure_past_performance_dataset(SAMPLE_HORSE_ID)

    assert not result.empty
    assert result.equals(existing)


def test_ensure_past_performance_dataset_creates_when_missing(new_data_root):
    assert new_past_performance.get_past_performance_dataset(SAMPLE_HORSE_ID).empty

    result = new_past_performance.ensure_past_performance_dataset(SAMPLE_HORSE_ID)

    assert result.shape == (3, 24)
    saved = new_past_performance.get_past_performance_dataset(SAMPLE_HORSE_ID)
    assert saved.equals(result)


# --- get_past_race_info ----------------------------------------------------------


def test_get_past_race_info_returns_expected_shape(new_data_root):
    new_data_root.mkdir(parents=True, exist_ok=True)
    src = os.path.join(paths.HORSE_DATA_PATH, "past_performance", f"{SAMPLE_HORSE_ID}.csv")
    shutil.copy(src, new_data_root / f"{SAMPLE_HORSE_ID}.csv")

    target_df = pd.read_csv(src, dtype=str)
    race_id = target_df.iloc[0]["race_id"]

    result = new_past_performance.get_past_race_info(SAMPLE_HORSE_ID, race_id, 5)

    assert result.shape == (5, 23)
