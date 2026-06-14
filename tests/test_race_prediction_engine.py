"""src/logic/prediction/race_prediction_engine.py の出力が旧実装と一致することを
確認するテスト（オフライン）。

- get_race_time_msec / calc_time_diff: 旧 src/legacy_datasets/average_time.py の
  対応関数と同一結果を返すことを確認する。
- get_time_diff: 新 src/managers/race_info_dataset_manager.py と旧
  src/legacy_datasets/average_time.py が、同一のtotal_avg_time.csvを参照した場合に
  同一結果を返すことを確認する（旧実装はname_header.DATA_PATH(ver2.0側)を参照するため、
  tmp_path配下にコピーしてname_header.DATA_PATH / AVERAGE_TIMES_DATA_PATHを向ける）。
- make_dataset_for_lightgbm / rank_prediction: 実データ（確定済みのrace_id/horse_id）を
  用いて、想定どおりの列数・出力形式となること、および
  src/RacePrediction/day_race_prediction.rank_prediction（新エンジンへのリダイレクト後）
  が新エンジンと同一結果を返すことを確認する。
"""

import os
import shutil

import numpy as np
import pandas as pd
import pytest

from src.logic.prediction import race_prediction_engine as engine
from src.RacePrediction import day_race_prediction
from src.datasets.race_info import transform as race_info_transform
from src.legacy_datasets import average_time as old_avg_time
from src.managers import past_performance_dataset_manager, race_info_dataset_manager

SAMPLE_PLACE_ID = 2
SAMPLE_PLACE = "02_hakodate"
SAMPLE_RACE_ID = 202302010101
SAMPLE_HORSE_ID = "2020102879"
SAMPLE_COURSE_INFO = [SAMPLE_PLACE_ID, "芝", "1200", "稍重", "未勝利"]


# --- 純粋関数の比較（旧 src/legacy_datasets/average_time.py vs 新 src/datasets/race_info/transform.py） ---


@pytest.mark.parametrize("time_str", ["1:11.2", "1:39.8", "2:00.0"])
def test_get_race_time_msec_matches_old(time_str):
    old_result = old_avg_time.get_race_time_msec(time_str)
    new_result = race_info_transform.get_race_time_msec(time_str)

    assert old_result == new_result


def test_get_race_time_msec_nan_matches_old():
    old_result = old_avg_time.get_race_time_msec(np.nan)
    new_result = race_info_transform.get_race_time_msec(np.nan)

    assert np.isnan(old_result) and np.isnan(new_result)


@pytest.mark.parametrize(("base_time", "race_time"), [("70000", 71000.0), (70000.0, 69000.0)])
def test_calc_time_diff_matches_old(base_time, race_time):
    old_result = old_avg_time.calc_time_diff(base_time, race_time)
    new_result = race_info_transform.calc_time_diff(base_time, race_time)

    assert old_result == new_result


def test_calc_time_diff_nan_matches_old():
    old_result = old_avg_time.calc_time_diff(np.nan, 70000.0)
    new_result = race_info_transform.calc_time_diff(np.nan, 70000.0)

    assert np.isnan(old_result) and np.isnan(new_result)


# --- get_time_diff（旧 average_time.get_time_diff vs 新 race_info_dataset_manager.get_time_diff） ---


@pytest.fixture
def avg_time_root(tmp_path):
    """新実装で生成済みのtotal_avg_time.csv/pickleをtmp_path/AverageTimes配下にコピーする"""
    src_dir = os.path.join(race_info_dataset_manager.AVERAGE_TIMES_DATA_PATH, SAMPLE_PLACE)
    dst_dir = tmp_path / "AverageTimes" / SAMPLE_PLACE
    dst_dir.mkdir(parents=True)
    for filename in ("total_avg_time.csv", "total_avg_time.pickle"):
        shutil.copy(os.path.join(src_dir, filename), dst_dir / filename)
    return tmp_path


def test_get_time_diff_matches_old(avg_time_root, monkeypatch):
    monkeypatch.setattr(old_avg_time.name_header, "DATA_PATH", str(avg_time_root) + "/")
    monkeypatch.setattr(race_info_dataset_manager, "AVERAGE_TIMES_DATA_PATH", str(avg_time_root / "AverageTimes"))

    race_time = 70000.0

    old_result = old_avg_time.get_time_diff(race_time, SAMPLE_COURSE_INFO)
    new_result = race_info_dataset_manager.get_time_diff(race_time, SAMPLE_COURSE_INFO)

    assert old_result == new_result
    assert old_result != [0, 0]


def test_get_time_diff_returns_zero_when_no_data(tmp_path, monkeypatch):
    monkeypatch.setattr(race_info_dataset_manager, "AVERAGE_TIMES_DATA_PATH", str(tmp_path / "average_times"))

    result = race_info_dataset_manager.get_time_diff(70000.0, SAMPLE_COURSE_INFO)

    assert result == [0, 0]


# --- get_past_race_info_data / make_dataset_for_lightgbm（実データ） ---


def test_get_past_race_info_data_returns_12_values():
    race_info_df = past_performance_dataset_manager.get_past_race_info(SAMPLE_HORSE_ID, SAMPLE_RACE_ID, race_num=3)

    result = engine.get_past_race_info_data(race_info_df)

    assert len(result) == 12


def test_make_dataset_for_lightgbm_shape():
    df = engine.make_dataset_for_lightgbm(SAMPLE_RACE_ID, SAMPLE_COURSE_INFO, SAMPLE_HORSE_ID)

    # 血統着度数(36列) + 過去3走分のタイム差・人気・着順(12列) = 48列、1行
    assert df.shape == (1, 48)
    assert not df.isna().all(axis=None)


# --- get_lightgbm_model / prediction_race_score（data/prediction/models参照） ---


def test_get_lightgbm_model_loads_from_prediction_data_path():
    model_path = os.path.join(engine.paths.PREDICTION_MODEL_PATH, SAMPLE_PLACE, "turf1200_lambdarank_model.txt")
    assert os.path.isfile(model_path)

    model = engine.get_lightgbm_model(SAMPLE_PLACE_ID, "芝", "1200")

    assert model.num_feature() == 50


# --- rank_prediction（エンドツーエンド、実データ） -------------------------------


@pytest.fixture(scope="module")
def sample_race_args():
    horse_ids = [
        2020102879, 2020101122, 2020105791, 2020104087, 2020101229, 2020102877,
        2020104758, 2020110014, 2020110061, 2020106121, 2020106679, 2020102360,
        2020110139, 2020110124, 2020105711, 2020103949,
    ]
    race_info_df = pd.DataFrame([{"race_type": "芝", "course_len": "1200", "ground_state": "稍重", "class": "未勝利"}])
    waku_df = pd.DataFrame(
        {
            "枠番": [1, 6, 1, 6, 4, 7, 3, 5, 2, 3, 5, 7, 8, 8, 4, 2],
            "馬番": [1, 12, 2, 11, 8, 14, 6, 10, 4, 5, 9, 13, 16, 15, 7, 3],
        }
    )
    return horse_ids, race_info_df, waku_df


def test_rank_prediction_returns_score_and_rank(sample_race_args):
    horse_ids, race_info_df, waku_df = sample_race_args

    result = engine.rank_prediction(SAMPLE_RACE_ID, horse_ids, race_info_df, waku_df)

    assert list(result.columns) == ["score", "rank"]
    assert len(result) == len(horse_ids)
    assert sorted(result["rank"].tolist()) == list(range(1, len(horse_ids) + 1))


def test_day_race_prediction_delegates_to_engine(sample_race_args):
    horse_ids, race_info_df, waku_df = sample_race_args

    old_result = day_race_prediction.rank_prediction(SAMPLE_RACE_ID, horse_ids, race_info_df, waku_df)
    new_result = engine.rank_prediction(SAMPLE_RACE_ID, horse_ids, race_info_df, waku_df)

    assert old_result.equals(new_result)
