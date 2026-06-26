"""src/logic/prediction/race_prediction_engine.py のテスト（オフライン）。

- get_time_diff: src/managers/race_info_dataset_manager.py の新実装単体を、
  生成済みのtotal_avg_time.csvをtmp_path配下にコピーして
  AVERAGE_TIMES_DATA_PATHを向けた状態で検証する。
- make_dataset_for_lightgbm / rank_prediction: 実データ（確定済みのrace_id/horse_id）を
  用いて、想定どおりの列数・出力形式となることを確認する。

get_race_time_msec / calc_time_diff の新実装単体検証は tests/test_average_calculator.py
に分離している。
"""

import os
import shutil

import pandas as pd
import pytest

from src.logic.prediction import race_prediction_engine as engine
from src.managers import past_performance_dataset_manager, race_info_dataset_manager

SAMPLE_PLACE_ID = 2
SAMPLE_PLACE = "02_hakodate"
SAMPLE_RACE_ID = 202302010101
SAMPLE_HORSE_ID = "2020102879"
SAMPLE_COURSE_INFO = [SAMPLE_PLACE_ID, "芝", "1200", "稍重", "未勝利"]


# --- get_time_diff（race_info_dataset_manager.get_time_diff） ---


@pytest.fixture
def avg_time_root(tmp_path):
    """新実装で生成済みのtotal_avg_time.csv/pickleをtmp_path/AverageTimes配下にコピーする"""
    src_dir = os.path.join(race_info_dataset_manager.AVERAGE_TIMES_DATA_PATH, SAMPLE_PLACE)
    dst_dir = tmp_path / "AverageTimes" / SAMPLE_PLACE
    dst_dir.mkdir(parents=True)
    for filename in ("total_avg_time.csv", "total_avg_time.pickle"):
        shutil.copy(os.path.join(src_dir, filename), dst_dir / filename)
    return tmp_path


def test_get_time_diff_returns_expected(avg_time_root, monkeypatch):
    monkeypatch.setattr(race_info_dataset_manager, "AVERAGE_TIMES_DATA_PATH", str(avg_time_root / "AverageTimes"))

    race_time = 70000.0

    result = race_info_dataset_manager.get_time_diff(race_time, SAMPLE_COURSE_INFO)

    print(f"\n--- get_time_diff(race_time={race_time}, course_info={SAMPLE_COURSE_INFO}) ---")
    print(f"  結果（[平均タイムとの差, 標準偏差換算]）: {result}")

    assert result == [0.001184310032390166, 0.004366563784544924]


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

    print(f"\n--- make_dataset_for_lightgbm(race_id={SAMPLE_RACE_ID}, horse_id={SAMPLE_HORSE_ID}) ---")
    print(f"  shape: {df.shape}")
    print(df.to_string())

    # 血統着度数(36列) + 過去3走分のタイム差・人気・着順(12列) = 48列、1行
    assert df.shape == (1, 48)
    assert not df.isna().all(axis=None)


# --- get_lightgbm_model / prediction_race_score（data/prediction/models参照） ---


def test_get_lightgbm_model_loads_from_prediction_data_path():
    model_path = os.path.join(engine.paths.PREDICTION_MODEL_PATH, SAMPLE_PLACE, "turf1200_lambdarank_model.txt")
    assert os.path.isfile(model_path)

    model = engine.get_lightgbm_model(SAMPLE_PLACE_ID, "芝", "1200")

    print(f"\n--- get_lightgbm_model(place_id={SAMPLE_PLACE_ID}, race_type=芝, course_len=1200) ---")
    print(f"  モデルパス: {model_path}")
    print(f"  特徴量数: {model.num_feature()}")

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

    print(f"\n--- rank_prediction(race_id={SAMPLE_RACE_ID}, 出走{len(horse_ids)}頭) ---")
    print(result.assign(horse_id=horse_ids).to_string())

    assert list(result.columns) == ["score", "rank"]
    assert len(result) == len(horse_ids)
    # rank=1はscore最大の馬に付く。モデル次第でスコアが同値になり順位が
    # 重複することもあるため、1〜出走数のきれいな整列までは厳密に求めない
    assert result["rank"].min() == 1
    assert result["rank"].max() <= len(horse_ids)
    top_rank_score = result.loc[result["rank"] == 1, "score"].iloc[0]
    assert top_rank_score == result["score"].max()


# --- blended_rank_prediction（的中率重視/サブA・回収率重視/サブB の合成、オフライン） -----


def test_normalized_score_top_rank_is_one_and_last_is_zero():
    ranks = pd.Series([3, 1, 2])

    result = engine._normalized_score(ranks)

    assert result.tolist() == [0.0, 1.0, 0.5]


def test_normalized_score_single_horse_returns_one():
    result = engine._normalized_score(pd.Series([1]))

    assert result.tolist() == [1.0]


def test_blended_rank_prediction_falls_back_to_hitrate_only_when_no_popularity(monkeypatch):
    hitrate_df = pd.DataFrame({"score": [0.5, -0.2, 0.1], "rank": [1, 3, 2]})
    monkeypatch.setattr(engine, "rank_prediction", lambda *a, **k: hitrate_df)

    result = engine.blended_rank_prediction(SAMPLE_RACE_ID, [1, 2, 3], pd.DataFrame(), pd.DataFrame())

    assert result["rank"].tolist() == hitrate_df["rank"].tolist()
    assert result["score_value"].isna().all()
    assert result["rank_value"].isna().all()


def test_blended_rank_prediction_blends_hitrate_and_value_scores(monkeypatch):
    # 的中率重視は1番(rank1)推し、回収率重視は3番(rank1)推し。重み0.5で中間の結果になる
    hitrate_df = pd.DataFrame({"score": [1.0, 0.5, 0.0], "rank": [1, 2, 3]})
    value_df = pd.DataFrame({"score": [0.0, 0.5, 1.0], "rank": [3, 2, 1]})
    monkeypatch.setattr(engine, "rank_prediction", lambda *a, **k: hitrate_df)
    monkeypatch.setattr(engine, "rank_prediction_value", lambda *a, **k: value_df)

    result = engine.blended_rank_prediction(
        SAMPLE_RACE_ID, [1, 2, 3], pd.DataFrame(), pd.DataFrame(), popularity_series=pd.Series([1, 2, 3]),
    )

    # 正規化スコアは両モデルとも[1.0, 0.5, 0.0]の組（順序が逆）になるため、
    # 重み0.5で平均すると全頭0.5で並ぶ
    assert result["score"].tolist() == [0.5, 0.5, 0.5]
    assert result["score_hitrate"].tolist() == hitrate_df["score"].tolist()
    assert result["score_value"].tolist() == value_df["score"].tolist()
