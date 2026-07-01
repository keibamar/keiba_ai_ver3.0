"""RacePredictionEngine（Oracle）の日次予想ロジック

旧 src/PredictionModels/LightGBM/{make_dataset,prediction}.py と
src/RacePrediction/day_race_prediction.py の日次予想パスを移植したもの。
学習済みLightGBMモデルへの入力の列の並び・連結順は旧実装と完全に一致させている。
オフライン学習パイプライン（make_dataset_for_train等）は対象外で、
旧実装(src/PredictionModels/LightGBM/*.py)のまま残る。
"""

import os
import re

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import rankdata

from src.config import paths
from src.config.constants import PLACE_LIST
from src.datasets.race_result import transform as race_result_transform
from src.logic.calculators import average_calculator
from src.managers import (
    horse_peds_dataset_manager,
    past_performance_dataset_manager,
    peds_results_dataset_manager,
    race_info_dataset_manager,
)


def prediction_error(e):
    print(__name__ + ":" + __file__)
    print(f"{e.__class__.__name__}: {e}")


def get_time_info(race_info):
    """過去レース結果よりタイム差を取得する（タイム差/平均タイム[ms]）

    Args:
        race_info (pd.Series): レース結果（1走分）

    Returns:
        list: [time_diff, time_diff_class]
    """
    course_info = race_result_transform.get_course_info(race_info)
    if course_info[0] < 0:
        return [0, 0]

    race_time = average_calculator.get_race_time_msec(race_info["タイム"])
    return race_info_dataset_manager.get_time_diff(race_time, course_info)


def get_past_race_info_data(race_info_df):
    """過去最大3走分のタイム差・人気・着順を抽出する

    Args:
        race_info_df (pd.DataFrame): 過去レース結果（直近順、最大3走）

    Returns:
        list: 過去3走分のタイム差・人気・着順を連結したリスト（要素数12）
    """
    race_score_list = []
    for i in range(3):
        if i < len(race_info_df.index):
            df_time = get_time_info(race_info_df.iloc[i])

            raw_rank = race_info_df.at[i, "着順"]
            raw_pop = race_info_df.at[i, "人気"]
            rank_str = str(raw_rank)

            if "除" in rank_str or "取" in rank_str:
                df_time.append(np.nan)
                df_time.append(np.nan)
            elif "中" in rank_str or "失" in rank_str:
                try:
                    pop_val = float(re.sub(r"\D", "", str(raw_pop)))
                except Exception:
                    pop_val = np.nan
                df_time.append(pop_val)
                df_time.append(np.nan)
            else:
                try:
                    pop_val = float(re.sub(r"\D", "", str(raw_pop)))
                except Exception:
                    pop_val = np.nan
                try:
                    rank_val = float(re.sub(r"\D", "", rank_str))
                except Exception:
                    rank_val = np.nan
                df_time.append(pop_val)
                df_time.append(rank_val)
            race_score_list.append(df_time)
        else:
            race_score_list.append([np.nan, np.nan, np.nan, np.nan])

    return sum(race_score_list, [])


def make_dataset_for_lightgbm(race_id, course_info, horse_id):
    """1馬分のLightGBM用特徴量行を作成する

    Args:
        race_id (int): race_id
        course_info (list): [place_id, race_type, course_len, ground_state, race_class]
        horse_id (int): horse_id

    Returns:
        pd.DataFrame: 1行のLightGBM用データセット（取得失敗時は空のDataFrame）
    """
    try:
        race_year = int(str(race_id)[0:4])

        peds_info = horse_peds_dataset_manager.get_peds_info(horse_id)
        df_peds = peds_results_dataset_manager.peds_index(peds_info[0], peds_info[1], course_info, race_year)
        df_peds = sum(df_peds.T.values.tolist(), [])

        race_info = past_performance_dataset_manager.get_past_race_info(horse_id, race_id, race_num=3)
        df_race = get_past_race_info_data(race_info)

        df_lightgbm = df_peds + df_race
        return pd.DataFrame(df_lightgbm).T
    except Exception as e:
        prediction_error(e)
        return pd.DataFrame()


def get_lightgbm_model(place_id, race_type, length):
    """LightGBMの学習済みモデルを取得する

    Args:
        place_id (int): 開催コースid
        race_type (str): 芝/ダート
        length (int): キョリ

    Returns:
        lgb.Booster: 学習済みモデル
    """
    type_str = "turf" if race_type == "芝" else "dirt"
    model_path = os.path.join(
        paths.PREDICTION_MODEL_PATH, PLACE_LIST[place_id - 1], f"{type_str}{length}_lambdarank_model.txt"
    )
    return lgb.Booster(model_file=model_path)


def rank_index(score_list):
    """予想スコアを降順の順位に変換する

    Args:
        score_list (list): 予想スコアのリスト

    Returns:
        list: 降順の順位リスト
    """
    rank = rankdata(score_list)
    rank = (len(rank) - rank + 1).astype(int)
    return list(rank)


def prediction_race_score(place_id, race_type, length, race_dataset):
    """LightGBMモデルでスコア・順位を推定する

    Args:
        place_id (int): 開催コースid
        race_type (str): 芝/ダート
        length (int): キョリ
        race_dataset (pd.DataFrame): LightGBM用データセット

    Returns:
        pd.DataFrame: score, rank列を持つDataFrame（失敗時は空のDataFrame）
    """
    try:
        race_dataset = race_dataset.fillna(-1)

        model = get_lightgbm_model(place_id, race_type, length)

        y_pred = model.predict(race_dataset, num_iteration=model.best_iteration)

        rank = rank_index(y_pred)
        result_df = pd.concat([pd.DataFrame(y_pred, columns=["score"]), pd.DataFrame(rank, columns=["rank"])], axis=1)

        return result_df
    except Exception as e:
        prediction_error(e)
        return pd.DataFrame()


def _make_base_race_dataset(race_id, horse_ids, race_info_df):
    """的中率重視・回収率重視の両モデルで共通の特徴量（血統+過去3走）を組み立てる

    Args:
        race_id (int): race_id
        horse_ids (list): レース出走馬のhorse_idリスト
        race_info_df (pd.DataFrame): レース情報(race_type, course_len, ground_state, class)

    Returns:
        tuple: (race_dataset(pd.DataFrame), place_id(int), course_info(list))
    """
    race_dataset = pd.DataFrame()
    place_id = int(str(race_id)[4] + str(race_id)[5])
    course_info = [
        place_id,
        race_info_df.at[0, "race_type"],
        race_info_df.at[0, "course_len"],
        race_info_df.at[0, "ground_state"],
        race_info_df.at[0, "class"],
    ]

    for horse_id in horse_ids:
        df_result = make_dataset_for_lightgbm(race_id, course_info, horse_id)
        race_dataset = pd.concat([race_dataset.reset_index(drop=True), df_result.reset_index(drop=True)])

    return race_dataset, place_id, course_info


def rank_prediction(race_id, horse_ids, race_info_df, waku_df):
    """出走馬のAI予想ランキングを計算する（的中率重視モデル/サブA）

    Args:
        race_id (int): race_id
        horse_ids (list): レース出走馬のhorse_idリスト
        race_info_df (pd.DataFrame): レース情報(race_type, course_len, ground_state, class)
        waku_df (pd.DataFrame): 枠番・馬番のデータセット

    Returns:
        pd.DataFrame: 予想結果データセット（score, rank列。失敗時は空のDataFrame）
    """
    try:
        race_dataset, place_id, course_info = _make_base_race_dataset(race_id, horse_ids, race_info_df)
        race_dataset = pd.concat([race_dataset.reset_index(drop=True), waku_df.reset_index(drop=True)], axis=1)

        return prediction_race_score(place_id, course_info[1], course_info[2], race_dataset)
    except Exception as e:
        prediction_error(e)
        return pd.DataFrame()


def get_lightgbm_value_model(place_id, race_type, length):
    """回収率重視モデル（サブB）の学習済みLightGBMモデルを取得する

    Args:
        place_id (int): 開催コースid
        race_type (str): 芝/ダート
        length (int): キョリ

    Returns:
        lgb.Booster: 学習済みモデル
    """
    type_str = "turf" if race_type == "芝" else "dirt"
    model_path = os.path.join(
        paths.PREDICTION_MODEL_PATH, PLACE_LIST[place_id - 1], f"{type_str}{length}_lambdarank_model_value.txt"
    )
    return lgb.Booster(model_file=model_path)


def prediction_race_score_value(place_id, race_type, length, race_dataset):
    """回収率重視モデル（サブB）でスコア・順位を推定する

    Args:
        place_id (int): 開催コースid
        race_type (str): 芝/ダート
        length (int): キョリ
        race_dataset (pd.DataFrame): LightGBM用データセット（自レース人気列を含む）

    Returns:
        pd.DataFrame: score, rank列を持つDataFrame（失敗時は空のDataFrame）
    """
    try:
        race_dataset = race_dataset.fillna(-1)

        model = get_lightgbm_value_model(place_id, race_type, length)

        y_pred = model.predict(race_dataset, num_iteration=model.best_iteration)

        rank = rank_index(y_pred)
        result_df = pd.concat([pd.DataFrame(y_pred, columns=["score"]), pd.DataFrame(rank, columns=["rank"])], axis=1)

        return result_df
    except Exception as e:
        prediction_error(e)
        return pd.DataFrame()


def rank_prediction_value(race_id, horse_ids, race_info_df, waku_df, popularity_series):
    """出走馬のAI予想ランキングを計算する（回収率重視モデル/サブB）

    的中率重視モデルと同じ血統・過去3走の特徴量に「そのレース自身の人気」を
    1列追加して予測する。人気はレース当日、発走20分前頃の出馬表再取得時に
    スクレイピングされる値を使う想定（src.logic.scraping.netkeiba_scraper.scrape_race_card）。

    Args:
        race_id (int): race_id
        horse_ids (list): レース出走馬のhorse_idリスト
        race_info_df (pd.DataFrame): レース情報(race_type, course_len, ground_state, class)
        waku_df (pd.DataFrame): 枠番・馬番のデータセット
        popularity_series (pd.Series): horse_idsと同じ順序の人気（数値化できない値はNaN）

    Returns:
        pd.DataFrame: 予想結果データセット（score, rank列。失敗時は空のDataFrame）
    """
    try:
        race_dataset, place_id, course_info = _make_base_race_dataset(race_id, horse_ids, race_info_df)
        race_dataset = pd.concat(
            [
                race_dataset.reset_index(drop=True),
                waku_df.reset_index(drop=True),
                pd.Series(popularity_series, name="self_popularity").reset_index(drop=True),
            ],
            axis=1,
        )

        return prediction_race_score_value(place_id, course_info[1], course_info[2], race_dataset)
    except Exception as e:
        prediction_error(e)
        return pd.DataFrame()


def _zscore(score_series):
    """レース内でスコアを標準化（平均0・標準偏差1）する

    的中率重視・回収率重視モデルのスコアはスケールが全く異なる（LightGBMの
    生スコアでモデルごとに値の範囲が違う）ため、ブレンド前にそろえる。
    以前は「レース内の順位を0〜1に変換」する方式だったが、これだと
    元のスコアが違っていても同じ順位なら同じ値になり、さらに出走頭数で
    決まる不自然に丸い数字（0.5, 0.8667等）になってしまっていた。
    生スコアの差をそのまま反映する標準化に変更し、表示するScoreとRankが
    常に対応する（Scoreの大小がそのままRankの順序になる）ようにする。

    Args:
        score_series (pd.Series): 生スコア

    Returns:
        pd.Series: 標準化スコア（全頭が同値、または1頭のみの場合は全て0）
    """
    std = score_series.std()
    if not std or pd.isna(std):
        return pd.Series([0.0] * len(score_series), index=score_series.index)
    return (score_series - score_series.mean()) / std


def score_to_index(score):
    """標準化スコア（z-score）を、馴染みやすい0〜100の「AI指数」に変換する

    偏差値と同じ計算式（50 + 10 × z）を使う。blended_rank_predictionのscore列は
    レース内で平均0・標準偏差1に標準化されているため、score=0（平均的な馬）が
    指数50、score=+1（平均より1標準偏差上）が指数60になる。0〜100の範囲に
    クリップする（理論上は範囲外の値も起こり得るが、表示上のキリの良さを優先する）。
    score_calibration.py（指数帯ごとの単勝/複勝的中率の実績テーブル）と対になる。

    Args:
        score (float): blended_rank_prediction等が返す標準化スコア（score列の値）。

    Returns:
        int: 0〜100のAI指数（10の位で意味のある区切りになるよう整数に丸める）
    """
    if pd.isna(score):
        return None
    index = 50 + 10 * score
    # 整数に丸めると z-score の差が 0.05 未満の馬が同じ指数になって識別不能になる
    # （例: score=+0.004 と -0.027 は両方 int→50）ため、小数点1桁に変更する。
    # これにより 0.1 点差で区別でき、16頭 ÷ 40点幅（±2σ = 20〜80 点の範囲）≈
    # 平均2.5頭/点 → ほぼ全馬が個別の指数値を持てる解像度になる。
    return round(max(0.0, min(100.0, index)), 1)


def blended_rank_prediction(race_id, horse_ids, race_info_df, waku_df, popularity_series=None, value_weight=0.5):
    """的中率重視モデル（サブA）・回収率重視モデル（サブB）を合成したバランス型のAI予想を計算する

    各モデルのスコアはレース内で標準化した上で重み付き平均する
    （LightGBMのスコアはモデル間で直接比較できるスケールではないため）。
    表示されるScore（この関数のscore列）とRank（rank列）は常にこの同じ値から
    計算されるため、「Scoreが違うのにRankが同じ」「Scoreの大小とRankの順序が
    合わない」という不整合は起きない。
    人気が未確定（popularity_seriesが無い、または回収率重視モデルが未学習）の場合は
    的中率重視モデルのみのスコアにフォールバックする。

    Args:
        race_id (int): race_id
        horse_ids (list): レース出走馬のhorse_idリスト
        race_info_df (pd.DataFrame): レース情報(race_type, course_len, ground_state, class)
        waku_df (pd.DataFrame): 枠番・馬番のデータセット
        popularity_series (pd.Series): horse_idsと同じ順序の人気（Noneなら回収率重視モデルをスキップ）
        value_weight (float): 回収率重視モデルの重み（0〜1。的中率重視モデルの重みは1-value_weight）

    Returns:
        pd.DataFrame: score, rank（バランス型。表示用のScore/Rankはこの2列を使う）、
            score_hitrate, rank_hitrate（的中率重視、参考用）、
            score_value, rank_value（回収率重視、参考用。未算出時はNaN）列を持つDataFrame
    """
    hitrate_df = rank_prediction(race_id, horse_ids, race_info_df, waku_df)
    if hitrate_df.empty:
        return pd.DataFrame()

    value_df = pd.DataFrame()
    if popularity_series is not None:
        value_df = rank_prediction_value(race_id, horse_ids, race_info_df, waku_df, popularity_series)

    n = len(hitrate_df)
    if value_df.empty or len(value_df) != n:
        blended_score = _zscore(hitrate_df["score"])
        value_df = pd.DataFrame({"score": [np.nan] * n, "rank": [np.nan] * n})
    else:
        blended_score = (
            (1 - value_weight) * _zscore(hitrate_df["score"])
            + value_weight * _zscore(value_df["score"])
        )

    result_df = pd.DataFrame({"score": blended_score})
    result_df["rank"] = rank_index(result_df["score"].tolist())
    result_df["score_hitrate"] = hitrate_df["score"]
    result_df["rank_hitrate"] = hitrate_df["rank"]
    result_df["score_value"] = value_df["score"]
    result_df["rank_value"] = value_df["rank"]
    return result_df
