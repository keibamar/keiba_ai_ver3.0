"""RacePredictionEngine（Oracle）の日次予想ロジック

旧 src/PredictionModels/LightGBM/{make_dataset,prediction}.py と
src/RacePrediction/day_race_prediction.py の日次予想パスを移植したもの。
学習済みLightGBMモデルへの入力の列の並び・連結順は旧実装と完全に一致させている。
オフライン学習パイプライン（make_dataset_for_train等）は対象外で、
旧実装(src/PredictionModels/LightGBM/*.py)のまま残る。
"""

import os
import re
import sys as _sys

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import rankdata

from datetime import date

from src.config import paths
from src.config.constants import PLACE_LIST

# ── v2.0 ライブラリ（マルチモデル用） ──
for _v2p in [r"C:\keiba_ai\keiba_ai_ver2.0\libs",
             r"C:\keiba_ai\keiba_ai_ver2.0\src\PredictionModels\LightGBM"]:
    if _v2p not in _sys.path:
        _sys.path.append(_v2p)
try:
    from src.PredictionModels.LightGBM.make_dataset_v5 import (
        build_pedigree_vocab as _mm_build_pedigree_vocab,
        get_pedigree_cats   as _mm_get_pedigree_cats,
    )
    from src.PredictionModels.LightGBM.make_dataset_v6 import (
        get_extra_past_race_features_v6 as _mm_get_v6_extra,
    )
    from src.PredictionModels.LightGBM.make_dataset_v7 import _parse_kinryo as _mm_parse_kinryo
    from src.PredictionModels.LightGBM.make_dataset_v8 import (
        get_extra_past_race_features_v8 as _mm_get_v8_extra,
    )
    from src.PredictionModels.LightGBM.make_dataset_v9 import (
        get_extra_past_race_features_v9 as _mm_get_v9_extra,
        index_v9 as _mm_index_v9,
    )
    _MM_AVAILABLE = True
except ImportError:
    _MM_AVAILABLE = False

# ── マルチモデル列定義（初回インポート時に構築） ──
if _MM_AVAILABLE:
    _MM_IX9_NO_RACE = [c for c in _mm_index_v9 if c != "race_id"]   # 104列（重複あり）
    _MM_NODDS_EXCL  = {"race_id", "current_odds", "current_popularity"}
    _MM_NODDS_DUPS  = [c for c in _mm_index_v9 if c not in _MM_NODDS_EXCL]  # 102列（重複あり）
    _mm_seen = set(); _MM_NODDS_COLS = []
    for _c in _MM_NODDS_DUPS:
        if _c not in _mm_seen:
            _MM_NODDS_COLS.append(_c); _mm_seen.add(_c)
    # 100列（重複排除後）
    _MM_V11N_COLS = _MM_NODDS_COLS[:_MM_NODDS_COLS.index("horse_weight_abs_1") + 1]  # 86列
    _MM_V12N_COLS = _MM_NODDS_COLS[:_MM_NODDS_COLS.index("time_diff_trend_5")  + 1]  # 93列
from src.datasets.race_result import transform as race_result_transform
from src.logic.calculators import average_calculator
from src.managers import (
    horse_peds_dataset_manager,
    past_performance_dataset_manager,
    peds_results_dataset_manager,
    race_info_dataset_manager,
    race_result_dataset_manager,
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
    """レース内でスコアを標準化（平均0・標準偏差1）する（フォールバック用）

    的中率重視・回収率重視モデルのスコアはスケールが全く異なる（LightGBMの
    生スコアでモデルごとに値の範囲が違う）ため、ブレンド前にそろえる。

    Args:
        score_series (pd.Series): 生スコア

    Returns:
        pd.Series: 標準化スコア（全頭が同値、または1頭のみの場合は全て0）
    """
    std = score_series.std()
    if not std or pd.isna(std):
        return pd.Series([0.0] * len(score_series), index=score_series.index)
    return (score_series - score_series.mean()) / std


# コース別生スコア分布テーブルのメモリキャッシュ（モジュール初回ロード時に1回だけ読む）
_raw_score_dist_cache = None

# 絶対スコア正規化の有効フラグ。raw_score_distribution.csvが現在のモデルバージョンの
# スコア分布で構築されたら True に切り替える。
# それまでは従来のper-race z-scoreにフォールバックする。
USE_GLOBAL_ZSCORE = True


def _load_raw_score_distribution():
    """コース別の生スコア（LightGBM raw output）分布テーブルを読み込んでキャッシュする

    data/prediction/performance/raw_score_distribution.csv から読み込み、
    {(place_id, race_type, course_len): {"mean_hitrate", "std_hitrate", ...}} の
    辞書形式で返す。ファイルが無い場合は空辞書。
    """
    global _raw_score_dist_cache
    if _raw_score_dist_cache is None:
        path = paths.RAW_SCORE_DISTRIBUTION_PATH
        if os.path.isfile(path):
            df = pd.read_csv(path, dtype=str)
            _raw_score_dist_cache = {}
            for _, row in df.iterrows():
                try:
                    key = (int(row["place_id"]), str(row["race_type"]), str(row["course_len"]))
                    _raw_score_dist_cache[key] = {
                        "mean_hitrate": float(row["mean_hitrate"]),
                        "std_hitrate": max(float(row["std_hitrate"]), 1e-6),
                        "mean_value": float(row["mean_value"]) if pd.notna(row.get("mean_value", "")) and row.get("mean_value", "") != "nan" else None,
                        "std_value": max(float(row["std_value"]), 1e-6) if pd.notna(row.get("std_value", "")) and row.get("std_value", "") != "nan" else None,
                    }
                except (ValueError, KeyError):
                    continue
        else:
            _raw_score_dist_cache = {}
    return _raw_score_dist_cache


def _global_zscore(score_series, place_id, race_type, course_len, model_type="hitrate"):
    """コース別のグローバル分布でスコアを絶対スコアとして標準化する

    raw_score_distribution.csvに当該コースのデータがある場合、
    「コース全体を通じた平均・標準偏差」で正規化する（絶対スコア）。
    これにより、異なるレース・異なるメンバーでも同じコース種別の馬同士で
    スコアを比較できる（「この馬はこのコースで歴史的に上位X%」という意味になる）。
    データ不足（コースの組み合わせがテーブルに無い）場合はper-race _zscore にフォールバック。

    Args:
        score_series (pd.Series): LightGBMの生スコア列
        place_id (int): 開催場ID
        race_type (str): 芝/ダート
        course_len (str): 距離
        model_type (str): "hitrate"（サブA）または "value"（サブB）

    Returns:
        pd.Series: 絶対z-score（またはフォールバック時はper-race z-score）
    """
    dist = _load_raw_score_distribution()
    key = (int(place_id), str(race_type), str(course_len))
    if key in dist:
        entry = dist[key]
        mean_val = entry.get(f"mean_{model_type}")
        std_val = entry.get(f"std_{model_type}")
        if mean_val is not None and std_val is not None and std_val > 0:
            return (score_series - mean_val) / std_val
    return _zscore(score_series)


# ============================================================
# v3 モデル用ヘルパー
# ============================================================

def _parse_agari(val):
    try:
        return float(str(val).strip())
    except Exception:
        return np.nan


def _parse_margin(val):
    s = str(val).strip()
    if s in ("", "nan", "NaN"):
        return np.nan
    jp_map = {"ハナ": 0.05, "クビ": 0.1, "アタマ": 0.15, "1/2": 0.3, "3/4": 0.5}
    for key, sec in jp_map.items():
        if key in s:
            return sec
    try:
        v = float(s.replace("−", "-").replace("ー", "-"))
        return max(0.0, v)
    except Exception:
        return np.nan


def _parse_corner_ratio(通過_val, headcount_val):
    try:
        s = str(通過_val).strip()
        first = int(s.split("-")[0]) if "-" in s else int(float(s))
        hc = int(float(str(headcount_val).strip()))
        return first / hc if hc > 0 else np.nan
    except Exception:
        return np.nan


def _parse_weight(val):
    try:
        return float(re.search(r"\d+", str(val)).group())
    except Exception:
        return np.nan


def _rank_trend_v3(ranks):
    pairs = [(i, r) for i, r in enumerate(ranks) if not (isinstance(r, float) and np.isnan(r))]
    if len(pairs) < 2:
        return np.nan
    xs = [-(len(pairs) - 1 - i) for i in range(len(pairs))]
    ys = [r for _, r in pairs]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    num = sum((xs[i] - mx) * (ys[i] - my) for i in range(len(xs)))
    den = sum((x - mx) ** 2 for x in xs)
    return num / den if den != 0 else 0.0


def get_past_race_info_data_v3(race_info_df, current_course_len):
    """v3特徴量: 過去3走（7列×3）+ rank_trend + weight_change + dist_change = 24要素"""
    race_score_list = []
    weight_list = []
    rank_list = []

    for i in range(3):
        if i < len(race_info_df.index):
            row = race_info_df.iloc[i]
            course_info_row = race_result_transform.get_course_info(row)
            if course_info_row[0] < 0:
                df_time = [0, 0]
            else:
                try:
                    race_time = average_calculator.get_race_time_msec(row["タイム"])
                    df_time = race_info_dataset_manager.get_time_diff(race_time, course_info_row)
                except Exception:
                    df_time = [np.nan, np.nan]

            raw_rank = row.get("着順", "")
            raw_pop = row.get("人気", "")
            rank_str = str(raw_rank)

            if "除" in rank_str or "取" in rank_str:
                df_time.extend([np.nan, np.nan])
                rank_val = np.nan
            elif "中" in rank_str or "失" in rank_str:
                try:
                    pop_val = float(re.sub(r"\D", "", str(raw_pop)))
                except Exception:
                    pop_val = np.nan
                df_time.extend([pop_val, np.nan])
                rank_val = np.nan
            else:
                try:
                    pop_val = float(re.sub(r"\D", "", str(raw_pop)))
                except Exception:
                    pop_val = np.nan
                try:
                    rank_val = float(re.sub(r"\D", "", rank_str))
                except Exception:
                    rank_val = np.nan
                df_time.extend([pop_val, rank_val])

            df_time.append(_parse_agari(row.get("上り", "")))
            df_time.append(_parse_margin(row.get("着差", "")))
            df_time.append(_parse_corner_ratio(
                row.get("通過", ""), row.get("頭数", row.get("頭 数", ""))
            ))

            race_score_list.append(df_time)
            weight_list.append(_parse_weight(row.get("馬体重", "")))
            rank_list.append(rank_val if not ("除" in rank_str or "取" in rank_str) else np.nan)
        else:
            race_score_list.append([np.nan] * 7)
            weight_list.append(np.nan)
            rank_list.append(np.nan)

    result = sum(race_score_list, [])
    result.append(_rank_trend_v3(rank_list))

    wc = (weight_list[0] - weight_list[1]
          if not (np.isnan(weight_list[0]) or np.isnan(weight_list[1]))
          else np.nan)
    result.append(wc)

    try:
        prev_len = float(race_info_df.iloc[0]["course_len"]) if len(race_info_df) > 0 else np.nan
        dist_change = float(current_course_len) - prev_len
    except Exception:
        dist_change = np.nan
    result.append(dist_change)

    return result  # 24要素


# 騎手×コース成績キャッシュ {place_id: {(jockey_id, race_type, course_len): (win_rate, place_rate)}}
_jockey_stats_cache: dict = {}


def _build_jockey_course_stats(place_id):
    """過去5年の結果から騎手×コースの通算勝率・複勝率を構築してキャッシュする"""
    if place_id in _jockey_stats_cache:
        return _jockey_stats_cache[place_id]

    current_year = date.today().year
    dfs = []
    for yr in range(max(2015, current_year - 5), current_year + 1):
        df = race_result_dataset_manager.get_race_results_csv(place_id, yr)
        if not df.empty:
            dfs.append(df.reset_index())

    if not dfs:
        _jockey_stats_cache[place_id] = {}
        return {}

    df_all = pd.concat(dfs, ignore_index=True)
    rank_col = df_all.columns[1]  # reset_index後の1列目が元indexの着順列

    def _to_rank(v):
        s = re.sub(r"\D", "", str(v))
        return int(s) if s else None

    df_all["_rank"] = df_all[rank_col].apply(_to_rank)
    df_all = df_all.dropna(subset=["_rank", "jockey_id", "race_type", "course_len"])

    stats = {}
    for (jid, rt, cl), grp in df_all.groupby(["jockey_id", "race_type", "course_len"]):
        ranks = grp["_rank"].tolist()
        n = len(ranks)
        if n == 0:
            continue
        win_rate   = sum(1 for r in ranks if r == 1) / n
        place_rate = sum(1 for r in ranks if r <= 3) / n
        stats[(str(jid), str(rt), str(cl))] = (win_rate, place_rate)

    _jockey_stats_cache[place_id] = stats
    return stats


def make_dataset_for_lightgbm_v3(race_id, course_info, horse_id):
    """v3特徴量: 血統(36) + 過去3走+dist_change(24) = 60列（騎手率・枠は外で追加）"""
    try:
        race_year = int(str(race_id)[0:4])

        peds_info = horse_peds_dataset_manager.get_peds_info(horse_id)
        df_peds = peds_results_dataset_manager.peds_index(peds_info[0], peds_info[1], course_info, race_year)
        df_peds = sum(df_peds.T.values.tolist(), [])

        race_info = past_performance_dataset_manager.get_past_race_info(horse_id, race_id, race_num=3)
        df_race = get_past_race_info_data_v3(race_info, course_info[2])

        df_lightgbm = df_peds + df_race  # 36 + 24 = 60
        return pd.DataFrame(df_lightgbm).T
    except Exception as e:
        prediction_error(e)
        return pd.DataFrame()


def get_lightgbm_v3_model(place_id, race_type, length):
    """v3モデル (_v3 サフィックス) を取得する"""
    type_str = "turf" if race_type == "芝" else "dirt"
    model_path = os.path.join(
        paths.PREDICTION_MODEL_PATH, PLACE_LIST[place_id - 1],
        f"{type_str}{length}_lambdarank_model_v3.txt"
    )
    return lgb.Booster(model_file=model_path)


def _softmax(x):
    x = np.array(x, dtype=float)
    x = x - np.nanmax(x)
    e = np.exp(x)
    return e / np.nansum(e)


def _blend_proba_v3(v3_scores, odds_vals, ai_weight=0.5):
    """v3_score と 1/odds を確率ベースでブレンドして合成確率を返す（高い=有力）"""
    n = len(v3_scores)
    v3 = pd.to_numeric(v3_scores, errors="coerce").values
    od = pd.to_numeric(odds_vals,  errors="coerce").values

    valid_v3 = ~np.isnan(v3)
    p_ai = np.full(n, np.nan)
    if valid_v3.any():
        p_ai[valid_v3] = _softmax(v3[valid_v3])

    valid_od = (~np.isnan(od)) & (od > 0)
    p_market = np.full(n, np.nan)
    if valid_od.any():
        inv = 1.0 / od[valid_od]
        p_market[valid_od] = inv / inv.sum()

    both    = ~np.isnan(p_ai) & ~np.isnan(p_market)
    ai_only = ~np.isnan(p_ai) & np.isnan(p_market)
    mk_only = np.isnan(p_ai)  & ~np.isnan(p_market)

    p_combined = np.full(n, np.nan)
    p_combined[both]    = ai_weight * p_ai[both]    + (1 - ai_weight) * p_market[both]
    p_combined[ai_only] = p_ai[ai_only]
    p_combined[mk_only] = p_market[mk_only]

    return p_combined, p_ai, p_market


def v3_rank_prediction(race_id, horse_ids, race_info_df, waku_df, jockey_ids=None):
    """v3モデルでスコア・順位を計算する

    Returns:
        pd.DataFrame: v3_score, v3_rank 列（失敗時は空のDataFrame）
    """
    try:
        place_id = int(str(race_id)[4:6])
        course_info = [
            place_id,
            race_info_df.at[0, "race_type"],
            race_info_df.at[0, "course_len"],
            race_info_df.at[0, "ground_state"],
            race_info_df.at[0, "class"],
        ]
        race_type  = course_info[1]
        course_len = str(course_info[2])

        race_dataset = pd.DataFrame()
        for horse_id in horse_ids:
            df_result = make_dataset_for_lightgbm_v3(race_id, course_info, horse_id)
            race_dataset = pd.concat([race_dataset.reset_index(drop=True), df_result.reset_index(drop=True)])

        n = len(race_dataset)

        # 騎手×コース成績を追加（取得できない場合は NaN → fillna(-1) で対応）
        jockey_stats = _build_jockey_course_stats(place_id)
        win_rates, place_rates = [], []
        for i, horse_id in enumerate(horse_ids):
            jid = str(jockey_ids[i]) if jockey_ids is not None and i < len(jockey_ids) else None
            wr, pr = np.nan, np.nan
            if jid:
                wr, pr = jockey_stats.get((jid, race_type, course_len), (np.nan, np.nan))
            win_rates.append(wr)
            place_rates.append(pr)

        jockey_df = pd.DataFrame({
            "jockey_win_rate":   win_rates,
            "jockey_place_rate": place_rates,
        })

        # 特徴量結合: 血統+過去(60) + 騎手率(2) + 枠番+馬番(2) = 64
        race_dataset = pd.concat([
            race_dataset.reset_index(drop=True),
            jockey_df.reset_index(drop=True),
            waku_df.reset_index(drop=True),
        ], axis=1)
        race_dataset = race_dataset.fillna(-1)

        model = get_lightgbm_v3_model(place_id, race_type, course_len)
        y_pred = model.predict(race_dataset, num_iteration=model.best_iteration)
        v3_rank = rank_index(y_pred)

        return pd.DataFrame({"v3_score": y_pred, "v3_rank": v3_rank})
    except Exception as e:
        prediction_error(e)
        return pd.DataFrame()


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


def blended_rank_prediction(race_id, horse_ids, race_info_df, waku_df,
                            popularity_series=None, value_weight=0.5,
                            odds_series=None, jockey_ids=None):
    """AI予想ランキングを計算する

    v3モデルが利用可能な場合は softmax(v3_score) + 1/odds のブレンドをメインの
    score/rank として使用する。v3モデルが存在しない競馬場・コースでは
    v1的中率重視 + v2回収率重視のブレンドにフォールバックする。

    Args:
        race_id (int): race_id
        horse_ids (list): レース出走馬のhorse_idリスト
        race_info_df (pd.DataFrame): レース情報(race_type, course_len, ground_state, class)
        waku_df (pd.DataFrame): 枠番・馬番のデータセット
        popularity_series (pd.Series): 人気（v2回収率重視モデル用）
        value_weight (float): v2回収率重視モデルの重み
        odds_series (pd.Series): オッズ（v3ブレンド用。Noneならv3スコアのみ使用）
        jockey_ids (list): 騎手ID（v3の騎手×コース成績特徴量用）

    Returns:
        pd.DataFrame: score, rank（メイン表示用）、
            score_hitrate, rank_hitrate（v1的中率重視）、
            score_value, rank_value（v2回収率重視）列を持つDataFrame。
            v3モデル使用時は v3_score, v3_rank, p_ai, p_market 列も追加される。
    """
    hitrate_df = rank_prediction(race_id, horse_ids, race_info_df, waku_df)
    if hitrate_df.empty:
        return pd.DataFrame()

    value_df = pd.DataFrame()
    if popularity_series is not None:
        value_df = rank_prediction_value(race_id, horse_ids, race_info_df, waku_df, popularity_series)

    place_id = int(str(race_id)[4:6])
    race_type = race_info_df.at[0, "race_type"] if not race_info_df.empty and "race_type" in race_info_df.columns else ""
    course_len = str(race_info_df.at[0, "course_len"]) if not race_info_df.empty and "course_len" in race_info_df.columns else ""

    n = len(hitrate_df)
    normalize = _global_zscore if USE_GLOBAL_ZSCORE else lambda s, *a, **k: _zscore(s)
    if value_df.empty or len(value_df) != n:
        v1v2_score = normalize(hitrate_df["score"], place_id, race_type, course_len, "hitrate")
        value_df = pd.DataFrame({"score": [np.nan] * n, "rank": [np.nan] * n})
    else:
        v1v2_score = (
            (1 - value_weight) * normalize(hitrate_df["score"], place_id, race_type, course_len, "hitrate")
            + value_weight * normalize(value_df["score"], place_id, race_type, course_len, "value")
        )

    # v3モデルによる予測を試みる
    v3_df = v3_rank_prediction(race_id, horse_ids, race_info_df, waku_df, jockey_ids=jockey_ids)

    if not v3_df.empty and len(v3_df) == n:
        # v3スコア + オッズのブレンドをメインのスコアとして採用
        odds_for_blend = odds_series if odds_series is not None else pd.Series([np.nan] * n)
        p_comb, p_ai, p_market = _blend_proba_v3(v3_df["v3_score"], odds_for_blend)

        valid = ~pd.isna(pd.Series(p_comb))
        blended_rank = np.full(n, np.nan)
        if valid.any():
            blended_rank[valid.values] = rankdata(-np.array(p_comb)[valid.values], method="min").astype(int)

        score_display = _zscore(pd.Series(p_comb)).values
        result_df = pd.DataFrame({"score": score_display, "rank": blended_rank})
        result_df["rank"] = result_df["rank"].astype("Int64")
        result_df["score_hitrate"] = hitrate_df["score"]
        result_df["rank_hitrate"]  = hitrate_df["rank"]
        result_df["score_value"]   = value_df["score"]
        result_df["rank_value"]    = value_df["rank"]
        result_df["v3_score"]      = v3_df["v3_score"].values
        result_df["v3_rank"]       = v3_df["v3_rank"].values
        result_df["p_ai"]          = p_ai
        result_df["p_market"]      = p_market
        print(f"  ✓ v3ブレンド予想 (race_id={race_id})")
        return result_df

    # v3モデルなし → v1/v2フォールバック
    result_df = pd.DataFrame({"score": v1v2_score})
    result_df["rank"] = rank_index(result_df["score"].tolist())
    result_df["score_hitrate"] = hitrate_df["score"]
    result_df["rank_hitrate"]  = hitrate_df["rank"]
    result_df["score_value"]   = value_df["score"]
    result_df["rank_value"]    = value_df["rank"]
    return result_df


# ============================================================
# マルチモデル予測（3戦略: 的中率重視 / 回収率重視 / MAR推奨）
# ============================================================

_mm_model_cache: dict = {}
_mm_ped_vocab = None


def _mm_get_model(place_id, race_type, length, suffix):
    key = (place_id, race_type, length, suffix)
    if key not in _mm_model_cache:
        type_str = "turf" if race_type == "芝" else "dirt"
        mp = os.path.join(paths.PREDICTION_MODEL_PATH, PLACE_LIST[place_id - 1],
                          f"{type_str}{length}_lambdarank_model{suffix}.txt")
        if not os.path.isfile(mp):
            raise FileNotFoundError(mp)
        _mm_model_cache[key] = lgb.Booster(model_file=mp)
    return _mm_model_cache[key]


def _mm_vocab():
    global _mm_ped_vocab
    if _mm_ped_vocab is None and _MM_AVAILABLE:
        _mm_ped_vocab = _mm_build_pedigree_vocab()
    return _mm_ped_vocab


def _mm_norm(arr):
    arr = np.array(arr, dtype=float)
    mn, mx = np.nanmin(arr), np.nanmax(arr)
    return (arr - mn) / (mx - mn + 1e-12)


def _mm_blend(s_vx, s_v7, alpha):
    return (1 - alpha) * _mm_norm(s_vx) + alpha * _mm_norm(s_v7)


def _mm_strategy_score(s_tan, s_san):
    """本命が1位・3連複上位が2〜5位になる統合スコア。"""
    s_tan_n = _mm_norm(s_tan)
    s_san_n = _mm_norm(s_san)
    honmei  = int(np.argmax(s_tan_n))
    combined = s_san_n.copy()
    spread = combined.max() - combined.min() + 1e-6
    combined[honmei] = combined.max() + spread
    return combined


def multi_model_rank_prediction(race_id, horse_ids, race_info_df, waku_df,
                                kinryo_series=None, jockey_ids=None, odds_series=None):
    """3戦略（的中率重視/回収率重視/MAR推奨）の統合指数とランクを計算する。

    戦略:
      ②的中率重視: 単複=v11α0.6  / 3連複=v15α0.5
      ①③回収率重視: 単複=v12_nodds / 3連複=v12_nodds
      ④MAR推奨:    単複=v11_nodds / 3連複=v12α0.4

    Returns:
        pd.DataFrame: idx_hitrate, rank_hitrate, idx_value, rank_value,
                      idx_mar, rank_mar 列（失敗時は空のDataFrame）
    """
    if not _MM_AVAILABLE:
        return pd.DataFrame()
    try:
        place_id   = int(str(race_id)[4:6])
        race_type  = race_info_df.at[0, "race_type"]
        course_len = str(race_info_df.at[0, "course_len"])
        course_info = [place_id, race_type, course_len,
                       race_info_df.at[0, "ground_state"], race_info_df.at[0, "class"]]
        n     = len(horse_ids)
        vocab = _mm_vocab()
        js    = _build_jockey_course_stats(place_id)
        today = date.today()

        feat_rows_v15   = []   # 104列（オッズあり、重複あり）
        feat_rows_nodds = []   # 102列（オッズなし、重複あり）

        for idx, hid in enumerate(horse_ids):
            # v3 base (60列)
            try:
                row_v3 = make_dataset_for_lightgbm_v3(race_id, course_info, hid)
                v3_vals = row_v3.values[0].tolist() if not row_v3.empty else [np.nan] * 60
            except Exception:
                v3_vals = [np.nan] * 60

            # pedigree cats
            f_cat, mf_cat, pgf_cat = _mm_get_pedigree_cats(hid, vocab)

            # 過去5走
            try:
                r5 = past_performance_dataset_manager.get_past_race_info(hid, race_id, race_num=5)
            except Exception:
                r5 = pd.DataFrame()

            ev6 = _mm_get_v6_extra(r5) if not r5.empty else [np.nan] * 16
            ev8 = _mm_get_v8_extra(r5) if not r5.empty else [np.nan] * 7
            ev9 = _mm_get_v9_extra(r5) if not r5.empty else [np.nan] * 7

            # v7 extra（前走情報）
            days_since = n_horses_1 = hw_abs_1 = np.nan
            if not r5.empty:
                try:
                    prev_dt = pd.to_datetime(
                        str(r5.iloc[0].get("日付", "")).strip().replace("/", "-"),
                        format="%Y-%m-%d",
                    )
                    days_since = float((today - prev_dt.date()).days)
                except Exception:
                    pass
                try:
                    n_horses_1 = float(str(r5.iloc[0].get("頭数", "")).strip())
                except Exception:
                    pass
                hw_abs_1 = _parse_weight(r5.iloc[0].get("馬体重", ""))

            # 騎手成績
            jid = str(jockey_ids[idx]) if jockey_ids and idx < len(jockey_ids) else None
            wr, pr = js.get((jid, race_type, course_len), (np.nan, np.nan)) if jid else (np.nan, np.nan)

            # 斤量
            kry = np.nan
            if kinryo_series is not None and idx < len(kinryo_series):
                try:
                    kry = _mm_parse_kinryo(kinryo_series.iloc[idx])
                except Exception:
                    pass

            # 枠・馬番
            waku_v   = float(waku_df["枠"].iloc[idx])  if "枠"  in waku_df.columns else np.nan
            umaban_v = float(waku_df["馬番"].iloc[idx]) if "馬番" in waku_df.columns else np.nan

            # オッズ・人気
            odds_v = np.nan
            if odds_series is not None and idx < len(odds_series):
                try:
                    odds_v = float(odds_series.iloc[idx])
                except Exception:
                    pass

            n_hs = float(n)

            def _ev(lst, i, default=np.nan):
                return lst[i] if len(lst) > i else default

            # 104列（オッズあり）
            feat_rows_v15.append(
                v3_vals
                + [wr, pr, waku_v, umaban_v, odds_v, np.nan]           # jockey+waku+odds+pop
                + [f_cat, mf_cat, pgf_cat]                              # ped_cats
                + list(ev6)                                              # v6_extra
                + [kry, days_since, n_hs, n_horses_1, hw_abs_1]        # v7_extra
                + list(ev8[:5])                                         # corner_chase
                + [_ev(ev8, 5), _ev(ev8, 6)]                           # agari_trend, time_diff_trend
                + list(ev9[:5])                                         # agari_df_course
                + [_ev(ev9, 5), _ev(ev9, 6)]                           # corner_ratio_std5, agari_std5
            )  # 104列

            # 102列（オッズなし: current_odds/pop を除く）
            feat_rows_nodds.append(
                v3_vals
                + [wr, pr, waku_v, umaban_v]                           # jockey+waku（oddsなし）
                + [f_cat, mf_cat, pgf_cat]
                + list(ev6)
                + [kry, days_since, n_hs, n_horses_1, hw_abs_1]
                + list(ev8[:5])
                + [_ev(ev8, 5), _ev(ev8, 6)]
                + list(ev9[:5])
                + [_ev(ev9, 5), _ev(ev9, 6)]
            )  # 102列

        # ── オッズあり DataFrame（列名付き 104列）──
        df_v15 = pd.DataFrame(feat_rows_v15, columns=_MM_IX9_NO_RACE).fillna(-1)

        def _score_v(suffix, n_cols):
            m = _mm_get_model(place_id, race_type, course_len, suffix)
            return m.predict(df_v15.iloc[:, :n_cols], num_iteration=m.best_iteration)

        s_v7  = _score_v("_v7_no26",  66)
        s_v11 = _score_v("_v11_no26", 90)
        s_v12 = _score_v("_v12_no26", 97)
        s_v15 = _score_v("_v15_no26", 104)

        # ── オッズなし DataFrame（重複排除 100列）──
        df_nodds = pd.DataFrame(feat_rows_nodds, columns=_MM_NODDS_DUPS).fillna(-1)
        df_nodds = df_nodds.loc[:, ~df_nodds.columns.duplicated(keep="first")]

        def _score_n(suffix, cols):
            m = _mm_get_model(place_id, race_type, course_len, suffix)
            return m.predict(df_nodds[cols], num_iteration=m.best_iteration)

        s_v11n = _score_n("_v11nodds_no26", _MM_V11N_COLS)
        s_v12n = _score_n("_v12nodds_no26", _MM_V12N_COLS)

        # ── 戦略ブレンドスコア ──
        # ②的中率重視: 単複=v11α0.6 / 3連複=v15α0.5
        s_tan_hr  = _mm_blend(s_v11, s_v7, 0.6)
        s_san_hr  = _mm_blend(s_v15, s_v7, 0.5)
        # ①③回収率重視: 単複=v12n / 3連複=v12n
        s_tan_val = _mm_norm(s_v12n)
        s_san_val = _mm_norm(s_v12n)
        # ④MAR推奨: 単複=v11n / 3連複=v12α0.4
        s_tan_mar = _mm_norm(s_v11n)
        s_san_mar = _mm_blend(s_v12, s_v7, 0.4)

        # ── 統合スコア → ランク・指数 ──
        def _to_rank_idx(s_tan, s_san):
            combined = _mm_strategy_score(s_tan, s_san)
            z   = _zscore(pd.Series(combined))
            idx = [score_to_index(v) for v in z]
            rnk = rank_index(combined.tolist())
            return idx, rnk

        idx_hr,  rnk_hr  = _to_rank_idx(s_tan_hr,  s_san_hr)
        idx_val, rnk_val = _to_rank_idx(s_tan_val, s_san_val)
        idx_mar, rnk_mar = _to_rank_idx(s_tan_mar, s_san_mar)

        return pd.DataFrame({
            "idx_hitrate":  idx_hr,
            "rank_hitrate": rnk_hr,
            "idx_value":    idx_val,
            "rank_value":   rnk_val,
            "idx_mar":      idx_mar,
            "rank_mar":     rnk_mar,
        })

    except Exception as e:
        prediction_error(e)
        return pd.DataFrame()
