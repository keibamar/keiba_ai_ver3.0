"""LightGBM学習データセット生成 v3

make_dataset_v2.py の特徴量に以下を追加する（Phase 2）:
- 騎手×コース勝率 / 複勝率（jockey_win_rate, jockey_place_rate）
- 距離変更差分（dist_change: 今回距離 - 前走距離, m）

データは "_v3" サフィックスで保存。
v1=51列、v2=62列、v3=65列。
"""

import os
import re
import sys

from datetime import date
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.dont_write_bytecode = True
sys.path.append(r"C:\keiba_ai\keiba_ai_ver2.0\libs")
sys.path.append(r"C:\keiba_ai\keiba_ai_ver2.0\src\Datasets")
import name_header
import horse_peds
import race_results
import peds_results
import past_performance
import average_time

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
from src.config import paths as paths_v3
from src.PredictionModels.LightGBM.make_dataset_v2 import (
    _parse_agari, _parse_margin, _parse_corner_ratio,
    _parse_weight, _rank_trend, _get_time_info,
)

# v3 特徴量列名（v2の62列 + Phase2の3列 = 65列）
index_v3 = [
    "race_id",
    # 血統（36列）
    "f_course1", "f_course2", "f_course3", "f_courseout",
    "mf_course1", "mf_course2", "mf_course3", "mf_courseout",
    "f_mf_course1", "f_mf_course2", "f_mf_course3", "f_mf_courseout",
    "f_len1", "f_len2", "f_len3", "f_lenout",
    "mf_len1", "mf_len2", "mf_len3", "mf_classout",
    "f_mf_len1", "f_mf_len2", "f_mf_len3", "f_mf_classout",
    "f_class1", "f_class2", "f_class3", "f_classout",
    "mf_class1", "mf_class2", "mf_class3", "mf_classout",
    "f_mf_class1", "f_mf_class2", "f_mf_class3", "f_mf_classout",
    # 過去3走（7列×3走 = 21列）
    "time_df_course1", "time_df_class1", "ninki_1", "result_1",
    "agari_1", "margin_1", "corner_ratio_1",
    "time_df_course2", "time_df_class2", "ninki_2", "result_2",
    "agari_2", "margin_2", "corner_ratio_2",
    "time_df_course3", "time_df_class3", "ninki_3", "result_3",
    "agari_3", "margin_3", "corner_ratio_3",
    # 全体特徴量（2列）
    "rank_trend",
    "weight_change",
    # Phase 2 新規（3列）
    "dist_change",          # 今回距離 - 前走距離（m）
    "jockey_win_rate",      # 同コースでの騎手勝率
    "jockey_place_rate",    # 同コースでの騎手複勝率
    # 枠順（2列）
    "waku", "umaban",
]


# ---------- 日付パース ----------

def _parse_date_jp(s):
    """'2023年01月05日' → pd.Timestamp。失敗時は NaT。"""
    try:
        return pd.to_datetime(re.sub(r"[年月]", "-", str(s)).replace("日", ""), format="%Y-%m-%d")
    except Exception:
        return pd.NaT


# ---------- 騎手×コース成績テーブル構築 ----------

def build_jockey_course_stats(place_id, target_year, lookback_years=5):
    """
    target_year の各馬について、そのレース直前時点での
    騎手×（race_type, course_len）の勝率・複勝率を計算する。

    Returns:
        dict: {(race_id_str, horse_id_str): (win_rate, place_rate)}
    """
    start_year = max(2015, target_year - lookback_years)
    dfs = []
    for yr in range(start_year, target_year + 1):
        df = race_results.get_race_results_csv(place_id, yr)
        if not df.empty:
            # race_id を列に移してリセット
            dfs.append(df.reset_index(names=["race_id"]))
    if not dfs:
        return {}

    df_all = pd.concat(dfs, ignore_index=True)

    # reset_index 後: col0=race_id, col1=着順（元col0）
    rank_col = df_all.columns[1]

    def _to_rank(v):
        s = re.sub(r"\D", "", str(v))
        return int(s) if s else None

    df_all["_rank"] = df_all[rank_col].apply(_to_rank)
    df_all["_is_win"] = (df_all["_rank"] == 1).astype(float)
    df_all["_is_place"] = (df_all["_rank"] <= 3).astype(float)
    df_all["_date_dt"] = df_all["date"].apply(_parse_date_jp)

    df_all = df_all.dropna(subset=["_rank", "_date_dt", "jockey_id"]).copy()
    df_all = df_all.sort_values("_date_dt").reset_index(drop=True)

    win_arr = np.full(len(df_all), np.nan)
    place_arr = np.full(len(df_all), np.nan)

    for _, grp in df_all.groupby(["jockey_id", "race_type", "course_len"]):
        idx = grp.index.tolist()
        wins = grp["_is_win"].values
        places = grp["_is_place"].values
        cum_w = np.cumsum(wins)
        cum_p = np.cumsum(places)
        counts = np.arange(1, len(wins) + 1, dtype=float)
        # shift(1): 当該レース直前まで（初出走は NaN のまま）
        win_arr[idx[1:]] = cum_w[:-1] / counts[:-1]
        place_arr[idx[1:]] = cum_p[:-1] / counts[:-1]

    df_all["_win_rate"] = win_arr
    df_all["_place_rate"] = place_arr

    lookup = {}
    for _, row in df_all.iterrows():
        key = (str(row["race_id"]), str(row["horse_id"]))
        lookup[key] = (row["_win_rate"], row["_place_rate"])
    return lookup


# ---------- v3 過去レース特徴量取得 ----------

def get_past_race_info_data_v3(race_info_df, current_course_len):
    """
    v2 の特徴量 + dist_change（今回距離 - 前走距離）を返す。
    Returns: list 24要素（7×3 + rank_trend + weight_change + dist_change）
    """
    race_score_list = []
    weight_list = []
    rank_list = []

    for i in range(3):
        if i < len(race_info_df.index):
            row = race_info_df.iloc[i]
            df_time = _get_time_info(row)

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

    # 近走着順トレンド・体重変化（v2と同じ）
    result.append(_rank_trend(rank_list))
    wc = (weight_list[0] - weight_list[1]
          if not (np.isnan(weight_list[0]) or np.isnan(weight_list[1]))
          else np.nan)
    result.append(wc)

    # 距離変更差分
    try:
        prev_len = float(race_info_df.iloc[0]["course_len"]) if len(race_info_df) > 0 else np.nan
        dist_change = float(current_course_len) - prev_len
    except Exception:
        dist_change = np.nan
    result.append(dist_change)

    return result  # 24要素


# ---------- データセット作成 ----------

def make_dataset_for_lightGBM_v3(race_id, course_info, horse_id):
    """v3特徴量の1行（血統36 + 過去24 = 60列）を作成する。"""
    try:
        race_year = int(str(race_id)[:4])
        peds_info = horse_peds.get_peds_info(horse_id)
        df_peds = peds_results.peds_index(peds_info[0], peds_info[1], course_info, race_year)
        df_peds = sum(df_peds.T.values.tolist(), [])

        race_info = past_performance.get_past_race_info(horse_id, race_id, race_num=3)
        df_race = get_past_race_info_data_v3(race_info, course_info[2])

        row = df_peds + df_race  # 36 + 24 = 60
        return pd.DataFrame([row])
    except Exception as e:
        print(f"  make_dataset_v3 error ({race_id}, {horse_id}): {e}")
        return pd.DataFrame()


def save_dataset_v3(place_id, year, race_type, length, df, flag_list):
    out_dir = os.path.join(paths_v3.PREDICTION_DATASET_PATH, name_header.PLACE_LIST[place_id - 1])
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, f"{year}_{race_type}{length}_ai_dataset")
    df.reset_index(drop=True).to_csv(base + "_for_rank_v3.csv")
    pd.DataFrame(flag_list, columns=["result_flag"]).to_csv(base + "_flag_v3.csv")


def load_dataset_v3(place_id, year, race_type, length):
    out_dir = os.path.join(paths_v3.PREDICTION_DATASET_PATH, name_header.PLACE_LIST[place_id - 1])
    base = os.path.join(out_dir, f"{year}_{race_type}{length}_ai_dataset")
    df_path = base + "_for_rank_v3.csv"
    flag_path = base + "_flag_v3.csv"
    df = pd.read_csv(df_path, index_col=0, dtype=float) if os.path.isfile(df_path) else pd.DataFrame()
    flag = pd.read_csv(flag_path, index_col=0, dtype=int) if os.path.isfile(flag_path) else pd.DataFrame()
    return df, flag


def make_dataset_for_train_v3(place_id, year=date.today().year, course_filter=None):
    """指定競馬場・年の v3 学習データセットを作成・保存する。"""
    from src.PredictionModels.LightGBM.make_dataset import relevance_grade

    df_results = race_results.get_race_results_csv(place_id, year)
    if df_results.empty:
        print(f"  データなし: {year} {name_header.PLACE_LIST[place_id - 1]}")
        return

    courses = name_header.COURSE_LISTS[place_id - 1]
    if course_filter is not None:
        courses = [(t, l) for t, l in courses if (t, l) in course_filter]

    # 騎手×コース成績テーブルを年単位で1回だけ構築
    print(f"  騎手×コース成績テーブル構築中...")
    jockey_lookup = build_jockey_course_stats(place_id, year)
    print(f"  ルックアップエントリ数: {len(jockey_lookup)}")

    for race_type, length in courses:
        df_course = df_results[
            (df_results["race_type"] == race_type) & (df_results["course_len"] == length)
        ]
        if df_course.empty:
            continue

        print(f"  {race_type}{length}m ({len(df_course)}走)")
        race_id_list = df_course.index.tolist()
        df_dataset = pd.DataFrame()
        flag_list = []
        jockey_wins = []
        jockey_places = []

        for i in tqdm(range(len(df_course))):
            df_result = df_course.iloc[i:i+1]
            race_id = int(df_result.index[0])
            flag_list.append(relevance_grade(df_result, race_id))

            course_info = [
                place_id,
                df_result.iloc[0]["race_type"],
                df_result.iloc[0]["course_len"],
                df_result.iloc[0]["ground_state"],
                df_result.iloc[0]["class"],
            ]
            horse_id = df_result.iloc[0]["horse_id"]
            row = make_dataset_for_lightGBM_v3(race_id, course_info, horse_id)
            df_dataset = pd.concat([df_dataset, row])

            # jockey stats をルックアップ（ループ内で収集し後でまとめて結合）
            key = (str(race_id), str(horse_id))
            jw, jp = jockey_lookup.get(key, (np.nan, np.nan))
            jockey_wins.append(jw)
            jockey_places.append(jp)

        # 全特徴量を結合: race_id(1) + peds+past(60) + jockey(2) + waku+umaban(2) = 65
        df_dataset = pd.concat([
            pd.DataFrame(race_id_list, columns=["race_id"]),
            df_dataset.reset_index(drop=True),
            pd.Series(jockey_wins, name="jockey_win_rate"),
            pd.Series(jockey_places, name="jockey_place_rate"),
            df_course["枠番"].reset_index(drop=True),
            df_course["馬番"].reset_index(drop=True),
        ], axis=1)
        df_dataset.columns = index_v3

        save_dataset_v3(place_id, year, race_type, length, df_dataset, flag_list)
        print(f"    保存完了: {len(df_dataset)}行")
