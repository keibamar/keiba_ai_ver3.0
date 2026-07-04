"""LightGBM学習データセット生成 v4

v3 の特徴量に以下を追加（的中率重視モデル v4）:
  - current_odds       : 今走の単勝オッズ（訓練時=race_results.単勝, ライブ時=スクレイピング値）
  - current_popularity : 今走の人気順位（訓練時=race_results.人気, ライブ時=スクレイピング値）

v1=51列、v2=62列、v3=65列、v4=67列。
データは "_v4" サフィックスで保存。
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
import race_results

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
from src.config import paths as paths_v3
from src.PredictionModels.LightGBM.make_dataset_v3 import (
    make_dataset_for_lightGBM_v3,
    build_jockey_course_stats,
    index_v3,
)

# v4 特徴量列名（v3 の 65列 + 2列）
index_v4 = index_v3 + ["current_odds", "current_popularity"]


def _parse_odds(v):
    """単勝文字列 → float。'---' や空文字は NaN。"""
    s = str(v).strip()
    if s in ("---", "---.-", "", "nan"):
        return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


def _parse_popularity(v):
    """人気文字列 → float。空や非数値は NaN。'1.0' → 1.0 に正しく変換。"""
    try:
        return float(int(float(str(v).strip())))
    except (ValueError, TypeError):
        return np.nan


def save_dataset_v4(place_id, year, race_type, length, df, flag_list):
    out_dir = os.path.join(paths_v3.PREDICTION_DATASET_PATH, name_header.PLACE_LIST[place_id - 1])
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, f"{year}_{race_type}{length}_ai_dataset")
    df.reset_index(drop=True).to_csv(base + "_for_rank_v4.csv")
    pd.DataFrame(flag_list, columns=["result_flag"]).to_csv(base + "_flag_v4.csv")


def load_dataset_v4(place_id, year, race_type, length):
    out_dir = os.path.join(paths_v3.PREDICTION_DATASET_PATH, name_header.PLACE_LIST[place_id - 1])
    base = os.path.join(out_dir, f"{year}_{race_type}{length}_ai_dataset")
    df_path = base + "_for_rank_v4.csv"
    flag_path = base + "_flag_v4.csv"
    df = pd.read_csv(df_path, index_col=0, dtype=float) if os.path.isfile(df_path) else pd.DataFrame()
    flag = pd.read_csv(flag_path, index_col=0, dtype=int) if os.path.isfile(flag_path) else pd.DataFrame()
    return df, flag


def make_dataset_for_train_v4(place_id, year=date.today().year, course_filter=None):
    """指定競馬場・年の v4 学習データセットを作成・保存する。"""
    from src.PredictionModels.LightGBM.make_dataset import relevance_grade

    df_results = race_results.get_race_results_csv(place_id, year)
    if df_results.empty:
        print(f"  データなし: {year} {name_header.PLACE_LIST[place_id - 1]}")
        return

    courses = name_header.COURSE_LISTS[place_id - 1]
    if course_filter is not None:
        courses = [(t, l) for t, l in courses if (t, l) in course_filter]

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
        odds_list = []
        popularity_list = []

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

            key = (str(race_id), str(horse_id))
            jw, jp = jockey_lookup.get(key, (np.nan, np.nan))
            jockey_wins.append(jw)
            jockey_places.append(jp)

            # current_odds / current_popularity（確定オッズ＝発走前オッズの近似）
            odds_list.append(_parse_odds(df_result.iloc[0].get("単勝", np.nan)))
            popularity_list.append(_parse_popularity(df_result.iloc[0].get("人気", "")))

        df_dataset = pd.concat([
            pd.DataFrame(race_id_list, columns=["race_id"]),
            df_dataset.reset_index(drop=True),
            pd.Series(jockey_wins,    name="jockey_win_rate"),
            pd.Series(jockey_places,  name="jockey_place_rate"),
            df_course["枠番"].reset_index(drop=True),
            df_course["馬番"].reset_index(drop=True),
            pd.Series(odds_list,       name="current_odds"),
            pd.Series(popularity_list, name="current_popularity"),
        ], axis=1)
        df_dataset.columns = index_v4

        save_dataset_v4(place_id, year, race_type, length, df_dataset, flag_list)
        print(f"    保存完了: {len(df_dataset)}行")
