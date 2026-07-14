"""LightGBM学習データセット生成 v7

v6 の特徴量（86列）に以下の5列を追加:
  - kinryo              : 今走の斤量（kg）
  - days_since_last_race: 前走からの日数（初出走はNaN）
  - n_horses_today      : 今走の出走頭数
  - n_horses_1          : 前走の出走頭数
  - horse_weight_abs_1  : 前走時の馬体重絶対値（kg）

days_since_last_race（間隔）は競馬では重要な特徴量:
  中1週（7日）= 疲労懸念、中2〜4週（14〜28日）= 標準的な間隔、
  中10週以上（70日+）= 長期休養明けで状態不確実。

v6=86列 → v7=91列。データは "_v7" サフィックスで保存。
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
import past_performance

_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.config import paths as paths_v3
from src.PredictionModels.LightGBM.make_dataset_v2 import (
    _parse_agari, _parse_margin, _parse_corner_ratio, _get_time_info, _parse_weight,
)
from src.PredictionModels.LightGBM.make_dataset_v3 import build_jockey_course_stats
from src.PredictionModels.LightGBM.make_dataset_v4 import _parse_odds, _parse_popularity
from src.PredictionModels.LightGBM.make_dataset_v5 import (
    index_v5,
    make_dataset_for_lightGBM_v3,
    build_pedigree_vocab,
    get_pedigree_cats,
)
from src.PredictionModels.LightGBM.make_dataset_v6 import (
    index_v6,
    get_extra_past_race_features_v6,
)

# v7 特徴量列名: v6（86列）+ 5列 = 91列
index_v7 = index_v6 + [
    "kinryo",               # 今走の斤量（kg）
    "days_since_last_race", # 前走からの日数
    "n_horses_today",       # 今走の出走頭数
    "n_horses_1",           # 前走の出走頭数
    "horse_weight_abs_1",   # 前走時の馬体重絶対値（kg）
]


def _parse_date_flex(s):
    """'2025年02月01日' or '2025/02/01' → pd.Timestamp。失敗時は NaT。"""
    try:
        s = str(s).strip()
        s = re.sub(r"[年月]", "-", s).replace("日", "").replace("/", "-")
        return pd.to_datetime(s, format="%Y-%m-%d")
    except Exception:
        return pd.NaT


def _parse_kinryo(val):
    """斤量文字列 → float。失敗時は NaN。"""
    try:
        return float(str(val).strip())
    except Exception:
        return np.nan


def _parse_n_horses(val):
    """出走頭数文字列 → float。失敗時は NaN。"""
    try:
        v = float(str(val).strip())
        return v if v > 0 else np.nan
    except Exception:
        return np.nan


def get_v7_extra_features(current_date_str, race_info_df_5):
    """v7 追加の5特徴量を返す（past_performance 5走分データを受け取る）。

    Args:
        current_date_str: 今走の日付文字列（race_results["date"] 形式: '2025年02月01日'）
        race_info_df_5  : past_performance.get_past_race_info の結果（最大5走）

    Returns:
        list of 4 elements:
            [days_since_last_race, n_horses_1, horse_weight_abs_1]
        ※ kinryo と n_horses_today は呼び出し元で付与する（race_results から取得）
    """
    # 前走からの日数
    days_since = np.nan
    if not race_info_df_5.empty:
        current_dt = _parse_date_flex(current_date_str)
        prev_date_str = race_info_df_5.iloc[0].get("日付", "")
        prev_dt = _parse_date_flex(prev_date_str)
        if not pd.isna(current_dt) and not pd.isna(prev_dt):
            days_since = float((current_dt - prev_dt).days)

    # 前走の出走頭数
    n_horses_1 = np.nan
    if not race_info_df_5.empty:
        n_horses_1 = _parse_n_horses(race_info_df_5.iloc[0].get("頭数", ""))

    # 前走の馬体重絶対値
    horse_weight_abs_1 = np.nan
    if not race_info_df_5.empty:
        horse_weight_abs_1 = _parse_weight(race_info_df_5.iloc[0].get("馬体重", ""))

    return [days_since, n_horses_1, horse_weight_abs_1]


def make_dataset_for_lightGBM_v7(race_id, course_info, horse_id, vocab, current_date_str):
    """v7特徴量の1行（v3の60列 + v6追加16列 = 76列）を作成する。
    kinryo / n_horses_today は呼び出し元で付与。days_since 等3列もここで付与。
    """
    try:
        # v3 ベース（血統36 + 過去3走24 = 60列）
        row_v3 = make_dataset_for_lightGBM_v3(race_id, course_info, horse_id)
        if row_v3.empty:
            return pd.DataFrame(), [np.nan, np.nan, np.nan]

        # 5走分データ取得
        race_info_5 = past_performance.get_past_race_info(horse_id, race_id, race_num=5)

        # v6 追加特徴量（16列: 4走前5走前 + trend集計）
        extra_v6 = get_extra_past_race_features_v6(race_info_5)

        # v7 追加特徴量（3列: days_since/n_horses_1/horse_weight_abs_1）
        extra_v7 = get_v7_extra_features(current_date_str, race_info_5)

        extra_df = pd.DataFrame([extra_v6])
        row_combined = pd.concat(
            [row_v3.reset_index(drop=True), extra_df.reset_index(drop=True)], axis=1
        )
        return row_combined, extra_v7
    except Exception as e:
        print(f"  make_dataset_v7 error ({race_id}, {horse_id}): {e}")
        return pd.DataFrame(), [np.nan, np.nan, np.nan]


def save_dataset_v7(place_id, year, race_type, length, df, flag_list):
    out_dir = os.path.join(
        paths_v3.PREDICTION_DATASET_PATH, name_header.PLACE_LIST[place_id - 1]
    )
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, f"{year}_{race_type}{length}_ai_dataset")
    df.reset_index(drop=True).to_csv(base + "_for_rank_v7.csv")
    pd.DataFrame(flag_list, columns=["result_flag"]).to_csv(base + "_flag_v7.csv")


def load_dataset_v7(place_id, year, race_type, length):
    out_dir = os.path.join(
        paths_v3.PREDICTION_DATASET_PATH, name_header.PLACE_LIST[place_id - 1]
    )
    base = os.path.join(out_dir, f"{year}_{race_type}{length}_ai_dataset")
    df_path   = base + "_for_rank_v7.csv"
    flag_path = base + "_flag_v7.csv"
    df   = pd.read_csv(df_path,   index_col=0, dtype=float) if os.path.isfile(df_path) else pd.DataFrame()
    flag = pd.read_csv(flag_path, index_col=0, dtype=int)   if os.path.isfile(flag_path) else pd.DataFrame()
    return df, flag


def make_dataset_for_train_v7(place_id, year=date.today().year, vocab=None, course_filter=None):
    """指定競馬場・年の v7 学習データセットを作成・保存する。"""
    from src.PredictionModels.LightGBM.make_dataset import relevance_grade

    if vocab is None:
        vocab = build_pedigree_vocab()

    df_results = race_results.get_race_results_csv(place_id, year)
    if df_results.empty:
        print(f"  データなし: {year} {name_header.PLACE_LIST[place_id - 1]}")
        return

    # 今走の出走頭数を race_id ごとに事前計算
    n_horses_map = df_results.groupby(df_results.index).size().to_dict()

    courses = name_header.COURSE_LISTS[place_id - 1]
    if course_filter is not None:
        courses = [(t, l) for t, l in courses if (t, l) in course_filter]

    print(f"  騎手×コース成績テーブル構築中...")
    jockey_lookup = build_jockey_course_stats(place_id, year)
    print(f"  ルックアップエントリ数: {len(jockey_lookup)}")

    _V3_NCOLS = 60

    for race_type, length in courses:
        df_course = df_results[
            (df_results["race_type"] == race_type) & (df_results["course_len"] == length)
        ]
        if df_course.empty:
            continue

        print(f"  {race_type}{length}m ({len(df_course)}走)")
        race_id_list = df_course.index.tolist()
        flag_list = []
        jockey_wins, jockey_places = [], []
        odds_list, popularity_list = [], []
        father_ids, mf_ids, pgf_ids = [], [], []
        kinryo_list = []
        n_horses_today_list = []
        days_since_list, n_horses_1_list, hw_abs_1_list = [], [], []

        df_v3_base  = pd.DataFrame()
        df_v6_extra = pd.DataFrame()

        for i in tqdm(range(len(df_course))):
            df_result = df_course.iloc[i:i + 1]
            race_id   = int(df_result.index[0])
            flag_list.append(relevance_grade(df_result, race_id))

            course_info = [
                place_id,
                df_result.iloc[0]["race_type"],
                df_result.iloc[0]["course_len"],
                df_result.iloc[0]["ground_state"],
                df_result.iloc[0]["class"],
            ]
            horse_id         = df_result.iloc[0]["horse_id"]
            current_date_str = df_result.iloc[0].get("date", "")

            row, extra_v7 = make_dataset_for_lightGBM_v7(
                race_id, course_info, horse_id, vocab, current_date_str
            )

            if not row.empty:
                df_v3_base  = pd.concat([df_v3_base,  row.iloc[:, :_V3_NCOLS]])
                df_v6_extra = pd.concat([df_v6_extra, row.iloc[:, _V3_NCOLS:]])
            else:
                df_v3_base  = pd.concat([df_v3_base,  pd.DataFrame([[np.nan] * _V3_NCOLS])])
                df_v6_extra = pd.concat([df_v6_extra, pd.DataFrame([[np.nan] * 16])])

            key = (str(race_id), str(horse_id))
            jw, jp = jockey_lookup.get(key, (np.nan, np.nan))
            jockey_wins.append(jw)
            jockey_places.append(jp)

            odds_list.append(_parse_odds(df_result.iloc[0].get("単勝", np.nan)))
            popularity_list.append(_parse_popularity(df_result.iloc[0].get("人気", "")))

            f_cat, mf_cat, pgf_cat = get_pedigree_cats(horse_id, vocab)
            father_ids.append(f_cat)
            mf_ids.append(mf_cat)
            pgf_ids.append(pgf_cat)

            kinryo_list.append(_parse_kinryo(df_result.iloc[0].get("斤量", "")))
            n_horses_today_list.append(float(n_horses_map.get(race_id, np.nan)))
            days_since_list.append(extra_v7[0])
            n_horses_1_list.append(extra_v7[1])
            hw_abs_1_list.append(extra_v7[2])

        # 列順: race_id(1) + v3_base(60) + jockey(2) + waku/umaban(2)
        #       + odds/pop(2) + ped_cat(3) + v6_extra(16) + v7_extra(5) = 91列
        df_dataset = pd.concat([
            pd.DataFrame(race_id_list,       columns=["race_id"]),
            df_v3_base.reset_index(drop=True),
            pd.Series(jockey_wins,           name="jockey_win_rate"),
            pd.Series(jockey_places,         name="jockey_place_rate"),
            df_course["枠番"].reset_index(drop=True),
            df_course["馬番"].reset_index(drop=True),
            pd.Series(odds_list,             name="current_odds"),
            pd.Series(popularity_list,       name="current_popularity"),
            pd.Series(father_ids,            name="father_cat"),
            pd.Series(mf_ids,               name="mother_father_cat"),
            pd.Series(pgf_ids,              name="paternal_gf_cat"),
            df_v6_extra.reset_index(drop=True),
            pd.Series(kinryo_list,           name="kinryo"),
            pd.Series(days_since_list,       name="days_since_last_race"),
            pd.Series(n_horses_today_list,   name="n_horses_today"),
            pd.Series(n_horses_1_list,       name="n_horses_1"),
            pd.Series(hw_abs_1_list,         name="horse_weight_abs_1"),
        ], axis=1)
        df_dataset.columns = index_v7

        save_dataset_v7(place_id, year, race_type, length, df_dataset, flag_list)
        print(f"    保存完了: {len(df_dataset)}行")
