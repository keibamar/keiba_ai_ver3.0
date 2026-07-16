"""LightGBM学習データセット生成 v8

v7 の特徴量（91列）にコーナー追走効率・上がり/タイムのトレンドを追加:
  - corner_chase_1〜5  : 各走の（最終コーナー順位 - 最初コーナー順位）/ 頭数
                         負=後方から追い込み、正=先行から失速
  - agari_trend_5      : 直近5走の上がり3F の線形回帰傾き（負=短縮傾向=末脚強化中）
                         上がり3Fは常に最後の600mなので距離非依存
  - time_diff_trend_5  : 直近5走のタイム差（コース平均差=距離補正済み）の線形回帰傾き
                         負=タイム改善中

v7=91列 → v8=98列。データは "_v8" サフィックスで保存。
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
from src.PredictionModels.LightGBM.make_dataset_v7 import (
    index_v7,
    _parse_date_flex,
    _parse_kinryo,
    _parse_n_horses,
    get_v7_extra_features,
    make_dataset_for_lightGBM_v7,
)

# v8 特徴量列名: v7（91列）+ 7列 = 98列
index_v8 = index_v7 + [
    "corner_chase_1",   # 1走前: (最終コーナー順位 - 最初コーナー順位) / 頭数
    "corner_chase_2",   # 2走前
    "corner_chase_3",   # 3走前
    "corner_chase_4",   # 4走前
    "corner_chase_5",   # 5走前
    "agari_trend_5",    # 直近5走の上がり3F 線形回帰傾き（距離非依存）
    "time_diff_trend_5", # 直近5走のタイム差（コース平均差=距離補正済み）線形回帰傾き
]


def _parse_corner_chase(通過_val, headcount_val):
    """(最終コーナー順位 - 最初コーナー順位) / 頭数。
    通過='11-11-11-11' or '5-3' 等の形式に対応。
    負=後方から追い込み（末脚発揮）、正=先行から失速。
    """
    try:
        s = str(通過_val).strip()
        parts = [int(x) for x in s.split("-") if x.strip().isdigit()]
        if len(parts) < 2:
            return np.nan
        hc = int(float(str(headcount_val).strip()))
        return (parts[-1] - parts[0]) / hc if hc > 0 else np.nan
    except Exception:
        return np.nan


def _linear_trend(values):
    """値リスト（古い→新しい順）から線形回帰傾きを返す（NaNは除外）。"""
    try:
        valid = [(i, float(v)) for i, v in enumerate(values)
                 if v is not None and not np.isnan(float(v))]
        if len(valid) < 2:
            return np.nan
        n = len(valid)
        xs = [-(n - 1 - i) for i in range(n)]
        ys = [v for _, v in valid]
        mx = sum(xs) / n
        my = sum(ys) / n
        num = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
        den = sum((x - mx) ** 2 for x in xs)
        return num / den if den != 0 else 0.0
    except Exception:
        return np.nan


def get_extra_past_race_features_v8(race_info_df_5):
    """v8 追加の7列を返す。

    Args:
        race_info_df_5: past_performance.get_past_race_info の結果（最大5走）

    Returns:
        list of 7 elements:
            [corner_chase_1〜5（5列）, agari_trend_5, time_diff_trend_5]

    タイム差（time_diff_course）はコース×距離別平均との差なので距離補正済み。
    上がり3Fは常に最後の600mなので距離非依存。
    """
    corner_chases = []
    agaris = []
    time_diffs = []

    for i in range(5):
        if i < len(race_info_df_5.index):
            row = race_info_df_5.iloc[i]
            corner_chases.append(_parse_corner_chase(
                row.get("通過", ""), row.get("頭数", row.get("頭 数", ""))
            ))
            agaris.append(_parse_agari(row.get("上り", "")))
            td = _get_time_info(row)
            # td[0] = time_df_course（コース平均差=距離補正済み）
            td0 = td[0] if td and len(td) > 0 else np.nan
            time_diffs.append(td0 if not np.isnan(float(td0)) else np.nan)
        else:
            corner_chases.append(np.nan)
            agaris.append(np.nan)
            time_diffs.append(np.nan)

    # トレンドは古い走（インデックス大）→新しい走（インデックス0）の逆順で入力
    agari_trend = _linear_trend(list(reversed(agaris)))
    time_diff_trend = _linear_trend(list(reversed(time_diffs)))

    return corner_chases + [agari_trend, time_diff_trend]  # 5 + 1 + 1 = 7列


def save_dataset_v8(place_id, year, race_type, length, df, flag_list):
    out_dir = os.path.join(
        paths_v3.PREDICTION_DATASET_PATH, name_header.PLACE_LIST[place_id - 1]
    )
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, f"{year}_{race_type}{length}_ai_dataset")
    df.reset_index(drop=True).to_csv(base + "_for_rank_v8.csv")
    pd.DataFrame(flag_list, columns=["result_flag"]).to_csv(base + "_flag_v8.csv")


def load_dataset_v8(place_id, year, race_type, length):
    out_dir = os.path.join(
        paths_v3.PREDICTION_DATASET_PATH, name_header.PLACE_LIST[place_id - 1]
    )
    base = os.path.join(out_dir, f"{year}_{race_type}{length}_ai_dataset")
    df_path   = base + "_for_rank_v8.csv"
    flag_path = base + "_flag_v8.csv"
    df   = pd.read_csv(df_path,   index_col=0, dtype=float) if os.path.isfile(df_path)   else pd.DataFrame()
    flag = pd.read_csv(flag_path, index_col=0, dtype=int)   if os.path.isfile(flag_path) else pd.DataFrame()
    return df, flag


def make_dataset_for_train_v8(place_id, year=date.today().year, vocab=None, course_filter=None):
    """指定競馬場・年の v8 学習データセットを作成・保存する。"""
    from src.PredictionModels.LightGBM.make_dataset import relevance_grade

    if vocab is None:
        vocab = build_pedigree_vocab()

    df_results = race_results.get_race_results_csv(place_id, year)
    if df_results.empty:
        print(f"  データなし: {year} {name_header.PLACE_LIST[place_id - 1]}")
        return

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
        race_id_list = []
        flag_list = []
        jockey_wins, jockey_places = [], []
        odds_list, popularity_list = [], []
        father_ids, mf_ids, pgf_ids = [], [], []
        kinryo_list = []
        n_horses_today_list = []
        days_since_list, n_horses_1_list, hw_abs_1_list = [], [], []
        corner_chase_lists = [[] for _ in range(5)]
        agari_trend_list = []
        time_diff_trend_list = []

        df_v3_base  = pd.DataFrame()
        df_v6_extra = pd.DataFrame()

        for i in tqdm(range(len(df_course))):
            df_result = df_course.iloc[i:i + 1]
            race_id   = int(df_result.index[0])
            race_id_list.append(race_id)
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

            # v7 extras
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

            # v8 extras
            try:
                race_info_5 = past_performance.get_past_race_info(horse_id, race_id, race_num=5)
                extra_v8 = get_extra_past_race_features_v8(race_info_5)
            except Exception:
                extra_v8 = [np.nan] * 7

            for ci in range(5):
                corner_chase_lists[ci].append(extra_v8[ci])
            agari_trend_list.append(extra_v8[5])
            time_diff_trend_list.append(extra_v8[6])

        df_dataset = pd.concat([
            pd.DataFrame(race_id_list,         columns=["race_id"]),
            df_v3_base.reset_index(drop=True),
            pd.Series(jockey_wins,             name="jockey_win_rate"),
            pd.Series(jockey_places,           name="jockey_place_rate"),
            df_course["枠番"].reset_index(drop=True),
            df_course["馬番"].reset_index(drop=True),
            pd.Series(odds_list,               name="current_odds"),
            pd.Series(popularity_list,         name="current_popularity"),
            pd.Series(father_ids,              name="father_cat"),
            pd.Series(mf_ids,                  name="mother_father_cat"),
            pd.Series(pgf_ids,                 name="paternal_gf_cat"),
            df_v6_extra.reset_index(drop=True),
            pd.Series(kinryo_list,             name="kinryo"),
            pd.Series(days_since_list,         name="days_since_last_race"),
            pd.Series(n_horses_today_list,     name="n_horses_today"),
            pd.Series(n_horses_1_list,         name="n_horses_1"),
            pd.Series(hw_abs_1_list,           name="horse_weight_abs_1"),
            pd.Series(corner_chase_lists[0],   name="corner_chase_1"),
            pd.Series(corner_chase_lists[1],   name="corner_chase_2"),
            pd.Series(corner_chase_lists[2],   name="corner_chase_3"),
            pd.Series(corner_chase_lists[3],   name="corner_chase_4"),
            pd.Series(corner_chase_lists[4],   name="corner_chase_5"),
            pd.Series(agari_trend_list,        name="agari_trend_5"),
            pd.Series(time_diff_trend_list,    name="time_diff_trend_5"),
        ], axis=1)
        df_dataset.columns = index_v8

        save_dataset_v8(place_id, year, race_type, length, df_dataset, flag_list)
        print(f"    保存完了: {len(df_dataset)}行")
