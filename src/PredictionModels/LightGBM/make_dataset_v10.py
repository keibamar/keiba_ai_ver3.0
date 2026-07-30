"""LightGBM学習データセット生成 v10

v9 の特徴量（105列）にコース実績・斤量変化特徴量を追加:
  - kinryo_diff_1    : 前走からの斤量変化 (正=重くなった, 負=軽くなった)
                       ※ v3の weight_change(前走-2走前の馬体重変化) / dist_change(今回-前走距離)
                         はv3ベースに既存のため除外。hw_change_1 は race_card に馬体重列がないため除外。
  - same_course_cnt5 : 過去5走中、今走と同一コース(種別×距離)での出走数
  - same_course_pr5  : 過去5走中、同一コースでの複勝率 (3着以内 / 同コース出走数)

v9=105列 → v10=108列。データは "_v10" サフィックスで保存。
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
from src.PredictionModels.LightGBM.make_dataset_v2 import _parse_weight
from src.PredictionModels.LightGBM.make_dataset_v5 import build_pedigree_vocab, get_pedigree_cats
from src.PredictionModels.LightGBM.make_dataset_v7 import (
    _parse_kinryo, make_dataset_for_lightGBM_v7,
)
from src.PredictionModels.LightGBM.make_dataset_v8 import (
    index_v8, get_extra_past_race_features_v8,
)
from src.PredictionModels.LightGBM.make_dataset_v9 import (
    index_v9, get_extra_past_race_features_v9, _load_avg_agari,
)
from src.PredictionModels.LightGBM.make_dataset_v3 import build_jockey_course_stats

# v10 特徴量列名: v9（105列）+ 3列 = 108列
index_v10 = index_v9 + [
    "kinryo_diff_1",     # 前走からの斤量変化 (g)
    "same_course_cnt5",  # 過去5走中の同コース出走数
    "same_course_pr5",   # 過去5走中の同コース複勝率
]


def _safe_float(v, default=np.nan):
    try:
        return float(v)
    except Exception:
        return default


def _parse_rank_val(rank_str):
    """着順文字列を float に変換。除外・取消は NaN。"""
    s = str(rank_str)
    if any(c in s for c in ("除", "取", "中", "失")):
        return np.nan
    try:
        return float(re.sub(r"\D", "", s))
    except Exception:
        return np.nan


def get_extra_past_race_features_v10(
    race_info_5, current_kinryo, current_course_len, current_race_type
):
    """v10 追加の3列を返す。

    Args:
        race_info_5       : past_performance.get_past_race_info の結果（最大5走）
        current_kinryo    : 今走の斤量 (float, 例: 55.0)
        current_course_len: 今走の距離 (int/str, 例: 1800)
        current_race_type : 今走のコース種別 ('芝' or 'ダ' など)

    Returns:
        list of 3 elements: [kinryo_diff_1, same_course_cnt5, same_course_pr5]

    注: dist_change_1 は v3 の dist_change（col 60）と同一のため除外。
        hw_change_1 は race_card CSV に馬体重列がないため予測時に使用不可→除外。
    """
    if race_info_5.empty:
        return [np.nan, 0.0, np.nan]

    prev_row = race_info_5.iloc[0]

    # kinryo_diff_1
    prev_kinryo = _safe_float(prev_row.get("斤量", ""))
    kinryo_diff = (
        current_kinryo - prev_kinryo
        if not (np.isnan(current_kinryo) or np.isnan(prev_kinryo))
        else np.nan
    )

    # same_course_cnt5 / same_course_pr5
    cnt = 0
    hits = 0
    try:
        target_rt = str(current_race_type).strip()
        target_cl = int(float(str(current_course_len)))
    except Exception:
        return [kinryo_diff, 0.0, np.nan]

    for i in range(min(5, len(race_info_5))):
        row = race_info_5.iloc[i]
        rt = str(row.get("race_type", "")).strip()
        try:
            cl = int(float(str(row.get("course_len", "")).strip()))
            same = (rt == target_rt) and (cl == target_cl)
        except Exception:
            same = False
        if same:
            cnt += 1
            rank = _parse_rank_val(row.get("着順", ""))
            if not np.isnan(rank) and rank <= 3:
                hits += 1

    same_cnt = float(cnt)
    same_pr = float(hits / cnt) if cnt > 0 else np.nan

    return [kinryo_diff, same_cnt, same_pr]


def save_dataset_v10(place_id, year, race_type, length, df, flag_list):
    out_dir = os.path.join(
        paths_v3.PREDICTION_DATASET_PATH, name_header.PLACE_LIST[place_id - 1]
    )
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, f"{year}_{race_type}{length}_ai_dataset")
    df.reset_index(drop=True).to_csv(base + "_for_rank_v10.csv")
    pd.DataFrame(flag_list, columns=["result_flag"]).to_csv(base + "_flag_v10.csv")


def load_dataset_v10(place_id, year, race_type, length):
    out_dir = os.path.join(
        paths_v3.PREDICTION_DATASET_PATH, name_header.PLACE_LIST[place_id - 1]
    )
    base = os.path.join(out_dir, f"{year}_{race_type}{length}_ai_dataset")
    df_path   = base + "_for_rank_v10.csv"
    flag_path = base + "_flag_v10.csv"
    df   = pd.read_csv(df_path,   index_col=0, dtype=float) if os.path.isfile(df_path)   else pd.DataFrame()
    flag = pd.read_csv(flag_path, index_col=0, dtype=int)   if os.path.isfile(flag_path) else pd.DataFrame()
    return df, flag


def make_dataset_for_train_v10(place_id, year=date.today().year, vocab=None, course_filter=None):
    """指定競馬場・年の v10 学習データセットを作成・保存する。"""
    from src.PredictionModels.LightGBM.make_dataset import relevance_grade
    from src.PredictionModels.LightGBM.make_dataset_v4 import _parse_odds, _parse_popularity

    if vocab is None:
        vocab = build_pedigree_vocab()

    _load_avg_agari()

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
        agari_df_course_lists = [[] for _ in range(5)]
        corner_ratio_std_list = []
        agari_std_list = []
        # v10 extras
        kinryo_diff_list = []
        same_course_cnt_list = []
        same_course_pr_list = []

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

            current_kinryo = _parse_kinryo(df_result.iloc[0].get("斤量", ""))
            kinryo_list.append(current_kinryo)
            n_horses_today_list.append(float(n_horses_map.get(race_id, np.nan)))
            days_since_list.append(extra_v7[0])
            n_horses_1_list.append(extra_v7[1])
            hw_abs_1 = extra_v7[2]
            hw_abs_1_list.append(hw_abs_1)

            try:
                race_info_5 = past_performance.get_past_race_info(horse_id, race_id, race_num=5)
                extra_v8 = get_extra_past_race_features_v8(race_info_5)
                extra_v9 = get_extra_past_race_features_v9(race_info_5)
            except Exception:
                race_info_5 = pd.DataFrame()
                extra_v8 = [np.nan] * 7
                extra_v9 = [np.nan] * 7

            for ci in range(5):
                corner_chase_lists[ci].append(extra_v8[ci])
            agari_trend_list.append(extra_v8[5])
            time_diff_trend_list.append(extra_v8[6])
            for ci in range(5):
                agari_df_course_lists[ci].append(extra_v9[ci])
            corner_ratio_std_list.append(extra_v9[5])
            agari_std_list.append(extra_v9[6])

            # v10 extras
            extra_v10 = get_extra_past_race_features_v10(
                race_info_5, current_kinryo, length, race_type
            )
            kinryo_diff_list.append(extra_v10[0])
            same_course_cnt_list.append(extra_v10[1])
            same_course_pr_list.append(extra_v10[2])

        df_dataset = pd.concat([
            pd.DataFrame(race_id_list,              columns=["race_id"]),
            df_v3_base.reset_index(drop=True),
            pd.Series(jockey_wins,                  name="jockey_win_rate"),
            pd.Series(jockey_places,                name="jockey_place_rate"),
            df_course["枠番"].reset_index(drop=True),
            df_course["馬番"].reset_index(drop=True),
            pd.Series(odds_list,                    name="current_odds"),
            pd.Series(popularity_list,              name="current_popularity"),
            pd.Series(father_ids,                   name="father_cat"),
            pd.Series(mf_ids,                       name="mother_father_cat"),
            pd.Series(pgf_ids,                      name="paternal_gf_cat"),
            df_v6_extra.reset_index(drop=True),
            pd.Series(kinryo_list,                  name="kinryo"),
            pd.Series(days_since_list,              name="days_since_last_race"),
            pd.Series(n_horses_today_list,          name="n_horses_today"),
            pd.Series(n_horses_1_list,              name="n_horses_1"),
            pd.Series(hw_abs_1_list,                name="horse_weight_abs_1"),
            pd.Series(corner_chase_lists[0],        name="corner_chase_1"),
            pd.Series(corner_chase_lists[1],        name="corner_chase_2"),
            pd.Series(corner_chase_lists[2],        name="corner_chase_3"),
            pd.Series(corner_chase_lists[3],        name="corner_chase_4"),
            pd.Series(corner_chase_lists[4],        name="corner_chase_5"),
            pd.Series(agari_trend_list,             name="agari_trend_5"),
            pd.Series(time_diff_trend_list,         name="time_diff_trend_5"),
            pd.Series(agari_df_course_lists[0],     name="agari_df_course_1"),
            pd.Series(agari_df_course_lists[1],     name="agari_df_course_2"),
            pd.Series(agari_df_course_lists[2],     name="agari_df_course_3"),
            pd.Series(agari_df_course_lists[3],     name="agari_df_course_4"),
            pd.Series(agari_df_course_lists[4],     name="agari_df_course_5"),
            pd.Series(corner_ratio_std_list,        name="corner_ratio_std5"),
            pd.Series(agari_std_list,               name="agari_std5"),
            # v10 extras
            pd.Series(kinryo_diff_list,             name="kinryo_diff_1"),
            pd.Series(same_course_cnt_list,         name="same_course_cnt5"),
            pd.Series(same_course_pr_list,          name="same_course_pr5"),
        ], axis=1)
        df_dataset.columns = index_v10

        save_dataset_v10(place_id, year, race_type, length, df_dataset, flag_list)
        print(f"    保存完了: {len(df_dataset)}行")
