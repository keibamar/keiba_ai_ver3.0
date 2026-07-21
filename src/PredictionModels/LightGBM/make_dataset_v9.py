"""LightGBM学習データセット生成 v9

v8 の特徴量（98列）にペース適性・脚質安定性を追加:
  - agari_df_course_1〜5 : 各走の上り3Fとコース×馬場別平均上り3Fとの差
                           負=平均より末脚を使えた（末脚強), 正=平均より遅かった
  - corner_ratio_std5    : 過去5走の1コーナー通過順位/頭数の標準偏差（位置取り安定性）
  - agari_std5           : 過去5走の上り3Fの標準偏差（末脚安定性）

v8=98列 → v9=105列。データは "_v9" サフィックスで保存。
"""

import os
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
    _parse_agari, _parse_corner_ratio,
)
from src.PredictionModels.LightGBM.make_dataset_v5 import (
    build_pedigree_vocab,
    get_pedigree_cats,
)
from src.PredictionModels.LightGBM.make_dataset_v3 import build_jockey_course_stats
from src.PredictionModels.LightGBM.make_dataset_v4 import _parse_odds, _parse_popularity
from src.PredictionModels.LightGBM.make_dataset_v7 import (
    _parse_kinryo,
    make_dataset_for_lightGBM_v7,
)
from src.PredictionModels.LightGBM.make_dataset_v8 import (
    index_v8,
    get_extra_past_race_features_v8,
    _parse_corner_chase,
)

# v9 特徴量列名: v8（98列）+ 7列 = 105列
index_v9 = index_v8 + [
    "agari_df_course_1",  # 1走前: 上り3F - コース×馬場平均上り3F
    "agari_df_course_2",  # 2走前
    "agari_df_course_3",  # 3走前
    "agari_df_course_4",  # 4走前
    "agari_df_course_5",  # 5走前
    "corner_ratio_std5",  # 過去5走の位置取り標準偏差
    "agari_std5",         # 過去5走の上り3F標準偏差
]

_AVG_AGARI_PATH = os.path.join(
    _PROJECT_ROOT, "data", "race_info", "average_agari", "avg_agari.csv"
)

# グローバルに読み込む（race_type, course_len, ground_state) → avg_agari
_avg_agari_lookup = {}


def _load_avg_agari():
    global _avg_agari_lookup
    if _avg_agari_lookup:
        return
    if not os.path.isfile(_AVG_AGARI_PATH):
        print(f"警告: avg_agari.csv が見つかりません: {_AVG_AGARI_PATH}")
        return
    df = pd.read_csv(_AVG_AGARI_PATH, encoding="utf-8-sig")
    for _, row in df.iterrows():
        key = (str(row["race_type"]), int(row["course_len"]), str(row["ground_state"]))
        _avg_agari_lookup[key] = float(row["avg_agari"])


def _get_avg_agari(race_type, course_len, ground_state):
    """コース×馬場別の平均上り3Fを返す。見つからない場合は '全' にフォールバック。"""
    _load_avg_agari()
    key = (str(race_type), int(course_len), str(ground_state))
    if key in _avg_agari_lookup:
        return _avg_agari_lookup[key]
    key_all = (str(race_type), int(course_len), "全")
    return _avg_agari_lookup.get(key_all, np.nan)


def get_extra_past_race_features_v9(race_info_5):
    """v9 追加の7列を返す。

    Args:
        race_info_5: past_performance.get_past_race_info の結果（最大5走）

    Returns:
        list of 7 elements:
            [agari_df_course_1〜5（5列）, corner_ratio_std5, agari_std5]
    """
    agari_df_courses = []
    corner_ratios = []
    agaris = []

    for i in range(5):
        if i < len(race_info_5.index):
            row = race_info_5.iloc[i]
            agari = _parse_agari(row.get("上り", ""))
            agaris.append(agari)

            rt  = row.get("race_type", "")
            cl  = row.get("course_len", "")
            gs  = row.get("ground_state", "")
            cr  = _parse_corner_ratio(
                row.get("通過", ""), row.get("頭数", row.get("頭 数", ""))
            )
            corner_ratios.append(cr)

            try:
                avg = _get_avg_agari(rt, int(float(str(cl))), gs)
                df_val = (agari - avg) if (not np.isnan(agari) and not np.isnan(avg)) else np.nan
            except Exception:
                df_val = np.nan
            agari_df_courses.append(df_val)
        else:
            agari_df_courses.append(np.nan)
            corner_ratios.append(np.nan)
            agaris.append(np.nan)

    valid_cr = [v for v in corner_ratios if not np.isnan(v)]
    corner_ratio_std = float(np.std(valid_cr)) if len(valid_cr) >= 2 else np.nan

    valid_ag = [v for v in agaris if not np.isnan(v)]
    agari_std = float(np.std(valid_ag)) if len(valid_ag) >= 2 else np.nan

    return agari_df_courses + [corner_ratio_std, agari_std]  # 5 + 1 + 1 = 7列


def save_dataset_v9(place_id, year, race_type, length, df, flag_list):
    out_dir = os.path.join(
        paths_v3.PREDICTION_DATASET_PATH, name_header.PLACE_LIST[place_id - 1]
    )
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, f"{year}_{race_type}{length}_ai_dataset")
    df.reset_index(drop=True).to_csv(base + "_for_rank_v9.csv")
    pd.DataFrame(flag_list, columns=["result_flag"]).to_csv(base + "_flag_v9.csv")


def load_dataset_v9(place_id, year, race_type, length):
    out_dir = os.path.join(
        paths_v3.PREDICTION_DATASET_PATH, name_header.PLACE_LIST[place_id - 1]
    )
    base = os.path.join(out_dir, f"{year}_{race_type}{length}_ai_dataset")
    df_path   = base + "_for_rank_v9.csv"
    flag_path = base + "_flag_v9.csv"
    df   = pd.read_csv(df_path,   index_col=0, dtype=float) if os.path.isfile(df_path)   else pd.DataFrame()
    flag = pd.read_csv(flag_path, index_col=0, dtype=int)   if os.path.isfile(flag_path) else pd.DataFrame()
    return df, flag


def make_dataset_for_train_v9(place_id, year=date.today().year, vocab=None, course_filter=None):
    """指定競馬場・年の v9 学習データセットを作成・保存する。"""
    from src.PredictionModels.LightGBM.make_dataset import relevance_grade

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
        # v9 extras
        agari_df_course_lists = [[] for _ in range(5)]
        corner_ratio_std_list = []
        agari_std_list = []

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

            kinryo_list.append(_parse_kinryo(df_result.iloc[0].get("斤量", "")))
            n_horses_today_list.append(float(n_horses_map.get(race_id, np.nan)))
            days_since_list.append(extra_v7[0])
            n_horses_1_list.append(extra_v7[1])
            hw_abs_1_list.append(extra_v7[2])

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
            # v9 extras
            pd.Series(agari_df_course_lists[0],     name="agari_df_course_1"),
            pd.Series(agari_df_course_lists[1],     name="agari_df_course_2"),
            pd.Series(agari_df_course_lists[2],     name="agari_df_course_3"),
            pd.Series(agari_df_course_lists[3],     name="agari_df_course_4"),
            pd.Series(agari_df_course_lists[4],     name="agari_df_course_5"),
            pd.Series(corner_ratio_std_list,        name="corner_ratio_std5"),
            pd.Series(agari_std_list,               name="agari_std5"),
        ], axis=1)
        df_dataset.columns = index_v9

        save_dataset_v9(place_id, year, race_type, length, df_dataset, flag_list)
        print(f"    保存完了: {len(df_dataset)}行")
