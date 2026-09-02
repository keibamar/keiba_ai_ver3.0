"""LightGBM学習データセット生成 v11

v10 の特徴量（108列）に以下を追加:
  - trainer_win_rate   : 調教師×コース勝率（過去5年）
  - trainer_place_rate : 調教師×コース複勝率
  - same_cond_cnt5     : 過去5走中、同コース×距離×馬場状態での出走数
  - same_cond_pr5      : 過去5走中、同条件複勝率（v10 same_course_pr5 の馬場状態拡張版）
  - f_ground_delta     : 父の道悪適性指数（重/稍重/不良複勝率 − 良複勝率、会場×race_type内）
  - mf_ground_delta    : 母父の道悪適性指数（同上）

v10=108列 → v11=114列。データは "_v11" サフィックスで保存。
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
from src.managers import peds_results_dataset_manager, horse_peds_dataset_manager
from src.PredictionModels.LightGBM.make_dataset_v2 import _parse_weight
from src.PredictionModels.LightGBM.make_dataset_v3 import _parse_date_jp, build_jockey_course_stats
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
from src.PredictionModels.LightGBM.make_dataset_v10 import (
    index_v10, get_extra_past_race_features_v10, _safe_float, _parse_rank_val,
)

# v11 特徴量列名: v10（108列）+ 6列 = 114列
index_v11 = index_v10 + [
    "trainer_win_rate",   # 調教師×コース勝率（過去5年）
    "trainer_place_rate", # 調教師×コース複勝率
    "same_cond_cnt5",     # 過去5走中の同条件(コース×距離×馬場)出走数
    "same_cond_pr5",      # 過去5走中の同条件複勝率
    "f_ground_delta",     # 父の道悪適性指数（重系 − 良）
    "mf_ground_delta",    # 母父の道悪適性指数
]

_MIN_GROUND_SAMPLES = 5  # 道悪指数の計算に必要な最低出走数


# ============================================================
# 調教師×コース成績テーブル構築
# ============================================================

def build_trainer_course_stats(place_id, target_year, lookback_years=5):
    """
    target_year の各馬について、そのレース直前時点での
    調教師×（race_type, course_len）の勝率・複勝率を計算する。

    Returns:
        dict: {(race_id_str, horse_id_str): (win_rate, place_rate)}
    """
    start_year = max(2015, target_year - lookback_years)
    dfs = []
    for yr in range(start_year, target_year + 1):
        df = race_results.get_race_results_csv(place_id, yr)
        if not df.empty:
            dfs.append(df.reset_index(names=["race_id"]))
    if not dfs:
        return {}

    df_all = pd.concat(dfs, ignore_index=True)

    if "調教師" not in df_all.columns:
        return {}

    rank_col = df_all.columns[1]

    def _to_rank(v):
        s = re.sub(r"\D", "", str(v))
        return int(s) if s else None

    df_all["_rank"] = df_all[rank_col].apply(_to_rank)
    df_all["_is_win"] = (df_all["_rank"] == 1).astype(float)
    df_all["_is_place"] = (df_all["_rank"] <= 3).astype(float)
    df_all["_date_dt"] = df_all["date"].apply(_parse_date_jp)
    df_all["_trainer"] = df_all["調教師"].astype(str).str.strip()

    df_all = df_all.dropna(subset=["_rank", "_date_dt"]).copy()
    df_all = df_all[df_all["_trainer"] != ""].copy()
    df_all = df_all.sort_values("_date_dt").reset_index(drop=True)

    win_arr = np.full(len(df_all), np.nan)
    place_arr = np.full(len(df_all), np.nan)

    for _, grp in df_all.groupby(["_trainer", "race_type", "course_len"]):
        idx = grp.index.tolist()
        wins = grp["_is_win"].values
        places = grp["_is_place"].values
        cum_w = np.cumsum(wins)
        cum_p = np.cumsum(places)
        counts = np.arange(1, len(wins) + 1, dtype=float)
        win_arr[idx[1:]] = cum_w[:-1] / counts[:-1]
        place_arr[idx[1:]] = cum_p[:-1] / counts[:-1]

    df_all["_win_rate"] = win_arr
    df_all["_place_rate"] = place_arr

    lookup = {}
    for _, row in df_all.iterrows():
        key = (str(row["race_id"]), str(row["horse_id"]))
        lookup[key] = (row["_win_rate"], row["_place_rate"])
    return lookup


# ============================================================
# 道悪適性指数ルックアップ構築
# ============================================================

def _calc_place_rate(df):
    """DataFrame の 着順列から複勝率を返す。サンプル不足時は NaN。"""
    if len(df) < _MIN_GROUND_SAMPLES:
        return np.nan
    total = len(df)
    placed = df[df["着順"].isin(["1", "2", "3"])].shape[0]
    return placed / total


def build_ground_delta_lookup(place_id, target_year, race_type, lookback_years=5):
    """
    会場×race_type 内で父・母父ごとに道悪適性指数を計算する。
    道悪適性 = (重+稍重+不良での複勝率) − (良での複勝率)

    Returns:
        dict: {sire_name: delta_float}
            peds_0列（父）と peds_4列（母父）で別々に呼び出すことを想定。
    """
    start_year = max(2019, target_year - lookback_years)
    df_peds = pd.DataFrame()
    for yr in range(start_year, target_year):
        df_y = peds_results_dataset_manager.get_peds_data_dataset_csv(place_id, yr)
        if not df_y.empty:
            df_peds = pd.concat([df_peds, df_y], ignore_index=True)

    if df_peds.empty:
        return {}

    if "race_type" not in df_peds.columns or "ground_state" not in df_peds.columns:
        return {}

    df_rt = df_peds[df_peds["race_type"] == race_type].copy()
    if df_rt.empty:
        return {}

    delta_map = {}
    for col in ["peds_0", "peds_4"]:
        if col not in df_rt.columns:
            continue
        for sire, grp in df_rt.groupby(col):
            good_rate = _calc_place_rate(grp[grp["ground_state"] == "良"])
            heavy_rate = _calc_place_rate(
                grp[grp["ground_state"].isin(["重", "稍重", "不良"])]
            )
            if not np.isnan(good_rate) and not np.isnan(heavy_rate):
                delta_map[(col, str(sire))] = heavy_rate - good_rate

    return delta_map


# ============================================================
# 同条件（コース×距離×馬場状態）成績
# ============================================================

def get_extra_past_race_features_v11(
    race_info_5, current_race_type, current_course_len, current_ground_state
):
    """v11 追加の same_cond 2列を返す。

    Args:
        race_info_5       : get_past_race_info の結果（最大5走）
        current_race_type : 今走コース種別
        current_course_len: 今走距離
        current_ground_state: 今走馬場状態

    Returns:
        list of 2: [same_cond_cnt5, same_cond_pr5]
    """
    if race_info_5.empty:
        return [0.0, np.nan]

    try:
        target_rt = str(current_race_type).strip()
        target_cl = int(float(str(current_course_len)))
        target_gs = str(current_ground_state).strip()
    except Exception:
        return [0.0, np.nan]

    cnt = 0
    hits = 0
    for i in range(min(5, len(race_info_5))):
        row = race_info_5.iloc[i]
        rt = str(row.get("race_type", "")).strip()
        gs = str(row.get("ground_state", "")).strip()
        try:
            cl = int(float(str(row.get("course_len", "")).strip()))
            same = (rt == target_rt) and (cl == target_cl) and (gs == target_gs)
        except Exception:
            same = False
        if same:
            cnt += 1
            rank = _parse_rank_val(row.get("着順", ""))
            if not np.isnan(rank) and rank <= 3:
                hits += 1

    same_cond_cnt = float(cnt)
    same_cond_pr = float(hits / cnt) if cnt > 0 else np.nan
    return [same_cond_cnt, same_cond_pr]


# ============================================================
# データセット保存・読み込み
# ============================================================

def save_dataset_v11(place_id, year, race_type, length, df, flag_list):
    out_dir = os.path.join(
        paths_v3.PREDICTION_DATASET_PATH, name_header.PLACE_LIST[place_id - 1]
    )
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, f"{year}_{race_type}{length}_ai_dataset")
    df.reset_index(drop=True).to_csv(base + "_for_rank_v11.csv")
    pd.DataFrame(flag_list, columns=["result_flag"]).to_csv(base + "_flag_v11.csv")


def load_dataset_v11(place_id, year, race_type, length):
    out_dir = os.path.join(
        paths_v3.PREDICTION_DATASET_PATH, name_header.PLACE_LIST[place_id - 1]
    )
    base = os.path.join(out_dir, f"{year}_{race_type}{length}_ai_dataset")
    df_path   = base + "_for_rank_v11.csv"
    flag_path = base + "_flag_v11.csv"
    df   = pd.read_csv(df_path,   index_col=0, dtype=float) if os.path.isfile(df_path)   else pd.DataFrame()
    flag = pd.read_csv(flag_path, index_col=0, dtype=int)   if os.path.isfile(flag_path) else pd.DataFrame()
    return df, flag


# ============================================================
# メイン学習データセット生成
# ============================================================

def make_dataset_for_train_v11(place_id, year=date.today().year, vocab=None, course_filter=None):
    """指定競馬場・年の v11 学習データセットを作成・保存する。"""
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

    print(f"  調教師×コース成績テーブル構築中...")
    trainer_lookup = build_trainer_course_stats(place_id, year)

    _V3_NCOLS = 60

    for race_type, length in courses:
        df_course = df_results[
            (df_results["race_type"] == race_type) & (df_results["course_len"] == length)
        ]
        if df_course.empty:
            continue

        # 道悪適性ルックアップ（コース×race_type単位で一度だけ構築）
        print(f"  {race_type}{length}m 道悪適性ルックアップ構築中...")
        ground_delta_lookup = build_ground_delta_lookup(place_id, year, race_type)

        print(f"  {race_type}{length}m ({len(df_course)}走)")
        race_id_list = []
        flag_list = []
        jockey_wins, jockey_places = [], []
        trainer_wins, trainer_places = [], []
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
        kinryo_diff_list = []
        same_course_cnt_list = []
        same_course_pr_list = []
        same_cond_cnt_list = []
        same_cond_pr_list = []
        f_ground_delta_list = []
        mf_ground_delta_list = []

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
            current_gs       = str(df_result.iloc[0]["ground_state"]).strip()

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

            tw, tp = trainer_lookup.get(key, (np.nan, np.nan))
            trainer_wins.append(tw)
            trainer_places.append(tp)

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

            extra_v10 = get_extra_past_race_features_v10(
                race_info_5, current_kinryo, length, race_type
            )
            kinryo_diff_list.append(extra_v10[0])
            same_course_cnt_list.append(extra_v10[1])
            same_course_pr_list.append(extra_v10[2])

            # v11 extras
            extra_v11 = get_extra_past_race_features_v11(
                race_info_5, race_type, length, current_gs
            )
            same_cond_cnt_list.append(extra_v11[0])
            same_cond_pr_list.append(extra_v11[1])

            # 道悪適性指数（父・母父名をhorse_pedsから取得）
            try:
                peds_csv = horse_peds_dataset_manager.get_horse_peds_csv(horse_id)
                father_name = str(peds_csv.loc["peds_0"].iloc[0]) if not peds_csv.empty and "peds_0" in peds_csv.index else ""
                mf_name     = str(peds_csv.loc["peds_4"].iloc[0]) if not peds_csv.empty and "peds_4" in peds_csv.index else ""
            except Exception:
                father_name = ""
                mf_name = ""
            f_ground_delta_list.append(ground_delta_lookup.get(("peds_0", father_name), np.nan))
            mf_ground_delta_list.append(ground_delta_lookup.get(("peds_4", mf_name), np.nan))

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
            pd.Series(kinryo_diff_list,             name="kinryo_diff_1"),
            pd.Series(same_course_cnt_list,         name="same_course_cnt5"),
            pd.Series(same_course_pr_list,          name="same_course_pr5"),
            # v11 extras
            pd.Series(trainer_wins,                 name="trainer_win_rate"),
            pd.Series(trainer_places,               name="trainer_place_rate"),
            pd.Series(same_cond_cnt_list,           name="same_cond_cnt5"),
            pd.Series(same_cond_pr_list,            name="same_cond_pr5"),
            pd.Series(f_ground_delta_list,          name="f_ground_delta"),
            pd.Series(mf_ground_delta_list,         name="mf_ground_delta"),
        ], axis=1)
        df_dataset.columns = index_v11

        save_dataset_v11(place_id, year, race_type, length, df_dataset, flag_list)
        print(f"    保存完了: {len(df_dataset)}行")
