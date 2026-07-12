"""LightGBM学習データセット生成 v6

v5 の特徴量（70列）に近走トレンド拡張特徴量を追加:
  - 4走前・5走前の走行情報 (7列 × 2 = 14列)
  - rank_trend_5   : 直近5走着順の傾き（線形回帰）
  - win_rate_recent5: 直近5走3着以内率

v1=51列、v2=62列、v3=65列、v4=67列、v5=70列、v6=86列。
データは "_v6" サフィックスで保存。
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
    _parse_agari, _parse_margin, _parse_corner_ratio, _get_time_info,
)
from src.PredictionModels.LightGBM.make_dataset_v3 import build_jockey_course_stats
from src.PredictionModels.LightGBM.make_dataset_v4 import _parse_odds, _parse_popularity
from src.PredictionModels.LightGBM.make_dataset_v5 import (
    index_v5,
    make_dataset_for_lightGBM_v3,
    build_pedigree_vocab,
    get_pedigree_cats,
)

# v6 特徴量列名（v5 の 70列 + 16列）
index_v6 = index_v5 + [
    # 4走前（7列）
    "time_df_course4", "time_df_class4", "ninki_4", "result_4",
    "agari_4", "margin_4", "corner_ratio_4",
    # 5走前（7列）
    "time_df_course5", "time_df_class5", "ninki_5", "result_5",
    "agari_5", "margin_5", "corner_ratio_5",
    # 5走集計（2列）
    "rank_trend_5",        # 直近5走着順の線形回帰傾き（負=改善中）
    "win_rate_recent5",    # 直近5走3着以内率
]


def _rank_trend_n(ranks):
    """N走分の着順リストから線形回帰傾きを返す（負=改善傾向）。"""
    pairs = [(i, r) for i, r in enumerate(ranks) if not (isinstance(r, float) and np.isnan(r))]
    if len(pairs) < 2:
        return np.nan
    xs = [-(len(pairs) - 1 - i) for i in range(len(pairs))]
    ys = [r for _, r in pairs]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    num = sum((xs[i] - mx) * (ys[i] - my) for i in range(len(xs)))
    den = sum((x - mx) ** 2 for x in xs)
    return num / den if den != 0 else 0.0


def _parse_rank_val(rank_str):
    s = str(rank_str)
    if any(c in s for c in ("除", "取", "中", "失")):
        return np.nan
    try:
        return float(re.sub(r"\D", "", s))
    except Exception:
        return np.nan


def get_extra_past_race_features_v6(race_info_df_5):
    """
    5走分の past_race_info DataFrame から v6 追加特徴量を返す。

    - インデックス 3, 4（4走前・5走前）の7列×2
    - rank_trend_5  : 5走着順の線形回帰傾き
    - win_rate_recent5: 5走中3着以内の割合

    Returns: list of 16 elements
    """
    # まず5走分の着順を収集（rank_trend_5, win_rate_recent5 計算用）
    rank_list_5 = []
    valid_count = 0
    place_count = 0
    for i in range(5):
        if i < len(race_info_df_5.index):
            rank_val = _parse_rank_val(race_info_df_5.iloc[i].get("着順", ""))
        else:
            rank_val = np.nan
        rank_list_5.append(rank_val)
        if not (isinstance(rank_val, float) and np.isnan(rank_val)):
            valid_count += 1
            if rank_val <= 3:
                place_count += 1

    extra = []

    # 4走前・5走前（インデックス 3, 4）の7列
    for i in [3, 4]:
        if i < len(race_info_df_5.index):
            row = race_info_df_5.iloc[i]
            df_time = _get_time_info(row)

            raw_rank = row.get("着順", "")
            raw_pop  = row.get("人気", "")
            rank_str = str(raw_rank)

            if "除" in rank_str or "取" in rank_str:
                df_time.extend([np.nan, np.nan])
            elif "中" in rank_str or "失" in rank_str:
                try:
                    pop_val = float(re.sub(r"\D", "", str(raw_pop)))
                except Exception:
                    pop_val = np.nan
                df_time.extend([pop_val, np.nan])
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
            extra.extend(df_time)
        else:
            extra.extend([np.nan] * 7)

    # 5走集計
    extra.append(_rank_trend_n(rank_list_5))
    extra.append(place_count / valid_count if valid_count > 0 else np.nan)

    return extra  # 7+7+2 = 16 要素


def make_dataset_for_lightGBM_v6(race_id, course_info, horse_id, vocab):
    """v6特徴量の1行（v3の60列 + v6追加16列 = 76列）を作成する。
    jockey/waku/umaban/odds/popularity/pedigree_cat は make_dataset_for_train_v6 で付与。
    """
    try:
        # v3 ベース（血統36 + 過去3走特徴量24 = 60列）
        row_v3 = make_dataset_for_lightGBM_v3(race_id, course_info, horse_id)
        if row_v3.empty:
            return pd.DataFrame()

        # 5走分データ取得
        race_info_5 = past_performance.get_past_race_info(horse_id, race_id, race_num=5)
        extra = get_extra_past_race_features_v6(race_info_5)

        extra_df = pd.DataFrame([extra])
        return pd.concat(
            [row_v3.reset_index(drop=True), extra_df.reset_index(drop=True)], axis=1
        )
    except Exception as e:
        print(f"  make_dataset_v6 error ({race_id}, {horse_id}): {e}")
        return pd.DataFrame()


def save_dataset_v6(place_id, year, race_type, length, df, flag_list):
    out_dir = os.path.join(
        paths_v3.PREDICTION_DATASET_PATH, name_header.PLACE_LIST[place_id - 1]
    )
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, f"{year}_{race_type}{length}_ai_dataset")
    df.reset_index(drop=True).to_csv(base + "_for_rank_v6.csv")
    pd.DataFrame(flag_list, columns=["result_flag"]).to_csv(base + "_flag_v6.csv")


def load_dataset_v6(place_id, year, race_type, length):
    out_dir = os.path.join(
        paths_v3.PREDICTION_DATASET_PATH, name_header.PLACE_LIST[place_id - 1]
    )
    base = os.path.join(out_dir, f"{year}_{race_type}{length}_ai_dataset")
    df_path   = base + "_for_rank_v6.csv"
    flag_path = base + "_flag_v6.csv"
    df   = pd.read_csv(df_path,   index_col=0, dtype=float) if os.path.isfile(df_path)   else pd.DataFrame()
    flag = pd.read_csv(flag_path, index_col=0, dtype=int)   if os.path.isfile(flag_path) else pd.DataFrame()
    return df, flag


def make_dataset_for_train_v6(place_id, year=date.today().year, vocab=None, course_filter=None):
    """指定競馬場・年の v6 学習データセットを作成・保存する。"""
    from src.PredictionModels.LightGBM.make_dataset import relevance_grade

    if vocab is None:
        vocab = build_pedigree_vocab()

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
        jockey_wins, jockey_places = [], []
        odds_list, popularity_list = [], []
        father_ids, mf_ids, pgf_ids = [], [], []

        df_v3_base  = pd.DataFrame()  # v3 base 60 cols
        df_v6_extra = pd.DataFrame()  # v6 extra 16 cols
        _V3_NCOLS = 60

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
            horse_id = df_result.iloc[0]["horse_id"]

            # v3 base (60cols) + v6 extra (16cols) = 76 cols
            row = make_dataset_for_lightGBM_v6(race_id, course_info, horse_id, vocab)
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

        # 列順: race_id(1) + v3_base(60) + jockey(2) + waku/umaban(2)
        #       + odds/pop(2) + ped_cat(3) + v6_extra(16) = 86列  ← index_v6 と一致
        df_dataset = pd.concat([
            pd.DataFrame(race_id_list,     columns=["race_id"]),
            df_v3_base.reset_index(drop=True),
            pd.Series(jockey_wins,         name="jockey_win_rate"),
            pd.Series(jockey_places,       name="jockey_place_rate"),
            df_course["枠番"].reset_index(drop=True),
            df_course["馬番"].reset_index(drop=True),
            pd.Series(odds_list,           name="current_odds"),
            pd.Series(popularity_list,     name="current_popularity"),
            pd.Series(father_ids,          name="father_cat"),
            pd.Series(mf_ids,              name="mother_father_cat"),
            pd.Series(pgf_ids,             name="paternal_gf_cat"),
            df_v6_extra.reset_index(drop=True),
        ], axis=1)
        df_dataset.columns = index_v6

        save_dataset_v6(place_id, year, race_type, length, df_dataset, flag_list)
        print(f"    保存完了: {len(df_dataset)}行")
