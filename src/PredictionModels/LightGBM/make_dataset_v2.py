"""LightGBM学習データセット生成 v2

make_dataset.py の特徴量に以下を追加する:
- 過去3走の上り3ハロン
- 過去3走の着差（秒）
- 過去3走の通過順位比率（1角順位 / 頭数）
- 近走着順トレンド（傾き: 負 = 着順改善）
- 馬体重変化（前走 - 2走前）

データは "_v2" サフィックスで保存し既存データと分離する。
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
import get_race_id
import horse_peds
import race_results
import peds_results
import past_performance
import average_time

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
from src.config import paths as paths_v3

# v2 特徴量列名
index_v2 = [
    "race_id",
    # 血統（v1と同じ 36列）
    "f_course1", "f_course2", "f_course3", "f_courseout",
    "mf_course1", "mf_course2", "mf_course3", "mf_courseout",
    "f_mf_course1", "f_mf_course2", "f_mf_course3", "f_mf_courseout",
    "f_len1", "f_len2", "f_len3", "f_lenout",
    "mf_len1", "mf_len2", "mf_len3", "mf_classout",
    "f_mf_len1", "f_mf_len2", "f_mf_len3", "f_mf_classout",
    "f_class1", "f_class2", "f_class3", "f_classout",
    "mf_class1", "mf_class2", "mf_class3", "mf_classout",
    "f_mf_class1", "f_mf_class2", "f_mf_class3", "f_mf_classout",
    # 過去3走（v1の4列 + 新規3列 = 7列 × 3走）
    "time_df_course1", "time_df_class1", "ninki_1", "result_1",
    "agari_1", "margin_1", "corner_ratio_1",
    "time_df_course2", "time_df_class2", "ninki_2", "result_2",
    "agari_2", "margin_2", "corner_ratio_2",
    "time_df_course3", "time_df_class3", "ninki_3", "result_3",
    "agari_3", "margin_3", "corner_ratio_3",
    # 全体特徴量（新規 2列）
    "rank_trend",     # 近走着順の傾き（負=改善中）
    "weight_change",  # 馬体重変化（前走 - 2走前, kg）
    # 枠順（v1と同じ）
    "waku", "umaban",
]


# ---------- 新規パース関数 ----------

def _parse_agari(val):
    """上り3ハロンを float に変換。失敗時は NaN。"""
    try:
        return float(str(val).strip())
    except Exception:
        return np.nan


def _parse_margin(val):
    """着差を秒（float）に変換。1着のマイナス値は 0 に統一。失敗時は NaN。"""
    s = str(val).strip()
    if s in ("", "nan", "NaN"):
        return np.nan
    # 日本語着差表記 → 近似秒数
    jp_map = {"ハナ": 0.05, "クビ": 0.1, "アタマ": 0.15, "1/2": 0.3, "3/4": 0.5}
    for key, sec in jp_map.items():
        if key in s:
            return sec
    try:
        v = float(s.replace("−", "-").replace("ー", "-"))
        return max(0.0, v)  # 1着のマイナス値を 0 に統一
    except Exception:
        return np.nan


def _parse_corner_ratio(通過_val, headcount_val):
    """1コーナー通過順位 / 頭数。先行=小さい値。失敗時は NaN。"""
    try:
        s = str(通過_val).strip()
        first = int(s.split("-")[0]) if "-" in s else int(float(s))
        hc = int(float(str(headcount_val).strip()))
        return first / hc if hc > 0 else np.nan
    except Exception:
        return np.nan


def _parse_weight(val):
    """馬体重を float に変換。"460(+2)" 形式にも対応。"""
    try:
        return float(re.search(r"\d+", str(val)).group())
    except Exception:
        return np.nan


def _rank_trend(ranks):
    """直近3走の着順リスト [1走前, 2走前, 3走前] から傾きを返す。
    負の傾き = 着順が良くなっている（改善中）。NaN が多ければ NaN を返す。
    """
    pairs = [(i, r) for i, r in enumerate(ranks) if not (isinstance(r, float) and np.isnan(r))]
    if len(pairs) < 2:
        return np.nan
    xs = [-(len(pairs) - 1 - i) for i in range(len(pairs))]
    ys = [r for _, r in pairs]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    num = sum((xs[i] - mx) * (ys[i] - my) for i in range(len(xs)))
    den = sum((x - mx) ** 2 for x in xs)
    return num / den if den != 0 else 0.0


# ---------- v2 過去レース特徴量取得 ----------

def get_past_race_info_data_v2(race_info_df):
    """過去3レースの特徴量を取得（v2: 上り・着差・通過順位を追加）

    Returns:
        list: 7列 × 3走 + 2（トレンド・体重変化）= 23 要素
    """
    race_score_list = []
    weight_list = []
    rank_list = []

    for i in range(3):
        if i < len(race_info_df.index):
            row = race_info_df.iloc[i]

            # --- v1 と同じタイム差 ---
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

            # --- 新規: 上り / 着差 / 通過順位 ---
            df_time.append(_parse_agari(row.get("上り", "")))
            df_time.append(_parse_margin(row.get("着差", "")))
            df_time.append(_parse_corner_ratio(row.get("通過", ""), row.get("頭数", row.get("頭 数", ""))))

            race_score_list.append(df_time)
            weight_list.append(_parse_weight(row.get("馬体重", "")))
            rank_list.append(rank_val if not ("除" in rank_str or "取" in rank_str) else np.nan)
        else:
            race_score_list.append([np.nan] * 7)
            weight_list.append(np.nan)
            rank_list.append(np.nan)

    result = sum(race_score_list, [])

    # 近走着順トレンド
    result.append(_rank_trend(rank_list))

    # 馬体重変化（前走 - 2走前）
    wc = (weight_list[0] - weight_list[1]
          if not np.isnan(weight_list[0]) and not np.isnan(weight_list[1])
          else np.nan)
    result.append(wc)

    return result


def _get_time_info(race_info_row):
    """過去レース行からタイム差を取得（make_dataset.get_time_info 相当）。"""
    try:
        course_info = race_results.get_course_info(race_info_row)
        if course_info[0] < 0:
            return [0, 0]
        race_time = average_time.get_race_time_msec(race_info_row["タイム"])
        return average_time.get_time_diff(race_time, course_info)
    except Exception:
        return [np.nan, np.nan]


# ---------- データセット作成 ----------

def make_dataset_for_lightGBM_v2(race_id, course_info, horse_id):
    """v2特徴量の1行を作成する。"""
    try:
        race_year = int(str(race_id)[:4])
        peds_info = horse_peds.get_peds_info(horse_id)
        df_peds = peds_results.peds_index(peds_info[0], peds_info[1], course_info, race_year)
        df_peds = sum(df_peds.T.values.tolist(), [])

        race_info = past_performance.get_past_race_info(horse_id, race_id, race_num=3)
        df_race = get_past_race_info_data_v2(race_info)

        row = df_peds + df_race
        return pd.DataFrame([row])
    except Exception as e:
        print(f"  make_dataset_v2 error ({race_id}, {horse_id}): {e}")
        return pd.DataFrame()


def save_dataset_v2(place_id, year, race_type, length, df, flag_list):
    out_dir = os.path.join(paths_v3.PREDICTION_DATASET_PATH, name_header.PLACE_LIST[place_id - 1])
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, f"{year}_{race_type}{length}_ai_dataset")
    df.reset_index(drop=True).to_csv(base + "_for_rank_v2.csv")
    pd.DataFrame(flag_list, columns=["result_flag"]).to_csv(base + "_flag_v2.csv")


def load_dataset_v2(place_id, year, race_type, length):
    out_dir = os.path.join(paths_v3.PREDICTION_DATASET_PATH, name_header.PLACE_LIST[place_id - 1])
    base = os.path.join(out_dir, f"{year}_{race_type}{length}_ai_dataset")
    df_path = base + "_for_rank_v2.csv"
    flag_path = base + "_flag_v2.csv"
    df = pd.read_csv(df_path, index_col=0, dtype=float) if os.path.isfile(df_path) else pd.DataFrame()
    flag = pd.read_csv(flag_path, index_col=0, dtype=int) if os.path.isfile(flag_path) else pd.DataFrame()
    return df, flag


def make_dataset_for_train_v2(place_id, year=date.today().year, course_filter=None):
    """指定競馬場・年の v2 学習データセットを作成・保存する。

    Args:
        course_filter: [(race_type, length_str), ...] で処理対象コースを絞れる。
                       None の場合は全コースを処理する。
    """
    from src.PredictionModels.LightGBM.make_dataset import relevance_grade

    df_results = race_results.get_race_results_csv(place_id, year)
    if df_results.empty:
        print(f"  データなし: {year} {name_header.PLACE_LIST[place_id - 1]}")
        return

    courses = name_header.COURSE_LISTS[place_id - 1]
    if course_filter is not None:
        courses = [(t, l) for t, l in courses if (t, l) in course_filter]

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

        for i in tqdm(range(len(df_course))):
            df_result = df_course.iloc[i:i + 1]
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
            row = make_dataset_for_lightGBM_v2(race_id, course_info, horse_id)
            df_dataset = pd.concat([df_dataset, row])

        # race_id・枠番・馬番を結合
        df_dataset = pd.concat([
            pd.DataFrame(race_id_list, columns=["race_id"]),
            df_dataset.reset_index(drop=True),
            df_course["枠番"].reset_index(drop=True),
            df_course["馬番"].reset_index(drop=True),
        ], axis=1)
        df_dataset.columns = index_v2

        save_dataset_v2(place_id, year, race_type, length, df_dataset, flag_list)
        print(f"    保存完了: {len(df_dataset)}行")
