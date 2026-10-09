"""LightGBM学習データセット生成 v12

v11 の特徴量（114列）に誕生月を追加:
  - birth_month : 馬の誕生月 (1〜12、不明時はNaN)

v11=114列 → v12=115列。データは "_v12birth" サフィックスで保存。

生成方式: v11データセットCSVを読み込み、birth_monthを付加（高速拡張）。
  v11データセットが存在しない場合はスキップ（先に make_v11_datasets_no26.py を実行すること）。
"""

import os
import sys

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.dont_write_bytecode = True
sys.path.append(r"C:\keiba_ai\keiba_ai_ver2.0\libs")
sys.path.append(r"C:\keiba_ai\keiba_ai_ver2.0\src\Datasets")
import name_header
import race_results

_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.config import paths as paths_v3
from src.managers import horse_profile_dataset_manager
from src.PredictionModels.LightGBM.make_dataset_v11 import index_v11

# v12 特徴量列名: v11（114列）+ 1列 = 115列
index_v12 = index_v11 + [
    "birth_month",  # 馬の誕生月 (1〜12)
]


# ============================================================
# データセット保存・読み込み
# ============================================================

def save_dataset_v12(place_id, year, race_type, length, df, flag_list):
    out_dir = os.path.join(
        paths_v3.PREDICTION_DATASET_PATH, name_header.PLACE_LIST[place_id - 1]
    )
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, f"{year}_{race_type}{length}_ai_dataset")
    df.reset_index(drop=True).to_csv(base + "_for_rank_v12.csv")
    flag_df = pd.read_csv(base + "_flag_v11.csv", index_col=0) if os.path.isfile(base + "_flag_v11.csv") else pd.DataFrame(flag_list, columns=["result_flag"])
    flag_df.to_csv(base + "_flag_v12.csv")


def load_dataset_v12(place_id, year, race_type, length):
    out_dir = os.path.join(
        paths_v3.PREDICTION_DATASET_PATH, name_header.PLACE_LIST[place_id - 1]
    )
    base = os.path.join(out_dir, f"{year}_{race_type}{length}_ai_dataset")
    df_path   = base + "_for_rank_v12.csv"
    flag_path = base + "_flag_v12.csv"
    df   = pd.read_csv(df_path,   index_col=0, dtype=float) if os.path.isfile(df_path)   else pd.DataFrame()
    flag = pd.read_csv(flag_path, index_col=0, dtype=int)   if os.path.isfile(flag_path) else pd.DataFrame()
    return df, flag


# ============================================================
# メイン: v11 データセットに birth_month を追加
# ============================================================

def make_dataset_for_train_v12(place_id, year, course_filter=None):
    """指定競馬場・年の v12 学習データセットを作成・保存する。

    v11 データセット CSV を読み込み、birth_month 列を追加して v12 として保存。
    v11 データが存在しないコースはスキップ。

    Args:
        place_id      : 競馬場ID (1〜10)
        year          : 対象年
        course_filter : [(race_type, length), ...] で絞り込む場合に指定。None で全コース。
    """
    out_dir = os.path.join(
        paths_v3.PREDICTION_DATASET_PATH, name_header.PLACE_LIST[place_id - 1]
    )

    # race_results から horse_id を取得するためにロード
    df_results = race_results.get_race_results_csv(place_id, year)
    if df_results.empty:
        print(f"  データなし: {year} {name_header.PLACE_LIST[place_id - 1]}")
        return

    courses = name_header.COURSE_LISTS[place_id - 1]
    if course_filter is not None:
        courses = [(t, l) for t, l in courses if (t, l) in course_filter]

    for race_type, length in courses:
        base = os.path.join(out_dir, f"{year}_{race_type}{length}_ai_dataset")
        v11_path  = base + "_for_rank_v11.csv"
        flag_path = base + "_flag_v11.csv"
        v12_path  = base + "_for_rank_v12.csv"
        v12_flag  = base + "_flag_v12.csv"

        if os.path.isfile(v12_path):
            print(f"  スキップ（作成済み）: {year}_{race_type}{length}")
            continue

        if not os.path.isfile(v11_path):
            print(f"  スキップ（v11なし）: {year}_{race_type}{length}")
            continue

        # v11 データセット読み込み
        df_v11 = pd.read_csv(v11_path, index_col=0, dtype=float)
        if df_v11.empty:
            print(f"  スキップ（v11空）: {year}_{race_type}{length}")
            continue

        # 同コースのレース結果（horse_id取得用）
        df_course = df_results[
            (df_results["race_type"] == race_type) & (df_results["course_len"] == length)
        ]
        if len(df_course) != len(df_v11):
            print(f"  警告: 行数不一致 {year}_{race_type}{length} "
                  f"(v11={len(df_v11)}, results={len(df_course)})")
            # 行数が合わない場合はrace_idで照合を試みる
            race_ids_v11 = df_v11["race_id"].values.astype(int)
            horse_id_series = []
            for rid in tqdm(race_ids_v11, desc=f"horse_id照合 {year}_{race_type}{length}"):
                matching = df_course[df_course.index == rid]
                if not matching.empty:
                    horse_id_series.append(str(matching.iloc[0]["horse_id"]))
                else:
                    horse_id_series.append(None)
        else:
            horse_id_series = [str(hid) for hid in df_course["horse_id"].values]

        # birth_month を一括取得
        birth_months = []
        print(f"  {year}_{race_type}{length}m ({len(df_v11)}行) birth_month 付加中...")
        for hid in tqdm(horse_id_series, desc="birth_month"):
            if hid is None:
                birth_months.append(np.nan)
                continue
            bm = horse_profile_dataset_manager.get_birth_month_cached(str(hid))
            birth_months.append(float(bm) if bm is not None else np.nan)

        df_v12 = df_v11.copy()
        df_v12["birth_month"] = birth_months

        # 保存（v12）
        df_v12.reset_index(drop=True).to_csv(v12_path)
        # flag は v11 と共通
        if os.path.isfile(flag_path):
            import shutil
            shutil.copy2(flag_path, v12_flag)
        print(f"    保存完了: {len(df_v12)}行 (birth_month NaN率: "
              f"{sum(1 for v in birth_months if np.isnan(v)) / max(len(birth_months),1):.1%})")
