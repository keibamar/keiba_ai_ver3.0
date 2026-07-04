"""v4 モデル学習スクリプト（的中率重視・オッズ/人気特徴量追加）

v3 の 65 特徴量 + current_odds + current_popularity = 67 特徴量で
LightGBM LambdaRank モデルを学習し _v4 サフィックスで保存する。

対象: 全10場

実行:
    python scripts/train_v4_models.py
"""

import os
import sys
import traceback
import warnings
warnings.simplefilter("ignore")

from datetime import date

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

sys.path.append(r"C:\keiba_ai\keiba_ai_ver2.0\libs")
sys.path.append(r"C:\keiba_ai\keiba_ai_ver2.0\src\Datasets")
sys.path.append(r"C:\keiba_ai\keiba_ai_ver2.0\src\PredictionModels\LightGBM")

import name_header
from src.PredictionModels.LightGBM.prediction import (
    _band_lengths, data_group, split_dataframe,
    tune_hyperparameters, _fit_ranker_final, save_lightGBM_model,
)
from src.PredictionModels.LightGBM.make_dataset_v4 import (
    load_dataset_v4, make_dataset_for_train_v4,
)

TRAIN_YEARS   = list(range(2020, date.today().year + 1))
TARGET_VENUES = list(range(1, 11))  # 全10場
MODEL_SUFFIX  = "_v4"
N_TRIALS      = 20


# ---------- データセット生成 ----------

def generate_missing_datasets():
    for place_id in TARGET_VENUES:
        place_name = name_header.PLACE_LIST[place_id - 1]
        courses    = name_header.COURSE_LISTS[place_id - 1]

        for year in TRAIN_YEARS:
            missing = [
                (rt, ln) for rt, ln in courses
                if load_dataset_v4(place_id, year, rt, ln)[0].empty
            ]
            if not missing:
                continue

            print(f"\n[データ生成] {place_name} {year}年 ({len(missing)}コース欠損)")
            try:
                make_dataset_for_train_v4(place_id, year, course_filter=missing)
            except Exception:
                print(f"  ERROR: {place_name} {year}")
                traceback.print_exc()


# ---------- モデル学習 ----------

def train_all_courses():
    for place_id in TARGET_VENUES:
        place_name = name_header.PLACE_LIST[place_id - 1]
        print(f"\n{'='*55}")
        print(f"[学習] {place_name} (place_id={place_id})")
        print(f"{'='*55}")

        for race_type, length in name_header.COURSE_LISTS[place_id - 1]:
            print(f"\n  {race_type}{length}m ...")
            try:
                race_data = pd.DataFrame()
                race_flag = pd.DataFrame()

                band_lengths = _band_lengths(place_id, race_type, length)
                for year in TRAIN_YEARS:
                    for band_length in band_lengths:
                        df, flag = load_dataset_v4(place_id, year, race_type, band_length)
                        if not df.empty and not flag.empty:
                            race_data = pd.concat([race_data, df])
                            race_flag = pd.concat([race_flag, flag])

                if race_data.empty or race_flag.empty:
                    print(f"    データなし: スキップ")
                    continue

                race_data = race_data.fillna(-1).reset_index(drop=True)
                race_flag = race_flag.reset_index(drop=True)

                data_train, data_test, flag_train, flag_test = split_dataframe(race_data, race_flag)
                if data_test.empty:
                    print(f"    テストデータなし: スキップ")
                    continue

                data_train, train_group = data_group(data_train)
                data_test,  test_group  = data_group(data_test)

                print(f"    訓練 {len(data_train)}行 / テスト {len(data_test)}行")
                best_params = tune_hyperparameters(
                    data_train, flag_train, train_group,
                    data_test,  flag_test,  test_group,
                    n_trials=N_TRIALS,
                )
                print(f"    最良パラメータ: {best_params}")

                model = _fit_ranker_final(best_params, data_train, flag_train, train_group)
                save_lightGBM_model(model, place_id, race_type, length, model_suffix=MODEL_SUFFIX)
                print(f"    保存完了: {race_type}{length}m{MODEL_SUFFIX}")

            except Exception:
                print(f"    ERROR: {race_type}{length}m")
                traceback.print_exc()


# ---------- エントリポイント ----------

if __name__ == "__main__":
    print("=" * 55)
    print("v4 モデル学習（的中率重視 + オッズ/人気特徴量）")
    print(f"対象場: 全10場")
    print(f"訓練年: {TRAIN_YEARS[0]}〜{TRAIN_YEARS[-1]}")
    print("=" * 55)

    print("\n[Step 1] 欠損データセットを生成...")
    generate_missing_datasets()

    print("\n[Step 2] モデル学習...")
    train_all_courses()

    print("\n" + "=" * 55)
    print("完了")
    print("=" * 55)
