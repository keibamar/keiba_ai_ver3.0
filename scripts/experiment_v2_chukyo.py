"""v2特徴量の効果検証スクリプト（中京ダート）

v1（現行）と v2（上り・着差・通過順位・トレンド・体重変化追加）を
同じデータ・分割条件で比較し AUC と特徴量重要度を表示する。

実行:
    python scripts/experiment_v2_chukyo.py
"""

import os
import sys
import warnings

warnings.simplefilter("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

sys.path.append(r"C:\keiba_ai\keiba_ai_ver2.0\libs")
sys.path.append(r"C:\keiba_ai\keiba_ai_ver2.0\src\Datasets")
sys.path.append(r"C:\keiba_ai\keiba_ai_ver2.0\src\PredictionModels\LightGBM")

import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
from sklearn.metrics import roc_auc_score
optuna.logging.set_verbosity(optuna.logging.WARNING)

import name_header
from src.PredictionModels.LightGBM import make_dataset, make_dataset_v2
from src.PredictionModels.LightGBM.prediction import (
    data_group, split_dataframe, tune_hyperparameters, _fit_ranker_final
)

PLACE_ID = 7   # 中京
TARGET = [("ダート", "1400"), ("ダート", "1800"), ("ダート", "1900")]
YEARS = list(range(2020, 2026))
N_TRIALS = 20


def load_combined(place_id, years, race_type, length, version="v1"):
    """複数年のデータを結合して返す。"""
    data_all = pd.DataFrame()
    flag_all = pd.DataFrame()

    for year in years:
        if version == "v1":
            df = make_dataset.get_LightGBM_dataset_csv(place_id, year, race_type, length)
            flag = make_dataset.get_LightGBM_dataset_flag_csv(place_id, year, race_type, length)
        else:
            df, flag = make_dataset_v2.load_dataset_v2(place_id, year, race_type, length)

        if not df.empty and not flag.empty:
            data_all = pd.concat([data_all, df])
            flag_all = pd.concat([flag_all, flag])

    return data_all.reset_index(drop=True), flag_all.reset_index(drop=True)


def evaluate(data, flag, label=""):
    """LambdaRank モデルを学習・評価してAUCと特徴量重要度を返す。"""
    data = data.fillna(-1).reset_index(drop=True)
    flag = flag.reset_index(drop=True)

    data_train, data_test, flag_train, flag_test = split_dataframe(data, flag)
    if data_test.empty:
        print(f"  {label}: テストデータなし、スキップ")
        return None, None

    data_train, train_group = data_group(data_train)
    data_test, test_group = data_group(data_test)

    best_params = tune_hyperparameters(
        data_train, flag_train, train_group,
        data_test, flag_test, test_group,
        n_trials=N_TRIALS,
    )
    model = _fit_ranker_final(best_params, data_train, flag_train, train_group)

    # AUC（3着以内 = 1, それ以外 = 0）
    y_true = (flag_test["result_flag"] >= 2).astype(int).values
    scores = model.predict(data_test)
    auc = roc_auc_score(y_true, scores)

    # 特徴量重要度（gain）
    fi = pd.Series(
        model.feature_importances_,
        index=data_train.columns,
    ).sort_values(ascending=False)

    return auc, fi


def build_v2_datasets(force=False):
    """v2データセットが未生成の場合のみ作成する。"""
    from src.PredictionModels.LightGBM.make_dataset_v2 import make_dataset_for_train_v2

    for year in YEARS:
        for race_type, length in TARGET:
            df, _ = make_dataset_v2.load_dataset_v2(PLACE_ID, year, race_type, length)
            if df.empty or force:
                print(f"v2データセット作成: {year} ダート{length}m")
                make_dataset_for_train_v2(PLACE_ID, year)
                break  # make_dataset_for_train_v2 は全コースを処理するので1回でOK
        else:
            continue
        break


if __name__ == "__main__":
    print("=" * 60)
    print("v2特徴量 効果検証: 中京ダート")
    print("=" * 60)

    # v2 データセットを必要なら生成
    print("\n[Step 1] v2 データセット確認・生成")
    need_build = False
    for year in YEARS:
        for race_type, length in TARGET:
            df, _ = make_dataset_v2.load_dataset_v2(PLACE_ID, year, race_type, length)
            if df.empty:
                need_build = True
                break

    if need_build:
        print("v2 データセットを生成します（時間がかかります）")
        from src.PredictionModels.LightGBM.make_dataset_v2 import make_dataset_for_train_v2
        for year in YEARS:
            print(f"\n--- {year}年 ---")
            make_dataset_for_train_v2(PLACE_ID, year)
    else:
        print("v2 データセット: 全て存在します")

    # 評価
    print("\n[Step 2] v1 vs v2 比較")
    results = []

    for race_type, length in TARGET:
        course_label = f"ダート{length}m"
        print(f"\n--- {course_label} ---")

        # v1 (現行)
        data_v1, flag_v1 = load_combined(PLACE_ID, YEARS, race_type, length, "v1")
        if data_v1.empty:
            print(f"  v1データなし: スキップ")
            continue
        print(f"  v1: {len(data_v1)}行 × {data_v1.shape[1]}列")
        auc_v1, fi_v1 = evaluate(data_v1, flag_v1, f"v1 {course_label}")

        # v2 (新規)
        data_v2, flag_v2 = load_combined(PLACE_ID, YEARS, race_type, length, "v2")
        if data_v2.empty:
            print(f"  v2データなし: スキップ")
            continue
        print(f"  v2: {len(data_v2)}行 × {data_v2.shape[1]}列")
        auc_v2, fi_v2 = evaluate(data_v2, flag_v2, f"v2 {course_label}")

        if auc_v1 is not None and auc_v2 is not None:
            diff = auc_v2 - auc_v1
            symbol = "↑" if diff > 0 else "↓" if diff < 0 else "→"
            results.append((course_label, auc_v1, auc_v2, diff))
            print(f"  AUC: v1={auc_v1:.4f}  v2={auc_v2:.4f}  差={diff:+.4f} {symbol}")

            # 新規特徴量の重要度
            new_cols = ["agari_1", "agari_2", "agari_3",
                        "margin_1", "margin_2", "margin_3",
                        "corner_ratio_1", "corner_ratio_2", "corner_ratio_3",
                        "rank_trend", "weight_change"]
            print(f"\n  [新規特徴量の重要度（gain）]")
            for col in new_cols:
                if col in fi_v2.index:
                    rank = (fi_v2.index.tolist().index(col) + 1)
                    print(f"    {col:20s}: {fi_v2[col]:8.1f}  (全{len(fi_v2)}列中 {rank}位)")

    # サマリ
    print("\n" + "=" * 60)
    print("サマリ")
    print("=" * 60)
    print(f"{'コース':<12} {'v1 AUC':>8} {'v2 AUC':>8} {'差':>8}")
    print("-" * 40)
    for course_label, a1, a2, diff in results:
        symbol = "↑" if diff > 0 else "↓"
        print(f"{course_label:<12} {a1:>8.4f} {a2:>8.4f} {diff:>+8.4f} {symbol}")
    print("=" * 60)
