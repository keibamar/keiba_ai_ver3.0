"""Phase 2 特徴量の効果検証スクリプト（中京ダート）

v2（上り・着差・通過順位・トレンド・体重変化）vs
v3（v2 + 騎手×コース勝率/複勝率 + 距離変更差分）を比較。

実行:
    python scripts/experiment_v3_chukyo.py
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
import optuna
from sklearn.metrics import roc_auc_score
optuna.logging.set_verbosity(optuna.logging.WARNING)

import name_header
from src.PredictionModels.LightGBM import make_dataset_v2, make_dataset_v3
from src.PredictionModels.LightGBM.prediction import (
    data_group, split_dataframe, tune_hyperparameters, _fit_ranker_final
)

PLACE_ID = 7   # 中京
TARGET = [("ダート", "1400"), ("ダート", "1800"), ("ダート", "1900")]
YEARS = list(range(2020, 2026))
N_TRIALS = 20


def load_combined(place_id, years, race_type, length, version="v2"):
    data_all = pd.DataFrame()
    flag_all = pd.DataFrame()
    for year in years:
        if version == "v2":
            df, flag = make_dataset_v2.load_dataset_v2(place_id, year, race_type, length)
        else:
            df, flag = make_dataset_v3.load_dataset_v3(place_id, year, race_type, length)
        if not df.empty and not flag.empty:
            data_all = pd.concat([data_all, df])
            flag_all = pd.concat([flag_all, flag])
    return data_all.reset_index(drop=True), flag_all.reset_index(drop=True)


def evaluate(data, flag, label=""):
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

    y_true = (flag_test["result_flag"] >= 2).astype(int).values
    scores = model.predict(data_test)
    auc = roc_auc_score(y_true, scores)

    fi = pd.Series(
        model.feature_importances_,
        index=data_train.columns,
    ).sort_values(ascending=False)

    return auc, fi


if __name__ == "__main__":
    print("=" * 60)
    print("v3特徴量 効果検証: 中京ダート（v2 vs v3）")
    print("=" * 60)

    # v3 データセット生成（未生成のみ）
    print("\n[Step 1] v3 データセット確認・生成")
    need_build = any(
        make_dataset_v3.load_dataset_v3(PLACE_ID, year, rt, ln)[0].empty
        for year in YEARS for rt, ln in TARGET
    )

    if need_build:
        print("v3 データセットを生成します")
        from src.PredictionModels.LightGBM.make_dataset_v3 import make_dataset_for_train_v3
        for year in YEARS:
            # 不足しているコースのみ生成
            missing = [
                (rt, ln) for rt, ln in TARGET
                if make_dataset_v3.load_dataset_v3(PLACE_ID, year, rt, ln)[0].empty
            ]
            if not missing:
                print(f"  {year}: 全コース生成済み")
                continue
            print(f"\n--- {year}年 ---")
            make_dataset_for_train_v3(PLACE_ID, year, course_filter=TARGET)
    else:
        print("v3 データセット: 全て存在します")

    # 評価
    print("\n[Step 2] v2 vs v3 比較")
    results = []

    for race_type, length in TARGET:
        course_label = f"ダート{length}m"
        print(f"\n--- {course_label} ---")

        # v2
        data_v2, flag_v2 = load_combined(PLACE_ID, YEARS, race_type, length, "v2")
        if data_v2.empty:
            print(f"  v2データなし: スキップ")
            continue
        print(f"  v2: {len(data_v2)}行 × {data_v2.shape[1]}列")
        auc_v2, fi_v2 = evaluate(data_v2, flag_v2, f"v2 {course_label}")

        # v3
        data_v3, flag_v3 = load_combined(PLACE_ID, YEARS, race_type, length, "v3")
        if data_v3.empty:
            print(f"  v3データなし: スキップ")
            continue
        print(f"  v3: {len(data_v3)}行 × {data_v3.shape[1]}列")
        auc_v3, fi_v3 = evaluate(data_v3, flag_v3, f"v3 {course_label}")

        if auc_v2 is not None and auc_v3 is not None:
            diff = auc_v3 - auc_v2
            symbol = "↑" if diff > 0 else "↓" if diff < 0 else "→"
            results.append((course_label, auc_v2, auc_v3, diff))
            print(f"  AUC: v2={auc_v2:.4f}  v3={auc_v3:.4f}  差={diff:+.4f} {symbol}")

            # Phase 2 新規特徴量の重要度
            new_cols = ["dist_change", "jockey_win_rate", "jockey_place_rate"]
            print(f"\n  [Phase 2 新規特徴量の重要度（gain）]")
            for col in new_cols:
                if col in fi_v3.index:
                    rank = fi_v3.index.tolist().index(col) + 1
                    print(f"    {col:25s}: {fi_v3[col]:8.1f}  (全{len(fi_v3)}列中 {rank}位)")

    # サマリ
    print("\n" + "=" * 60)
    print("サマリ")
    print("=" * 60)
    print(f"{'コース':<12} {'v2 AUC':>8} {'v3 AUC':>8} {'差':>8}")
    print("-" * 40)
    for course_label, a2, a3, diff in results:
        symbol = "↑" if diff > 0 else "↓"
        print(f"{course_label:<12} {a2:>8.4f} {a3:>8.4f} {diff:>+8.4f} {symbol}")
    print("=" * 60)
