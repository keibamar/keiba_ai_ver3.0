"""v6hit_binary モデル学習スクリプト（binary classification・1着予測）

v4 の CSV（67列）を使い、LGBMClassifier (binary) で
「1着かどうか」を直接予測することで単勝的中率を最大化する。

  ラベル変換: relevance_grade=4（1着） → 1、それ以外 → 0
  目的関数: binary_crossentropy（LightGBM binary classification）
  予測値: 1着確率（0〜1）→ 高い順に並べてランキング

データセットは v4 のCSVを流用（再生成不要）。
モデルは "_v6hit_binary" サフィックスで保存。

実行:
    python scripts/train_v6hit_binary_models.py
"""

import os
import sys
import traceback
import warnings

warnings.simplefilter("ignore")

from datetime import date

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd

optuna.logging.set_verbosity(optuna.logging.WARNING)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

sys.path.append(r"C:\keiba_ai\keiba_ai_ver2.0\libs")
sys.path.append(r"C:\keiba_ai\keiba_ai_ver2.0\src\Datasets")
sys.path.append(r"C:\keiba_ai\keiba_ai_ver2.0\src\PredictionModels\LightGBM")

import name_header
from src.config.constants import USE_GPU_TRAINING, LIGHTGBM_DEVICE
from src.config import paths as paths_v3
from src.PredictionModels.LightGBM.prediction import (
    _band_lengths, split_dataframe,
)
from src.PredictionModels.LightGBM.make_dataset_v4 import load_dataset_v4

TRAIN_YEARS   = list(range(2020, date.today().year + 1))
TARGET_VENUES = list(range(1, 11))
MODEL_SUFFIX  = "_v6hit_binary"
N_TRIALS      = 20

_DEVICE_KWARGS = {"device": LIGHTGBM_DEVICE} if USE_GPU_TRAINING else {"force_col_wise": True}


def _to_binary_flag(flag_df):
    """relevance_grade=4（1着）を1、それ以外を0に変換"""
    return (flag_df["result_flag"] == 4).astype(int)


def _save_model(model, place_id, race_type, length):
    type_str = "turf" if race_type == "芝" else "dirt"
    model_dir = os.path.join(paths_v3.PREDICTION_MODEL_PATH, name_header.PLACE_LIST[place_id - 1])
    os.makedirs(model_dir, exist_ok=True)
    path = os.path.join(model_dir, f"{type_str}{length}_lambdarank_model{MODEL_SUFFIX}.txt")
    model.booster_.save_model(path)
    return path


def _fit_classifier(params, X_train, y_train, X_test, y_test):
    model = lgb.LGBMClassifier(
        objective="binary", random_state=0, verbosity=-1,
        **_DEVICE_KWARGS, **params,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    return model


def _fit_classifier_final(params, X_train, y_train):
    model = lgb.LGBMClassifier(
        objective="binary", random_state=0, verbosity=-1,
        **_DEVICE_KWARGS, **params,
    )
    model.fit(X_train, y_train, callbacks=[lgb.log_evaluation(period=0)])
    return model


def _tune(X_train, y_train, X_test, y_test, n_trials=20):
    min_cs_high = max(2, min(30, len(X_train) // 10))

    def objective(trial):
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 100, 500),
            "learning_rate":     trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
            "num_leaves":        trial.suggest_int("num_leaves", 16, 64),
            "max_depth":         trial.suggest_int("max_depth", 3, 10),
            "min_child_samples": trial.suggest_int("min_child_samples", 2, min_cs_high),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-3, 3.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-3, 3.0, log=True),
            "scale_pos_weight":  trial.suggest_float("scale_pos_weight", 1.0, 20.0),
        }
        model = _fit_classifier(params, X_train, y_train, X_test, y_test)
        # AUCで評価
        auc = model.evals_result_["valid_0"]["binary_logloss"][model.best_iteration_ - 1]
        return -auc  # loglossは小さいほど良い

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=0)
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


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
                    print(f"    v4データなし: スキップ")
                    continue

                _num_cols = [c for c in race_data.columns if c != "race_id"]
                race_data[_num_cols] = race_data[_num_cols].fillna(-1)
                race_data = race_data.reset_index(drop=True)
                race_flag = race_flag.reset_index(drop=True)

                # race_idを除いた特徴量
                X_all = race_data.drop(columns=["race_id"])
                y_all = _to_binary_flag(race_flag)

                # 時系列分割（split_dataframeと同じ比率: 後ろ20%をテスト）
                n = len(X_all)
                split = int(n * 0.8)
                X_train, X_test = X_all.iloc[:split], X_all.iloc[split:]
                y_train, y_test = y_all.iloc[:split], y_all.iloc[split:]

                if len(X_test) == 0:
                    print(f"    テストデータなし: スキップ")
                    continue

                pos_ratio = y_train.mean()
                print(f"    訓練 {len(X_train)}行 / テスト {len(X_test)}行 / 1着率 {pos_ratio:.3f}")

                best_params = _tune(X_train, y_train, X_test, y_test, n_trials=N_TRIALS)
                print(f"    最良パラメータ: {best_params}")

                model = _fit_classifier_final(best_params, X_train, y_train)
                path = _save_model(model, place_id, race_type, length)
                print(f"    保存完了: {path}")

            except Exception:
                print(f"    ERROR: {race_type}{length}m")
                traceback.print_exc()


if __name__ == "__main__":
    print("=" * 55)
    print("v6hit_binary モデル学習（binary classification・1着予測）")
    print(f"対象場: 全10場  訓練年: {TRAIN_YEARS[0]}〜{TRAIN_YEARS[-1]}")
    print("=" * 55)

    train_all_courses()

    print("\n" + "=" * 55)
    print("完了")
    print("=" * 55)
