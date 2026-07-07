"""v5' モデル学習スクリプト（v5データセット流用・オッズ/人気列を除外）

v5 の CSV（70列）から current_odds / current_popularity を除いた 68列で学習する。
データセットの再生成は不要。

対象: 全10場
実行:
    python scripts/train_v5prime_models.py
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
from src.PredictionModels.LightGBM.prediction import (
    _band_lengths, data_group, split_dataframe, save_lightGBM_model,
)
from src.PredictionModels.LightGBM.make_dataset_v5 import (
    CAT_COLS, load_dataset_v5,
)

# v5データセットからこの2列を除いてv5'とする
EXCLUDE_COLS = ["current_odds", "current_popularity"]

TRAIN_YEARS   = list(range(2020, date.today().year + 1))
TARGET_VENUES = list(range(1, 11))
MODEL_SUFFIX  = "_v5prime"
N_TRIALS      = 20

_DEVICE_KWARGS = {"device": LIGHTGBM_DEVICE} if USE_GPU_TRAINING else {"force_col_wise": True}


def _fit_ranker(params, data_train, flag_train, train_group,
                data_test, flag_test, test_group):
    model = lgb.LGBMRanker(random_state=0, verbosity=-1, **_DEVICE_KWARGS, **params)
    model.fit(
        data_train, flag_train, group=train_group,
        categorical_feature=CAT_COLS,
        eval_set=[(data_test, flag_test)],
        eval_group=[list(test_group)],
        eval_at=[1, 3, 5],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    return model


def _fit_ranker_final(params, data_train, flag_train, train_group):
    model = lgb.LGBMRanker(random_state=0, verbosity=-1, **_DEVICE_KWARGS, **params)
    model.fit(
        data_train, flag_train, group=train_group,
        categorical_feature=CAT_COLS,
        callbacks=[lgb.log_evaluation(period=0)],
    )
    return model


def _tune_hyperparameters(data_train, flag_train, train_group,
                          data_test, flag_test, test_group, n_trials=20):
    min_cs_low  = 2
    min_cs_high = max(min_cs_low, min(30, len(data_train) // 10))

    def objective(trial):
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 100, 500),
            "learning_rate":     trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
            "num_leaves":        trial.suggest_int("num_leaves", 16, 64),
            "max_depth":         trial.suggest_int("max_depth", 3, 10),
            "min_child_samples": trial.suggest_int("min_child_samples", min_cs_low, min_cs_high),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-3, 3.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-3, 3.0, log=True),
        }
        model = _fit_ranker(
            params, data_train, flag_train, train_group,
            data_test, flag_test, test_group,
        )
        ndcg1 = model.evals_result_["valid_0"]["ndcg@1"][model.best_iteration_ - 1]
        val_scores = model.predict(data_test, num_iteration=model.best_iteration_)
        unique_ratio = (
            len(np.unique(np.round(val_scores, 6))) / len(val_scores)
            if len(val_scores) > 0 else 1.0
        )
        return ndcg1 + 0.4 * unique_ratio

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
                        df, flag = load_dataset_v5(place_id, year, race_type, band_length)
                        if not df.empty and not flag.empty:
                            race_data = pd.concat([race_data, df])
                            race_flag = pd.concat([race_flag, flag])

                if race_data.empty or race_flag.empty:
                    print(f"    v5データなし: スキップ")
                    continue

                # オッズ・人気を除外
                drop_cols = [c for c in EXCLUDE_COLS if c in race_data.columns]
                if drop_cols:
                    race_data = race_data.drop(columns=drop_cols)

                # 欠損補完
                _actual_num_cols = [c for c in race_data.columns if c not in ["race_id"] + CAT_COLS]
                race_data[_actual_num_cols] = race_data[_actual_num_cols].fillna(-1)
                race_data[CAT_COLS] = race_data[CAT_COLS].fillna(0).astype(int)
                race_data = race_data.reset_index(drop=True)
                race_flag = race_flag.reset_index(drop=True)

                data_train, data_test, flag_train, flag_test = split_dataframe(race_data, race_flag)
                if data_test.empty:
                    print(f"    テストデータなし: スキップ")
                    continue

                data_train, train_group = data_group(data_train)
                data_test,  test_group  = data_group(data_test)

                for col in CAT_COLS:
                    if col in data_train.columns:
                        data_train[col] = data_train[col].astype(int)
                        data_test[col]  = data_test[col].astype(int)

                print(f"    訓練 {len(data_train)}行 / テスト {len(data_test)}行 / 特徴量 {data_train.shape[1]}列")
                best_params = _tune_hyperparameters(
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


if __name__ == "__main__":
    print("=" * 55)
    print("v5' モデル学習（v5データセット流用・オッズ/人気除外）")
    print(f"対象場: 全10場  訓練年: {TRAIN_YEARS[0]}〜{TRAIN_YEARS[-1]}")
    print(f"除外列: {EXCLUDE_COLS}")
    print("=" * 55)

    train_all_courses()

    print("\n" + "=" * 55)
    print("完了")
    print("=" * 55)
