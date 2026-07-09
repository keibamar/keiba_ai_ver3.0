"""v6hit_extreme モデル学習（2026年除外・時系列分割修正版）

  - TRAIN_YEARS: 2020〜2025（2026除外）
  - split: 先頭80%=train（古い）、後ろ20%=validation（新しい）← 時系列的に正しい方向
  - label_gain: [0, 0, 0, 0, 100]（1着のみ評価）
  - データ: v4のCSVを流用
  - サフィックス: _v6hit_extreme_no26

実行:
    python scripts/train_v6hit_extreme_no26.py
"""

import os, sys, traceback, warnings
warnings.simplefilter("ignore")

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
    _band_lengths, data_group, save_lightGBM_model,
)
from src.PredictionModels.LightGBM.make_dataset_v4 import load_dataset_v4

LABEL_GAIN    = [0, 0, 0, 0, 100]
TRAIN_YEARS   = list(range(2020, 2026))   # 2026年を除外
TARGET_VENUES = list(range(1, 11))
MODEL_SUFFIX  = "_v6hit_extreme_no26"
N_TRIALS      = 20

_DEVICE_KWARGS = {"device": LIGHTGBM_DEVICE} if USE_GPU_TRAINING else {"force_col_wise": True}


def split_timeseries(race_data, race_flag, train_ratio=0.8):
    """先頭80%=train（古い）、後ろ20%=validation（新しい）に分割。"""
    n = len(race_data)
    split = int(n * train_ratio)
    while split < n - 1 and race_data.at[split - 1, "race_id"] == race_data.at[split, "race_id"]:
        split += 1
    if split >= n:
        return race_data, pd.DataFrame(), race_flag, pd.DataFrame()
    return (
        race_data.iloc[:split].reset_index(drop=True),
        race_data.iloc[split:].reset_index(drop=True),
        race_flag.iloc[:split].reset_index(drop=True),
        race_flag.iloc[split:].reset_index(drop=True),
    )


def _fit_ranker(params, data_train, flag_train, train_group,
                data_test, flag_test, test_group):
    model = lgb.LGBMRanker(random_state=0, verbosity=-1,
                            label_gain=LABEL_GAIN, **_DEVICE_KWARGS, **params)
    model.fit(
        data_train, flag_train, group=train_group,
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
    model = lgb.LGBMRanker(random_state=0, verbosity=-1,
                            label_gain=LABEL_GAIN, **_DEVICE_KWARGS, **params)
    model.fit(
        data_train, flag_train, group=train_group,
        callbacks=[lgb.log_evaluation(period=0)],
    )
    return model


def _tune(data_train, flag_train, train_group,
          data_test, flag_test, test_group, n_trials=20):
    min_cs_high = max(2, min(30, len(data_train) // 10))

    def objective(trial):
        params = {
            "n_estimators":      trial.suggest_int("n_estimators", 100, 500),
            "learning_rate":     trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
            "num_leaves":        trial.suggest_int("num_leaves", 16, 64),
            "max_depth":         trial.suggest_int("max_depth", 3, 10),
            "min_child_samples": trial.suggest_int("min_child_samples", 2, min_cs_high),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-3, 3.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-3, 3.0, log=True),
        }
        model = _fit_ranker(params, data_train, flag_train, train_group,
                            data_test, flag_test, test_group)
        ndcg1 = model.evals_result_["valid_0"]["ndcg@1"][model.best_iteration_ - 1]
        val_scores = model.predict(data_test, num_iteration=model.best_iteration_)
        unique_ratio = len(np.unique(np.round(val_scores, 6))) / max(len(val_scores), 1)
        return ndcg1 + 0.4 * unique_ratio

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=0))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def train_all_courses():
    for place_id in TARGET_VENUES:
        place_name = name_header.PLACE_LIST[place_id - 1]
        print(f"\n{'='*55}\n[学習] {place_name}\n{'='*55}")

        for race_type, length in name_header.COURSE_LISTS[place_id - 1]:
            print(f"\n  {race_type}{length}m ...")
            try:
                race_data, race_flag = pd.DataFrame(), pd.DataFrame()
                for year in TRAIN_YEARS:
                    for band_length in _band_lengths(place_id, race_type, length):
                        df, flag = load_dataset_v4(place_id, year, race_type, band_length)
                        if not df.empty and not flag.empty:
                            race_data = pd.concat([race_data, df])
                            race_flag = pd.concat([race_flag, flag])

                if race_data.empty:
                    print(f"    データなし: スキップ"); continue

                _num_cols = [c for c in race_data.columns if c != "race_id"]
                race_data[_num_cols] = race_data[_num_cols].fillna(-1)
                race_data = race_data.reset_index(drop=True)
                race_flag = race_flag.reset_index(drop=True)

                data_train, data_test, flag_train, flag_test = split_timeseries(race_data, race_flag)
                if data_test.empty:
                    print(f"    validationなし: スキップ"); continue

                data_train, train_group = data_group(data_train)
                data_test,  test_group  = data_group(data_test)

                print(f"    train={len(data_train)}行 / val={len(data_test)}行 / {data_train.shape[1]}列")
                best_params = _tune(data_train, flag_train, train_group,
                                    data_test, flag_test, test_group, N_TRIALS)
                print(f"    params: {best_params}")
                model = _fit_ranker_final(best_params, data_train, flag_train, train_group)
                save_lightGBM_model(model, place_id, race_type, length, model_suffix=MODEL_SUFFIX)
                print(f"    保存: {race_type}{length}m{MODEL_SUFFIX}")

            except Exception:
                print(f"    ERROR: {race_type}{length}m")
                traceback.print_exc()


if __name__ == "__main__":
    print("=" * 55)
    print(f"v6hit_extreme_no26  学習年: {TRAIN_YEARS[0]}〜{TRAIN_YEARS[-1]}")
    print(f"label_gain: {LABEL_GAIN}")
    print(f"split: 先頭80%=train / 後ろ20%=validation（時系列正順）")
    print("=" * 55)
    train_all_courses()
    print("\n" + "=" * 55 + "\n完了\n" + "=" * 55)
