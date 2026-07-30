"""v19_no26 モデル学習スクリプト（状態変化特徴量 + オッズ重み付き二値分類）

v10 データセット（110列）を使用:
  - v9 の 105列（v8 + ペース適性7列）
  - v10 追加の 5列:
      kinryo_diff_1  : 前走からの斤量変化
      hw_change_1    : 前走からの馬体重変化
      dist_change_1  : 前走からの距離変化
      same_course_cnt5: 過去5走中の同コース出走数
      same_course_pr5 : 過去5走中の同コース複勝率

目的関数: binary + オッズ重み付き（v15pace と同一）
TRAIN_YEARS: 2020〜2025（リークなし）
サフィックス: _v19_no26

実行: python scripts/train_v19_no26.py
"""

import os
import sys
import traceback
import warnings
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
from src.config import paths as paths_v3
from src.PredictionModels.LightGBM.prediction import _band_lengths
from src.PredictionModels.LightGBM.make_dataset_v10 import load_dataset_v10

TRAIN_YEARS   = list(range(2020, 2026))
TARGET_VENUES = list(range(1, 11))
MODEL_SUFFIX  = "_v19_no26"
N_TRIALS      = 20

_DEVICE_KWARGS = {"device": LIGHTGBM_DEVICE} if USE_GPU_TRAINING else {"force_col_wise": True}


def _to_binary_label(flag_df):
    return (flag_df["result_flag"] == 4).astype(int)


def _make_sample_weight(y_binary, X_df):
    if "current_odds" in X_df.columns:
        odds = X_df["current_odds"].values.copy()
        valid = np.isfinite(odds) & (odds > 0)
        odds[~valid] = 1.0
    else:
        odds = np.ones(len(y_binary))
    return np.where(y_binary == 1, odds, 1.0)


def split_timeseries(race_data, race_flag, train_ratio=0.8):
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


def _fit(params, X_tr, y_tr, w_tr, X_va, y_va, w_va):
    model = lgb.LGBMClassifier(
        objective="binary", random_state=0, verbosity=-1,
        **_DEVICE_KWARGS, **params,
    )
    model.fit(
        X_tr, y_tr, sample_weight=w_tr,
        eval_set=[(X_va, y_va)],
        eval_sample_weight=[w_va],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    return model


def _fit_final(params, X_tr, y_tr, w_tr):
    model = lgb.LGBMClassifier(
        objective="binary", random_state=0, verbosity=-1,
        **_DEVICE_KWARGS, **params,
    )
    model.fit(X_tr, y_tr, sample_weight=w_tr, callbacks=[lgb.log_evaluation(period=0)])
    return model


def _tune(X_tr, y_tr, w_tr, X_va, y_va, w_va, n_trials=N_TRIALS):
    min_cs_high = max(2, min(30, len(X_tr) // 10))

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
        model = _fit(params, X_tr, y_tr, w_tr, X_va, y_va, w_va)
        logloss = model.evals_result_["valid_0"]["binary_logloss"][model.best_iteration_ - 1]
        return -logloss

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=0)
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def _save_model(model, place_id, race_type, length):
    type_str = "turf" if race_type == "芝" else "dirt"
    model_dir = os.path.join(paths_v3.PREDICTION_MODEL_PATH, name_header.PLACE_LIST[place_id - 1])
    os.makedirs(model_dir, exist_ok=True)
    path = os.path.join(model_dir, f"{type_str}{length}_lambdarank_model{MODEL_SUFFIX}.txt")
    model.booster_.save_model(path)
    return path


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
                        df, flag = load_dataset_v10(place_id, year, race_type, band_length)
                        if not df.empty and not flag.empty:
                            race_data = pd.concat([race_data, df])
                            race_flag = pd.concat([race_flag, flag])

                if race_data.empty:
                    print("    データなし: スキップ"); continue

                _num_cols = [c for c in race_data.columns if c != "race_id"]
                race_data[_num_cols] = race_data[_num_cols].fillna(-1)
                race_data = race_data.reset_index(drop=True)
                race_flag = race_flag.reset_index(drop=True)

                data_tr, data_va, flag_tr, flag_va = split_timeseries(race_data, race_flag)
                if data_va.empty:
                    print("    validationなし: スキップ"); continue

                X_tr = data_tr.drop(columns=["race_id"])
                X_va = data_va.drop(columns=["race_id"])
                y_tr = _to_binary_label(flag_tr)
                y_va = _to_binary_label(flag_va)
                w_tr = _make_sample_weight(y_tr.values, X_tr)
                w_va = _make_sample_weight(y_va.values, X_va)

                pos_ratio = y_tr.mean()
                mean_win_odds = (
                    X_tr.loc[y_tr == 1, "current_odds"].replace(-1, np.nan).mean()
                    if "current_odds" in X_tr.columns else float("nan")
                )
                print(f"    train={len(X_tr)}行 / val={len(X_va)}行 / "
                      f"1着率={pos_ratio:.3f} / 平均勝馬オッズ={mean_win_odds:.1f}")

                best_params = _tune(X_tr, y_tr, w_tr, X_va, y_va, w_va)
                print(f"    params: {best_params}")

                model = _fit_final(best_params, X_tr, y_tr, w_tr)
                path = _save_model(model, place_id, race_type, length)
                print(f"    保存: {path}")

            except Exception:
                print(f"    ERROR: {race_type}{length}m")
                traceback.print_exc()


if __name__ == "__main__":
    print("=" * 55)
    print(f"v19_no26  学習年: {TRAIN_YEARS[0]}〜{TRAIN_YEARS[-1]}")
    print("データセット: v10（110列: v9 + 状態変化5列）")
    print("目的関数: binary + オッズ重み付き")
    print("=" * 55)
    train_all_courses()
    print("\n" + "=" * 55 + "\n完了\n" + "=" * 55)
