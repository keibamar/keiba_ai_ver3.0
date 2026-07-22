"""train_nodds_no26.py  —  オッズ不使用 3モデル同時学習

前日予想用（current_odds / current_popularity を特徴量から除外）。
v9データセット（105列）から上記2列と race_id を除いた102列のサブセットで
v11_nodds / v12_nodds / v15_nodds を順番に学習する。

特徴量列数:
  v11_nodds : 88列  = v3_base(60) + jockey(2) + waku/umaban(2) + ped_cats(3) + v6(16) + v7_extra(5)
  v12_nodds : 95列  = v11_nodds + v8_extra(7)
  v15_nodds : 102列 = v12_nodds + v9_extra(7)

サンプルウェイト: オッズなしのため均一（全て1.0）。
               → 純粋な1着確率最大化（的中率重視傾向）

サフィックス:
  _v11nodds_no26
  _v12nodds_no26
  _v15nodds_no26

実行: python scripts/train_nodds_no26.py
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
from src.PredictionModels.LightGBM.make_dataset_v9 import load_dataset_v9, index_v9

TRAIN_YEARS   = list(range(2020, 2026))
TARGET_VENUES = list(range(1, 11))
N_TRIALS      = 10

_DEVICE_KWARGS = {"device": LIGHTGBM_DEVICE} if USE_GPU_TRAINING else {"force_col_wise": True}

# ── オッズ不使用列リスト（重複排除）──
_EXCLUDE = {"race_id", "current_odds", "current_popularity"}
_seen = set()
NODDS_COLS = []
for _c in index_v9:
    if _c not in _EXCLUDE and _c not in _seen:
        NODDS_COLS.append(_c)
        _seen.add(_c)
# 重複排除後 100列

from src.PredictionModels.LightGBM.make_dataset_v7 import index_v7
from src.PredictionModels.LightGBM.make_dataset_v8 import index_v8

# v7_extra / v8_extra / v9_extra の末尾列でスライス境界を決定
_v7_last  = "horse_weight_abs_1"   # v7_extra の末尾
_v8_last  = "time_diff_trend_5"    # v8_extra の末尾
_v7_end   = NODDS_COLS.index(_v7_last) + 1   # v11_nodds 境界
_v8_end   = NODDS_COLS.index(_v8_last) + 1   # v12_nodds 境界

V11_NODDS_COLS = NODDS_COLS[:_v7_end]
V12_NODDS_COLS = NODDS_COLS[:_v8_end]
V15_NODDS_COLS = NODDS_COLS        # 100列

MODELS_TO_TRAIN = [
    ("_v11nodds_no26", V11_NODDS_COLS, "v11_nodds (88列)"),
    ("_v12nodds_no26", V12_NODDS_COLS, "v12_nodds (95列)"),
    ("_v15nodds_no26", V15_NODDS_COLS, "v15_nodds (102列)"),
]

print("モデル列数検証:")
for suffix, cols, label in MODELS_TO_TRAIN:
    print(f"  {label}: {len(cols)}列  最後の列={cols[-1]}")


def _to_binary_label(flag_df):
    return (flag_df["result_flag"] == 4).astype(int)


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


def _fit(params, X_tr, y_tr, X_va, y_va):
    model = lgb.LGBMClassifier(
        objective="binary", random_state=0, verbosity=-1,
        **_DEVICE_KWARGS, **params,
    )
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    return model


def _fit_final(params, X_tr, y_tr):
    model = lgb.LGBMClassifier(
        objective="binary", random_state=0, verbosity=-1,
        **_DEVICE_KWARGS, **params,
    )
    model.fit(X_tr, y_tr, callbacks=[lgb.log_evaluation(period=0)])
    return model


def _tune(X_tr, y_tr, X_va, y_va, n_trials=N_TRIALS):
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
        model = _fit(params, X_tr, y_tr, X_va, y_va)
        logloss = model.evals_result_["valid_0"]["binary_logloss"][model.best_iteration_ - 1]
        return -logloss

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=0)
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def _save_model(model, place_id, race_type, length, suffix):
    type_str = "turf" if race_type == "芝" else "dirt"
    model_dir = os.path.join(paths_v3.PREDICTION_MODEL_PATH, name_header.PLACE_LIST[place_id - 1])
    os.makedirs(model_dir, exist_ok=True)
    path = os.path.join(model_dir, f"{type_str}{length}_lambdarank_model{suffix}.txt")
    model.booster_.save_model(path)
    return path


def train_all_courses():
    for place_id in TARGET_VENUES:
        place_name = name_header.PLACE_LIST[place_id - 1]
        print(f"\n{'='*60}\n[学習] {place_name}\n{'='*60}")

        for race_type, length in name_header.COURSE_LISTS[place_id - 1]:
            print(f"\n  {race_type}{length}m ...")
            try:
                # v9データセット読み込み（全特徴量）
                race_data, race_flag = pd.DataFrame(), pd.DataFrame()
                for year in TRAIN_YEARS:
                    for band_length in _band_lengths(place_id, race_type, length):
                        df, flag = load_dataset_v9(place_id, year, race_type, band_length)
                        if not df.empty and not flag.empty:
                            race_data = pd.concat([race_data, df])
                            race_flag = pd.concat([race_flag, flag])

                if race_data.empty:
                    print("    データなし: スキップ"); continue

                # 重複列を排除（index_v9 に重複列が含まれるため）
                race_data = race_data.loc[:, ~race_data.columns.duplicated(keep="first")]
                num_cols = [c for c in race_data.columns if c != "race_id"]
                race_data[num_cols] = race_data[num_cols].fillna(-1)
                race_data = race_data.reset_index(drop=True)
                race_flag = race_flag.reset_index(drop=True)

                data_tr, data_va, flag_tr, flag_va = split_timeseries(race_data, race_flag)
                if data_va.empty:
                    print("    validationなし: スキップ"); continue

                y_tr = _to_binary_label(flag_tr)
                y_va = _to_binary_label(flag_va)
                pos_ratio = y_tr.mean()
                print(f"    train={len(data_tr)}行 / val={len(data_va)}行 / 1着率={pos_ratio:.3f}")

                # ── 3モデルを順番に学習 ──
                for suffix, feat_cols, label in MODELS_TO_TRAIN:
                    # データセット内にない列はスキップ（make_dataset_v9 の問題を回避）
                    valid_cols = [c for c in feat_cols if c in data_tr.columns]
                    if len(valid_cols) < len(feat_cols):
                        missing = set(feat_cols) - set(valid_cols)
                        print(f"      {label}: 列不足 {missing} → スキップ")
                        continue

                    model_path = os.path.join(
                        paths_v3.PREDICTION_MODEL_PATH,
                        name_header.PLACE_LIST[place_id - 1],
                        f"{'turf' if race_type == '芝' else 'dirt'}{length}_lambdarank_model{suffix}.txt"
                    )
                    if os.path.exists(model_path):
                        print(f"      {label}: 既存スキップ {model_path}")
                        continue

                    print(f"      {label} ({len(valid_cols)}列) チューニング中...")
                    X_tr = data_tr[valid_cols]
                    X_va = data_va[valid_cols]

                    best_params = _tune(X_tr, y_tr, X_va, y_va)
                    print(f"        params: {best_params}")

                    model = _fit_final(best_params, X_tr, y_tr)
                    path = _save_model(model, place_id, race_type, length, suffix)
                    print(f"        保存: {path}")

            except Exception:
                print(f"    ERROR: {race_type}{length}m")
                traceback.print_exc()


if __name__ == "__main__":
    import time
    t0 = time.time()
    print("=" * 60)
    print(f"オッズ不使用 3モデル学習  TRAIN_YEARS={TRAIN_YEARS[0]}〜{TRAIN_YEARS[-1]}")
    print(f"  v11_nodds: {len(V11_NODDS_COLS)}列  {V11_NODDS_COLS[-1]}")
    print(f"  v12_nodds: {len(V12_NODDS_COLS)}列  {V12_NODDS_COLS[-1]}")
    print(f"  v15_nodds: {len(V15_NODDS_COLS)}列  {V15_NODDS_COLS[-1]}")
    print("=" * 60)
    train_all_courses()
    elapsed = time.time() - t0
    print(f"\n完了: {elapsed/3600:.1f}時間")
