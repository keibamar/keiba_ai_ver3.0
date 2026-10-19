"""v19 + 調教特徴量 全コース学習スクリプト

train_v19_no26.py と同じ構造だが TRAINING_FEATURES (11列) を加えて学習。
horse_id は race_results CSVの (race_id, 馬番) から取得。

保存先:
  data/prediction/models/{place}/{type}{length}_lambdarank_model_v19tr_no26.txt

実行:
    python scripts/train_v19_training_no26.py
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
from src.PredictionModels.LightGBM.make_dataset_shinba_training import (
    TRAINING_FEATURES,
    _get_course_stats,
    _zscore,
    _course_cat,
    _grade_code,
    _leg_code,
    _parse_times,
)

TRAIN_YEARS   = list(range(2020, 2026))
TARGET_VENUES = list(range(1, 11))
MODEL_SUFFIX  = "_v19tr_no26"
N_TRIALS      = 15
RACE_RESULTS_BASE = r"C:\keiba_ai\keiba_ai_ver2.0\data\RaceResults"

_DEVICE_KWARGS = {"device": LIGHTGBM_DEVICE} if USE_GPU_TRAINING else {"force_col_wise": True}


# ======================================================================
# horse_id ルックアップ (起動時に1回だけ構築)
# ======================================================================

def build_horse_id_lookup(years):
    print("  horse_id ルックアップ構築中...")
    lookup = {}
    for place_dir in sorted(os.listdir(RACE_RESULTS_BASE)):
        for year in years:
            path = os.path.join(RACE_RESULTS_BASE, place_dir, f"{year}_race_results.csv")
            if not os.path.isfile(path):
                continue
            try:
                df = pd.read_csv(path, index_col=0, dtype=str, low_memory=False)
                uma_col = "umaban" if "umaban" in df.columns else "馬番"
                if uma_col not in df.columns or "horse_id" not in df.columns:
                    continue
                for rid, row in df.iterrows():
                    try:
                        uma = int(float(row[uma_col]))
                        hid = str(row["horse_id"]).strip()
                        if hid and hid != "nan":
                            lookup[(str(rid), uma)] = hid
                    except Exception:
                        pass
            except Exception:
                pass
    print(f"  ルックアップ完成: {len(lookup):,} エントリ")
    return lookup


# ======================================================================
# 調教特徴量バッチキャッシュ
# ======================================================================

_TRAINING_CACHE: dict = {}


def _load_sessions(horse_id: str) -> pd.DataFrame:
    if horse_id in _TRAINING_CACHE:
        return _TRAINING_CACHE[horse_id]
    path = os.path.join(paths_v3.TRAINING_DATA_PATH, f"{horse_id}.csv")
    if not os.path.isfile(path) or os.path.getsize(path) == 0:
        _TRAINING_CACHE[horse_id] = pd.DataFrame()
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, encoding="utf-8-sig")
        df["race_id"] = df["race_id"].astype(str).str.strip()
        df["training_date"] = pd.to_datetime(df["training_date"], errors="coerce")
        _TRAINING_CACHE[horse_id] = df
        return df
    except Exception:
        _TRAINING_CACHE[horse_id] = pd.DataFrame()
        return pd.DataFrame()


def _training_feats(horse_id: str, race_id: str) -> dict:
    default = {c: np.nan for c in TRAINING_FEATURES}
    default["train_has_data"] = 0.0
    df = _load_sessions(horse_id)
    if df.empty:
        return default
    sess = df[df["race_id"] == str(race_id)].dropna(subset=["training_date"])
    if sess.empty:
        return default
    sess = sess.sort_values("training_date")
    last = sess.iloc[-1]
    course = str(last.get("course", "")).strip()
    times = _parse_times(last.get("times_list", ""))
    ll  = float(times[-1]) if len(times) >= 1 else np.nan
    l4  = float(times[-2]) if len(times) >= 2 else np.nan
    st  = _get_course_stats().get(course, {})
    return {
        "train_has_data":       1.0,
        "train_last_lap":       ll,
        "train_last_400m":      l4,
        "train_n_poles":        float(len(times)),
        "train_grade_code":     _grade_code(last.get("grade", "")),
        "train_leg_code":       _leg_code(last.get("leg_type", "")),
        "train_sessions_count": float(len(sess)),
        "train_best_grade":     max(
            (_grade_code(g) for g in sess["grade"].fillna("").astype(str)),
            default=0.0,
        ),
        "train_last_lap_z":     _zscore(ll, st.get("ll_mean"), st.get("ll_std")),
        "train_last_400m_z":    _zscore(l4, st.get("l4_mean"), st.get("l4_std")),
        "train_course_cat":     _course_cat(course),
    }


def add_training_cols(df: pd.DataFrame, lookup: dict) -> pd.DataFrame:
    """race_id + umaban を使って調教特徴量列を付加する。"""
    records = []
    for _, row in df.iterrows():
        rid = str(int(row["race_id"]))
        uma = int(row["umaban"])
        hid = lookup.get((rid, uma))
        if hid:
            records.append(_training_feats(hid, rid))
        else:
            d = {c: np.nan for c in TRAINING_FEATURES}
            d["train_has_data"] = 0.0
            records.append(d)
    feat_df = pd.DataFrame(records, index=df.index)
    return pd.concat([df, feat_df], axis=1)


# ======================================================================
# 学習ユーティリティ (v19 と同一)
# ======================================================================

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
        eval_set=[(X_va, y_va)], eval_sample_weight=[w_va],
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
    model.fit(X_tr, y_tr, sample_weight=w_tr,
              callbacks=[lgb.log_evaluation(period=0)])
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
        m = _fit(params, X_tr, y_tr, w_tr, X_va, y_va, w_va)
        ll = m.evals_result_["valid_0"]["binary_logloss"][m.best_iteration_ - 1]
        return -ll

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=0)
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def _save_model(model, place_id, race_type, length):
    type_str  = "turf" if race_type == "芝" else "dirt"
    model_dir = os.path.join(paths_v3.PREDICTION_MODEL_PATH,
                             name_header.PLACE_LIST[place_id - 1])
    os.makedirs(model_dir, exist_ok=True)
    path = os.path.join(model_dir,
                        f"{type_str}{length}_lambdarank_model{MODEL_SUFFIX}.txt")
    model.booster_.save_model(path)
    return path


# ======================================================================
# メイン
# ======================================================================

def train_all_courses(lookup):
    total_courses = done = skipped = 0
    for place_id in TARGET_VENUES:
        total_courses += len(name_header.COURSE_LISTS[place_id - 1])

    for place_id in TARGET_VENUES:
        place_name = name_header.PLACE_LIST[place_id - 1]
        print(f"\n{'='*60}\n[{place_name}]\n{'='*60}")

        for race_type, length in name_header.COURSE_LISTS[place_id - 1]:
            print(f"\n  {race_type}{length}m  ({done+skipped+1}/{total_courses})")
            try:
                race_data = pd.DataFrame()
                race_flag = pd.DataFrame()
                for year in TRAIN_YEARS:
                    for band_length in _band_lengths(place_id, race_type, length):
                        df, flag = load_dataset_v10(
                            place_id, year, race_type, band_length
                        )
                        if not df.empty and not flag.empty:
                            race_data = pd.concat([race_data, df])
                            race_flag = pd.concat([race_flag, flag])

                if race_data.empty:
                    print("    データなし: スキップ")
                    skipped += 1
                    continue

                race_data = race_data.reset_index(drop=True)
                race_flag = race_flag.reset_index(drop=True)

                # 調教特徴量を付加
                if "umaban" in race_data.columns:
                    race_data = add_training_cols(race_data, lookup)
                    has_data_pct = race_data["train_has_data"].mean() * 100
                    print(f"    調教データあり: {has_data_pct:.0f}%")

                # fillna (race_id を除く)
                num_cols = [c for c in race_data.columns if c != "race_id"]
                race_data[num_cols] = race_data[num_cols].fillna(-1)

                data_tr, data_va, flag_tr, flag_va = split_timeseries(
                    race_data, race_flag
                )
                if data_va.empty:
                    print("    validationなし: スキップ")
                    skipped += 1
                    continue

                X_tr = data_tr.drop(columns=["race_id"])
                X_va = data_va.drop(columns=["race_id"])
                y_tr = _to_binary_label(flag_tr)
                y_va = _to_binary_label(flag_va)
                w_tr = _make_sample_weight(y_tr.values, X_tr)
                w_va = _make_sample_weight(y_va.values, X_va)

                pos_ratio = y_tr.mean()
                print(f"    train={len(X_tr)}行 / val={len(X_va)}行 / "
                      f"1着率={pos_ratio:.3f} / 特徴量={X_tr.shape[1]}列")

                best_params = _tune(X_tr, y_tr, w_tr, X_va, y_va, w_va)
                model = _fit_final(best_params, X_tr, y_tr, w_tr)
                path  = _save_model(model, place_id, race_type, length)
                print(f"    保存: {os.path.basename(path)}")
                done += 1

            except Exception:
                print(f"    ERROR: {race_type}{length}m")
                traceback.print_exc()
                skipped += 1

    return done, skipped


if __name__ == "__main__":
    print("=" * 60)
    print(f"v19 + 調教特徴量  全コース学習  (suffix={MODEL_SUFFIX})")
    print(f"学習年: {TRAIN_YEARS[0]}〜{TRAIN_YEARS[-1]}  N_TRIALS={N_TRIALS}")
    print("=" * 60)

    lookup = build_horse_id_lookup(TRAIN_YEARS)
    done, skipped = train_all_courses(lookup)

    print("\n" + "=" * 60)
    print(f"完了: {done}コース学習  {skipped}コーススキップ")
    print("=" * 60)
