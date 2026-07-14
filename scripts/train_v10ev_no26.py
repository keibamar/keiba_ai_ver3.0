"""v10ev_no26 モデル学習スクリプト（期待値回帰・EV Regression）

目的関数を「期待値（EV）の直接回帰」に変更する。

  ターゲット: 勝ち馬 = current_odds、負け馬 = 0
              → モデルは P(win) × odds ≈ 期待回収額 を予測する
  損失関数:   regression（MSE）
              MAEは94%がゼロのスパースターゲットで中央値=0を予測してしまうためMSEを使用。
              オッズ上限50でアウトライアの影響を抑制する。
  サンプル重み: 不使用（ターゲット自体に価値情報が含まれるため）
  評価指標:   バリデーション期間の本命TOP1単勝シミュレーション回収率（ROI）
  TRAIN_YEARS: 2020〜2025（リークなし）
  特徴量:     v4データセット（66列、v7oddsと同一）
  サフィックス: _v10ev_no26

実行:
    python scripts/train_v10ev_no26.py
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
from src.config import paths as paths_v3
from src.PredictionModels.LightGBM.prediction import _band_lengths
from src.PredictionModels.LightGBM.make_dataset_v4 import load_dataset_v4

TRAIN_YEARS   = list(range(2020, 2026))
TARGET_VENUES = list(range(1, 11))
MODEL_SUFFIX  = "_v10ev_no26"
N_TRIALS      = 20
ODDS_CAP      = 50.0   # 外れ値対策のオッズ上限

_DEVICE_KWARGS = {"device": LIGHTGBM_DEVICE} if USE_GPU_TRAINING else {"force_col_wise": True}


def _make_ev_target(flag_df, X_df):
    """EV回帰ターゲット: 勝ち馬=current_odds（上限あり）、負け馬=0"""
    is_winner = (flag_df["result_flag"] == 4).values
    if "current_odds" in X_df.columns:
        odds = X_df["current_odds"].values.copy()
        odds = np.where((odds > 0) & np.isfinite(odds), odds, 1.1)
        odds = np.clip(odds, 1.0, ODDS_CAP)
    else:
        odds = np.full(len(flag_df), 1.1)
    return np.where(is_winner, odds, 0.0)


def make_roi_eval(race_ids_val):
    """バリデーション期間の単勝シミュレーション回収率を評価指標に使う。"""
    unique_ids = np.unique(race_ids_val)

    def feval(preds, dataset):
        total_pay = 0.0
        n_race = 0
        y_enc = dataset.get_label()
        for rid in unique_ids:
            mask = race_ids_val == rid
            pred_sub = preds[mask]
            y_sub = y_enc[mask]
            if len(pred_sub) == 0:
                continue
            top_idx = np.argmax(pred_sub)
            n_race += 1
            if y_sub[top_idx] > 0:
                total_pay += float(y_sub[top_idx]) * 100.0
        roi = total_pay / (n_race * 100.0) if n_race > 0 else 0.0
        return "roi", roi, True
    return feval


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


def _fit(params, ds_tr, ds_va, feval_fn, n_estimators):
    full_params = {
        "verbosity":     -1,
        "objective":     "regression",
        "num_iterations": n_estimators,
        **_DEVICE_KWARGS,
        **params,
    }
    model = lgb.train(
        full_params,
        ds_tr,
        feval=feval_fn,
        valid_sets=[ds_va],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    return model


def _tune(X_tr, y_tr, X_va, y_va, feval_fn, n_trials=20):
    min_cs_high = max(2, min(30, len(X_tr) // 10))

    def objective(trial):
        params = {
            "learning_rate":     trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
            "num_leaves":        trial.suggest_int("num_leaves", 16, 64),
            "max_depth":         trial.suggest_int("max_depth", 3, 10),
            "min_child_samples": trial.suggest_int("min_child_samples", 2, min_cs_high),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-3, 3.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-3, 3.0, log=True),
        }
        n_est = trial.suggest_int("n_estimators", 100, 500)
        try:
            ds_tr_t = lgb.Dataset(X_tr, label=y_tr)
            ds_va_t = lgb.Dataset(X_va, label=y_va, reference=ds_tr_t)
            model = _fit(params, ds_tr_t, ds_va_t, feval_fn, n_est)
            roi_val = model.best_score.get("valid_0", {}).get("roi", None)
            return roi_val if roi_val is not None else 0.0
        except Exception:
            return 0.0

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
    model.save_model(path)
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

                data_tr, data_va, flag_tr, flag_va = split_timeseries(race_data, race_flag)
                if data_va.empty:
                    print(f"    validationなし: スキップ"); continue

                rid_va = data_va["race_id"].values
                X_tr = data_tr.drop(columns=["race_id"])
                X_va = data_va.drop(columns=["race_id"])
                y_tr = _make_ev_target(flag_tr, X_tr)
                y_va = _make_ev_target(flag_va, X_va)

                n_win = (y_tr > 0).sum()
                mean_odds = y_tr[y_tr > 0].mean() if n_win > 0 else float("nan")
                print(f"    train={len(X_tr)}行 / val={len(X_va)}行 / 勝ち馬平均オッズ={mean_odds:.1f} (上限{ODDS_CAP})")

                feval_fn = make_roi_eval(rid_va)

                best_params = _tune(X_tr, y_tr, X_va, y_va, feval_fn, n_trials=N_TRIALS)
                print(f"    params: {best_params}")

                n_est = best_params.pop("n_estimators", 300)
                ds_tr = lgb.Dataset(X_tr, label=y_tr)
                ds_va = lgb.Dataset(X_va, label=y_va, reference=ds_tr)
                model = _fit(best_params, ds_tr, ds_va, feval_fn, n_est)
                path  = _save_model(model, place_id, race_type, length)
                print(f"    保存: {path}")

            except Exception:
                print(f"    ERROR: {race_type}{length}m")
                traceback.print_exc()


if __name__ == "__main__":
    print("=" * 55)
    print(f"v10ev_no26  学習年: {TRAIN_YEARS[0]}〜{TRAIN_YEARS[-1]}")
    print("目的関数: EV回帰（ターゲット=odds×win, 損失=MSE）")
    print(f"オッズ上限: {ODDS_CAP}")
    print("=" * 55)
    train_all_courses()
    print("\n" + "=" * 55 + "\n完了\n" + "=" * 55)
