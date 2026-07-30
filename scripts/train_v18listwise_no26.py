"""v18listwise_no26 モデル学習スクリプト（Listwise Softmax ROI最大化 × v9特徴量）

レース単位（Listwise）で回収率を直接最大化するカスタム目的関数を実装。
個別馬の binary ではなく「このレースで最高スコアの馬に賭けたときの期待回収率」を
Softmax で微分可能な形に定式化して学習する。

  各レースで:
    q_i = softmax(score_i for i in race)
    期待ROI = q_winner × (odds_winner + 1) - 1
    Loss = 1 - q_winner × (odds_winner + 1)

  勾配 wrt score_i:
    i == k (winner): grad = -w_k × q_k × (1 - q_k)    [w_k = odds + 1]
    i != k:          grad = +w_k × q_k × q_i

  ヘッセ: |grad| + ε（符号の安定な近似）

  評価: バリデーション期間の本命TOP1単勝シミュレーション回収率（ROI）
  データセット: v9（105列）
  TRAIN_YEARS: 2020〜2025
  サフィックス: _v18listwise_no26

実行: python scripts/train_v18listwise_no26.py
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
from src.PredictionModels.LightGBM.make_dataset_v9 import load_dataset_v9

TRAIN_YEARS   = list(range(2020, 2026))
TARGET_VENUES = list(range(1, 11))
MODEL_SUFFIX  = "_v18listwise_no26"
N_TRIALS      = 20
ODDS_CAP      = 50.0

_DEVICE_KWARGS = {"device": LIGHTGBM_DEVICE} if USE_GPU_TRAINING else {"force_col_wise": True}


# ---------- カスタム目的関数 ----------

def make_listwise_roi_objective(race_ids_arr, winner_odds_arr, odds_cap=ODDS_CAP):
    """Listwise softmax ROI最大化。

    race_ids_arr:     shape (N,) 各馬のレースID
    winner_odds_arr:  shape (N,) 勝ち馬=current_odds, 非勝ち馬=0
    """
    unique_races, inverse_idx = np.unique(race_ids_arr, return_inverse=True)
    wo = np.where(winner_odds_arr > 0,
                  np.clip(winner_odds_arr, 1.0, odds_cap),
                  0.0)

    def objective(y_pred, dataset):
        grad = np.zeros(len(y_pred), dtype=np.float64)
        hess = np.full(len(y_pred), 1e-3, dtype=np.float64)

        for i in range(len(unique_races)):
            mask = inverse_idx == i
            if mask.sum() == 0:
                continue
            s  = y_pred[mask]
            wo_r = wo[mask]

            win_pos = np.where(wo_r > 0)[0]
            if len(win_pos) == 0:
                continue
            win_idx = win_pos[0]
            w_k = wo_r[win_idx] + 1.0  # effective weight = odds + 1

            # Numerically stable softmax
            exp_s = np.exp(np.clip(s - s.max(), -50.0, 50.0))
            q = exp_s / exp_s.sum()
            q_k = q[win_idx]

            # Gradient of Loss = 1 - q_k * w_k
            # dL/ds_i for i==k:   -w_k * q_k * (1 - q_k)
            # dL/ds_i for i!=k:   +w_k * q_k * q_i
            g = (w_k * q_k) * q.copy()
            g[win_idx] -= w_k * q_k

            h = np.clip(np.abs(g), 1e-4, 10.0)

            grad[mask] = g
            hess[mask] = h

        return grad, hess

    return objective


def make_roi_eval(race_ids_val, y_enc_val):
    """バリデーション期間の単勝シミュレーション回収率。"""
    unique_ids = np.unique(race_ids_val)

    def feval(preds, dataset):
        total_pay = 0.0
        n_race = 0
        for rid in unique_ids:
            mask = race_ids_val == rid
            if mask.sum() == 0:
                continue
            top_idx = np.argmax(preds[mask])
            n_race += 1
            if y_enc_val[mask][top_idx] > 0:
                total_pay += float(y_enc_val[mask][top_idx]) * 100.0
        roi = total_pay / (n_race * 100.0) if n_race > 0 else 0.0
        return "roi", roi, True
    return feval


# ---------- ユーティリティ ----------

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


def _encode_label(flag_df, X_df):
    """勝ち馬=current_odds, 非勝ち馬=0"""
    is_winner = (flag_df["result_flag"] == 4).values
    if "current_odds" in X_df.columns:
        odds = X_df["current_odds"].values.copy()
        odds = np.where((odds > 0) & np.isfinite(odds), odds, 1.1)
    else:
        odds = np.full(len(flag_df), 1.1)
    return np.where(is_winner, odds, 0.0)


def _fit(params, ds_tr, ds_va, feval_fn, n_estimators, listwise_obj):
    full_params = {
        "verbosity": -1,
        "objective": listwise_obj,
        "num_iterations": n_estimators,
        **_DEVICE_KWARGS,
        **params,
    }
    return lgb.train(
        full_params,
        ds_tr,
        feval=feval_fn,
        valid_sets=[ds_va],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )


def _tune(X_tr, y_tr_enc, rid_tr, X_va, y_va_enc, rid_va, n_trials):
    min_cs_high = max(2, min(30, len(X_tr) // 10))
    feval_fn = make_roi_eval(rid_va, y_va_enc)

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
            listwise_obj = make_listwise_roi_objective(rid_tr, y_tr_enc)
            ds_tr_t = lgb.Dataset(X_tr, label=y_tr_enc)
            ds_va_t = lgb.Dataset(X_va, label=y_va_enc, reference=ds_tr_t)
            model = _fit(params, ds_tr_t, ds_va_t, feval_fn, n_est, listwise_obj)
            roi = model.best_score.get("valid_0", {}).get("roi", None)
            return roi if roi is not None else 0.0
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


# ---------- メイン学習ループ ----------

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
                        df, flag = load_dataset_v9(place_id, year, race_type, band_length)
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

                rid_tr   = data_tr["race_id"].values
                rid_va   = data_va["race_id"].values
                X_tr = data_tr.drop(columns=["race_id"])
                X_va = data_va.drop(columns=["race_id"])
                y_tr_enc = _encode_label(flag_tr, X_tr)
                y_va_enc = _encode_label(flag_va, X_va)

                n_races_tr = len(np.unique(rid_tr))
                print(f"    train={len(X_tr)}行({n_races_tr}R) / val={len(X_va)}行")

                best_params = _tune(
                    X_tr.values, y_tr_enc, rid_tr,
                    X_va.values, y_va_enc, rid_va,
                    N_TRIALS,
                )
                print(f"    params: {best_params}")

                n_est = best_params.pop("n_estimators", 300)
                listwise_obj = make_listwise_roi_objective(rid_tr, y_tr_enc)
                ds_tr = lgb.Dataset(X_tr.values, label=y_tr_enc, feature_name=list(X_tr.columns))
                ds_va = lgb.Dataset(X_va.values, label=y_va_enc, reference=ds_tr)
                feval_fn = make_roi_eval(rid_va, y_va_enc)
                model = _fit(best_params, ds_tr, ds_va, feval_fn, n_est, listwise_obj)
                path = _save_model(model, place_id, race_type, length)
                print(f"    保存: {path}")

            except Exception:
                print(f"    ERROR: {race_type}{length}m")
                traceback.print_exc()


if __name__ == "__main__":
    print("=" * 55)
    print(f"v18listwise_no26  学習年: {TRAIN_YEARS[0]}〜{TRAIN_YEARS[-1]}")
    print("データセット: v9（105列）")
    print("目的関数: Listwise Softmax ROI最大化")
    print("=" * 55)
    train_all_courses()
    print("\n" + "=" * 55 + "\n完了\n" + "=" * 55)
