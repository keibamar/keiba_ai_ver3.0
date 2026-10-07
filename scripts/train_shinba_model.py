"""新馬戦専用モデル学習スクリプト

的中率重視 (hit): LambdaRank ndcg@3
回収率重視 (val): カスタム損失 profit_objective（単勝オッズ最大化）

出力:
  data/prediction/models/shinba/turf_shinba_hit.txt
  data/prediction/models/shinba/dirt_shinba_hit.txt
  data/prediction/models/shinba/turf_shinba_val.txt
  data/prediction/models/shinba/dirt_shinba_val.txt
  data/prediction/models/shinba/stats_tables/

実行:
    python scripts/train_shinba_model.py
"""

import os
import sys
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

from src.config import paths
from src.config.constants import USE_GPU_TRAINING, LIGHTGBM_DEVICE
from src.PredictionModels.LightGBM.make_dataset_shinba import (
    SHINBA_FEATURES,
    SHINBA_ALL_FEATURES,
    build_stats_tables,
    make_training_dataset,
)

N_TRIALS  = 30
MODEL_DIR = os.path.join(paths.PREDICTION_MODEL_PATH, "shinba")
STATS_DIR = os.path.join(MODEL_DIR, "stats_tables")
CACHE_DIR = os.path.join(MODEL_DIR, "cache")

_DEVICE_KWARGS = {"device": LIGHTGBM_DEVICE} if USE_GPU_TRAINING else {"force_col_wise": True}

ODDS_CAP = 30.0


# ---------- カスタム損失: 回収率重視 ----------

def profit_objective(y_pred, dataset):
    """期待回収率を最大化するカスタム損失。label = tan_odds(1着) or 0"""
    y_enc = dataset.get_label()
    p = 1.0 / (1.0 + np.exp(-y_pred))
    is_winner = y_enc > 0
    w = np.clip(y_enc - 1.0, 0.5, ODDS_CAP - 1.0)
    grad = np.where(is_winner, -w * (1.0 - p), p)
    hess = np.where(is_winner, w * p * (1.0 - p), p * (1.0 - p))
    hess = np.clip(hess, 1e-6, None)
    return grad, hess


# ---------- 時系列 Train/Val 分割 ----------

def split_timeseries(X, labels, groups, tan_odds=None, train_ratio=0.8):
    cum = np.cumsum(groups)
    total = cum[-1]
    split_row = int(total * train_ratio)
    split_race = np.searchsorted(cum, split_row, side="left")
    split_row = int(cum[split_race - 1]) if split_race > 0 else 0

    X_tr, X_va = X.iloc[:split_row], X.iloc[split_row:]
    y_tr, y_va = labels[:split_row], labels[split_row:]
    g_tr = groups[:split_race]
    g_va = groups[split_race:]

    if tan_odds is not None:
        to_tr = tan_odds[:split_row]
        to_va = tan_odds[split_row:]
        return X_tr, X_va, y_tr, y_va, g_tr, g_va, to_tr, to_va
    return X_tr, X_va, y_tr, y_va, g_tr, g_va


# ---------- HIT モデル（LambdaRank） ----------

def _fit_hit(params, ds_tr, ds_va, n_estimators):
    full_params = {
        "objective": "lambdarank",
        "label_gain": [0, 1, 2, 3],
        "eval_at": [1, 3],
        "metric": "ndcg",
        "verbosity": -1,
        "num_iterations": n_estimators,
        **_DEVICE_KWARGS,
        **params,
    }
    return lgb.train(
        full_params, ds_tr, valid_sets=[ds_va],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )


def _tune_hit(X_tr, y_tr, g_tr, X_va, y_va, g_va, n_trials):
    min_cs = max(2, min(20, len(X_tr) // 50))

    def objective(trial):
        params = {
            "learning_rate":     trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
            "num_leaves":        trial.suggest_int("num_leaves", 16, 64),
            "max_depth":         trial.suggest_int("max_depth", 3, 10),
            "min_child_samples": trial.suggest_int("min_child_samples", 2, min_cs),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-3, 3.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-3, 3.0, log=True),
            "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.6, 1.0),
        }
        n_est = trial.suggest_int("n_estimators", 100, 600)
        try:
            ds_tr_t = lgb.Dataset(X_tr.values, label=y_tr, group=g_tr, feature_name=list(X_tr.columns))
            ds_va_t = lgb.Dataset(X_va.values, label=y_va, group=g_va, reference=ds_tr_t)
            model = _fit_hit(params, ds_tr_t, ds_va_t, n_est)
            score = model.best_score.get("valid_0", {}).get("ndcg@3", None)
            return score if score is not None else 0.0
        except Exception:
            return 0.0

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


# ---------- VAL モデル（profit objective） ----------

def _fit_val(params, ds_tr, ds_va, n_estimators):
    full_params = {
        "objective": profit_objective,
        "metric": "binary_logloss",
        "verbosity": -1,
        "num_iterations": n_estimators,
        **_DEVICE_KWARGS,
        **params,
    }
    return lgb.train(
        full_params, ds_tr, valid_sets=[ds_va],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )


def _tune_val(X_tr, to_tr, g_tr, X_va, to_va, g_va, n_trials):
    # val の binary label (1着=1, 他=0) を metric 評価に使う
    y_tr_bin = (to_tr > 0).astype(float)
    y_va_bin = (to_va > 0).astype(float)
    min_cs = max(2, min(20, len(X_tr) // 50))

    def objective(trial):
        params = {
            "learning_rate":     trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
            "num_leaves":        trial.suggest_int("num_leaves", 16, 64),
            "max_depth":         trial.suggest_int("max_depth", 3, 10),
            "min_child_samples": trial.suggest_int("min_child_samples", 2, min_cs),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-3, 3.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-3, 3.0, log=True),
            "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.6, 1.0),
        }
        n_est = trial.suggest_int("n_estimators", 100, 600)
        try:
            # 学習 label は tan_odds（profit_objective 用）
            ds_tr_t = lgb.Dataset(X_tr.values, label=to_tr, group=g_tr, feature_name=list(X_tr.columns))
            # val metric は binary_logloss 用に binary label を使用
            ds_va_t = lgb.Dataset(X_va.values, label=y_va_bin, group=g_va, reference=ds_tr_t)
            model = _fit_val(params, ds_tr_t, ds_va_t, n_est)
            score = model.best_score.get("valid_0", {}).get("binary_logloss", None)
            return -score if score is not None else 0.0  # binary_logloss は minimize
        except Exception:
            return 0.0

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=43))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


# ---------- race_type フィルタリング ----------

def _filter_by_race_type(feat_df, labels, groups, tan_odds, race_type_code):
    mask = feat_df["race_type_code"] == race_type_code
    X_rt   = feat_df[mask].copy().reset_index(drop=True)
    y_rt   = labels[mask.values]
    to_rt  = tan_odds[mask.values]

    cum_orig = np.cumsum(groups)
    row_idx  = np.where(mask.values)[0]
    rt_groups = []
    for start, end in zip([0] + list(cum_orig[:-1]), cum_orig):
        race_rows = [i for i in row_idx if start <= i < end]
        if race_rows:
            rt_groups.append(len(race_rows))

    return X_rt, y_rt, to_rt, rt_groups


# ---------- メイン ----------

def train():
    print("=" * 60)
    print("新馬戦専用モデル学習 (hit + val)")
    print(f"特徴量: {len(SHINBA_FEATURES)}列 (V5)")
    print("=" * 60)

    os.makedirs(MODEL_DIR,  exist_ok=True)
    os.makedirs(STATS_DIR,  exist_ok=True)
    os.makedirs(CACHE_DIR,  exist_ok=True)

    _cache_feat   = os.path.join(CACHE_DIR, "feat.pkl")
    _cache_groups = os.path.join(CACHE_DIR, "groups.npy")
    _cache_labels = os.path.join(CACHE_DIR, "labels.npy")
    _cache_odds   = os.path.join(CACHE_DIR, "tan_odds.npy")

    all_cached = all(os.path.isfile(p) for p in [_cache_feat, _cache_groups, _cache_labels, _cache_odds])

    if all_cached:
        print("\n1. キャッシュからデータセットを読み込み中...")
        feat_df  = pd.read_pickle(_cache_feat)
        groups   = list(np.load(_cache_groups))
        labels   = np.load(_cache_labels)
        tan_odds = np.load(_cache_odds)
        print(f"  → {len(feat_df)}行 / {len(groups)}レース")
    else:
        print("\n1. 学習データセット生成中...")
        feat_df, groups, labels, tan_odds = make_training_dataset()
        groups = list(groups)
        feat_df.to_pickle(_cache_feat)
        np.save(_cache_groups, np.array(groups))
        np.save(_cache_labels, labels)
        np.save(_cache_odds,   tan_odds)
        print(f"  キャッシュ保存: {CACHE_DIR}")

    feat_df = feat_df[SHINBA_FEATURES].fillna(-1.0)
    labels  = labels.astype(float)

    for race_type_code, race_type_name, hit_name, val_name in [
        (0.0, "芝",     "turf_shinba_hit.txt", "turf_shinba_val.txt"),
        (1.0, "ダート", "dirt_shinba_hit.txt", "dirt_shinba_val.txt"),
    ]:
        print(f"\n{'='*55}")
        print(f"[{race_type_name}] 学習開始")

        X_rt, y_rt, to_rt, rt_groups = _filter_by_race_type(feat_df, labels, groups, tan_odds, race_type_code)
        n_races_rt = len(rt_groups)
        print(f"  {len(X_rt)}行 / {n_races_rt}レース")
        if len(X_rt) < 100 or n_races_rt < 20:
            print("  データ不足: スキップ")
            continue

        split = split_timeseries(X_rt, y_rt, rt_groups, to_rt)
        X_tr, X_va, y_tr, y_va, g_tr, g_va, to_tr, to_va = split
        if len(X_va) == 0:
            print("  validationなし: スキップ")
            continue
        print(f"  train={len(X_tr)}行({len(g_tr)}R) / val={len(X_va)}行({len(g_va)}R)")

        # --- HIT モデル ---
        print(f"\n  [hit] Optuna チューニング ({N_TRIALS} trials)...")
        best_hit = _tune_hit(X_tr, y_tr, g_tr, X_va, y_va, g_va, N_TRIALS)
        print(f"  [hit] best_params: {best_hit}")
        n_est = best_hit.pop("n_estimators", 300)
        ds_tr_hit = lgb.Dataset(X_tr.values, label=y_tr, group=g_tr, feature_name=list(X_tr.columns))
        ds_va_hit = lgb.Dataset(X_va.values, label=y_va, group=g_va, reference=ds_tr_hit)
        model_hit = _fit_hit(best_hit, ds_tr_hit, ds_va_hit, n_est)
        hit_path = os.path.join(MODEL_DIR, hit_name)
        model_hit.save_model(hit_path)
        ndcg = model_hit.best_score.get("valid_0", {}).get("ndcg@3", "N/A")
        print(f"  [hit] 保存: {hit_path}  (val ndcg@3={ndcg:.4f})" if isinstance(ndcg, float) else f"  [hit] 保存: {hit_path}")

        # --- VAL モデル ---
        print(f"\n  [val] Optuna チューニング ({N_TRIALS} trials)...")
        best_val = _tune_val(X_tr, to_tr, g_tr, X_va, to_va, g_va, N_TRIALS)
        print(f"  [val] best_params: {best_val}")
        n_est_v = best_val.pop("n_estimators", 300)
        y_va_bin = (to_va > 0).astype(float)
        ds_tr_val = lgb.Dataset(X_tr.values, label=to_tr, group=g_tr, feature_name=list(X_tr.columns))
        ds_va_val = lgb.Dataset(X_va.values, label=y_va_bin, group=g_va, reference=ds_tr_val)
        model_val = _fit_val(best_val, ds_tr_val, ds_va_val, n_est_v)
        val_path = os.path.join(MODEL_DIR, val_name)
        model_val.save_model(val_path)
        print(f"  [val] 保存: {val_path}")

    print("\n2. 統計テーブル生成・保存中...")
    tables = build_stats_tables()
    for name, df in tables.items():
        p = os.path.join(STATS_DIR, f"{name}.csv")
        df.to_csv(p, index=False)
        print(f"  保存: {p}  ({len(df)}件)")

    print("\n" + "=" * 60)
    print("完了")
    print("=" * 60)


if __name__ == "__main__":
    train()
