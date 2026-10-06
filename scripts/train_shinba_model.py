"""新馬戦専用 LambdaRank モデル学習スクリプト

特徴量: 19列（父統計 × 騎手 × 調教師 × レース属性 × オッズ）
目的関数: LambdaRank (ndcg @1,3)
モデル: 芝・ダート 各1モデル（全場共通、place_id を特徴量に含む）
出力: data/prediction/models/shinba/turf_shinba_model.txt
         data/prediction/models/shinba/dirt_shinba_model.txt
      + data/prediction/models/shinba/stats_tables/  （予測時参照テーブル）

実行:
    python scripts/train_shinba_model.py
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

from src.config import paths
from src.config.constants import USE_GPU_TRAINING, LIGHTGBM_DEVICE
from src.PredictionModels.LightGBM.make_dataset_shinba import (
    SHINBA_FEATURES,
    build_stats_tables,
    make_training_dataset,
)

N_TRIALS  = 30
MODEL_DIR = os.path.join(paths.PREDICTION_MODEL_PATH, "shinba")
STATS_DIR = os.path.join(MODEL_DIR, "stats_tables")
CACHE_DIR = os.path.join(MODEL_DIR, "cache")

_DEVICE_KWARGS = {"device": LIGHTGBM_DEVICE} if USE_GPU_TRAINING else {"force_col_wise": True}


# ---------- 時系列 Train/Val 分割 ----------

def split_timeseries(X, labels, groups, train_ratio=0.8):
    """
    全データを時系列順に train/val 分割する。
    groups はレースごとの頭数リストなので末尾境界でレースが切れないよう調整。
    """
    cum = np.cumsum(groups)
    total = cum[-1]
    split_row = int(total * train_ratio)
    # split_row 以下に収まる最後のレース末尾インデックスを探す
    split_race = np.searchsorted(cum, split_row, side="left")
    split_row = int(cum[split_race - 1]) if split_race > 0 else 0

    X_tr, X_va = X.iloc[:split_row], X.iloc[split_row:]
    y_tr, y_va = labels[:split_row], labels[split_row:]
    g_tr = groups[:split_race]
    g_va = groups[split_race:]
    return X_tr, X_va, y_tr, y_va, g_tr, g_va


# ---------- モデル学習 ----------

def _fit(params, ds_tr, ds_va, n_estimators):
    # lambdarank は eval_at で NDCG を built-in で計算する。
    # custom feval との併用は early_stopping の best_score 管理と競合するため使わない。
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
        full_params,
        ds_tr,
        valid_sets=[ds_va],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )


def _tune(X_tr, y_tr, g_tr, X_va, y_va, g_va, n_trials):
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
            model = _fit(params, ds_tr_t, ds_va_t, n_est)
            score = model.best_score.get("valid_0", {}).get("ndcg@3", None)
            return score if score is not None else 0.0
        except Exception:
            return 0.0

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler(seed=42)
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


# ---------- メイン ----------

def train():
    print("=" * 60)
    print("新馬戦専用モデル学習")
    print(f"特徴量: {len(SHINBA_FEATURES)}列")
    print("目的関数: LambdaRank (ndcg@3)")
    print("=" * 60)

    os.makedirs(MODEL_DIR,  exist_ok=True)
    os.makedirs(STATS_DIR,  exist_ok=True)
    os.makedirs(CACHE_DIR,  exist_ok=True)

    _cache_feat   = os.path.join(CACHE_DIR, "feat.pkl")
    _cache_groups = os.path.join(CACHE_DIR, "groups.npy")
    _cache_labels = os.path.join(CACHE_DIR, "labels.npy")

    if os.path.isfile(_cache_feat) and os.path.isfile(_cache_groups) and os.path.isfile(_cache_labels):
        print("\n1. キャッシュからデータセットを読み込み中...")
        feat_df = pd.read_pickle(_cache_feat)
        groups  = list(np.load(_cache_groups))
        labels  = np.load(_cache_labels)
        print(f"  → {len(feat_df)}行 / {len(groups)}レース")
    else:
        print("\n1. 学習データセット生成中...")
        feat_df, groups, labels = make_training_dataset()
        groups = list(groups)
        feat_df.to_pickle(_cache_feat)
        np.save(_cache_groups, np.array(groups))
        np.save(_cache_labels, labels)
        print(f"  キャッシュ保存: {CACHE_DIR}")

    feat_df = feat_df.fillna(-1.0)
    labels  = labels.astype(float)

    for race_type_code, race_type_name, model_name in [
        (0.0, "芝",     "turf_shinba_model.txt"),
        (1.0, "ダート", "dirt_shinba_model.txt"),
    ]:
        print(f"\n{'='*55}")
        print(f"[{race_type_name}] 学習開始")

        # race_type でフィルタ
        mask = feat_df["race_type_code"] == race_type_code
        X_rt = feat_df[mask].copy().reset_index(drop=True)
        y_rt = labels[mask.values]

        # groups も race_type で絞り込み（馬インデックスから再計算）
        cum_orig = np.cumsum(groups)
        row_idx  = np.where(mask.values)[0]
        rt_groups = []
        for start, end in zip([0] + list(cum_orig[:-1]), cum_orig):
            race_rows = [i for i in row_idx if start <= i < end]
            if race_rows:
                rt_groups.append(len(race_rows))

        n_races_rt = len(rt_groups)
        print(f"  {len(X_rt)}行 / {n_races_rt}レース")
        if len(X_rt) < 100 or n_races_rt < 20:
            print("  データ不足: スキップ")
            continue

        X_tr, X_va, y_tr, y_va, g_tr, g_va = split_timeseries(X_rt, y_rt, rt_groups)
        if len(X_va) == 0 or len(g_va) == 0:
            print("  validationなし: スキップ")
            continue

        print(f"  train={len(X_tr)}行({len(g_tr)}R) / val={len(X_va)}行({len(g_va)}R)")

        print(f"  Optuna チューニング ({N_TRIALS} trials)...")
        best_params = _tune(X_tr, y_tr, g_tr, X_va, y_va, g_va, N_TRIALS)
        print(f"  best_params: {best_params}")

        n_est = best_params.pop("n_estimators", 300)
        ds_tr = lgb.Dataset(X_tr.values, label=y_tr, group=g_tr, feature_name=list(X_tr.columns))
        ds_va = lgb.Dataset(X_va.values, label=y_va, group=g_va, reference=ds_tr)
        model = _fit(best_params, ds_tr, ds_va, n_est)

        model_path = os.path.join(MODEL_DIR, model_name)
        model.save_model(model_path)
        ndcg = model.best_score.get("valid_0", {}).get("ndcg@3", "N/A")
        print(f"  保存: {model_path}  (val ndcg@3={ndcg:.4f})" if isinstance(ndcg, float) else f"  保存: {model_path}")

    # 2. 統計テーブル保存（予測時参照用）
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
