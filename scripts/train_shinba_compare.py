"""新馬戦モデル 4バリアント比較学習スクリプト

v3/v4(ニック)/v5(クロス)/v6(母母父) を一度のデータロードで全部学習し、
各バリアントの芝・ダートモデルを別ファイルに保存する。

出力:
  data/prediction/models/shinba/compare/turf_v3.txt  dirt_v3.txt
  data/prediction/models/shinba/compare/turf_v4.txt  dirt_v4.txt
  data/prediction/models/shinba/compare/turf_v5.txt  dirt_v5.txt
  data/prediction/models/shinba/compare/turf_v6.txt  dirt_v6.txt
  data/prediction/models/shinba/compare/stats_tables/*.csv  (共通)

実行:
    python scripts/train_shinba_compare.py
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
    SHINBA_ALL_FEATURES,
    V3_FEATURES, V4_FEATURES, V5_FEATURES, V6_FEATURES,
    build_stats_tables,
    make_training_dataset,
)

N_TRIALS   = 30
MODEL_DIR  = os.path.join(paths.PREDICTION_MODEL_PATH, "shinba")
COMPARE_DIR = os.path.join(MODEL_DIR, "compare")
STATS_DIR  = os.path.join(COMPARE_DIR, "stats_tables")
CACHE_DIR  = os.path.join(MODEL_DIR, "cache")

_DEVICE_KWARGS = {"device": LIGHTGBM_DEVICE} if USE_GPU_TRAINING else {"force_col_wise": True}

VARIANTS = {
    "v3": V3_FEATURES,
    "v4": V4_FEATURES,
    "v5": V5_FEATURES,
    "v6": V6_FEATURES,
}


def split_timeseries(X, labels, groups, train_ratio=0.8):
    cum = np.cumsum(groups)
    total = cum[-1]
    split_row = int(total * train_ratio)
    split_race = np.searchsorted(cum, split_row, side="left")
    split_row = int(cum[split_race - 1]) if split_race > 0 else 0
    X_tr, X_va = X.iloc[:split_row], X.iloc[split_row:]
    y_tr, y_va = labels[:split_row], labels[split_row:]
    g_tr = groups[:split_race]
    g_va = groups[split_race:]
    return X_tr, X_va, y_tr, y_va, g_tr, g_va


def _fit(params, ds_tr, ds_va, n_estimators):
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


def train_variant(variant_name, features, feat_df_all, labels, groups):
    feat_df = feat_df_all[features].copy()
    results = {}
    for race_type_code, race_type_name, fname in [
        (0.0, "芝",     f"turf_{variant_name}.txt"),
        (1.0, "ダート", f"dirt_{variant_name}.txt"),
    ]:
        print(f"\n  [{race_type_name}] 学習開始  ({variant_name}, {len(features)}列)")
        mask = feat_df["race_type_code"] == race_type_code
        X_rt = feat_df[mask].copy().reset_index(drop=True)
        y_rt = labels[mask.values]

        cum_orig = np.cumsum(groups)
        row_idx  = np.where(mask.values)[0]
        rt_groups = []
        for start, end in zip([0] + list(cum_orig[:-1]), cum_orig):
            race_rows = [i for i in row_idx if start <= i < end]
            if race_rows:
                rt_groups.append(len(race_rows))

        if len(X_rt) < 100 or len(rt_groups) < 20:
            print("  データ不足: スキップ")
            continue

        X_tr, X_va, y_tr, y_va, g_tr, g_va = split_timeseries(X_rt, y_rt, rt_groups)
        print(f"  train={len(X_tr)}行({len(g_tr)}R) / val={len(X_va)}行({len(g_va)}R)")

        print(f"  Optuna チューニング ({N_TRIALS} trials)...")
        best_params = _tune(X_tr, y_tr, g_tr, X_va, y_va, g_va, N_TRIALS)

        n_est = best_params.pop("n_estimators", 300)
        ds_tr = lgb.Dataset(X_tr.values, label=y_tr, group=g_tr, feature_name=list(X_tr.columns))
        ds_va = lgb.Dataset(X_va.values, label=y_va, group=g_va, reference=ds_tr)
        model = _fit(best_params, ds_tr, ds_va, n_est)

        model_path = os.path.join(COMPARE_DIR, fname)
        model.save_model(model_path)
        ndcg = model.best_score.get("valid_0", {}).get("ndcg@3", "N/A")
        ndcg_str = f"{ndcg:.4f}" if isinstance(ndcg, float) else str(ndcg)
        print(f"  保存: {model_path}  (val ndcg@3={ndcg_str})")
        results[race_type_name] = ndcg_str

    return results


def main():
    os.makedirs(COMPARE_DIR, exist_ok=True)
    os.makedirs(STATS_DIR,   exist_ok=True)
    os.makedirs(CACHE_DIR,   exist_ok=True)

    _cache_feat   = os.path.join(CACHE_DIR, "feat_all.pkl")
    _cache_groups = os.path.join(CACHE_DIR, "groups_all.npy")
    _cache_labels = os.path.join(CACHE_DIR, "labels_all.npy")

    print("=" * 65)
    print("新馬戦モデル 4バリアント比較学習")
    print("v3(ベース) / v4(ニック) / v5(クロス) / v6(母母父)")
    print("=" * 65)

    if os.path.isfile(_cache_feat) and os.path.isfile(_cache_groups) and os.path.isfile(_cache_labels):
        print("\n1. キャッシュからデータセットを読み込み中...")
        feat_df_all = pd.read_pickle(_cache_feat)
        groups      = list(np.load(_cache_groups))
        labels      = np.load(_cache_labels)
        print(f"  → {len(feat_df_all)}行 / {len(groups)}レース / {len(feat_df_all.columns)}列")
    else:
        print("\n1. 学習データセット生成中（全列 SHINBA_ALL_FEATURES）...")
        feat_df_all, groups, labels, _ = make_training_dataset()
        groups = list(groups)
        feat_df_all.to_pickle(_cache_feat)
        np.save(_cache_groups, np.array(groups))
        np.save(_cache_labels, labels)
        print(f"  キャッシュ保存: {CACHE_DIR}")

    feat_df_all = feat_df_all.fillna(-1.0)
    labels = labels.astype(float)

    summary = {}
    for variant_name, features in VARIANTS.items():
        print(f"\n{'='*65}")
        print(f"バリアント: {variant_name}  ({len(features)}列)")
        print(f"  特徴量: {features}")
        print(f"{'='*65}")
        res = train_variant(variant_name, features, feat_df_all, labels, groups)
        summary[variant_name] = res

    print("\n\n" + "=" * 65)
    print("比較学習 完了サマリー (val ndcg@3)")
    print(f"{'バリアント':<10} {'芝':>10} {'ダート':>10}")
    print("-" * 40)
    for vname, res in summary.items():
        turf = res.get("芝", "-")
        dirt = res.get("ダート", "-")
        print(f"{vname:<10} {turf:>10} {dirt:>10}")

    print("\n2. 共通統計テーブル生成・保存中...")
    tables = build_stats_tables()
    for name, df in tables.items():
        p = os.path.join(STATS_DIR, f"{name}.csv")
        df.to_csv(p, index=False)
        print(f"  保存: {p}  ({len(df)}件)")

    print("\n完了")


if __name__ == "__main__":
    main()
