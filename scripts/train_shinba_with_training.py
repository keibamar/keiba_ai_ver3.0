"""新馬戦モデル + 調教特徴量 実験スクリプト

学習データ: 〜2025年末
評価データ: 2026年以降

ベースライン (SHINBA_FEATURES: 29列) と
調教特徴量追加版 (SHINBA_FEATURES + TRAINING_FEATURES: 37列) を比較。

出力:
  data/prediction/models/shinba_training/turf_shinba_training_hit.txt
  data/prediction/models/shinba_training/turf_shinba_training_val.txt
  data/prediction/models/shinba_training/dirt_shinba_training_hit.txt
  data/prediction/models/shinba_training/dirt_shinba_training_val.txt

実行:
    python scripts/train_shinba_with_training.py
"""

import os
import sys
import warnings

warnings.simplefilter("ignore")

import lightgbm as lgb
import numpy as np
import pandas as pd
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config import paths
from src.config.constants import LIGHTGBM_DEVICE, USE_GPU_TRAINING
from src.PredictionModels.LightGBM.make_dataset_shinba import (
    SHINBA_FEATURES,
    make_training_dataset_with_ids,
)
from src.PredictionModels.LightGBM.make_dataset_shinba_training import (
    TRAINING_FEATURES,
    add_training_features,
)

N_TRIALS   = 30
MODEL_DIR  = os.path.join(paths.PREDICTION_MODEL_PATH, "shinba_training")
CACHE_DIR  = os.path.join(MODEL_DIR, "cache")
EVAL_YEAR  = 2026   # 評価年
ODDS_CAP   = 30.0

_DEVICE_KWARGS = {"device": LIGHTGBM_DEVICE} if USE_GPU_TRAINING else {"force_col_wise": True}


# ---------- カスタム損失 ----------

def profit_objective(y_pred, dataset):
    y_enc = dataset.get_label()
    p = 1.0 / (1.0 + np.exp(-y_pred))
    is_winner = y_enc > 0
    w = np.clip(y_enc - 1.0, 0.5, ODDS_CAP - 1.0)
    grad = np.where(is_winner, -w * (1.0 - p), p)
    hess = np.where(is_winner, w * p * (1.0 - p), p * (1.0 - p))
    hess = np.clip(hess, 1e-6, None)
    return grad, hess


# ---------- モデル学習 ----------

def _fit_hit(params, ds_tr, ds_va, n_est):
    return lgb.train(
        {"objective": "lambdarank", "label_gain": [0,1,2,3],
         "eval_at": [1,3], "metric": "ndcg",
         "verbosity": -1, "num_iterations": n_est, **_DEVICE_KWARGS, **params},
        ds_tr, valid_sets=[ds_va],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )


def _fit_val(params, ds_tr, ds_va, n_est):
    return lgb.train(
        {"objective": profit_objective, "metric": "binary_logloss",
         "verbosity": -1, "num_iterations": n_est, **_DEVICE_KWARGS, **params},
        ds_tr, valid_sets=[ds_va],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )


def _tune_hit(X_tr, y_tr, g_tr, X_va, y_va, g_va):
    min_cs = max(2, min(20, len(X_tr) // 50))

    def objective(trial):
        p = {
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
            dtr = lgb.Dataset(X_tr.values, label=y_tr, group=g_tr, feature_name=list(X_tr.columns))
            dva = lgb.Dataset(X_va.values, label=y_va, group=g_va, reference=dtr)
            m = _fit_hit(p, dtr, dva, n_est)
            return m.best_score.get("valid_0", {}).get("ndcg@3", 0.0)
        except Exception:
            return 0.0

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
    return study.best_params


def _tune_val(X_tr, to_tr, g_tr, X_va, to_va, g_va):
    min_cs = max(2, min(20, len(X_tr) // 50))
    y_va_bin = (to_va > 0).astype(float)

    def objective(trial):
        p = {
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
            dtr = lgb.Dataset(X_tr.values, label=to_tr, group=g_tr, feature_name=list(X_tr.columns))
            dva = lgb.Dataset(X_va.values, label=y_va_bin, group=g_va, reference=dtr)
            m = _fit_val(p, dtr, dva, n_est)
            score = m.best_score.get("valid_0", {}).get("binary_logloss", None)
            return -score if score is not None else 0.0
        except Exception:
            return 0.0

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=43))
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=False)
    return study.best_params


# ---------- 評価ユーティリティ ----------

def eval_hit_model(model, X_eval, y_eval, groups_eval):
    """NDCG@1/@3 と 1着的中率を計算して返す。"""
    preds = model.predict(X_eval.values, num_iteration=model.best_iteration)
    ndcg1 = ndcg3 = win_acc = 0.0
    n_races = 0
    wins_hit = 0
    pos = 0
    for g in groups_eval:
        p_race  = preds[pos:pos+g]
        y_race  = y_eval[pos:pos+g]
        pos += g
        if g < 2:
            continue
        ranked_idx = np.argsort(-p_race)
        # NDCG@k
        def _dcg(k):
            return sum(
                (2**float(y_race[ranked_idx[i]]) - 1) / np.log2(i + 2)
                for i in range(min(k, g))
            )
        ideal_y = np.sort(y_race)[::-1]
        def _idcg(k):
            return sum(
                (2**float(ideal_y[i]) - 1) / np.log2(i + 2)
                for i in range(min(k, g))
            )
        idcg1 = _idcg(1); idcg3 = _idcg(3)
        ndcg1 += (_dcg(1) / idcg1) if idcg1 > 0 else 0
        ndcg3 += (_dcg(3) / idcg3) if idcg3 > 0 else 0
        n_races += 1
        top_pred = ranked_idx[0]
        if y_race[top_pred] == 3.0:   # label=3 が 1着
            wins_hit += 1
    ndcg1 /= max(n_races, 1)
    ndcg3 /= max(n_races, 1)
    win_acc = wins_hit / max(n_races, 1)
    return {"ndcg@1": ndcg1, "ndcg@3": ndcg3, "win_rate": win_acc, "n_races": n_races}


def eval_val_roi(model, X_eval, y_eval, tan_odds_eval, groups_eval):
    """回収率モデルの上位1頭単勝購入ベースの的中率・ROIを計算。"""
    preds = model.predict(X_eval.values, num_iteration=model.best_iteration)
    pos = 0; wins = 0; total = 0; total_return = 0.0
    for g in groups_eval:
        p_race = preds[pos:pos+g]
        y_race = y_eval[pos:pos+g]
        o_race = tan_odds_eval[pos:pos+g]
        pos += g
        if g < 2:
            continue
        top = int(np.argmax(p_race))
        total += 1
        if y_race[top] == 3.0:
            wins += 1
            # tan_odds_eval の値は "1着の単勝オッズ" (1着以外は0)
            # 各レースの1着オッズを取得
            win_odds = max(o_race)   # 1着馬のオッズ
            total_return += win_odds
    win_rate = wins / max(total, 1)
    roi = total_return / max(total, 1)
    return {"win_rate": win_rate, "roi": roi, "n_races": total}


# ---------- 時系列 train/val 分割 ----------

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


# ---------- データ分割: 年 ----------

def split_by_year(feat_df, labels, groups, tan_odds, eval_year):
    """race_id の先頭4桁で train/eval 分割 (eval_year 以降を評価セット)。"""
    if "race_id" not in feat_df.columns:
        raise ValueError("feat_df に race_id 列が必要です")

    years = feat_df["race_id"].astype(str).str[:4].astype(int)
    tr_mask = years < eval_year
    ev_mask = years >= eval_year

    def _sub(mask):
        X_sub  = feat_df[mask].drop(columns=["race_id"]).copy().reset_index(drop=True)
        y_sub  = labels[mask.values]
        to_sub = tan_odds[mask.values]
        # groups を再構築
        cum = np.cumsum(groups)
        sub_idx = np.where(mask.values)[0]
        sub_groups = []
        for start, end in zip([0] + list(cum[:-1]), cum):
            race_rows = [i for i in sub_idx if start <= i < end]
            if race_rows:
                sub_groups.append(len(race_rows))
        return X_sub, y_sub, to_sub, sub_groups

    return _sub(tr_mask), _sub(ev_mask)


# ---------- race_type フィルタ ----------

def _filter_race_type(X, y, to, grps, rt_code):
    mask = (X["race_type_code"] == rt_code)
    X_rt  = X[mask].copy().reset_index(drop=True)
    y_rt  = y[mask.values]
    to_rt = to[mask.values]
    cum = np.cumsum(grps)
    idx = np.where(mask.values)[0]
    rt_grps = []
    for s, e in zip([0] + list(cum[:-1]), cum):
        r = [i for i in idx if s <= i < e]
        if r:
            rt_grps.append(len(r))
    return X_rt, y_rt, to_rt, rt_grps


# ---------- メイン ----------

def main():
    print("=" * 65)
    print("新馬戦モデル + 調教特徴量 実験")
    print(f"  Train: ~{EVAL_YEAR-1}   Eval: {EVAL_YEAR}~")
    print(f"  ベース特徴量: {len(SHINBA_FEATURES)}列  追加特徴量: {len(TRAINING_FEATURES)}列")
    print("=" * 65)

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

    # ---- 1. データセット生成 ----
    _cache_feat   = os.path.join(CACHE_DIR, "feat_tr.pkl")
    _cache_groups = os.path.join(CACHE_DIR, "groups.npy")
    _cache_labels = os.path.join(CACHE_DIR, "labels.npy")
    _cache_odds   = os.path.join(CACHE_DIR, "tan_odds.npy")
    _cache_rids   = os.path.join(CACHE_DIR, "race_ids.npy")

    all_cached = all(os.path.isfile(p) for p in [
        _cache_feat, _cache_groups, _cache_labels, _cache_odds, _cache_rids
    ])

    if all_cached:
        print("\n1. キャッシュから読み込み中...")
        feat_df  = pd.read_pickle(_cache_feat)
        groups   = list(np.load(_cache_groups))
        labels   = np.load(_cache_labels)
        tan_odds = np.load(_cache_odds)
        race_ids = np.load(_cache_rids, allow_pickle=True)
        print(f"  → {len(feat_df)}行 / {len(groups)}レース")
    else:
        print("\n1. 基本データセット生成中...")
        feat_base, groups, labels, tan_odds, race_ids, _, horse_ids = make_training_dataset_with_ids()
        groups = list(groups)
        print(f"  → {len(feat_base)}行 / {len(groups)}レース")

        print("\n2. 調教特徴量を抽出中...")
        feat_df = feat_base.copy()
        feat_df["race_id"]  = race_ids
        feat_df["horse_id"] = horse_ids
        feat_df = add_training_features(feat_df)
        feat_df = feat_df.drop(columns=["race_id", "horse_id"], errors="ignore")

        feat_df.to_pickle(_cache_feat)
        np.save(_cache_groups, np.array(groups))
        np.save(_cache_labels, labels)
        np.save(_cache_odds,   tan_odds)
        np.save(_cache_rids,   race_ids)
        print(f"  キャッシュ保存: {CACHE_DIR}")

    # train_has_data の分布確認
    if "train_has_data" in feat_df.columns:
        n_with = int(feat_df["train_has_data"].sum())
        print(f"\n  調教データあり: {n_with} / {len(feat_df)} 馬 ({100*n_with/len(feat_df):.1f}%)")

    # ---- 2. 特徴量セット定義 ----
    ALL_FEATURES = SHINBA_FEATURES + TRAINING_FEATURES
    # 特徴量列のうち feat_df に存在するものだけ使う
    use_feats_base = [c for c in SHINBA_FEATURES if c in feat_df.columns]
    use_feats_all  = [c for c in ALL_FEATURES      if c in feat_df.columns]

    # ---- 3. 年ベース分割 ----
    print(f"\n3. 年で分割 (train: ~{EVAL_YEAR-1} / eval: {EVAL_YEAR}~)...")
    (X_tr_all, y_tr, to_tr, g_tr), (X_ev_all, y_ev, to_ev, g_ev) = split_by_year(
        feat_df, labels.astype(float), groups, tan_odds, EVAL_YEAR
    )
    print(f"  train: {len(X_tr_all)}行 / {len(g_tr)}レース")
    print(f"  eval:  {len(X_ev_all)}行 / {len(g_ev)}レース")

    if len(X_ev_all) == 0:
        print("  ⚠️ 評価データが空です。2026年の新馬戦データを確認してください。")
        return

    results = {}   # {label: {race_type: {metric: value}}}

    for rt_code, rt_name in [(0.0, "turf"), (1.0, "dirt")]:
        print(f"\n{'='*55}")
        print(f"[{rt_name}]")

        X_tr_rt, y_tr_rt, to_tr_rt, g_tr_rt = _filter_race_type(
            X_tr_all.assign(race_type_code=X_tr_all["race_type_code"]),
            y_tr, to_tr, g_tr, rt_code
        )
        X_ev_rt, y_ev_rt, to_ev_rt, g_ev_rt = _filter_race_type(
            X_ev_all.assign(race_type_code=X_ev_all["race_type_code"]),
            y_ev, to_ev, g_ev, rt_code
        )
        print(f"  train: {len(X_tr_rt)}行 / {len(g_tr_rt)}R  eval: {len(X_ev_rt)}行 / {len(g_ev_rt)}R")

        if len(X_tr_rt) < 100 or len(g_tr_rt) < 20:
            print("  データ不足: スキップ")
            continue
        if len(X_ev_rt) < 10:
            print("  評価データ不足: スキップ")
            continue

        # Optunaで内部 val 分割 (train の最後 20%)
        split_b = split_timeseries(
            X_tr_rt[use_feats_base].fillna(-1), y_tr_rt, g_tr_rt, to_tr_rt
        )
        X_tv_b, X_vv_b, y_tv, y_vv, g_tv, g_vv, to_tv, to_vv = split_b

        split_a = split_timeseries(
            X_tr_rt[use_feats_all].fillna(-1), y_tr_rt, g_tr_rt, to_tr_rt
        )
        X_tv_a, X_vv_a = split_a[0], split_a[1]

        results[rt_name] = {}

        for label, X_tv, X_vv, X_ev_feat, feats in [
            ("base",     X_tv_b, X_vv_b, X_ev_rt[use_feats_base].fillna(-1), use_feats_base),
            ("training", X_tv_a, X_vv_a, X_ev_rt[use_feats_all].fillna(-1),  use_feats_all),
        ]:
            print(f"\n  [{label}] 特徴量: {len(feats)}列")

            # --- HIT ---
            print(f"  [{label}] HIT: Optunaチューニング ({N_TRIALS} trials)...")
            best_hit = _tune_hit(X_tv, y_tv, g_tv, X_vv, y_vv, g_vv)
            n_est_h = best_hit.pop("n_estimators", 300)
            dtr = lgb.Dataset(X_tv.values, label=y_tv, group=g_tv, feature_name=feats)
            dva = lgb.Dataset(X_vv.values, label=y_vv, group=g_vv, reference=dtr)
            model_hit = _fit_hit(best_hit, dtr, dva, n_est_h)

            # eval
            ev_hit = eval_hit_model(model_hit, X_ev_feat, y_ev_rt, g_ev_rt)
            print(f"  [{label}] HIT eval → NDCG@1={ev_hit['ndcg@1']:.4f}  NDCG@3={ev_hit['ndcg@3']:.4f}  WinRate={ev_hit['win_rate']:.3f}  ({ev_hit['n_races']}R)")

            # save
            hit_path = os.path.join(MODEL_DIR, f"{rt_name}_shinba_{label}_hit.txt")
            model_hit.save_model(hit_path)

            # --- VAL ---
            print(f"  [{label}] VAL: Optunaチューニング ({N_TRIALS} trials)...")
            best_val = _tune_val(X_tv, to_tv, g_tv, X_vv, to_vv, g_vv)
            n_est_v = best_val.pop("n_estimators", 300)
            y_vv_bin = (to_vv > 0).astype(float)
            dtr_v = lgb.Dataset(X_tv.values, label=to_tv, group=g_tv, feature_name=feats)
            dva_v = lgb.Dataset(X_vv.values, label=y_vv_bin, group=g_vv, reference=dtr_v)
            model_val = _fit_val(best_val, dtr_v, dva_v, n_est_v)

            ev_val = eval_val_roi(model_val, X_ev_feat, y_ev_rt, to_ev_rt, g_ev_rt)
            print(f"  [{label}] VAL eval → WinRate={ev_val['win_rate']:.3f}  ROI={ev_val['roi']:.2f}  ({ev_val['n_races']}R)")

            val_path = os.path.join(MODEL_DIR, f"{rt_name}_shinba_{label}_val.txt")
            model_val.save_model(val_path)

            # feature importance (training model のみ)
            if label == "training":
                imp = pd.Series(model_hit.feature_importance(importance_type="gain"),
                                index=feats).sort_values(ascending=False)
                print(f"\n  [training] HIT 特徴量重要度 Top10:")
                for feat, score in imp.head(10).items():
                    marker = " ★" if feat in TRAINING_FEATURES else ""
                    print(f"    {feat:<35s}: {score:.1f}{marker}")

            results[rt_name][label] = {
                "hit": ev_hit, "val": ev_val,
            }

    # ---- 4. サマリ ----
    print("\n" + "=" * 65)
    print("【評価サマリ】")
    for rt in ["turf", "dirt"]:
        if rt not in results:
            continue
        print(f"\n[{rt}]")
        for label in ["base", "training"]:
            if label not in results[rt]:
                continue
            h = results[rt][label]["hit"]
            v = results[rt][label]["val"]
            print(f"  {label:<10s} HIT: NDCG@3={h['ndcg@3']:.4f}  WinRate={h['win_rate']:.3f} ({h['n_races']}R)"
                  f"  |  VAL: WinRate={v['win_rate']:.3f}  ROI={v['roi']:.2f}")
    print("\n保存先:", MODEL_DIR)
    print("=" * 65)


if __name__ == "__main__":
    main()
