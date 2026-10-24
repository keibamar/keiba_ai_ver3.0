"""新馬モデルの horse_weight 有無による精度比較

3パターン比較:
  A     : horse_weight あり で学習・評価 (理論上限)
  A_nan : horse_weight あり で学習、評価時は NaN (現在の実態)
  B     : horse_weight なし で学習・評価 (除外案)

実行:
    python scripts/eval_shinba_horse_weight.py
"""

import os
import sys
import warnings
warnings.simplefilter("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.PredictionModels.LightGBM.make_dataset_shinba import (
    SHINBA_FEATURES, make_training_dataset,
)

FEATURES_NO_WEIGHT = [f for f in SHINBA_FEATURES if f != "horse_weight"]

FIXED_PARAMS = {
    "objective": "lambdarank",
    "label_gain": [0, 1, 2, 3],
    "eval_at": [3],
    "metric": "ndcg",
    "verbosity": -1,
    "num_iterations": 400,
    "learning_rate": 0.02,
    "num_leaves": 31,
    "max_depth": 6,
    "min_child_samples": 5,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
    "force_col_wise": True,
}

TRAIN_RATIO = 0.8


def filter_by_race_type(feat_df, labels, groups, tan_odds, race_type_code):
    mask = feat_df["race_type_code"] == race_type_code
    X_rt = feat_df[mask].copy().reset_index(drop=True)
    y_rt = labels[mask.values]
    to_rt = tan_odds[mask.values]

    cum_orig = np.cumsum(groups)
    row_idx = np.where(mask.values)[0]
    rt_groups = []
    for start, end in zip([0] + list(cum_orig[:-1]), cum_orig):
        race_rows = [i for i in row_idx if start <= i < end]
        if race_rows:
            rt_groups.append(len(race_rows))

    return X_rt, y_rt, to_rt, np.array(rt_groups)


def split_ts(X, labels, tan_odds, groups):
    cum = np.cumsum(groups)
    split_row = int(cum[-1] * TRAIN_RATIO)
    split_race = np.searchsorted(cum, split_row, side="left")
    split_row = int(cum[split_race - 1]) if split_race > 0 else 0

    return (
        X.iloc[:split_row], X.iloc[split_row:],
        labels[:split_row], labels[split_row:],
        groups[:split_race], groups[split_race:],
        tan_odds[:split_row], tan_odds[split_row:],
    )


def train_model(X_tr, y_tr, g_tr, X_va, y_va, g_va, feature_cols):
    ds_tr = lgb.Dataset(X_tr[feature_cols].values, label=y_tr, group=g_tr,
                        feature_name=feature_cols)
    ds_va = lgb.Dataset(X_va[feature_cols].values, label=y_va, group=g_va,
                        reference=ds_tr)
    model = lgb.train(
        FIXED_PARAMS, ds_tr, valid_sets=[ds_va],
        callbacks=[
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )
    return model


def evaluate(model, X_va, y_va, g_va, tan_odds_va, feature_cols, label):
    scores = model.predict(X_va[feature_cols].values)
    cum = np.concatenate([[0], np.cumsum(g_va)])

    hit_1, hit_fuku, total = 0, 0, 0
    ret_1 = 0.0

    for i in range(len(g_va)):
        s = int(cum[i]); e = int(cum[i + 1])
        if e - s < 2:
            continue
        sc = scores[s:e]
        lb = y_va[s:e]
        to = tan_odds_va[s:e]

        best_idx = np.argmax(sc)
        is_win  = lb[best_idx] == 3   # label 3=1着
        is_fuku = lb[best_idx] >= 1   # label 1以上=複勝圏(3着以内)
        odds_win = to[best_idx] if to[best_idx] > 0 else np.nan

        hit_1    += int(is_win)
        hit_fuku += int(is_fuku)
        if is_win and not np.isnan(odds_win):
            ret_1 += odds_win
        total += 1

    hit_rate_1    = hit_1 / total * 100 if total else 0
    hit_rate_fuku = hit_fuku / total * 100 if total else 0
    roi_1 = ret_1 / total * 100 if total else 0

    print(f"  [{label}]  レース数={total}  "
          f"単勝的中率={hit_rate_1:.1f}%  単勝ROI={roi_1:.1f}%  "
          f"複勝的中率={hit_rate_fuku:.1f}%")
    return hit_rate_1, roi_1, hit_rate_fuku


def run_surface(feat_df, labels, groups, tan_odds, race_type_code, name):
    print(f"\n{'='*60}")
    print(f"  {name} (race_type_code={race_type_code})")
    print(f"{'='*60}")

    X, y, to, g = filter_by_race_type(feat_df, labels, groups, tan_odds, race_type_code)
    if len(X) == 0:
        print("  データなし → スキップ")
        return

    print(f"  総エントリ: {len(X)}  レース数: {len(g)}")
    hw_nan_rate = X["horse_weight"].isna().mean() * 100
    print(f"  horse_weight NaN率: {hw_nan_rate:.1f}%")

    X_tr, X_va, y_tr, y_va, g_tr, g_va, to_tr, to_va = split_ts(X, y, to, g)
    print(f"  学習: {len(g_tr)}レース / 評価: {len(g_va)}レース")

    # A: horse_weight あり で学習・評価（理論上限）
    print("\n  学習中 A (with horse_weight)...")
    model_a = train_model(X_tr, y_tr, g_tr, X_va, y_va, g_va, SHINBA_FEATURES)

    # A_nan: 評価時だけ horse_weight を NaN（現在の実態）
    X_va_nan = X_va.copy()
    X_va_nan["horse_weight"] = np.nan

    # B: horse_weight なし で学習・評価
    print("  学習中 B (without horse_weight)...")
    model_b = train_model(X_tr, y_tr, g_tr, X_va, y_va, g_va, FEATURES_NO_WEIGHT)

    print("\n--- 評価結果 ---")
    evaluate(model_a, X_va,     y_va, g_va, to_va, SHINBA_FEATURES,    "A    : あり/あり（理論上限）")
    evaluate(model_a, X_va_nan, y_va, g_va, to_va, SHINBA_FEATURES,    "A_nan: あり/NaN（現在実態）")
    evaluate(model_b, X_va,     y_va, g_va, to_va, FEATURES_NO_WEIGHT, "B    : なし/なし（除外案）  ")


if __name__ == "__main__":
    print("データ読み込み中...")
    feat_df, groups, labels, tan_odds = make_training_dataset()
    groups = np.array(groups)
    print(f"全データ: {len(feat_df)}エントリ / {len(groups)}レース")

    run_surface(feat_df, labels, groups, tan_odds, race_type_code=0.0, name="芝")
    run_surface(feat_df, labels, groups, tan_odds, race_type_code=1.0, name="ダート")

    print("\n完了")
