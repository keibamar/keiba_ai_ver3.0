"""障害戦専用モデルの学習・評価スクリプト

全場共通の障害戦モデルを2つ学習する（新馬戦専用モデルと同じ hit / val 構成）:
  hit: 的中率重視  … 過去成績 + 当日オッズ・人気（LambdaRank）
  val: 回収率重視  … 過去成績のみ、オッズを使わない（LambdaRank）
  MAR = 0.6 × hit指数 + 0.4 × val指数（shinba と同じ合成）

評価（年ごとの前進評価）:
  2023〜2026年の各年について「その年より前のデータだけで学習 → その年を予測」を繰り返し、
  合計 約450レースの成績を出す（テスト年のデータは学習に一切使わない）。
  比較用に「1番人気を買った場合」も出す。
保存するモデル:
  全期間（2019〜最新）で学習したもの。木の本数は前進評価で安定していた 100 本に固定
  （障害戦はデータが少なく、早期終了だと検証のブレで本数が大きく揺れるため）。

出力:
  data/prediction/models/shogai/shogai_hit.txt
  data/prediction/models/shogai/shogai_val.txt
  data/prediction/models/shogai/cache/dataset.pkl   （特徴量キャッシュ。--rebuild で作り直す）

実行:
    python scripts/train_shogai_model.py [--rebuild]
"""

import glob
import os
import sys
import warnings

warnings.simplefilter("ignore")

import lightgbm as lgb
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.PredictionModels.LightGBM.make_dataset_shogai import (  # noqa: E402
    HIT_FEATURES,
    VAL_FEATURES,
    make_training_dataset,
)

MODEL_DIR = os.path.join(PROJECT_ROOT, "data", "prediction", "models", "shogai")
CACHE_PATH = os.path.join(MODEL_DIR, "cache", "dataset.pkl")
RETURNS_DIR = os.path.join(PROJECT_ROOT, "data", "race_info", "race_returns")

EVAL_YEARS = [2023, 2024, 2025, 2026]
NUM_ROUNDS = 100

PARAMS = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "ndcg_eval_at": [1, 3],
    "label_gain": [0, 1, 2, 3],
    "learning_rate": 0.03,
    "num_leaves": 15,
    "min_data_in_leaf": 40,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "verbose": -1,
    "seed": 42,
    "force_col_wise": True,
}


# ---------- データ ----------

def load_dataset(rebuild=False):
    if not rebuild and os.path.isfile(CACHE_PATH):
        return pd.read_pickle(CACHE_PATH)
    df = make_training_dataset()
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    df.to_pickle(CACHE_PATH)
    return df


def _groups(df):
    return df.groupby("race_id", sort=False).size().values


def _ds(df, feats, ref=None):
    return lgb.Dataset(df[feats].values, label=df["label"].values, group=_groups(df),
                       feature_name=feats, reference=ref, free_raw_data=False)


def fit(train_df, feats):
    return lgb.train(PARAMS, _ds(train_df, feats), num_boost_round=NUM_ROUNDS)


def predict_race_scores(df, m_hit, m_val):
    df = df.copy()
    df["hit"] = m_hit.predict(df[HIT_FEATURES].values)
    df["val"] = m_val.predict(df[VAL_FEATURES].values)
    df["mar"] = np.nan
    for _, idx in df.groupby("race_id").groups.items():
        df.loc[idx, "mar"] = 0.6 * _index(df.loc[idx, "hit"]) + 0.4 * _index(df.loc[idx, "val"])
    return df


# ---------- 評価 ----------

def load_returns(years):
    frames = []
    for y in years:
        for p in glob.glob(os.path.join(RETURNS_DIR, "*", f"{y}_race_returns.csv")):
            d = pd.read_csv(p, index_col=0)
            d.index = d.index.astype(str)
            frames.append(d)
    return pd.concat(frames) if frames else pd.DataFrame()


def _index(scores):
    """レース内 z スコア → 0-100 指数（本番の score_to_index と同じ考え方の簡易版）"""
    s = np.asarray(scores, dtype=float)
    sd = s.std()
    z = (s - s.mean()) / sd if sd > 0 else np.zeros_like(s)
    return 50 + 10 * z


def evaluate(test_df, score_cols, returns):
    """各スコアの1位馬で 単勝・複勝、上位3頭で 3連複（1点）を買った場合の成績"""
    rets = {}
    for rid, g in returns.groupby(level=0):
        rets[rid] = g
    out = []
    for name in score_cols:
        n = win = place = trio = 0
        pay_win = pay_place = pay_trio = 0.0
        for rid, g in test_df.groupby("race_id", sort=False):
            r = rets.get(rid)
            if r is None or g["rank"].notna().sum() < 3:
                continue
            g = g.sort_values(name, ascending=(name == "popularity"))
            top = g.iloc[0]
            top3 = set(int(x) for x in g["umaban"].iloc[:3])
            n += 1
            um = str(int(top["umaban"]))
            t = r[r["式別"] == "単勝"]
            if (t["馬番"].astype(str) == um).any():
                win += 1
                pay_win += float(t[t["馬番"].astype(str) == um]["配当"].iloc[0])
            f = r[r["式別"] == "複勝"]
            if (f["馬番"].astype(str) == um).any():
                place += 1
                pay_place += float(f[f["馬番"].astype(str) == um]["配当"].iloc[0])
            s = r[r["式別"] == "三連複"]
            for _, row in s.iterrows():
                if set(int(x) for x in str(row["馬番"]).split("-")) == top3:
                    trio += 1
                    pay_trio += float(row["配当"])
        out.append({
            "model": name, "races": n,
            "単勝的中": win / n, "単勝回収": pay_win / (n * 100),
            "複勝的中": place / n, "複勝回収": pay_place / (n * 100),
            "3連複的中": trio / n, "3連複回収": pay_trio / (n * 100),
        })
    return pd.DataFrame(out)


def main():
    rebuild = "--rebuild" in sys.argv
    df = load_dataset(rebuild)
    df["year"] = pd.to_datetime(df["date"]).dt.year
    df = df.sort_values(["date", "race_id"]).reset_index(drop=True)
    print(f"学習データ: {df.race_id.nunique()}R / {len(df)}頭 "
          f"({df.year.min()}〜{df.year.max()})")

    # --- 前進評価 ---
    returns = load_returns(EVAL_YEARS)
    tests = []
    for ty in EVAL_YEARS:
        tr = df[df["year"] < ty]
        te = df[df["year"] == ty]
        if te.empty:
            continue
        tests.append(predict_race_scores(te, fit(tr, HIT_FEATURES), fit(tr, VAL_FEATURES)))
    res = evaluate(pd.concat(tests), ["hit", "val", "mar", "popularity"], returns)
    pd.set_option("display.width", 200)
    print(f"\n=== 前進評価 {EVAL_YEARS[0]}〜{EVAL_YEARS[-1]}年（各年、それより前だけで学習） ===")
    print(res.to_string(index=False, float_format=lambda x: f"{x:.1%}"))
    print("（popularity = 1番人気を買った場合の比較用）")

    # --- 本番用: 全期間で学習 ---
    m_hit = fit(df, HIT_FEATURES)
    m_val = fit(df, VAL_FEATURES)
    imp = pd.Series(m_val.feature_importance("gain"), index=VAL_FEATURES).sort_values(ascending=False)
    print("\nval 重要度 上位10:", ", ".join(imp.index[:10]))

    os.makedirs(MODEL_DIR, exist_ok=True)
    m_hit.save_model(os.path.join(MODEL_DIR, "shogai_hit.txt"))
    m_val.save_model(os.path.join(MODEL_DIR, "shogai_val.txt"))
    print(f"\n保存: {os.path.join(MODEL_DIR, 'shogai_hit.txt')} / shogai_val.txt")


if __name__ == "__main__":
    main()
