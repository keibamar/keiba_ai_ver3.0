"""非新馬モデルへの調教特徴量追加 実験スクリプト

既存の v10/v11 データセット (108/114列) に調教特徴量 (11列) を加えて
binary+オッズ重み付きモデルを比較学習・評価する。

学習: 2020〜2025年  評価: 2026年

評価指標:
  単勝的中率  単勝ROI  複勝的中率 (3着以内)

実行:
    python scripts/train_v_training_exp.py
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

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

sys.path.append(r"C:\keiba_ai\keiba_ai_ver2.0\libs")
sys.path.append(r"C:\keiba_ai\keiba_ai_ver2.0\src\Datasets")

import name_header
from src.config import paths as paths_v3
from src.config.constants import LIGHTGBM_DEVICE, USE_GPU_TRAINING
from src.PredictionModels.LightGBM.make_dataset_shinba_training import (
    TRAINING_FEATURES,
    _get_course_stats,
    _zscore,
    _course_cat,
    _grade_code,
    _leg_code,
    _parse_times,
)

TRAIN_YEARS       = list(range(2020, 2026))
EVAL_YEARS        = [2026]
DATASET_DIR       = paths_v3.PREDICTION_DATASET_PATH
RACE_RESULTS_BASE = r"C:\keiba_ai\keiba_ai_ver2.0\data\RaceResults"
N_TRIALS          = 20
MODEL_DIR         = os.path.join(paths_v3.PREDICTION_MODEL_PATH, "training_exp")

_DEVICE_KWARGS = {"device": LIGHTGBM_DEVICE} if USE_GPU_TRAINING else {"force_col_wise": True}


# ======================================================================
# 1. race_id + umaban -> horse_id ルックアップ構築
# ======================================================================

def build_horse_id_lookup(years):
    """すべての競馬場・年の race_results から (race_id_str, umaban_int) -> horse_id を構築。"""
    print("  horse_id ルックアップを構築中...")
    lookup = {}
    for place_dir in sorted(os.listdir(RACE_RESULTS_BASE)):
        for year in years:
            path = os.path.join(RACE_RESULTS_BASE, place_dir, f"{year}_race_results.csv")
            if not os.path.isfile(path):
                continue
            try:
                df = pd.read_csv(path, index_col=0, dtype=str, low_memory=False)
                if "umaban" in df.columns:
                    uma_col = "umaban"
                elif "馬番" in df.columns:
                    uma_col = "馬番"
                else:
                    continue
                if "horse_id" not in df.columns:
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
# 2. 調教特徴量バッチキャッシュ
# ======================================================================

_TRAINING_CACHE: dict = {}


def _load_horse_sessions(horse_id: str) -> pd.DataFrame:
    """horse_id の調教セッション DataFrame を返す（キャッシュあり）。"""
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


def _extract_features_fast(horse_id: str, race_id: str) -> dict:
    """キャッシュ済み DataFrame から調教特徴量を抽出する。"""
    default = {col: np.nan for col in TRAINING_FEATURES}
    default["train_has_data"] = 0.0
    df = _load_horse_sessions(horse_id)
    if df.empty:
        return default
    sessions = df[df["race_id"] == str(race_id)].dropna(subset=["training_date"])
    if sessions.empty:
        return default
    sessions = sessions.sort_values("training_date")
    last = sessions.iloc[-1]
    course_str = str(last.get("course", "")).strip()
    times = _parse_times(last.get("times_list", ""))
    last_lap  = float(times[-1]) if len(times) >= 1 else np.nan
    last_400m = float(times[-2]) if len(times) >= 2 else np.nan
    stats = _get_course_stats()
    st = stats.get(course_str, {})
    return {
        "train_has_data":       1.0,
        "train_last_lap":       last_lap,
        "train_last_400m":      last_400m,
        "train_n_poles":        float(len(times)),
        "train_grade_code":     _grade_code(last.get("grade", "")),
        "train_leg_code":       _leg_code(last.get("leg_type", "")),
        "train_sessions_count": float(len(sessions)),
        "train_best_grade":     max(
            (_grade_code(g) for g in sessions["grade"].fillna("").astype(str)),
            default=0.0
        ),
        "train_last_lap_z":     _zscore(last_lap,  st.get("ll_mean"), st.get("ll_std")),
        "train_last_400m_z":    _zscore(last_400m, st.get("l4_mean"), st.get("l4_std")),
        "train_course_cat":     _course_cat(course_str),
    }


# ======================================================================
# 3. データセット読み込み
# ======================================================================

def load_v_datasets(years, horse_lookup, with_training=True):
    """全競馬場・コースの v10/v11 データを一括読み込み。

    with_training=True のとき TRAINING_FEATURES を付加する。
    """
    rows = []
    missing_hid = found_hid = 0

    for place_id in range(1, 11):
        place_dir = name_header.PLACE_LIST[place_id - 1]
        dset_dir  = os.path.join(DATASET_DIR, place_dir)
        if not os.path.isdir(dset_dir):
            continue

        for year in years:
            for fname in sorted(os.listdir(dset_dir)):
                if not fname.startswith(str(year)):
                    continue
                # 2020-2025: v10 / 2026: v11
                if year < 2026 and "for_rank_v10" not in fname:
                    continue
                if year >= 2026 and "for_rank_v11" not in fname:
                    continue
                fpath = os.path.join(dset_dir, fname)

                try:
                    df = pd.read_csv(fpath, index_col=0, dtype=float, low_memory=False)
                except Exception:
                    continue
                if df.empty or "race_id" not in df.columns or "umaban" not in df.columns:
                    continue

                flag_path = fpath.replace("for_rank_v10", "flag_v10") \
                                 .replace("for_rank_v11", "flag_v11")
                if not os.path.isfile(flag_path):
                    continue
                flag_df = pd.read_csv(flag_path, index_col=0, dtype=float, low_memory=False)
                if flag_df.empty or "result_flag" not in flag_df.columns:
                    continue
                if len(flag_df) != len(df):
                    continue

                df = df.reset_index(drop=True)
                flag_df = flag_df.reset_index(drop=True)
                df["_result_flag"] = flag_df["result_flag"]

                if with_training and horse_lookup is not None:
                    feat_records = []
                    for _, row in df.iterrows():
                        rid = str(int(row["race_id"]))
                        uma = int(row["umaban"])
                        hid = horse_lookup.get((rid, uma))
                        if hid:
                            found_hid += 1
                            tf = _extract_features_fast(hid, rid)
                        else:
                            missing_hid += 1
                            tf = {col: np.nan for col in TRAINING_FEATURES}
                            tf["train_has_data"] = 0.0
                        feat_records.append(tf)
                    feat_df = pd.DataFrame(feat_records)
                    df = pd.concat([df.reset_index(drop=True),
                                    feat_df.reset_index(drop=True)], axis=1)
                rows.append(df)

    if not rows:
        return pd.DataFrame()
    result = pd.concat(rows, ignore_index=True)
    if with_training and horse_lookup is not None:
        total = found_hid + missing_hid
        print(f"  horse_id 取得: {found_hid:,}件  未取得: {missing_hid:,}件 "
              f"({100*missing_hid/max(total, 1):.1f}%)")
    return result


# ======================================================================
# 4. 学習ユーティリティ
# ======================================================================

def _make_weight(y_bin, df):
    if "current_odds" in df.columns:
        odds = df["current_odds"].values.copy().astype(float)
        valid = np.isfinite(odds) & (odds > 0)
        odds[~valid] = 1.0
    else:
        odds = np.ones(len(y_bin))
    return np.where(y_bin == 1, odds, 1.0)


def _fit(params, X_tr, y_tr, w_tr, X_va, y_va, w_va):
    m = lgb.LGBMClassifier(objective="binary", random_state=0, verbosity=-1,
                           **_DEVICE_KWARGS, **params)
    m.fit(X_tr, y_tr, sample_weight=w_tr,
          eval_set=[(X_va, y_va)], eval_sample_weight=[w_va],
          callbacks=[lgb.early_stopping(50, verbose=False),
                     lgb.log_evaluation(0)])
    return m


def _tune(X_tr, y_tr, w_tr, X_va, y_va, w_va):
    min_cs = max(2, min(30, len(X_tr) // 10))

    def obj(trial):
        p = {
            "n_estimators":      trial.suggest_int("n_estimators", 100, 500),
            "learning_rate":     trial.suggest_float("learning_rate", 0.005, 0.05, log=True),
            "num_leaves":        trial.suggest_int("num_leaves", 16, 64),
            "max_depth":         trial.suggest_int("max_depth", 3, 10),
            "min_child_samples": trial.suggest_int("min_child_samples", 2, min_cs),
            "reg_alpha":         trial.suggest_float("reg_alpha", 1e-3, 3.0, log=True),
            "reg_lambda":        trial.suggest_float("reg_lambda", 1e-3, 3.0, log=True),
            "scale_pos_weight":  trial.suggest_float("scale_pos_weight", 1.0, 20.0),
        }
        m = _fit(p, X_tr, y_tr, w_tr, X_va, y_va, w_va)
        ll = m.evals_result_["valid_0"]["binary_logloss"][m.best_iteration_ - 1]
        return -ll

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=0))
    study.optimize(obj, n_trials=N_TRIALS, show_progress_bar=False)
    return study.best_params


# ======================================================================
# 5. 評価ユーティリティ
# ======================================================================

def _evaluate(model, df_ev, feat_cols):
    """単勝的中率・単勝ROI・複勝的中率を計算。"""
    X_use  = df_ev[feat_cols].fillna(-1)
    preds  = model.predict_proba(X_use.values)[:, 1]
    df_ev  = df_ev.copy()
    df_ev["_pred"] = preds

    tan_wins = tan_ret = fuku_wins = n_races = 0

    for rid, grp in df_ev.groupby("race_id"):
        if len(grp) < 2:
            continue
        n_races += 1
        top_idx  = grp["_pred"].idxmax()
        flag     = grp.loc[top_idx, "_result_flag"]
        if flag == 4:
            tan_wins += 1
            if "current_odds" in grp.columns:
                odds_val = float(grp.loc[top_idx, "current_odds"])
                if np.isfinite(odds_val) and odds_val > 0:
                    tan_ret += odds_val
        if flag >= 2:
            fuku_wins += 1

    n = max(n_races, 1)
    return {
        "tan_hit":  tan_wins  / n,
        "tan_roi":  tan_ret   / n,
        "fuku_hit": fuku_wins / n,
        "n_races":  n_races,
    }


# ======================================================================
# 6. メイン
# ======================================================================

def main():
    print("=" * 65)
    print("非新馬モデル + 調教特徴量 実験")
    print(f"  Train: {TRAIN_YEARS[0]}-{TRAIN_YEARS[-1]}  Eval: {EVAL_YEARS}")
    print("=" * 65)

    os.makedirs(MODEL_DIR, exist_ok=True)

    # horse_id ルックアップ (train + eval)
    lookup = build_horse_id_lookup(TRAIN_YEARS + EVAL_YEARS)

    # ---- データ読み込み ----
    print("\n[学習データ] v10 読み込み中 ...")
    df_tr = load_v_datasets(TRAIN_YEARS, lookup, with_training=True)
    print(f"  学習データ: {len(df_tr):,}行")

    print("\n[評価データ] v11 読み込み中 ...")
    df_ev = load_v_datasets(EVAL_YEARS, lookup, with_training=True)
    print(f"  評価データ: {len(df_ev):,}行")

    if df_tr.empty or df_ev.empty:
        print("データ不足: 終了")
        return

    # ---- 特徴量列の決定 ----
    skip_cols = {"_result_flag", "race_id", "umaban"} | set(TRAINING_FEATURES)
    base_cols = [c for c in df_tr.columns
                 if c not in skip_cols
                 and c in df_ev.columns
                 and pd.api.types.is_numeric_dtype(df_tr[c])]
    tr_cols   = [c for c in TRAINING_FEATURES if c in df_tr.columns and c in df_ev.columns]
    all_cols  = base_cols + tr_cols

    print(f"\n  base 特徴量: {len(base_cols)}列  調教特徴量: {len(tr_cols)}列")

    # ---- ラベル作成 ----
    y_tr = (df_tr["_result_flag"] == 4).astype(int).values
    y_ev = (df_ev["_result_flag"] == 4).astype(int).values

    # ---- train/val 分割 ----
    split = int(len(df_tr) * 0.8)
    while split < len(df_tr) - 1 and df_tr.iloc[split - 1]["race_id"] == df_tr.iloc[split]["race_id"]:
        split += 1

    X_tr_b = df_tr[base_cols].iloc[:split].fillna(-1)
    X_va_b = df_tr[base_cols].iloc[split:].fillna(-1)
    X_tr_a = df_tr[all_cols ].iloc[:split].fillna(-1)
    X_va_a = df_tr[all_cols ].iloc[split:].fillna(-1)
    y_tv = y_tr[:split]
    y_vv = y_tr[split:]
    w_tv = _make_weight(y_tv, df_tr.iloc[:split])
    w_vv = _make_weight(y_vv, df_tr.iloc[split:])

    # ---- base モデル ----
    print("\n[base] Optuna チューニング中 ...")
    best_b  = _tune(X_tr_b, y_tv, w_tv, X_va_b, y_vv, w_vv)
    model_b = _fit(best_b, X_tr_b, y_tv, w_tv, X_va_b, y_vv, w_vv)
    ev_b    = _evaluate(model_b, df_ev, base_cols)
    print(f"  [base]  単勝的中={ev_b['tan_hit']:.1%}  ROI={ev_b['tan_roi']:.2f}"
          f"  複勝={ev_b['fuku_hit']:.1%}  ({ev_b['n_races']}R)")

    # ---- +調教 モデル ----
    print("\n[+調教] Optuna チューニング中 ...")
    best_a  = _tune(X_tr_a, y_tv, w_tv, X_va_a, y_vv, w_vv)
    model_a = _fit(best_a, X_tr_a, y_tv, w_tv, X_va_a, y_vv, w_vv)
    ev_a    = _evaluate(model_a, df_ev, all_cols)
    print(f"  [+調教]  単勝的中={ev_a['tan_hit']:.1%}  ROI={ev_a['tan_roi']:.2f}"
          f"  複勝={ev_a['fuku_hit']:.1%}  ({ev_a['n_races']}R)")

    # ---- 特徴量重要度 ----
    imp = pd.Series(
        model_a.booster_.feature_importance(importance_type="gain"),
        index=all_cols
    ).sort_values(ascending=False)
    print(f"\n  [+調教] 特徴量重要度 Top 15:")
    for feat, score in imp.head(15).items():
        marker = " [調教]" if feat in set(tr_cols) else ""
        print(f"    {feat:<40s}: {score:>9.1f}{marker}")

    # ---- 調教特徴量のみの重要度 ----
    print(f"\n  [+調教] 調教特徴量 重要度:")
    for feat in tr_cols:
        score = imp.get(feat, 0.0)
        rank  = list(imp.index).index(feat) + 1 if feat in imp.index else -1
        print(f"    {feat:<40s}: {score:>9.1f}  (rank={rank})")

    # ---- サマリ ----
    print("\n" + "=" * 65)
    print("【結果サマリ】")
    print(f"  {'':30s}  {'単勝的中':>8s}  {'単勝ROI':>8s}  {'複勝的中':>8s}  R数")
    print(f"  {'base':30s}  {ev_b['tan_hit']:>8.1%}  {ev_b['tan_roi']:>8.2f}"
          f"  {ev_b['fuku_hit']:>8.1%}  {ev_b['n_races']}")
    print(f"  {'+調教':30s}  {ev_a['tan_hit']:>8.1%}  {ev_a['tan_roi']:>8.2f}"
          f"  {ev_a['fuku_hit']:>8.1%}  {ev_a['n_races']}")
    print("=" * 65)

    # ---- モデル保存 ----
    model_b.booster_.save_model(os.path.join(MODEL_DIR, "global_base.txt"))
    model_a.booster_.save_model(os.path.join(MODEL_DIR, "global_training.txt"))
    print(f"\nモデル保存: {MODEL_DIR}")


if __name__ == "__main__":
    main()
