"""v19 vs v19tr バックテスト評価 (2026年)

v19_no26 (base) と v19tr_no26 (+調教特徴量) を全コースで比較。

評価指標:
  単勝的中率 / 単勝ROI / 複勝的中率 (3着以内)

実行:
    python scripts/eval_v19_training_no26.py
"""

import os
import sys
import warnings

warnings.simplefilter("ignore")

import lightgbm as lgb
import numpy as np
import pandas as pd

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

sys.path.append(r"C:\keiba_ai\keiba_ai_ver2.0\libs")
sys.path.append(r"C:\keiba_ai\keiba_ai_ver2.0\src\Datasets")
sys.path.append(r"C:\keiba_ai\keiba_ai_ver2.0\src\PredictionModels\LightGBM")

import name_header
from src.config import paths as paths_v3
from src.PredictionModels.LightGBM.prediction import _band_lengths
from src.PredictionModels.LightGBM.make_dataset_shinba_training import (
    TRAINING_FEATURES,
    _get_course_stats,
    _zscore,
    _course_cat,
    _grade_code,
    _leg_code,
    _parse_times,
)

EVAL_YEARS        = [2026]
DATASET_DIR       = paths_v3.PREDICTION_DATASET_PATH
MODEL_DIR         = paths_v3.PREDICTION_MODEL_PATH
RACE_RESULTS_BASE = r"C:\keiba_ai\keiba_ai_ver2.0\data\RaceResults"

V19_SUFFIX   = "_v19_no26"
V19TR_SUFFIX = "_v19tr_no26"


# ======================================================================
# horse_id ルックアップ
# ======================================================================

def build_horse_id_lookup(years):
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
    times  = _parse_times(last.get("times_list", ""))
    ll = float(times[-1]) if len(times) >= 1 else np.nan
    l4 = float(times[-2]) if len(times) >= 2 else np.nan
    st = _get_course_stats().get(course, {})
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
    records = []
    for _, row in df.iterrows():
        rid = str(int(row["race_id"]))
        # horse_id 列があれば直接使う。なければ race_results lookup にフォールバック
        if "horse_id" in df.columns and pd.notna(row.get("horse_id")):
            hid = str(row["horse_id"]).strip()
        else:
            uma = int(row["umaban"])
            hid = lookup.get((rid, uma))
        records.append(_training_feats(hid, rid) if hid else
                       dict({c: np.nan for c in TRAINING_FEATURES}, train_has_data=0.0))
    feat_df = pd.DataFrame(records, index=df.index)
    return pd.concat([df, feat_df], axis=1)


# ======================================================================
# モデルロード
# ======================================================================

def load_model(place_id, race_type, length, suffix):
    type_str  = "turf" if race_type == "芝" else "dirt"
    place_dir = name_header.PLACE_LIST[place_id - 1]
    path = os.path.join(MODEL_DIR, place_dir,
                        f"{type_str}{length}_lambdarank_model{suffix}.txt")
    if not os.path.isfile(path):
        return None
    return lgb.Booster(model_file=path)


# ======================================================================
# 評価ユーティリティ
# ======================================================================

def _evaluate_model(model, df_ev):
    """model.feature_name() に一致する列を自動選択して評価。"""
    feats = [f for f in model.feature_name() if f in df_ev.columns]
    X = df_ev[feats].fillna(-1)
    preds = model.predict(X.values, num_iteration=model.best_iteration)
    df_ev = df_ev.copy()
    df_ev["_pred"] = preds

    tan_wins = tan_ret = fuku_wins = n_races = 0
    for _, grp in df_ev.groupby("race_id"):
        if len(grp) < 2:
            continue
        n_races += 1
        top_idx = grp["_pred"].idxmax()
        flag    = grp.loc[top_idx, "_result_flag"]
        if flag == 4:
            tan_wins += 1
            if "current_odds" in grp.columns:
                odds_v = float(grp.loc[top_idx, "current_odds"])
                if np.isfinite(odds_v) and odds_v > 0:
                    tan_ret += odds_v
        if flag >= 2:
            fuku_wins += 1

    n = max(n_races, 1)
    return {
        "tan_hit":  tan_wins  / n,
        "tan_roi":  tan_ret   / n,
        "fuku_hit": fuku_wins / n,
        "n_races":  n_races,
    }


def _load_v11_with_flag(place_id, year, race_type, band_length):
    """v11 データと flag を読み込む。horse_id は _horseid_v11.csv から付加。"""
    place_dir = name_header.PLACE_LIST[place_id - 1]
    dset_dir  = os.path.join(DATASET_DIR, place_dir)
    base = os.path.join(dset_dir,
                        f"{year}_{race_type}{band_length}_ai_dataset")
    # v11 → v10 の順でフォールバック
    for ver in ["v11", "v10"]:
        df_path   = base + f"_for_rank_{ver}.csv"
        flag_path = base + f"_flag_{ver}.csv"
        if os.path.isfile(df_path) and os.path.isfile(flag_path):
            try:
                df   = pd.read_csv(df_path,   index_col=0, dtype=float, low_memory=False)
                flag = pd.read_csv(flag_path, index_col=0, dtype=float, low_memory=False)
                if not df.empty and not flag.empty and len(df) == len(flag):
                    df   = df.reset_index(drop=True)
                    flag = flag.reset_index(drop=True)
                    df["_result_flag"] = flag["result_flag"]
                    # horse_id を _horseid_v11.csv から付加
                    hid_path = base + f"_horseid_{ver}.csv"
                    if os.path.isfile(hid_path):
                        hid_df = pd.read_csv(hid_path, dtype=str)
                        if len(hid_df) == len(df):
                            df["horse_id"] = hid_df["horse_id"].values
                    return df
            except Exception:
                pass
    return pd.DataFrame()


# ======================================================================
# メイン
# ======================================================================

def main():
    print("=" * 70)
    print("v19 vs v19tr バックテスト評価 (2026年)")
    print("=" * 70)

    print("\nhorse_id ルックアップ構築中...")
    lookup = build_horse_id_lookup(EVAL_YEARS)
    print(f"  {len(lookup):,} エントリ")

    # 集計用
    results = {
        "v19":   {"turf": [], "dirt": []},
        "v19tr": {"turf": [], "dirt": []},
    }
    course_log = []   # per-course 結果

    TARGET_VENUES = list(range(1, 11))

    for place_id in TARGET_VENUES:
        place_name = name_header.PLACE_LIST[place_id - 1]

        for race_type, length in name_header.COURSE_LISTS[place_id - 1]:
            # モデルロード
            m_v19   = load_model(place_id, race_type, length, V19_SUFFIX)
            m_v19tr = load_model(place_id, race_type, length, V19TR_SUFFIX)
            if m_v19 is None and m_v19tr is None:
                continue

            # 2026 評価データ収集
            df_all = pd.DataFrame()
            for year in EVAL_YEARS:
                for band_length in _band_lengths(place_id, race_type, length):
                    df = _load_v11_with_flag(place_id, year, race_type, band_length)
                    if not df.empty:
                        df_all = pd.concat([df_all, df])

            if df_all.empty or "race_id" not in df_all.columns:
                continue

            df_all = df_all.reset_index(drop=True)

            # 調教特徴量付加 (v19tr 用)
            if m_v19tr is not None and "umaban" in df_all.columns:
                df_all = add_training_cols(df_all, lookup)

            rt_key = "turf" if race_type == "芝" else "dirt"

            ev_v19 = ev_v19tr = None
            if m_v19 is not None:
                ev_v19 = _evaluate_model(m_v19, df_all)
                results["v19"][rt_key].append(ev_v19)
            if m_v19tr is not None:
                ev_v19tr = _evaluate_model(m_v19tr, df_all)
                results["v19tr"][rt_key].append(ev_v19tr)

            course_log.append({
                "place": place_name,
                "course": f"{race_type}{length}",
                "n_races": ev_v19["n_races"] if ev_v19 else (ev_v19tr["n_races"] if ev_v19tr else 0),
                "v19_tan":   ev_v19["tan_hit"]  if ev_v19  else np.nan,
                "v19_roi":   ev_v19["tan_roi"]  if ev_v19  else np.nan,
                "v19_fuku":  ev_v19["fuku_hit"] if ev_v19  else np.nan,
                "v19tr_tan":  ev_v19tr["tan_hit"]  if ev_v19tr else np.nan,
                "v19tr_roi":  ev_v19tr["tan_roi"]  if ev_v19tr else np.nan,
                "v19tr_fuku": ev_v19tr["fuku_hit"] if ev_v19tr else np.nan,
            })

    # ======================================================================
    # 出力
    # ======================================================================

    def _agg(evs):
        if not evs:
            return None
        total_w = total_ret = total_fu = total_r = 0
        for e in evs:
            n = e["n_races"]
            total_w   += e["tan_hit"]  * n
            total_ret += e["tan_roi"]  * n
            total_fu  += e["fuku_hit"] * n
            total_r   += n
        n = max(total_r, 1)
        return {"tan_hit": total_w/n, "tan_roi": total_ret/n,
                "fuku_hit": total_fu/n, "n_races": total_r}

    HDR = f"  {'':8s}  {'単勝的中':>8s}  {'単勝ROI':>8s}  {'複勝的中':>8s}  R数"
    SEP = "  " + "-" * 52

    print(f"\n{'='*70}")
    print("【種別集計】")
    for rt in ["turf", "dirt"]:
        b  = _agg(results["v19"][rt])
        tr = _agg(results["v19tr"][rt])
        print(f"\n  [{rt.upper()}]")
        print(HDR); print(SEP)
        if b:
            print(f"  {'v19 (base)':<8s}  {b['tan_hit']:>8.1%}  {b['tan_roi']:>8.2f}"
                  f"  {b['fuku_hit']:>8.1%}  {b['n_races']}")
        if tr:
            diff_h = (tr['tan_hit'] - b['tan_hit']) if b else 0
            diff_r = (tr['tan_roi'] - b['tan_roi']) if b else 0
            diff_f = (tr['fuku_hit'] - b['fuku_hit']) if b else 0
            arrow_h = "↑" if diff_h > 0 else ("↓" if diff_h < 0 else "→")
            arrow_r = "↑" if diff_r > 0 else ("↓" if diff_r < 0 else "→")
            print(f"  {'v19tr':8s}  {tr['tan_hit']:>8.1%}  {tr['tan_roi']:>8.2f}"
                  f"  {tr['fuku_hit']:>8.1%}  {tr['n_races']}")
            print(f"  {'差分':8s}  {diff_h:>+8.1%}  {diff_r:>+8.2f}"
                  f"  {diff_f:>+8.1%}  ({arrow_r})")

    # 全体合計
    all_v19   = results["v19"]["turf"]   + results["v19"]["dirt"]
    all_v19tr = results["v19tr"]["turf"] + results["v19tr"]["dirt"]
    b_all  = _agg(all_v19)
    tr_all = _agg(all_v19tr)
    print(f"\n  [全体]")
    print(HDR); print(SEP)
    if b_all:
        print(f"  {'v19 (base)':<8s}  {b_all['tan_hit']:>8.1%}  {b_all['tan_roi']:>8.2f}"
              f"  {b_all['fuku_hit']:>8.1%}  {b_all['n_races']}")
    if tr_all:
        diff_h = tr_all['tan_hit'] - b_all['tan_hit']
        diff_r = tr_all['tan_roi'] - b_all['tan_roi']
        diff_f = tr_all['fuku_hit'] - b_all['fuku_hit']
        print(f"  {'v19tr':8s}  {tr_all['tan_hit']:>8.1%}  {tr_all['tan_roi']:>8.2f}"
              f"  {tr_all['fuku_hit']:>8.1%}  {tr_all['n_races']}")
        print(f"  {'差分':8s}  {diff_h:>+8.1%}  {diff_r:>+8.2f}"
              f"  {diff_f:>+8.1%}")

    # コース別 Top / Bottom
    if course_log:
        df_log = pd.DataFrame(course_log)
        df_log["roi_diff"] = df_log["v19tr_roi"] - df_log["v19_roi"]
        df_log["hit_diff"] = df_log["v19tr_tan"] - df_log["v19_tan"]

        print(f"\n{'='*70}")
        print("【ROI改善 上位10コース】")
        top = df_log.dropna(subset=["roi_diff"]).nlargest(10, "roi_diff")
        print(f"  {'コース':<22s}  {'v19 ROI':>8s}  {'v19tr ROI':>9s}  {'差分':>7s}  R数")
        for _, r in top.iterrows():
            print(f"  {r['place']+' '+r['course']:<22s}"
                  f"  {r['v19_roi']:>8.2f}  {r['v19tr_roi']:>9.2f}"
                  f"  {r['roi_diff']:>+7.2f}  {int(r['n_races'])}")

        print(f"\n【ROI低下 上位10コース】")
        bot = df_log.dropna(subset=["roi_diff"]).nsmallest(10, "roi_diff")
        for _, r in bot.iterrows():
            print(f"  {r['place']+' '+r['course']:<22s}"
                  f"  {r['v19_roi']:>8.2f}  {r['v19tr_roi']:>9.2f}"
                  f"  {r['roi_diff']:>+7.2f}  {int(r['n_races'])}")

    print(f"\n{'='*70}")


if __name__ == "__main__":
    main()
