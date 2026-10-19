"""新馬 + 調教特徴量モデル 詳細評価スクリプト
単勝 / 複勝 / 3連複 の的中率・ROI を base vs training で比較。
オッズフィルタ (買い目絞り込み) 付き評価も含む。
保存済みモデルとキャッシュデータを使うため再学習不要。
"""

import os
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.config import paths
from src.PredictionModels.LightGBM.make_dataset_shinba import SHINBA_ALL_FEATURES
from src.PredictionModels.LightGBM.make_dataset_shinba_training import TRAINING_FEATURES

SHINBA_FEATURES = [c for c in SHINBA_ALL_FEATURES if c not in (
    "nick_win", "nick_place", "nick_runs", "matdam_sire_win", "matdam_sire_place"
)]

MODEL_DIR = os.path.join(paths.PREDICTION_MODEL_PATH, "shinba_training")
CACHE_DIR = os.path.join(MODEL_DIR, "cache")
EVAL_YEAR = 2026
MIN_ODDS_FILTER = 5.0           # デフォルト閾値（後でスイープする）
THRESH_SWEEP = [3.0, 5.0, 7.0, 10.0, 15.0]  # しきい値スイープ候補


# ---- データ分割ユーティリティ ----

def split_by_year(feat_df, labels, groups, tan_odds, h_odds, eval_year):
    if "race_id" not in feat_df.columns:
        raise ValueError("race_id 列が必要です")
    years = feat_df["race_id"].astype(str).str[:4].astype(int)
    ev_mask = years >= eval_year
    cum = np.cumsum(groups)
    sub_idx = np.where(ev_mask.values)[0]
    X_sub = feat_df[ev_mask].drop(columns=["race_id"]).copy().reset_index(drop=True)
    y_sub  = labels[ev_mask.values]
    to_sub = tan_odds[ev_mask.values]
    ho_sub = h_odds[ev_mask.values]
    ev_groups = []
    for start, end in zip([0] + list(cum[:-1]), cum):
        race_rows = [i for i in sub_idx if start <= i < end]
        if race_rows:
            ev_groups.append(len(race_rows))
    return X_sub, y_sub, to_sub, ho_sub, ev_groups


def filter_race_type(X, y, to, ho, grps, rt_code):
    mask = (X["race_type_code"] == rt_code)
    X_rt  = X[mask].copy().reset_index(drop=True)
    y_rt  = y[mask.values]
    to_rt = to[mask.values]
    ho_rt = ho[mask.values]
    cum = np.cumsum(grps)
    sub_idx = np.where(mask.values)[0]
    rt_grps = []
    for start, end in zip([0] + list(cum[:-1]), cum):
        race_rows = [i for i in sub_idx if start <= i < end]
        if race_rows:
            rt_grps.append(len(race_rows))
    return X_rt, y_rt, to_rt, ho_rt, rt_grps


# ---- 評価ユーティリティ ----

def _calc_metrics(ranked_indices, y_race, o_race):
    top1 = ranked_indices[0]
    tan_win = int(y_race[top1] == 3.0)
    tan_ret = float(max(o_race)) if tan_win else 0.0
    fuku_win = int(y_race[top1] > 0.0)
    sanren_win = 0
    if len(ranked_indices) >= 3:
        pred_top3   = set(ranked_indices[:3])
        actual_top3 = set(np.where(y_race > 0.0)[0])
        sanren_win  = int(len(actual_top3) >= 3 and pred_top3 == actual_top3)
    return tan_win, tan_ret, fuku_win, sanren_win


def evaluate(model, X_eval, y_eval, tan_odds_eval, h_odds_eval, groups_eval,
             min_odds=None):
    """単勝/複勝/3連複を評価。min_odds を指定すると買い目絞り込み。"""
    feats = [f for f in model.feature_name() if f in X_eval.columns]
    X_use = X_eval[feats].fillna(-1)
    preds = model.predict(X_use.values, num_iteration=model.best_iteration)

    pos = 0
    n_total = n_bet = 0
    tan_wins = tan_ret = fuku_wins = sanren_wins = 0

    for g in groups_eval:
        p_race  = preds[pos:pos+g]
        y_race  = y_eval[pos:pos+g]
        o_race  = tan_odds_eval[pos:pos+g]
        ho_race = h_odds_eval[pos:pos+g]
        pos += g
        if g < 2:
            continue
        n_total += 1
        ranked = np.argsort(-p_race)
        top1   = ranked[0]

        if min_odds is not None:
            top_odds = ho_race[top1]
            if np.isnan(top_odds) or top_odds < min_odds:
                continue

        n_bet += 1
        tw, tr, fw, sw = _calc_metrics(ranked, y_race, o_race)
        tan_wins += tw; tan_ret += tr; fuku_wins += fw; sanren_wins += sw

    return {
        "tan_hit":        tan_wins  / max(n_bet, 1),
        "tan_roi":        tan_ret   / max(n_bet, 1),
        "fuku_hit":       fuku_wins / max(n_bet, 1),
        "sanrenpuku_hit": sanren_wins / max(n_bet, 1),
        "n_races":        n_bet,
        "n_total":        n_total,
        "_raw_preds":     preds,
    }


def evaluate_mar(preds_hit, preds_val, y_eval, tan_odds_eval, h_odds_eval, groups_eval,
                 hit_w=0.6, val_w=0.4, min_odds=None):
    """MAR = hit_w×z(hit) + val_w×z(val) の合成スコアで評価。"""
    def _z(arr):
        s = arr.std()
        return (arr - arr.mean()) / s if s > 0 else arr - arr.mean()

    mar_preds = hit_w * _z(preds_hit) + val_w * _z(preds_val)
    pos = 0
    n_total = n_bet = 0
    tan_wins = tan_ret = fuku_wins = sanren_wins = 0

    for g in groups_eval:
        p_race  = mar_preds[pos:pos+g]
        y_race  = y_eval[pos:pos+g]
        o_race  = tan_odds_eval[pos:pos+g]
        ho_race = h_odds_eval[pos:pos+g]
        pos += g
        if g < 2:
            continue
        n_total += 1
        ranked = np.argsort(-p_race)
        top1   = ranked[0]

        if min_odds is not None:
            top_odds = ho_race[top1]
            if np.isnan(top_odds) or top_odds < min_odds:
                continue

        n_bet += 1
        tw, tr, fw, sw = _calc_metrics(ranked, y_race, o_race)
        tan_wins += tw; tan_ret += tr; fuku_wins += fw; sanren_wins += sw

    return {
        "tan_hit":        tan_wins  / max(n_bet, 1),
        "tan_roi":        tan_ret   / max(n_bet, 1),
        "fuku_hit":       fuku_wins / max(n_bet, 1),
        "sanrenpuku_hit": sanren_wins / max(n_bet, 1),
        "n_races":        n_bet,
        "n_total":        n_total,
    }


# ---- モデルロード ----

def load_model(rt, label, mtype):
    """モデルを読み込む。存在しない場合は None。"""
    if mtype == "valogds":
        path = os.path.join(MODEL_DIR, f"{rt}_shinba_training_valogds_val.txt")
    else:
        path = os.path.join(MODEL_DIR, f"{rt}_shinba_{label}_{mtype}.txt")
    if not os.path.exists(path):
        return None
    return lgb.Booster(model_file=path)


# ---- 出力ユーティリティ ----

def _row(tag, r, marker=""):
    if r is None:
        return f"  {tag:<32s}{marker}  {'N/A':>7s}  {'N/A':>7s}  {'N/A':>7s}  {'N/A':>8s}"
    return (
        f"  {tag:<32s}{marker}"
        f"  {r['tan_hit']:>7.1%}"
        f"  {r['tan_roi']:>7.2f}"
        f"  {r['fuku_hit']:>7.1%}"
        f"  {r['sanrenpuku_hit']:>8.1%}"
        f"  {r['n_races']}R"
        + (f" / {r['n_total']}R全" if r.get('n_total', r['n_races']) != r['n_races'] else "")
    )


# ---- 買い目拡張評価（馬連・3連複） ----

def evaluate_umaren(model, X_eval, y_eval, groups_eval, min_odds=None, h_odds_eval=None):
    """馬連: 予測上位2頭が実際の1着・2着に含まれるか評価。"""
    feats = [f for f in model.feature_name() if f in X_eval.columns]
    X_use = X_eval[feats].fillna(-1)
    preds = model.predict(X_use.values, num_iteration=model.best_iteration)
    pos = 0
    n_total = n_bet = hit = 0
    for g in groups_eval:
        p_race  = preds[pos:pos+g]
        y_race  = y_eval[pos:pos+g]
        ho_race = h_odds_eval[pos:pos+g] if h_odds_eval is not None else None
        pos += g
        if g < 3:
            continue
        n_total += 1
        ranked = np.argsort(-p_race)
        top1 = ranked[0]
        if min_odds is not None and ho_race is not None:
            if np.isnan(ho_race[top1]) or ho_race[top1] < min_odds:
                continue
        n_bet += 1
        pred_top2 = set(ranked[:2])
        actual_top2 = set(np.where(y_race >= 2.0)[0])   # label 2=2着, 3=1着
        if len(actual_top2) >= 2 and pred_top2 == actual_top2:
            hit += 1
    return {"hit": hit / max(n_bet, 1), "n_races": n_bet, "n_total": n_total}


def evaluate_sanrenpuku_expanded(model, X_eval, y_eval, groups_eval,
                                  min_odds=None, h_odds_eval=None):
    """3連複: 予測上位3頭 = 実際の3着以内。"""
    feats = [f for f in model.feature_name() if f in X_eval.columns]
    X_use = X_eval[feats].fillna(-1)
    preds = model.predict(X_use.values, num_iteration=model.best_iteration)
    pos = 0
    n_total = n_bet = hit = 0
    for g in groups_eval:
        p_race  = preds[pos:pos+g]
        y_race  = y_eval[pos:pos+g]
        ho_race = h_odds_eval[pos:pos+g] if h_odds_eval is not None else None
        pos += g
        if g < 4:
            continue
        n_total += 1
        ranked = np.argsort(-p_race)
        top1 = ranked[0]
        if min_odds is not None and ho_race is not None:
            if np.isnan(ho_race[top1]) or ho_race[top1] < min_odds:
                continue
        n_bet += 1
        pred_top3   = set(ranked[:3])
        actual_top3 = set(np.where(y_race > 0.0)[0])
        if len(actual_top3) >= 3 and pred_top3 == actual_top3:
            hit += 1
    return {"hit": hit / max(n_bet, 1), "n_races": n_bet, "n_total": n_total}


# ---- しきい値スイープ ----

def sweep_thresholds(preds_hit, preds_val_base, preds_val_tr, preds_valogds,
                     y_eval, tan_odds_eval, h_odds_eval, groups_eval, thresholds):
    """複数しきい値で base MAR / hybrid MAR / tr MAR+ を比較。"""
    def _z(arr):
        s = arr.std(); return (arr - arr.mean()) / s if s > 0 else arr - arr.mean()

    mar_base    = 0.6 * _z(preds_hit) + 0.4 * _z(preds_val_base)
    mar_hybrid  = 0.6 * _z(preds_hit) + 0.4 * _z(preds_val_base)   # hit=tr, val=base
    mar_tr      = 0.6 * _z(preds_hit) + 0.4 * _z(preds_val_tr)
    mar_plus    = 0.6 * _z(preds_hit) + 0.4 * _z(preds_valogds)

    # hit は tr_hit を共通で使い、val の組み合わせで差異を見る
    results = {}
    for thr in thresholds:
        row = {}
        for name, mar_preds, ref_hit_preds in [
            ("base_MAR",   0.6 * _z(preds_hit) + 0.4 * _z(preds_val_base),  None),
            ("hybrid_MAR", 0.6 * _z(preds_hit) + 0.4 * _z(preds_val_base),  None),
            ("tr_MAR+",    0.6 * _z(preds_hit) + 0.4 * _z(preds_valogds),    None),
        ]:
            pos = 0; n_bet = n_total = 0; tw = tr_ = fw = sw = 0
            for g in groups_eval:
                p_race  = mar_preds[pos:pos+g]
                y_race  = y_eval[pos:pos+g]
                o_race  = tan_odds_eval[pos:pos+g]
                ho_race = h_odds_eval[pos:pos+g]
                pos += g
                if g < 2: continue
                n_total += 1
                ranked = np.argsort(-p_race)
                top1 = ranked[0]
                if np.isnan(ho_race[top1]) or ho_race[top1] < thr:
                    continue
                n_bet += 1
                t, r, f, s = _calc_metrics(ranked, y_race, o_race)
                tw += t; tr_ += r; fw += f; sw += s
            row[name] = {
                "tan_hit": tw / max(n_bet, 1), "tan_roi": tr_ / max(n_bet, 1),
                "fuku_hit": fw / max(n_bet, 1), "n_races": n_bet, "n_total": n_total,
            }
        results[thr] = row
    return results


# ---- メイン ----

def main():
    print("=" * 80)
    print("新馬モデル + 調教特徴量 詳細評価（買い目調整・しきい値スイープ）")
    print("=" * 80)

    # キャッシュ読み込み
    feat_df    = pd.read_pickle(os.path.join(CACHE_DIR, "feat_tr.pkl"))
    groups     = list(np.load(os.path.join(CACHE_DIR, "groups.npy")))
    labels     = np.load(os.path.join(CACHE_DIR, "labels.npy"))
    tan_odds   = np.load(os.path.join(CACHE_DIR, "tan_odds.npy"))
    race_ids   = np.load(os.path.join(CACHE_DIR, "race_ids.npy"), allow_pickle=True)
    hodds_path = os.path.join(CACHE_DIR, "horse_odds.npy")
    horse_odds = np.load(hodds_path) if os.path.exists(hodds_path) \
                 else np.full(len(feat_df), np.nan)

    if "race_id" not in feat_df.columns:
        feat_df = feat_df.copy()
        feat_df.insert(0, "race_id", race_ids)

    # グレード × オッズ交互作用列を復元
    log_odds = np.log1p(np.where(np.isnan(horse_odds) | (horse_odds <= 0), np.nan, horse_odds))
    feat_df["horse_log_odds"]         = log_odds
    feat_df["train_grade_x_logodds"]  = feat_df.get("train_grade_code",  np.nan) * feat_df["horse_log_odds"]
    feat_df["train_bgrade_x_logodds"] = feat_df.get("train_best_grade",  np.nan) * feat_df["horse_log_odds"]
    feat_df["train_lapz_x_logodds"]   = feat_df.get("train_last_lap_z",  np.nan) * feat_df["horse_log_odds"]

    # 評価セット (2026~)
    X_ev, y_ev, to_ev, ho_ev, g_ev = split_by_year(
        feat_df, labels.astype(float), groups, tan_odds,
        np.array(horse_odds, dtype=float), EVAL_YEAR
    )
    print(f"評価セット ({EVAL_YEAR}~): {len(X_ev)}行 / {len(g_ev)}レース\n")

    HDR = f"  {'モデル':<36s}  {'単勝的中':>7s}  {'単勝ROI':>7s}  {'複勝的中':>7s}  {'3連複的中':>8s}  R数"
    SEP = "  " + "-" * 75

    for rt_code, rt_name in [(0.0, "turf"), (1.0, "dirt")]:
        X_rt, y_rt, to_rt, ho_rt, g_rt = filter_race_type(
            X_ev, y_ev, to_ev, ho_ev, g_ev, rt_code
        )
        if len(g_rt) < 5:
            continue

        # モデル読み込み
        m = {
            "base_hit":   load_model(rt_name, "base",     "hit"),
            "base_val":   load_model(rt_name, "base",     "val"),
            "tr_hit":     load_model(rt_name, "training", "hit"),
            "tr_val":     load_model(rt_name, "training", "val"),
            "tr_valogds": load_model(rt_name, None,       "valogds"),
        }

        # 全件予測スコア取得（MAR計算・スイープ用）
        def _preds(key):
            if m[key] is None: return None
            feats = [f for f in m[key].feature_name() if f in X_rt.columns]
            return m[key].predict(X_rt[feats].fillna(-1).values,
                                  num_iteration=m[key].best_iteration)

        p = {k: _preds(k) for k in m}

        # 全件評価
        def ev(key):
            if m[key] is None: return None
            r = evaluate(m[key], X_rt, y_rt, to_rt, ho_rt, g_rt)
            r["_raw_preds"] = p[key]
            return r

        r = {k: ev(k) for k in m}

        def _z(arr):
            s = arr.std(); return (arr - arr.mean()) / s if s > 0 else arr - arr.mean()

        def mar_eval(hp, vp, min_odds=None):
            if hp is None or vp is None: return None
            return evaluate_mar(hp, vp, y_rt, to_rt, ho_rt, g_rt, min_odds=min_odds)

        # ---- 全件: base / training / hybrid ----
        r["base_mar"]    = mar_eval(p["base_hit"],  p["base_val"])
        r["tr_mar"]      = mar_eval(p["tr_hit"],    p["tr_val"])
        r["hybrid_mar"]  = mar_eval(p["tr_hit"],    p["base_val"])   # HITのみ調教
        r["tr_mar_plus"] = mar_eval(p["tr_hit"],    p["tr_valogds"])

        print(f"\n{'='*80}")
        print(f"[{rt_name.upper()}]  評価: {len(g_rt)}R  ({len(X_rt)}頭)")
        print(HDR)

        print(SEP)
        print("  【全件: base (調教なし)】")
        print(_row("HIT",         r["base_hit"]))
        print(_row("VAL",         r["base_val"]))
        print(_row("MAR (H×0.6+V×0.4)", r["base_mar"]))

        print(SEP)
        print("  【全件: +調教 (full)】")
        print(_row("HIT ★",           r["tr_hit"],      " ★"))
        print(_row("VAL ★",           r["tr_val"],      " ★"))
        print(_row("VAL+odds ★",      r["tr_valogds"],  " ★"))
        print(_row("MAR ★",           r["tr_mar"],      " ★"))
        print(_row("MAR+ ★",          r["tr_mar_plus"], " ★"))

        print(SEP)
        print("  【全件: HIT のみ調教 (hybrid)】")
        print(_row("hybrid MAR (trHIT×0.6+baseVAL×0.4)", r["hybrid_mar"], " ▲"))

        # ---- しきい値スイープ ----
        if p["tr_hit"] is not None and p["base_val"] is not None and p["tr_valogds"] is not None:
            print(SEP)
            print(f"  【しきい値スイープ (base MAR / hybrid MAR / MAR+)】")
            THR_HDR = f"  {'thr':>5s}  {'モデル':<22s}  {'的中':>6s}  {'ROI':>6s}  {'複勝':>6s}  R数 / 全"
            print(THR_HDR)
            print("  " + "-" * 60)
            sweep = sweep_thresholds(
                p["tr_hit"], p["base_val"], p["tr_val"], p["tr_valogds"],
                y_rt, to_rt, ho_rt, g_rt, THRESH_SWEEP
            )
            for thr in THRESH_SWEEP:
                row_thr = sweep[thr]
                for mname in ["base_MAR", "hybrid_MAR", "tr_MAR+"]:
                    rv = row_thr[mname]
                    label_str = f"odds≥{thr:.0f}  {mname}"
                    print(f"  {thr:>5.0f}  {mname:<22s}"
                          f"  {rv['tan_hit']:>6.1%}  {rv['tan_roi']:>6.2f}"
                          f"  {rv['fuku_hit']:>6.1%}"
                          f"  {rv['n_races']}R / {rv['n_total']}R全")
                print("  " + "·" * 55)

        # ---- 買い目拡張: base MAR / hybrid MAR ----
        if m["base_hit"] and m["base_val"] and m["tr_hit"]:
            print(SEP)
            print("  【買い目拡張 (馬連・3連複 的中率)】")
            def _umaren(model, thr=None):
                return evaluate_umaren(model, X_rt, y_rt, g_rt,
                                       min_odds=thr, h_odds_eval=ho_rt)
            def _sanren(model, thr=None):
                return evaluate_sanrenpuku_expanded(model, X_rt, y_rt, g_rt,
                                                    min_odds=thr, h_odds_eval=ho_rt)
            for thr in [None] + [5.0, 7.0]:
                label = f"odds≥{thr:.0f}" if thr else "全件"
                ub = _umaren(m["base_hit"], thr); sb = _sanren(m["base_hit"], thr)
                uh = _umaren(m["tr_hit"],   thr); sh = _sanren(m["tr_hit"], thr)
                print(f"  {label:<8s}  base HIT  馬連:{ub['hit']:>6.1%}  3連複:{sb['hit']:>6.1%}  ({ub['n_races']}R)")
                print(f"  {label:<8s}  +調HIT ★  馬連:{uh['hit']:>6.1%}  3連複:{sh['hit']:>6.1%}  ({uh['n_races']}R)")
                print("  " + "·" * 55)

    print(f"\n{'='*80}")


if __name__ == "__main__":
    main()
