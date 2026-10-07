"""新馬戦本番モデル評価スクリプト (hit / val / MAR)

今年(2026年)の新馬戦を対象に、
  hit (的中率重視, LambdaRank ndcg@3)
  val (回収率重視, profit objective)
  MAR (0.4×hit + 0.6×val ブレンド)
の単勝・複勝・3連複の的中率・回収率を算出する。

実行:
    python scripts/eval_shinba_models.py
"""

import glob
import os
import re
import sys
import warnings
warnings.simplefilter("ignore")

import numpy as np
import pandas as pd
import lightgbm as lgb
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import paths
from src.managers import horse_peds_dataset_manager, horse_profile_dataset_manager, race_info_dataset_manager
from src.PredictionModels.LightGBM.make_dataset_shinba import (
    V5_FEATURES,
    _safe_ped, _dist_band, _compute_cross_score,
)

MODEL_DIR = os.path.join(paths.PREDICTION_MODEL_PATH, "shinba")
STATS_DIR = os.path.join(MODEL_DIR, "stats_tables")


def _safe_ped_val(v):
    if v is None:
        return ""
    try:
        if pd.isna(v):
            return ""
    except Exception:
        pass
    s = str(v).strip()
    return s if s and s.lower() != "nan" else ""


def _rank_int(v):
    s = re.sub(r"\D", "", str(v))
    return int(s) if s else None


def _parse_weight(v):
    try:
        m = re.match(r"(\d+)", str(v))
        return float(m.group(1)) if m else np.nan
    except Exception:
        return np.nan


def _lookup(tbl, key_cols, key_vals, val_col, default=np.nan):
    if tbl is None or tbl.empty:
        return default
    mask = pd.Series([True] * len(tbl))
    for col, val in zip(key_cols, key_vals):
        mask &= tbl[col].astype(str) == str(val)
    rows = tbl[mask]
    return rows[val_col].iloc[0] if not rows.empty else default


def _zscore(s):
    std = s.std()
    if std < 1e-9:
        return [0.0] * len(s)
    return ((s - s.mean()) / std).tolist()


def load_stats():
    stats = {}
    for name in ("sire_course", "sire_global", "bms_course", "bms_global",
                 "dam", "grandsire", "matdam_sire", "nick", "jockey", "trainer"):
        p = os.path.join(STATS_DIR, f"{name}.csv")
        stats[name] = pd.read_csv(p) if os.path.isfile(p) else pd.DataFrame()
    return stats


def load_models():
    models = {}
    for race_type in ("turf", "dirt"):
        for mt in ("hit", "val"):
            key = f"{race_type}_{mt}"
            p = os.path.join(MODEL_DIR, f"{race_type}_shinba_{mt}.txt")
            models[key] = lgb.Booster(model_file=p) if os.path.isfile(p) else None
    return models


def make_row(hid, row, race_type, course_len, place_id, stats, peds_cache):
    dist_band = _dist_band(course_len)
    rt = str(race_type)

    sc_tbl = stats.get("sire_course")
    sg_tbl = stats.get("sire_global")
    bc_tbl = stats.get("bms_course")
    bg_tbl = stats.get("bms_global")
    dam_tbl = stats.get("dam")
    gs_tbl  = stats.get("grandsire")
    jk_tbl  = stats.get("jockey")
    tr_tbl  = stats.get("trainer")

    ped = peds_cache.get(str(hid), {})
    sire        = ped.get("sire", "")
    dam         = ped.get("dam", "")
    grandsire   = ped.get("grandsire", "")
    bms         = ped.get("bms", "")
    cross_score = ped.get("cross_score", 0.0)

    birth_month = horse_profile_dataset_manager.get_birth_month_cached(str(hid))

    waku   = float(row.get("枠番", np.nan)) if str(row.get("枠番","")).replace('.','').isdigit() else np.nan
    umaban = float(row.get("馬番", np.nan)) if str(row.get("馬番","")).replace('.','').isdigit() else np.nan
    kinryo_s = str(row.get("斤量", "")).strip()
    kinryo = float(kinryo_s) if kinryo_s.replace('.','').isdigit() else np.nan
    horse_weight = _parse_weight(row.get("馬体重", ""))
    seirei = str(row.get("性齢", ""))
    sex_code = 1.0 if seirei.startswith("牝") else 0.0
    jockey_id    = str(row.get("jockey_id", ""))
    trainer_name = str(row.get("調教師", ""))

    return {
        "sire_course_win":     _lookup(sc_tbl,  ["sire","race_type","dist_band"], [sire,rt,dist_band], "win_rate"),
        "sire_course_place":   _lookup(sc_tbl,  ["sire","race_type","dist_band"], [sire,rt,dist_band], "place_rate"),
        "sire_course_runs":    _lookup(sc_tbl,  ["sire","race_type","dist_band"], [sire,rt,dist_band], "runs", 0.0),
        "sire_global_win":     _lookup(sg_tbl,  ["sire"], [sire], "global_win"),
        "sire_global_place":   _lookup(sg_tbl,  ["sire"], [sire], "global_place"),
        "bms_course_win":      _lookup(bc_tbl,  ["bms","race_type","dist_band"], [bms,rt,dist_band], "win_rate"),
        "bms_course_place":    _lookup(bc_tbl,  ["bms","race_type","dist_band"], [bms,rt,dist_band], "place_rate"),
        "bms_course_runs":     _lookup(bc_tbl,  ["bms","race_type","dist_band"], [bms,rt,dist_band], "runs", 0.0),
        "bms_global_win":      _lookup(bg_tbl,  ["bms"], [bms], "global_win"),
        "bms_global_place":    _lookup(bg_tbl,  ["bms"], [bms], "global_place"),
        "dam_shinba_win":      _lookup(dam_tbl, ["dam"], [dam], "win_rate"),
        "dam_shinba_place":    _lookup(dam_tbl, ["dam"], [dam], "place_rate"),
        "dam_shinba_runs":     _lookup(dam_tbl, ["dam"], [dam], "runs", 0.0),
        "grandsire_win":       _lookup(gs_tbl,  ["grandsire"], [grandsire], "global_win"),
        "grandsire_place":     _lookup(gs_tbl,  ["grandsire"], [grandsire], "global_place"),
        "jockey_shinba_win":   _lookup(jk_tbl,  ["jockey_id"], [jockey_id],   "win_rate"),
        "jockey_shinba_place": _lookup(jk_tbl,  ["jockey_id"], [jockey_id],   "place_rate"),
        "trainer_shinba_win":  _lookup(tr_tbl,  ["trainer"],   [trainer_name], "win_rate"),
        "trainer_shinba_place":_lookup(tr_tbl,  ["trainer"],   [trainer_name], "place_rate"),
        "place_id":     float(place_id),
        "race_type_code": 1.0 if rt == "ダート" else 0.0,
        "course_len":   float(course_len),
        "waku":         waku,
        "umaban":       umaban,
        "kinryo":       kinryo,
        "sex_code":     sex_code,
        "horse_weight": horse_weight,
        "birth_month":  float(birth_month) if birth_month is not None else np.nan,
        "cross_score":  float(cross_score),
    }


def _get_fuku_payout(ret_df, umaban):
    if ret_df is None or ret_df.empty:
        return 0
    try:
        fuku = ret_df[ret_df["式別"] == "複勝"]
        target = str(int(float(umaban)))
        row = fuku[fuku["馬番"].str.strip() == target]
        if row.empty:
            return 0
        return int(str(row["配当"].iloc[0]).replace(",", ""))
    except Exception:
        return 0


def _get_trio_payout(ret_df, umaban_set):
    if ret_df is None or ret_df.empty:
        return 0
    try:
        trio = ret_df[ret_df["式別"] == "三連複"]
        if trio.empty:
            return 0
        combo = "-".join(str(u) for u in sorted(int(u) for u in umaban_set))
        row = trio[trio["馬番"].str.strip() == combo]
        if row.empty:
            return 0
        return int(str(row["配当"].iloc[0]).replace(",", ""))
    except Exception:
        return 0


def aggregate(results_list, returns_cache):
    """results_list: [{race_id, actual_rank, pred_top1, pred_top3, tan_odds, umaban}]"""
    if not results_list:
        return None
    res_df = pd.DataFrame(results_list)
    n_races = res_df["race_id"].nunique()

    # 単勝
    tan_bets = res_df[res_df["pred_top1"]].copy()
    tan_hits = tan_bets[tan_bets["actual_rank"] == 1]
    hitrate  = len(tan_hits) / len(tan_bets) * 100 if len(tan_bets) > 0 else 0.0
    roi      = tan_hits["tan_odds"].sum() * 100 / (len(tan_bets) * 100) * 100 if len(tan_bets) > 0 else 0.0

    # 複勝
    fuku_bets = tan_bets
    fuku_hits = fuku_bets[fuku_bets["actual_rank"] <= 3]
    fuku_hr   = len(fuku_hits) / len(fuku_bets) * 100 if len(fuku_bets) > 0 else 0.0
    fuku_pay_total = 0
    fuku_pay_bets  = 0
    for _, row in fuku_bets.iterrows():
        rid = str(row["race_id"])
        ret_df = returns_cache.get(rid)
        if ret_df is None:
            continue
        fuku_pay_bets += 1
        if row["actual_rank"] <= 3 and row["umaban"] is not None:
            fuku_pay_total += _get_fuku_payout(ret_df, row["umaban"])
    fuku_roi = fuku_pay_total / (fuku_pay_bets * 100) * 100 if fuku_pay_bets > 0 else 0.0

    # 3連複
    trio_hits = 0
    trio_pay_total = 0
    trio_bets = 0
    trio_pay_bets = 0
    for race_id in res_df["race_id"].unique():
        rdf = res_df[res_df["race_id"] == race_id]
        actual_top3 = set(rdf[rdf["actual_rank"] <= 3]["umaban"].dropna().apply(int))
        pred_top3   = set(rdf[rdf["pred_top3"]]["umaban"].dropna().apply(int))
        if len(actual_top3) < 3 or len(pred_top3) < 3:
            continue
        trio_bets += 1
        if actual_top3 == pred_top3:
            trio_hits += 1
        ret_df = returns_cache.get(str(race_id))
        if ret_df is not None:
            trio_pay_bets += 1
            if actual_top3 == pred_top3:
                trio_pay_total += _get_trio_payout(ret_df, actual_top3)
    trio_hr  = trio_hits / trio_bets * 100 if trio_bets > 0 else 0.0
    trio_roi = trio_pay_total / (trio_pay_bets * 100) * 100 if trio_pay_bets > 0 else 0.0

    return {
        "n_races":  n_races,
        "hitrate":  hitrate,
        "roi":      roi,
        "fuku_hr":  fuku_hr,
        "fuku_roi": fuku_roi,
        "trio_hr":  trio_hr,
        "trio_roi": trio_roi,
    }


def evaluate_all(shinba_df, models, stats, peds_cache, returns_cache):
    """hit / val / mar の3バリアントを1パスで評価"""
    results_hit = []
    results_val = []
    results_mar = []

    race_ids = shinba_df["race_id"].unique()

    for race_id in race_ids:
        race_df = shinba_df[shinba_df["race_id"] == race_id].copy().reset_index(drop=True)
        if len(race_df) < 2:
            continue

        race_type  = str(race_df["race_type"].iloc[0]) if "race_type" in race_df.columns else "芝"
        course_len_s = str(race_df["course_len"].iloc[0]) if "course_len" in race_df.columns else "1600"
        course_len = int(course_len_s) if course_len_s.isdigit() else 1600
        place_id   = int(str(race_id)[4:6]) if str(race_id)[4:6].isdigit() else 1

        rt_key = "turf" if race_type != "ダート" else "dirt"
        model_hit = models.get(f"{rt_key}_hit")
        model_val = models.get(f"{rt_key}_val")
        if model_hit is None and model_val is None:
            continue

        rows = []
        for _, h_row in race_df.iterrows():
            hid = str(h_row.get("horse_id", ""))
            d = make_row(hid, h_row, race_type, course_len, place_id, stats, peds_cache)
            rows.append(d)
        if not rows:
            continue

        X = pd.DataFrame(rows)[V5_FEATURES].fillna(-1.0)

        scores_hit = np.array(model_hit.predict(X.values, num_iteration=model_hit.best_iteration), dtype=float) if model_hit else np.zeros(len(X))
        scores_val = np.array(model_val.predict(X.values, num_iteration=model_val.best_iteration), dtype=float) if model_val else np.zeros(len(X))

        z_hit = _zscore(pd.Series(scores_hit))
        z_val = _zscore(pd.Series(scores_val))
        scores_mar = [0.4 * h + 0.6 * v for h, v in zip(z_hit, z_val)]

        def _build_results(scores):
            sorted_idx = np.argsort(scores)[::-1]
            pred_rank1_idx = int(sorted_idx[0])
            pred_top3_idx  = set(sorted_idx[:3].tolist())
            rows_out = []
            for i, (_, h_row) in enumerate(race_df.iterrows()):
                actual_rank = _rank_int(h_row.get("着順", ""))
                if actual_rank is None:
                    continue
                try:
                    tan_odds = float(str(h_row.get("単勝", "0")).replace(",",""))
                except Exception:
                    tan_odds = 0.0
                umaban_s = str(h_row.get("馬番", "")).strip()
                umaban = int(float(umaban_s)) if umaban_s.replace('.','').isdigit() else None
                rows_out.append({
                    "race_id":     race_id,
                    "actual_rank": actual_rank,
                    "pred_top1":   (i == pred_rank1_idx),
                    "pred_top3":   (i in pred_top3_idx),
                    "tan_odds":    tan_odds,
                    "umaban":      umaban,
                })
            return rows_out

        results_hit.extend(_build_results(scores_hit))
        results_val.extend(_build_results(scores_val))
        results_mar.extend(_build_results(scores_mar))

    return {
        "hit": aggregate(results_hit, returns_cache),
        "val": aggregate(results_val, returns_cache),
        "mar": aggregate(results_mar, returns_cache),
    }


def evaluate_blend(shinba_df, models, stats, peds_cache, returns_cache, hit_w, val_w):
    """任意のブレンド比率で評価。hit_w + val_w = 1.0 を想定"""
    results = []

    race_ids = shinba_df["race_id"].unique()

    for race_id in race_ids:
        race_df = shinba_df[shinba_df["race_id"] == race_id].copy().reset_index(drop=True)
        if len(race_df) < 2:
            continue

        race_type  = str(race_df["race_type"].iloc[0]) if "race_type" in race_df.columns else "芝"
        course_len_s = str(race_df["course_len"].iloc[0]) if "course_len" in race_df.columns else "1600"
        course_len = int(course_len_s) if course_len_s.isdigit() else 1600
        place_id   = int(str(race_id)[4:6]) if str(race_id)[4:6].isdigit() else 1

        rt_key = "turf" if race_type != "ダート" else "dirt"
        model_hit = models.get(f"{rt_key}_hit")
        model_val = models.get(f"{rt_key}_val")
        if model_hit is None and model_val is None:
            continue

        rows = []
        for _, h_row in race_df.iterrows():
            hid = str(h_row.get("horse_id", ""))
            d = make_row(hid, h_row, race_type, course_len, place_id, stats, peds_cache)
            rows.append(d)
        if not rows:
            continue

        X = pd.DataFrame(rows)[V5_FEATURES].fillna(-1.0)

        scores_hit = np.array(model_hit.predict(X.values, num_iteration=model_hit.best_iteration), dtype=float) if model_hit else np.zeros(len(X))
        scores_val = np.array(model_val.predict(X.values, num_iteration=model_val.best_iteration), dtype=float) if model_val else np.zeros(len(X))

        z_hit = _zscore(pd.Series(scores_hit))
        z_val = _zscore(pd.Series(scores_val))
        scores_blend = [hit_w * h + val_w * v for h, v in zip(z_hit, z_val)]

        sorted_idx = np.argsort(scores_blend)[::-1]
        pred_rank1_idx = int(sorted_idx[0])
        pred_top3_idx  = set(sorted_idx[:3].tolist())

        for i, (_, h_row) in enumerate(race_df.iterrows()):
            actual_rank = _rank_int(h_row.get("着順", ""))
            if actual_rank is None:
                continue
            try:
                tan_odds = float(str(h_row.get("単勝", "0")).replace(",",""))
            except Exception:
                tan_odds = 0.0
            umaban_s = str(h_row.get("馬番", "")).strip()
            umaban = int(float(umaban_s)) if umaban_s.replace('.','').isdigit() else None
            results.append({
                "race_id":     race_id,
                "actual_rank": actual_rank,
                "pred_top1":   (i == pred_rank1_idx),
                "pred_top3":   (i in pred_top3_idx),
                "tan_odds":    tan_odds,
                "umaban":      umaban,
            })

    return aggregate(results, returns_cache)


def main():
    print("2026年新馬戦レース結果読み込み中...")
    files = glob.glob(os.path.join(paths.RACE_RESULT_DATA_PATH, "*", "*_race_results.csv"))
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f, dtype=str, index_col=0)
            df.index.name = "race_id"
            dfs.append(df.reset_index())
        except Exception:
            pass
    all_df = pd.concat(dfs, ignore_index=True)
    shinba_2026 = all_df[
        all_df["race_id"].astype(str).str.startswith("2026") &
        (all_df["class"] == "新馬")
    ].copy()
    print(f"  2026: {shinba_2026['race_id'].nunique()}R / {len(shinba_2026)}頭")

    print("\n統計テーブル読み込み中...")
    stats = load_stats()
    print(f"  完了  ({len(stats)}テーブル)")

    print("\nモデル読み込み中...")
    models = load_models()
    for k, m in models.items():
        print(f"  {k}: {'OK' if m else 'なし'}")

    print("\n血統データ一括キャッシュ構築中...")
    horse_ids = shinba_2026["horse_id"].dropna().unique().tolist()
    peds_cache = {}
    for hid in tqdm(horse_ids, desc="血統", ncols=60):
        try:
            peds_data = horse_peds_dataset_manager.get_horse_peds_dataset(str(hid))
            peds_list = peds_data[str(hid)].tolist()
            sire        = _safe_ped_val(peds_list[0])  if len(peds_list) > 0  else ""
            dam         = _safe_ped_val(peds_list[1])  if len(peds_list) > 1  else ""
            grandsire   = _safe_ped_val(peds_list[2])  if len(peds_list) > 2  else ""
            bms         = _safe_ped_val(peds_list[4])  if len(peds_list) > 4  else ""
            cross_score = _compute_cross_score(peds_list)
        except Exception:
            sire = dam = grandsire = bms = ""
            cross_score = 0.0
        peds_cache[str(hid)] = {
            "sire": sire, "dam": dam, "grandsire": grandsire,
            "bms": bms, "cross_score": cross_score,
        }
    print(f"  {len(peds_cache)}頭分")

    print("\n配当データ読み込み中...")
    all_race_ids = shinba_2026["race_id"].dropna().unique().tolist()
    returns_cache = {}
    for rid in tqdm(all_race_ids, desc="配当", ncols=60):
        df = race_info_dataset_manager.get_race_return_csv_for_race(str(rid))
        returns_cache[str(rid)] = df if not df.empty else None
    n_avail = sum(1 for v in returns_cache.values() if v is not None)
    print(f"  配当データ取得: {n_avail}/{len(returns_cache)}R")

    # ---- hit/val 単体評価 ----
    print("\n評価中 (hit/val単体)...")
    result = evaluate_all(shinba_2026, models, stats, peds_cache, returns_cache)

    # ---- ブレンド比率グリッド ----
    BLENDS = [
        (1.0, 0.0),
        (0.9, 0.1),
        (0.8, 0.2),
        (0.7, 0.3),
        (0.6, 0.4),
        (0.5, 0.5),
        (0.6, 0.4),  # 現行MAR
        (0.3, 0.7),
        (0.2, 0.8),
        (0.1, 0.9),
        (0.0, 1.0),
    ]
    blend_results = {}
    print("評価中 (ブレンド比率グリッド)...")
    for hw, vw in tqdm(BLENDS, desc="ブレンド", ncols=50):
        key = f"h{int(hw*10):02d}_v{int(vw*10):02d}"
        blend_results[key] = (hw, vw, evaluate_blend(shinba_2026, models, stats, peds_cache, returns_cache, hw, vw))

    lines = [
        "",
        "=" * 90,
        "新馬戦本番モデル評価 - 2026年新馬戦  hit/val 単体",
        "=" * 90,
        f"{'モデル':<18} {'R数':>5} {'単勝的中':>8} {'単勝回収':>9} {'複勝的中':>8} {'複勝回収':>9} {'3連複的中':>9} {'3連複回収':>9}",
        "-" * 90,
    ]
    for key, label in [("hit", "hit(的中率重視)"), ("val", "val(回収率重視)")]:
        r = result.get(key)
        if r is None:
            lines.append(f"{label:<20}  データなし")
            continue
        lines.append(
            f"{label:<20} {r['n_races']:>4}R"
            f"  {r['hitrate']:>6.1f}%"
            f"  {r['roi']:>7.1f}%"
            f"  {r['fuku_hr']:>6.1f}%"
            f"  {r['fuku_roi']:>7.1f}%"
            f"  {r['trio_hr']:>7.1f}%"
            f"  {r['trio_roi']:>7.1f}%"
        )

    lines += [
        "",
        "=" * 90,
        "ブレンド比率グリッド  (hit比率 × val比率)",
        "=" * 90,
        f"{'hit:val':<10} {'R数':>5} {'単勝的中':>8} {'単勝回収':>9} {'複勝的中':>8} {'複勝回収':>9} {'3連複的中':>9} {'3連複回収':>9}",
        "-" * 90,
    ]
    for key, (hw, vw, r) in blend_results.items():
        if r is None:
            continue
        marker = "  ← 現行MAR" if (hw == 0.6 and vw == 0.4) else ""
        label = f"{hw:.1f}:{vw:.1f}"
        lines.append(
            f"{label:<10} {r['n_races']:>4}R"
            f"  {r['hitrate']:>6.1f}%"
            f"  {r['roi']:>7.1f}%"
            f"  {r['fuku_hr']:>6.1f}%"
            f"  {r['fuku_roi']:>7.1f}%"
            f"  {r['trio_hr']:>7.1f}%"
            f"  {r['trio_roi']:>7.1f}%"
            f"{marker}"
        )

    lines += [
        "",
        "* 単勝的中: 予測1位が1着  単勝回収: 全R1点買い",
        "* 複勝的中: 予測1位が3着内  複勝回収: 配当データあるRのみ",
        "* 3連複的中: 予測上位3頭=実際3着内3頭  3連複回収: 配当データあるRのみ",
        f"* 配当データ取得: {n_avail}/{len(returns_cache)}R",
    ]
    output = "\n".join(lines) + "\n"
    sys.stdout.buffer.write(output.encode("utf-8"))
    sys.stdout.flush()


if __name__ == "__main__":
    main()
