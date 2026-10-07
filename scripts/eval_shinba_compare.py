"""新馬戦モデル 4バリアント比較評価スクリプト

train_shinba_compare.py で学習した v3/v4/v5/v6 モデルを
2025-2026年新馬戦で評価し、単勝・複勝・3連複の的中率・回収率を比較する。

実行:
    python scripts/eval_shinba_compare.py
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
    V3_FEATURES, V4_FEATURES, V5_FEATURES, V6_FEATURES,
    _safe_ped, _dist_band, _compute_cross_score,
)

MODEL_DIR   = os.path.join(paths.PREDICTION_MODEL_PATH, "shinba")
COMPARE_DIR = os.path.join(MODEL_DIR, "compare")
STATS_DIR   = os.path.join(COMPARE_DIR, "stats_tables")

VARIANTS = {
    "v3": V3_FEATURES,
    "v4": V4_FEATURES,
    "v5": V5_FEATURES,
    "v6": V6_FEATURES,
}


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


def load_stats():
    stats = {}
    for name in ("sire_course", "sire_global", "bms_course", "bms_global",
                 "dam", "grandsire", "matdam_sire", "nick", "jockey", "trainer"):
        p = os.path.join(STATS_DIR, f"{name}.csv")
        stats[name] = pd.read_csv(p) if os.path.isfile(p) else pd.DataFrame()
    return stats


def load_models(variant):
    turf_path = os.path.join(COMPARE_DIR, f"turf_{variant}.txt")
    dirt_path = os.path.join(COMPARE_DIR, f"dirt_{variant}.txt")
    turf = lgb.Booster(model_file=turf_path) if os.path.isfile(turf_path) else None
    dirt = lgb.Booster(model_file=dirt_path) if os.path.isfile(dirt_path) else None
    return turf, dirt


def make_row(hid, row, race_type, course_len, place_id, stats, peds_cache):
    dist_band = _dist_band(course_len)
    rt = str(race_type)

    sc_tbl = stats.get("sire_course")
    sg_tbl = stats.get("sire_global")
    bc_tbl = stats.get("bms_course")
    bg_tbl = stats.get("bms_global")
    dam_tbl = stats.get("dam")
    gs_tbl  = stats.get("grandsire")
    ms_tbl  = stats.get("matdam_sire")
    nk_tbl  = stats.get("nick")
    jk_tbl  = stats.get("jockey")
    tr_tbl  = stats.get("trainer")

    ped = peds_cache.get(str(hid), {})
    sire        = ped.get("sire", "")
    dam         = ped.get("dam", "")
    grandsire   = ped.get("grandsire", "")
    bms         = ped.get("bms", "")
    matdam_sire = ped.get("matdam_sire", "")
    cross_score = ped.get("cross_score", 0.0)
    nick_key    = sire + "__" + bms

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

    d = {
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
        "grandsire_win":       _lookup(gs_tbl,  ["grandsire"],   [grandsire],   "global_win"),
        "grandsire_place":     _lookup(gs_tbl,  ["grandsire"],   [grandsire],   "global_place"),
        "jockey_shinba_win":   _lookup(jk_tbl,  ["jockey_id"], [jockey_id],    "win_rate"),
        "jockey_shinba_place": _lookup(jk_tbl,  ["jockey_id"], [jockey_id],    "place_rate"),
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
        # variant-specific
        "nick_win":          _lookup(nk_tbl, ["_nick_key"], [nick_key], "win_rate"),
        "nick_place":        _lookup(nk_tbl, ["_nick_key"], [nick_key], "place_rate"),
        "nick_runs":         _lookup(nk_tbl, ["_nick_key"], [nick_key], "runs", 0.0),
        "cross_score":       float(cross_score),
        "matdam_sire_win":   _lookup(ms_tbl, ["matdam_sire"], [matdam_sire], "global_win"),
        "matdam_sire_place": _lookup(ms_tbl, ["matdam_sire"], [matdam_sire], "global_place"),
    }
    return d


def _get_fuku_payout(ret_df, umaban):
    """複勝配当を返す（100円換算）。該当なし・データなしの場合は0"""
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
    """三連複配当を返す（100円換算）。該当なし・データなしの場合は0"""
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


def evaluate_variant(variant, features, shinba_df, stats, peds_cache, returns_cache):
    turf_model, dirt_model = load_models(variant)
    if turf_model is None and dirt_model is None:
        return None

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

        model = turf_model if race_type != "ダート" else dirt_model
        if model is None:
            continue

        rows = []
        for _, h_row in race_df.iterrows():
            hid = str(h_row.get("horse_id", ""))
            d = make_row(hid, h_row, race_type, course_len, place_id, stats, peds_cache)
            rows.append(d)

        if not rows:
            continue

        X = pd.DataFrame(rows)[features].fillna(-1.0)
        scores = model.predict(X.values, num_iteration=model.best_iteration)

        sorted_idx = np.argsort(scores)[::-1]
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

    if not results:
        return None

    res_df = pd.DataFrame(results)
    n_races = res_df["race_id"].nunique()

    # 単勝
    tan_bets = res_df[res_df["pred_top1"]].copy()
    tan_hits = tan_bets[tan_bets["actual_rank"] == 1]
    hitrate  = len(tan_hits) / len(tan_bets) * 100 if len(tan_bets) > 0 else 0.0
    roi      = tan_hits["tan_odds"].sum() * 100 / (len(tan_bets) * 100) * 100 if len(tan_bets) > 0 else 0.0

    # 複勝的中率
    fuku_bets = res_df[res_df["pred_top1"]].copy()
    fuku_hits = fuku_bets[fuku_bets["actual_rank"] <= 3]
    fuku_hr   = len(fuku_hits) / len(fuku_bets) * 100 if len(fuku_bets) > 0 else 0.0

    # 複勝回収率（配当データがあるレースのみ）
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

    # 3連複 (race単位で集計)
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
        "tan_hits": len(tan_hits),
        "tan_bets": len(tan_bets),
    }


def main():
    print("2025-2026年新馬戦レース結果読み込み中...")
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
    shinba_all = all_df[
        (
            all_df["race_id"].astype(str).str.startswith("2025") |
            all_df["race_id"].astype(str).str.startswith("2026")
        ) &
        (all_df["class"] == "新馬")
    ].copy()
    shinba_2025 = shinba_all[shinba_all["race_id"].astype(str).str.startswith("2025")].copy()
    shinba_2026 = shinba_all[shinba_all["race_id"].astype(str).str.startswith("2026")].copy()
    print(f"  2025: {shinba_2025['race_id'].nunique()}R / {len(shinba_2025)}頭")
    print(f"  2026: {shinba_2026['race_id'].nunique()}R / {len(shinba_2026)}頭")
    print(f"  合計: {shinba_all['race_id'].nunique()}R / {len(shinba_all)}頭")

    print("\n統計テーブル読み込み中...")
    stats = load_stats()
    print(f"  完了  ({len(stats)}テーブル)")

    print("\n血統データ一括キャッシュ構築中...")
    horse_ids = shinba_all["horse_id"].dropna().unique().tolist()
    peds_cache = {}
    for hid in horse_ids:
        try:
            peds_data = horse_peds_dataset_manager.get_horse_peds_dataset(str(hid))
            peds_list = peds_data[str(hid)].tolist()
            sire        = _safe_ped_val(peds_list[0])  if len(peds_list) > 0  else ""
            dam         = _safe_ped_val(peds_list[1])  if len(peds_list) > 1  else ""
            grandsire   = _safe_ped_val(peds_list[2])  if len(peds_list) > 2  else ""
            bms         = _safe_ped_val(peds_list[4])  if len(peds_list) > 4  else ""
            matdam_sire = _safe_ped_val(peds_list[12]) if len(peds_list) > 12 else ""
            cross_score = _compute_cross_score(peds_list)
        except Exception:
            sire = dam = grandsire = bms = matdam_sire = ""
            cross_score = 0.0
        peds_cache[str(hid)] = {
            "sire": sire, "dam": dam, "grandsire": grandsire,
            "bms": bms, "matdam_sire": matdam_sire, "cross_score": cross_score,
        }
    print(f"  {len(peds_cache)}頭分")

    print("\n配当データ読み込み中...")
    all_race_ids = shinba_all["race_id"].dropna().unique().tolist()
    returns_cache = {}
    for rid in tqdm(all_race_ids, desc="配当", ncols=60):
        df = race_info_dataset_manager.get_race_return_csv_for_race(str(rid))
        returns_cache[str(rid)] = df if not df.empty else None
    n_avail = sum(1 for v in returns_cache.values() if v is not None)
    print(f"  配当データ取得: {n_avail}/{len(returns_cache)}R")

    EVAL_SETS = [
        ("2025年", shinba_2025),
        ("2026年", shinba_2026),
        ("2025+2026年", shinba_all),
    ]

    all_results = {}  # {year_label: {variant: res}}
    for year_label, df_eval in EVAL_SETS:
        print(f"\n{year_label} ({df_eval['race_id'].nunique()}R) 評価中...")
        year_res = {}
        for variant, features in VARIANTS.items():
            res = evaluate_variant(variant, features, df_eval, stats, peds_cache, returns_cache)
            if res is not None:
                year_res[variant] = res
        all_results[year_label] = year_res

    LABELS = {"v3": "v3(ベース)", "v4": "v4(ニック)", "v5": "v5(クロス)", "v6": "v6(母母父)"}

    lines = [""]
    for year_label, year_res in all_results.items():
        lines += [
            "=" * 86,
            f"新馬戦専用モデル 比較評価 - {year_label}新馬戦",
            "=" * 86,
            f"{'バリアント':<14} {'R数':>5} {'単勝的中':>8} {'単勝回収':>9} {'複勝的中':>8} {'複勝回収':>9} {'3連複的中':>9} {'3連複回収':>9}",
            "-" * 86,
        ]
        for variant in ["v3", "v4", "v5", "v6"]:
            if variant not in year_res:
                lines.append(f"{LABELS[variant]:<16}  モデルなし")
                continue
            r = year_res[variant]
            lines.append(
                f"{LABELS[variant]:<16} {r['n_races']:>4}R"
                f"  {r['hitrate']:>6.1f}%"
                f"  {r['roi']:>7.1f}%"
                f"  {r['fuku_hr']:>6.1f}%"
                f"  {r['fuku_roi']:>7.1f}%"
                f"  {r['trio_hr']:>7.1f}%"
                f"  {r['trio_roi']:>7.1f}%"
            )
        lines.append("")

    lines += [
        "* 単勝的中: 予測1位が1着  単勝回収: 全R1点買い",
        "* 複勝的中: 予測1位が3着内  複勝回収: 配当データあるRのみ",
        "* 3連複的中: 予測上位3頭=実際3着内3頭  3連複回収: 配当データあるRのみ",
    ]
    output = "\n".join(lines) + "\n"
    sys.stdout.buffer.write(output.encode("utf-8"))
    sys.stdout.flush()


if __name__ == "__main__":
    main()
