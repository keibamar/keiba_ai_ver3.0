"""指数ベースモデル 探索スクリプト

rank固定ではなく確率値で軸・相手・BOXを動的に決定するモデルのパラメータ探索。
ベースライン（rank固定 v2）と複数の指数ベース設定を比較する。

使い方:
    python scripts/explore_score_based.py [--year 2026]
"""

import os
import sys
from pathlib import Path

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.logic.betting.ticket_advisor import (
    recommend_hitrate_v2, recommend_score_based,
)
from src.managers import race_info_dataset_manager, race_result_dataset_manager, race_card_dataset_manager
from src.config import paths


# ── データ取得 ─────────────────────────────────────────────────

def load_race_card(date_str, race_id):
    path = os.path.join(paths.RACE_CARD_DATA_PATH, date_str, f"{race_id}.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path, index_col=0, dtype=str)


def get_top3_and_odds(race_id):
    df = race_result_dataset_manager.get_race_id_result(race_id)
    if df.empty or "着順" not in df.columns or "馬番" not in df.columns:
        return None, {}
    df = df.copy()
    df["着順"] = pd.to_numeric(df["着順"], errors="coerce")
    df["馬番"] = pd.to_numeric(df["馬番"], errors="coerce")
    df = df.dropna(subset=["着順", "馬番"]).sort_values("着順")
    top3 = df[df["着順"] <= 3]["馬番"].astype(int).tolist()
    top3_tuple = tuple(top3[:3]) if len(top3) >= 3 else None

    odds_map: dict = {}
    if "単勝" in df.columns:
        for _, row in df.iterrows():
            try:
                bnum = int(row["馬番"])
                ov = float(pd.to_numeric(str(row["単勝"]), errors="coerce"))
                if ov > 0:
                    odds_map[str(bnum)] = ov
            except Exception:
                pass
    return top3_tuple, odds_map


def get_payout(race_id, bet_type, combo):
    df = race_info_dataset_manager.get_race_return_csv_for_race(race_id)
    if df.empty:
        return 0
    type_map = {"馬連": "馬連", "3連複": "三連複"}
    rows = df[df["式別"] == type_map.get(bet_type, "")]
    if rows.empty:
        return 0
    if bet_type == "馬連":
        target = f"{min(combo)}-{max(combo)}"
    elif bet_type == "3連複":
        target = "-".join(map(str, sorted(combo)))
    else:
        return 0
    m = rows[rows["馬番"].astype(str) == target]
    return int(m.iloc[0]["配当"]) if not m.empty else 0


def is_hit(bet_type, combo, top3):
    if bet_type == "馬連":
        return set(combo) == set(top3[:2])
    if bet_type == "3連複":
        return set(combo) == set(top3)
    return False


# ── 設定グリッド ───────────────────────────────────────────────

def _sb(label, *, um_axis_min, um_box_gap, um_partner_min,
        tp_axis_min, tp_dual_min, tp_box_gap, tp_partner_min,
        um_1jiku_max=25.0, um_box_max=25.0,
        tp_1jiku_max=50.0, tp_2jiku_max=50.0, tp_box_max=50.0,
        um_value_ratio_min=1.0, tp_value_ratio_min=1.0,
        um_min_exp_odds=1.0, tp_min_exp_odds=1.0,
        axis_prob_min=0.0, axis_prob_max=1.0, zones=None):
    return {"label": label, "mode": "sb",
            "um_axis_min": um_axis_min, "um_box_gap": um_box_gap, "um_partner_min": um_partner_min,
            "tp_axis_min": tp_axis_min, "tp_dual_min": tp_dual_min,
            "tp_box_gap": tp_box_gap,   "tp_partner_min": tp_partner_min,
            "um_1jiku_max": um_1jiku_max, "um_box_max": um_box_max,
            "tp_1jiku_max": tp_1jiku_max, "tp_2jiku_max": tp_2jiku_max, "tp_box_max": tp_box_max,
            "um_value_ratio_min": um_value_ratio_min, "tp_value_ratio_min": tp_value_ratio_min,
            "um_min_exp_odds": um_min_exp_odds, "tp_min_exp_odds": tp_min_exp_odds,
            "axis_prob_min": axis_prob_min, "axis_prob_max": axis_prob_max, "zones": zones}


_B_ARGS = dict(um_axis_min=0.12, um_box_gap=0.04, um_partner_min=0.05,
               tp_axis_min=0.12, tp_dual_min=0.09, tp_box_gap=0.04, tp_partner_min=0.04)
_B1_MAX  = dict(um_1jiku_max=25, um_box_max=15, tp_1jiku_max=50, tp_2jiku_max=30, tp_box_max=20)
_B1_BASE = dict(**_B_ARGS, **_B1_MAX)

CONFIGS = [
    {"label": "★rank固定 v2",         "mode": "v2", "v2_kwargs": {}},
    _sb("SB-B 案1(baseline)",          **_B1_BASE),
    _sb("SB-B axis>=0.28",             **_B1_BASE, axis_prob_min=0.28),
    _sb("SB-B axis>=0.30",             **_B1_BASE, axis_prob_min=0.30),
    _sb("SB-B [0.16, 0.20)",           **_B1_BASE, axis_prob_min=0.16, axis_prob_max=0.20),
    _sb("SB-B [0.17, 0.20)",           **_B1_BASE, axis_prob_min=0.17, axis_prob_max=0.20),
    _sb("SB-B [0.18, 0.20)",           **_B1_BASE, axis_prob_min=0.18, axis_prob_max=0.20),
    _sb("SB-B [0.15, 0.21)",           **_B1_BASE, axis_prob_min=0.15, axis_prob_max=0.21),
    _sb("SB-B [0.15, 0.20)",           **_B1_BASE, axis_prob_min=0.15, axis_prob_max=0.20),
    _sb("SB-B [0.16,0.20)+>=0.28",    **_B1_BASE,
        zones=[(0.16, 0.20), (0.28, 1.0)]),
    _sb("SB-B [0.16,0.20)+>=0.25",    **_B1_BASE,
        zones=[(0.16, 0.20), (0.25, 1.0)]),
]


# ── SB レコードのキャッシュ（recommend_score_based を1回だけ計算）─

def _load_race_meta(race_id):
    """race_info から馬場状態・種別・クラスを取得する"""
    try:
        info = race_card_dataset_manager.get_race_info_csv(race_id)
        if info.empty:
            return "?", "?", "?"
        row = info.iloc[0]
        ground = str(row.get("ground_state", "?")).strip()
        rtype  = str(row.get("race_type",    "?")).strip()
        cls    = str(row.get("class",        "?")).strip()
        return ground, rtype, cls
    except Exception:
        return "?", "?", "?"


def precompute_sb_recs(race_records):
    """全レースの recommend_score_based(_B1_BASE) 結果と race_info をキャッシュする。"""
    print("  SB予測計算中...", end="", flush=True)
    # (race_id, date_str, top3, odds_map, rec, ground, rtype, cls)
    cached = []
    for race_id, date_str, df, top3, odds_map in race_records:
        if odds_map:
            df = df.copy()
            df["単勝オッズ"] = df["馬番"].apply(
                lambda x: str(odds_map.get(str(int(float(str(x)))), "0"))
                if pd.to_numeric(str(x), errors="coerce") > 0 else "0"
            )
        rec = recommend_score_based(df, win_odds_col="単勝オッズ", **_B1_BASE)
        ground, rtype, cls = _load_race_meta(race_id)
        cached.append((race_id, date_str, top3, odds_map, rec, ground, rtype, cls))
    print(f" {len(cached)}R 完了\n")
    return cached


# ── axis_prob 分析 ─────────────────────────────────────────────

def analyze_axis_prob(sb_cache, total_races):
    """キャッシュ済み SB 結果を使って axis_prob バケット別 ROI を分析する"""
    rows = []
    for race_id, date_str, top3, odds_map, rec, ground, rtype, cls in sb_cache:
        if not rec:
            continue
        meta = rec.get("_meta", {})
        ap   = meta.get("axis_prob", 0)
        um_tickets = rec.get("馬連",  {}).get("tickets", [])
        tp_tickets = rec.get("3連複", {}).get("tickets", [])
        if not um_tickets and not tp_tickets:
            continue

        um_cost = um_ret = tp_cost = tp_ret = 0
        for t in um_tickets:
            hit = is_hit("馬連", t["組合せ"], top3)
            pay = get_payout(race_id, "馬連", t["組合せ"]) if hit else 0
            um_cost += 100; um_ret += pay
        for t in tp_tickets:
            hit = is_hit("3連複", t["組合せ"], top3)
            pay = get_payout(race_id, "3連複", t["組合せ"]) if hit else 0
            tp_cost += 100; tp_ret += pay

        rows.append({
            "race_id":    race_id,
            "date":       date_str,
            "month":      date_str[:6],
            "axis_prob":  ap,
            "ground":     ground,
            "rtype":      rtype,
            "cls":        cls,
            "um_cost":    um_cost, "um_ret":  um_ret,
            "tp_cost":    tp_cost, "tp_ret":  tp_ret,
            "cost":       um_cost + tp_cost,
            "ret":        um_ret  + tp_ret,
            "n_um":       len(um_tickets),
            "n_tp":       len(tp_tickets),
        })

    if not rows:
        return None

    df = pd.DataFrame(rows)
    print(f"  SB提案: {len(df)}R / {total_races}R ({len(df)/total_races*100:.1f}%)\n")

    # ── バケット別 ROI ──────────────────────────────────────────
    buckets = [0.00, 0.12, 0.14, 0.16, 0.18, 0.20, 0.22, 0.25, 0.28, 0.32, 1.01]
    print(f"  ── axis_prob バケット別 ROI ──────────────────────────────────")
    print(f"  {'範囲':<18} {'R数':>5} {'提案%':>7} {'点/R':>6} {'ROI':>8}")
    print(f"  {'-'*50}")
    for i in range(len(buckets)-1):
        lo = buckets[i]; hi = buckets[i+1]
        sub = df[(df["axis_prob"] >= lo) & (df["axis_prob"] < hi)]
        if sub.empty:
            continue
        r    = len(sub)
        cost = sub["cost"].sum(); ret = sub["ret"].sum()
        roi  = ret / cost * 100 if cost else 0
        pct  = r / total_races * 100
        avg_t= (sub["n_um"].sum() + sub["n_tp"].sum()) / r
        label = f"[{lo:.2f}, {hi:.2f})"
        print(f"  {label:<18} {r:>5}R {pct:>6.1f}% {avg_t:>5.1f}点 {roi:>7.1f}%")

    # ── 累積 ROI（axis_prob >= X）──────────────────────────────
    print(f"\n  ── axis_prob >= X での累積 ROI ──────────────────────────────")
    print(f"  {'閾値':>10} {'提案R':>7} {'提案%':>7} {'点/R':>6} {'ROI':>8}")
    print(f"  {'-'*45}")
    for th in [0.12, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19,
               0.20, 0.21, 0.22, 0.23, 0.24, 0.25, 0.27, 0.28, 0.30]:
        sub = df[df["axis_prob"] >= th]
        if sub.empty:
            break
        r    = len(sub)
        cost = sub["cost"].sum(); ret = sub["ret"].sum()
        roi  = ret / cost * 100 if cost else 0
        pct  = r / total_races * 100
        avg_t= (sub["n_um"].sum() + sub["n_tp"].sum()) / r
        flag = " HIT" if roi >= 100 else ""
        print(f"  >= {th:.2f}       {r:>6}R {pct:>6.1f}% {avg_t:>5.1f}点 {roi:>7.1f}%{flag}")

    print()

    # ── 四半期別 クロス検証 ─────────────────────────────────────
    print(f"  ── 四半期別 クロス検証 ──────────────────────────────────────────────────────────")
    print(f"  {'期間':<16} {'全提案':>6}  {'[0.18,0.20)':>12}  {'[0.16,0.20)':>12}  {'>=0.28':>12}  {'>=0.30':>12}")
    print(f"  {'-'*74}")

    quarters = [
        ("2026 Q1(1-3月)", "20260101", "20260401"),
        ("2026 Q2(4-6月)", "20260401", "20260701"),
        ("2026 Q3(7月~)",  "20260701", "20270101"),
    ]
    for qname, q_lo, q_hi in quarters:
        sub = df[(df["date"] >= q_lo) & (df["date"] < q_hi)]
        if sub.empty:
            continue
        q_total = len(sub)

        def band_roi(lo, hi, s=sub):
            ss = s[(s["axis_prob"] >= lo) & (s["axis_prob"] < hi)]
            if ss.empty:
                return f"{'---':>3}R    ---%"
            cost = ss["cost"].sum(); ret = ss["ret"].sum()
            roi  = ret / cost * 100 if cost else 0
            mark = "*" if roi >= 100 else " "
            return f"{len(ss):>3}R {roi:>6.0f}%{mark}"

        b1 = band_roi(0.18, 0.20)
        b2 = band_roi(0.16, 0.20)
        b3 = band_roi(0.28, 1.00)
        b4 = band_roi(0.30, 1.00)
        print(f"  {qname:<16} {q_total:>5}R  {b1:>12}  {b2:>12}  {b3:>12}  {b4:>12}")

    print()
    return df


# ── 深掘り分析（月別・馬場別・券種別）──────────────────────────────

def deep_analyze(df: pd.DataFrame, total_races: int):
    """analyze_axis_prob が返す per-race DataFrame を使って3軸を深掘りする"""
    if df is None or df.empty:
        return

    # ── ヘルパー ──────────────────────────────────────────────────
    def roi_str(cost, ret):
        if cost == 0:
            return "   ---"
        return f"{ret/cost*100:>6.0f}%"

    def band(df, lo, hi):
        return df[(df["axis_prob"] >= lo) & (df["axis_prob"] < hi)]

    BANDS = {
        "全体":       (0.00, 1.00),
        "[0.18,0.20)": (0.18, 0.20),
        "[0.16,0.20)": (0.16, 0.20),
        ">=0.28":      (0.28, 1.00),
    }

    def show_breakdown(label, group_col, groups, df_source):
        print(f"  ── {label} ────────────────────────────────────────────────────────────────────")
        header = f"  {'区分':<14}"
        for bn in BANDS:
            header += f"  {bn:>12}"
        print(header)
        print(f"  {'-'*70}")
        for g in groups:
            sub = df_source[df_source[group_col] == g]
            if sub.empty:
                continue
            row = f"  {str(g):<14}"
            for bn, (lo, hi) in BANDS.items():
                s = band(sub, lo, hi)
                if s.empty:
                    row += f"  {'---':>12}"
                else:
                    r = roi_str(s["cost"].sum(), s["ret"].sum())
                    row += f"  {str(len(s))+'R '+r:>12}"
            print(row)
        print()

    # ── 1. 月別 ──────────────────────────────────────────────────
    months = sorted(df["month"].unique())
    show_breakdown("月別 ROI", "month", months, df)

    # ── 2. 馬場状態別 ─────────────────────────────────────────────
    ground_order = ["良", "稍重", "重", "不良", "?"]
    grounds = [g for g in ground_order if g in df["ground"].unique()]
    show_breakdown("馬場状態別 ROI", "ground", grounds, df)

    # ── 3. 馬場種別（芝 / ダート）─────────────────────────────────
    rtype_order = ["芝", "ダート", "?"]
    rtypes = [r for r in rtype_order if r in df["rtype"].unique()]
    show_breakdown("馬場種別 ROI（芝 / ダート）", "rtype", rtypes, df)

    # ── 4. 券種別 ROI を四半期×フィルター で展開 ──────────────────
    print(f"  ── 券種別 × 四半期 ROI ──────────────────────────────────────────────────────────")
    quarters = [
        ("Q1(1-3月)", "20260101", "20260401"),
        ("Q2(4-6月)", "20260401", "20260701"),
        ("Q3(7月~)",  "20260701", "20270101"),
        ("全体",      "20260101", "20270101"),
    ]
    print(f"  {'フィルター':<16} {'期間':<12}  {'馬連 R/ROI':>14}  {'3複 R/ROI':>14}  {'合計ROI':>8}")
    print(f"  {'-'*72}")
    for bn, (lo, hi) in BANDS.items():
        b = band(df, lo, hi)
        for qname, q_lo, q_hi in quarters:
            sub = b[(b["date"] >= q_lo) & (b["date"] < q_hi)]
            if sub.empty:
                continue
            um_r = roi_str(sub["um_cost"].sum(), sub["um_ret"].sum())
            tp_r = roi_str(sub["tp_cost"].sum(), sub["tp_ret"].sum())
            tot  = roi_str(sub["cost"].sum(),    sub["ret"].sum())
            um_n = (sub["um_cost"] > 0).sum()
            tp_n = (sub["tp_cost"] > 0).sum()
            print(f"  {bn:<16} {qname:<12}  {str(um_n)+'R '+um_r:>14}  {str(tp_n)+'R '+tp_r:>14}  {tot:>8}")
        print()

    # ── 5. ダート良 / 芝良 / ダート道悪 / 芝道悪 詳細分析 ──────────
    GOOD  = {"良"}
    MUDDY = {"稍重", "重", "不良"}

    df["track_cat"] = df.apply(
        lambda r: (
            "ダート良"  if r["rtype"] == "ダート" and r["ground"] in GOOD  else
            "芝良"      if r["rtype"] == "芝"     and r["ground"] in GOOD  else
            "ダート道悪" if r["rtype"] == "ダート" and r["ground"] in MUDDY else
            "芝道悪"    if r["rtype"] == "芝"     and r["ground"] in MUDDY else
            "その他"
        ), axis=1
    )

    # axis_prob バンド別に各カテゴリのROIを展開
    DETAIL_BANDS = [
        ("全体",         0.00, 1.00),
        ("[0.14,0.16)",  0.14, 0.16),
        ("[0.16,0.18)",  0.16, 0.18),
        ("[0.18,0.20)",  0.18, 0.20),
        ("[0.20,0.25)",  0.20, 0.25),
        ("[0.25,0.28)",  0.25, 0.28),
        (">=0.28",       0.28, 1.00),
        (">=0.30",       0.30, 1.00),
    ]
    CAT_ORDER = ["ダート良", "芝良", "ダート道悪", "芝道悪"]
    CAT_COLS  = {c: df[df["track_cat"] == c] for c in CAT_ORDER}

    W2 = 100
    print(f"\n{'='*W2}")
    print(f"  ダート良/芝良/ダート道悪/芝道悪 × axis_prob バンド別 ROI  (HIT=100%+)")
    print(f"{'='*W2}")

    hdr = f"  {'バンド':<14}"
    for c in CAT_ORDER:
        n = len(CAT_COLS[c])
        hdr += f"  {c+'('+str(n)+'R)':<18}"
    print(hdr)
    print(f"  {'-'*90}")

    for bn, lo, hi in DETAIL_BANDS:
        row = f"  {bn:<14}"
        for c in CAT_ORDER:
            sub = band(CAT_COLS[c], lo, hi)
            if sub.empty:
                row += f"  {'  ---  ---':>18}"
            else:
                cost = sub["cost"].sum(); ret = sub["ret"].sum()
                roi  = ret / cost * 100 if cost else 0
                flag = "*" if roi >= 100 else " "
                row += f"  {str(len(sub))+'R '+f'{roi:.0f}%'+flag:>18}"
        print(row)

    # 馬連と3連複を分けて再表示
    print(f"\n  ── 馬連ROI ─────────────────────────────────────────────────────────")
    hdr2 = f"  {'バンド':<14}"
    for c in CAT_ORDER:
        hdr2 += f"  {c:<18}"
    print(hdr2)
    print(f"  {'-'*88}")
    for bn, lo, hi in DETAIL_BANDS:
        row = f"  {bn:<14}"
        for c in CAT_ORDER:
            sub = band(CAT_COLS[c], lo, hi)
            s = sub[sub["um_cost"] > 0]
            if s.empty:
                row += f"  {'  ---':>18}"
            else:
                r = roi_str(s["um_cost"].sum(), s["um_ret"].sum())
                row += f"  {str(len(s))+'R '+r:>18}"
        print(row)

    print(f"\n  ── 3連複ROI ────────────────────────────────────────────────────────")
    print(hdr2)
    print(f"  {'-'*88}")
    for bn, lo, hi in DETAIL_BANDS:
        row = f"  {bn:<14}"
        for c in CAT_ORDER:
            sub = band(CAT_COLS[c], lo, hi)
            s = sub[sub["tp_cost"] > 0]
            if s.empty:
                row += f"  {'  ---':>18}"
            else:
                r = roi_str(s["tp_cost"].sum(), s["tp_ret"].sum())
                row += f"  {str(len(s))+'R '+r:>18}"
        print(row)

    print()


# ── バックテスト ────────────────────────────────────────────────

def backtest_one(cfg, race_records, sb_cache, total_races):
    um = {"bets":0,"hits":0,"hit_races":0,"cost":0,"ret":0,"prop":0}
    tp = {"bets":0,"hits":0,"hit_races":0,"cost":0,"ret":0,"prop":0}
    combined_hit = 0
    proposed_races = 0
    method_counts = {"um": {}, "tp": {}}
    axis_prob_min = cfg.get("axis_prob_min", 0.0)
    axis_prob_max = cfg.get("axis_prob_max", 1.0)
    axis_zones    = cfg.get("zones", None)

    if cfg["mode"] == "v2":
        # v2 は race_records を直接使う
        for race_id, date_str, df, top3, odds_map in race_records:
            rec = recommend_hitrate_v2(df, **cfg.get("v2_kwargs", {}))  # type: ignore
            if not rec:
                continue
            um_tickets = rec.get("馬連",  {}).get("tickets", [])
            tp_tickets = rec.get("3連複", {}).get("tickets", [])
            if not um_tickets and not tp_tickets:
                continue
            meta = rec.get("_meta", {})
            um_m = meta.get("um_method", rec.get("馬連",{}).get("選択方式",""))
            tp_m = meta.get("tp_method", rec.get("3連複",{}).get("選択方式",""))
            proposed_races += 1
            method_counts["um"][um_m] = method_counts["um"].get(um_m, 0) + 1
            method_counts["tp"][tp_m] = method_counts["tp"].get(tp_m, 0) + 1
            race_um = race_tp = False
            for t in um_tickets:
                hit = is_hit("馬連", t["組合せ"], top3)
                pay = get_payout(race_id, "馬連", t["組合せ"]) if hit else 0
                um["bets"] += 1; um["cost"] += 100
                if hit: um["hits"] += 1; um["ret"] += pay; race_um = True
            if um_tickets: um["prop"] += 1
            if race_um: um["hit_races"] += 1
            for t in tp_tickets:
                hit = is_hit("3連複", t["組合せ"], top3)
                pay = get_payout(race_id, "3連複", t["組合せ"]) if hit else 0
                tp["bets"] += 1; tp["cost"] += 100
                if hit: tp["hits"] += 1; tp["ret"] += pay; race_tp = True
            if tp_tickets: tp["prop"] += 1
            if race_tp: tp["hit_races"] += 1
            if race_um or race_tp:
                combined_hit += 1

    else:
        # SB 系はキャッシュ済み結果を使ってフィルターだけ適用
        for race_id, date_str, top3, odds_map, rec, ground, rtype, cls in sb_cache:
            if not rec:
                continue
            meta = rec.get("_meta", {})
            ap = meta.get("axis_prob", 0)

            # axis_prob フィルター
            if axis_zones is not None:
                if not any(lo <= ap < hi for lo, hi in axis_zones):
                    continue
            else:
                if ap < axis_prob_min or ap >= axis_prob_max:
                    continue

            um_m = meta.get("um_method", rec.get("馬連",{}).get("選択方式",""))
            tp_m = meta.get("tp_method", rec.get("3連複",{}).get("選択方式",""))
            um_tickets = rec.get("馬連",  {}).get("tickets", [])
            tp_tickets = rec.get("3連複", {}).get("tickets", [])
            if not um_tickets and not tp_tickets:
                continue

            proposed_races += 1
            method_counts["um"][um_m] = method_counts["um"].get(um_m, 0) + 1
            method_counts["tp"][tp_m] = method_counts["tp"].get(tp_m, 0) + 1

            race_um = race_tp = False
            for t in um_tickets:
                hit = is_hit("馬連", t["組合せ"], top3)
                pay = get_payout(race_id, "馬連", t["組合せ"]) if hit else 0
                um["bets"] += 1; um["cost"] += 100
                if hit: um["hits"] += 1; um["ret"] += pay; race_um = True
            if um_tickets: um["prop"] += 1
            if race_um: um["hit_races"] += 1
            for t in tp_tickets:
                hit = is_hit("3連複", t["組合せ"], top3)
                pay = get_payout(race_id, "3連複", t["組合せ"]) if hit else 0
                tp["bets"] += 1; tp["cost"] += 100
                if hit: tp["hits"] += 1; tp["ret"] += pay; race_tp = True
            if tp_tickets: tp["prop"] += 1
            if race_tp: tp["hit_races"] += 1
            if race_um or race_tp:
                combined_hit += 1

    total_cost = um["cost"] + tp["cost"]
    total_ret  = um["ret"]  + tp["ret"]
    avg_bets   = (um["bets"] + tp["bets"]) / total_races if total_races else 0
    roi        = total_ret / total_cost * 100 if total_cost else 0
    um_roi     = um["ret"] / um["cost"] * 100 if um["cost"] else 0
    tp_roi     = tp["ret"] / tp["cost"] * 100 if tp["cost"] else 0
    prop_pct   = proposed_races / total_races * 100 if total_races else 0

    return {
        "avg_bets": avg_bets,
        "um_hr":  um["hit_races"] / total_races * 100,
        "tp_hr":  tp["hit_races"] / total_races * 100,
        "comb_hr": combined_hit / total_races * 100,
        "roi": roi, "um_roi": um_roi, "tp_roi": tp_roi,
        "um_prop": um["prop"], "tp_prop": tp["prop"],
        "proposed_races": proposed_races, "prop_pct": prop_pct,
        "method_counts": method_counts,
    }


def run(year="2026"):
    race_card_root = paths.RACE_CARD_DATA_PATH
    all_dates = sorted([
        d for d in os.listdir(race_card_root)
        if d.startswith(year) and os.path.isdir(os.path.join(race_card_root, d))
    ])

    print(f"  データ読み込み中...", end="", flush=True)
    total_races = 0; skipped = 0
    race_records = []
    for date_str in all_dates:
        day_dir = os.path.join(race_card_root, date_str)
        for fname in sorted(f for f in os.listdir(day_dir) if f.endswith(".csv")):
            race_id = fname.replace(".csv", "")
            df = load_race_card(date_str, race_id)
            if df.empty:
                continue
            needed = ["score", "score_hitrate", "score_value", "馬番"]
            if not all(c in df.columns for c in needed):
                skipped += 1; continue
            if pd.to_numeric(df["score"], errors="coerce").isna().all():
                skipped += 1; continue
            top3, odds_map = get_top3_and_odds(race_id)
            if top3 is None:
                skipped += 1; continue
            total_races += 1
            race_records.append((race_id, date_str, df, top3, odds_map))
    print(f" {total_races}R (skip:{skipped})\n")

    # ── SB予測を一括キャッシュ（全設定で共有）────────────────────
    sb_cache = precompute_sb_recs(race_records)

    # ── axis_prob 分析 ────────────────────────────────────────────
    ap_df = analyze_axis_prob(sb_cache, total_races)
    deep_analyze(ap_df, total_races)

    # ── 設定別 比較テーブル ────────────────────────────────────────
    W = 106
    print("=" * W)
    print(f"  {year} [axis_prob フィルター比較]  total:{total_races}R")
    print("=" * W)
    print(f"  {'設定':<26} {'提案R%':>7} {'点/R':>5} {'馬連的中%':>9} "
          f"{'3複的中%':>9} {'合計的中%':>10} "
          f"{'馬連ROI':>8} {'3複ROI':>8} {'全ROI':>7}")
    print(f"  {'-'*(W-2)}")

    results = []
    for cfg in CONFIGS:
        r = backtest_one(cfg, race_records, sb_cache, total_races)
        results.append((cfg, r))
        flag = " ***" if r["roi"] >= 100 else ""
        print(f"  {cfg['label']:<26} {r['prop_pct']:>6.1f}% {r['avg_bets']:>5.1f} "
              f"{r['um_hr']:>8.1f}% {r['tp_hr']:>8.1f}% {r['comb_hr']:>9.1f}% "
              f"{r['um_roi']:>7.1f}% {r['tp_roi']:>7.1f}% {r['roi']:>6.1f}%{flag}")

    print(f"  {'='*(W-2)}")
    print()

    # ── 方式別内訳 ────────────────────────────────────────────────
    print(f"  ── 方式別内訳 ──")
    print(f"  {'設定':<26} {'馬連:1軸':>8} {'馬連:BOX':>8} {'馬連:なし':>9} "
          f"{'3複:1軸':>8} {'3複:2軸':>8} {'3複:BOX':>8} {'3複:なし':>9}")
    print(f"  {'-'*(W-2)}")
    for cfg, r in results:
        if cfg["mode"] not in ("sb", "sb_zones"):
            continue
        mc = r["method_counts"]
        um_1 = mc["um"].get("1頭軸", 0); um_b = mc["um"].get("BOX", 0)
        um_n = total_races - um_1 - um_b
        tp_1 = mc["tp"].get("1頭軸", 0); tp_2 = mc["tp"].get("2頭軸", 0)
        tp_b = mc["tp"].get("BOX", 0);    tp_n = total_races - tp_1 - tp_2 - tp_b
        print(f"  {cfg['label']:<26} "
              f"{um_1:>7}R {um_b:>7}R {um_n:>8}R "
              f"{tp_1:>7}R {tp_2:>7}R {tp_b:>7}R {tp_n:>8}R")

    print()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", default="2026")
    args = parser.parse_args()
    run(args.year)
