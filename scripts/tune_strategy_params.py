"""
戦略別パラメータチューニング

ダート×良馬場 / 芝×良馬場 / 道悪 の3戦略に対してグリッドサーチを実行し
最良パラメータ組み合わせを探す。

使い方:
    python scripts/tune_strategy_params.py [--year 2026]
"""

import os
import sys
import itertools
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import paths
from src.logic.betting.ticket_advisor import recommend_score_based

from scripts.explore_score_based import (
    load_race_card, get_top3_and_odds, get_payout, is_hit, _load_race_meta,
)

# ── 戦略定義 ─────────────────────────────────────────────────────

STRATEGIES = {
    "ダート良": dict(rtype="ダート", muddy=False, ap_lo=0.18, ap_hi=0.20),
    "芝良":     dict(rtype="芝",     muddy=False, ap_lo=0.30, ap_hi=1.01),
    "道悪":     dict(rtype=None,     muddy=True,  ap_lo=0.00, ap_hi=1.01),
}

# ── 固定パラメータ（axis 閾値・gap 等） ────────────────────────────

FIXED = dict(
    um_axis_min=0.12,
    um_box_gap=0.04,
    tp_axis_min=0.12,
    tp_dual_min=0.09,
    tp_box_gap=0.04,
    tp_2jiku_max=30.0,
    um_value_ratio_min=1.0,
    tp_value_ratio_min=1.0,
    um_min_exp_odds=1.0,
    tp_min_exp_odds=1.0,
)

# ── グリッドサーチ対象パラメータ ──────────────────────────────────

GRID = {
    "um_partner_min": [0.03, 0.05, 0.07, 0.10],
    "um_1jiku_max":   [20.0, 30.0],
    "um_box_max":     [12.0, 15.0, 20.0],
    "tp_partner_min": [0.03, 0.05, 0.07],
    "tp_1jiku_max":   [30.0, 60.0],
    "tp_box_max":     [15.0, 20.0, 25.0],
}


def _make_configs():
    keys   = list(GRID.keys())
    values = list(GRID.values())
    configs = []
    for combo in itertools.product(*values):
        cfg = dict(zip(keys, combo))
        cfg.update(FIXED)
        configs.append(cfg)
    return configs


# ── レース評価（1レース × 1パラメータセット） ─────────────────────

def eval_race(df, top3, odds_map, params):
    """recommend_score_based を実行して (ap, um_cost, um_ret, tp_cost, tp_ret) を返す"""
    _df = df
    if odds_map:
        _df = df.copy()
        _df["単勝オッズ"] = _df["馬番"].apply(
            lambda x: str(odds_map.get(str(int(float(str(x)))), "0"))
            if pd.to_numeric(str(x), errors="coerce") > 0 else "0"
        )
    rec = recommend_score_based(_df, win_odds_col="単勝オッズ", **params)
    if not rec:
        return None

    meta = rec.get("_meta", {})
    ap   = meta.get("axis_prob", 0)
    um_tickets = rec.get("馬連",  {}).get("tickets", [])
    tp_tickets = rec.get("3連複", {}).get("tickets", [])
    if not um_tickets and not tp_tickets:
        return None

    um_cost = um_ret = 0
    for t in um_tickets:
        hit = is_hit("馬連", t["組合せ"], top3)
        pay = get_payout(race_id_dummy, "馬連", t["組合せ"]) if hit else 0
        um_cost += 100; um_ret += pay

    tp_cost = tp_ret = 0
    for t in tp_tickets:
        hit = is_hit("3連複", t["組合せ"], top3)
        pay = get_payout(race_id_dummy, "3連複", t["組合せ"]) if hit else 0
        tp_cost += 100; tp_ret += pay

    return ap, um_cost, um_ret, tp_cost, tp_ret


# ↑ race_id は get_payout に必要なので、以下の本体で渡す形にする


def eval_race_with_id(race_id, df, top3, odds_map, params):
    _df = df
    if odds_map:
        _df = df.copy()
        _df["単勝オッズ"] = _df["馬番"].apply(
            lambda x: str(odds_map.get(str(int(float(str(x)))), "0"))
            if pd.to_numeric(str(x), errors="coerce") > 0 else "0"
        )
    rec = recommend_score_based(_df, win_odds_col="単勝オッズ", **params)
    if not rec:
        return None

    meta = rec.get("_meta", {})
    ap   = meta.get("axis_prob", 0)
    um_tickets = rec.get("馬連",  {}).get("tickets", [])
    tp_tickets = rec.get("3連複", {}).get("tickets", [])
    if not um_tickets and not tp_tickets:
        return None

    um_cost = um_ret = 0
    for t in um_tickets:
        hit = is_hit("馬連", t["組合せ"], top3)
        pay = get_payout(race_id, "馬連", t["組合せ"]) if hit else 0
        um_cost += 100; um_ret += pay

    tp_cost = tp_ret = 0
    for t in tp_tickets:
        hit = is_hit("3連複", t["組合せ"], top3)
        pay = get_payout(race_id, "3連複", t["組合せ"]) if hit else 0
        tp_cost += 100; tp_ret += pay

    return ap, um_cost, um_ret, tp_cost, tp_ret


# ── メイン ────────────────────────────────────────────────────────

def run(year="2026"):
    race_card_root = paths.RACE_CARD_DATA_PATH
    all_dates = sorted([
        d for d in os.listdir(race_card_root)
        if d.startswith(year) and os.path.isdir(os.path.join(race_card_root, d))
    ])

    # ── Step1: レース読み込み & race_info 取得 ────────────────────
    print("  データ読み込み中...", end="", flush=True)
    race_meta = []  # (race_id, date_str, df, top3, odds_map, rtype, ground)
    skipped = 0
    for date_str in all_dates:
        day_dir = os.path.join(race_card_root, date_str)
        for fname in sorted(f for f in os.listdir(day_dir) if f.endswith(".csv")):
            race_id = fname.replace(".csv", "")
            df = load_race_card(date_str, race_id)
            if df.empty:
                continue
            needed = ["score_hitrate", "馬番"]
            if not all(c in df.columns for c in needed):
                skipped += 1; continue
            if pd.to_numeric(df["score_hitrate"], errors="coerce").isna().all():
                skipped += 1; continue
            top3, odds_map = get_top3_and_odds(race_id)
            if top3 is None:
                skipped += 1; continue
            ground, rtype, cls = _load_race_meta(race_id)
            race_meta.append((race_id, date_str, df, top3, odds_map, rtype, ground))
    print(f" {len(race_meta)}R (skip:{skipped})\n")

    # ── Step2: axis_prob を1回計算して各レースに紐付け ───────────
    # axis_prob はスコア分布から決まるため、パラメータに依存しない
    # → FIXED + 標準値で1回だけ計算してキャッシュ
    print("  axis_prob 計算中...", end="", flush=True)
    _ref_params = dict(**FIXED,
                       um_partner_min=0.05, um_1jiku_max=25, um_box_max=15,
                       tp_partner_min=0.04, tp_1jiku_max=50, tp_box_max=20)
    ap_cache = {}  # race_id -> axis_prob
    for race_id, date_str, df, top3, odds_map, rtype, ground in race_meta:
        _df = df
        if odds_map:
            _df = df.copy()
            _df["単勝オッズ"] = _df["馬番"].apply(
                lambda x: str(odds_map.get(str(int(float(str(x)))), "0"))
                if pd.to_numeric(str(x), errors="coerce") > 0 else "0"
            )
        rec = recommend_score_based(_df, win_odds_col="単勝オッズ", **_ref_params)
        ap  = rec.get("_meta", {}).get("axis_prob", 0) if rec else 0
        ap_cache[race_id] = ap
    print(f" {len(ap_cache)}R 完了\n")

    # ── Step3: 戦略ごとにレースをフィルタ ────────────────────────
    MUDDY = {"稍重", "重", "不良"}
    strategy_races: dict[str, list] = {}
    for sname, sdef in STRATEGIES.items():
        filtered = []
        for race_id, date_str, df, top3, odds_map, rtype, ground in race_meta:
            if sdef["muddy"]:
                if ground not in MUDDY:
                    continue
            else:
                if ground != "良":
                    continue
                if sdef["rtype"] and rtype != sdef["rtype"]:
                    continue
            ap = ap_cache.get(race_id, 0)
            if not (sdef["ap_lo"] <= ap < sdef["ap_hi"]):
                continue
            filtered.append((race_id, df, top3, odds_map))
        strategy_races[sname] = filtered
        print(f"  {sname}: {len(filtered)}R")
    print()

    # ── Step4: グリッドサーチ ─────────────────────────────────────
    configs = _make_configs()
    total_configs = len(configs)
    print(f"  グリッドサーチ: {total_configs} configs\n")

    for sname, races in strategy_races.items():
        if not races:
            print(f"  [{sname}] 対象レースなし\n")
            continue

        print(f"  {'='*80}")
        print(f"  [{sname}] {len(races)}R × {total_configs}configs 評価中...")
        print(f"  {'='*80}")

        results = []
        for i, cfg in enumerate(configs):
            if (i + 1) % 50 == 0:
                print(f"    ... {i+1}/{total_configs}", flush=True)

            um_cost_t = um_ret_t = tp_cost_t = tp_ret_t = 0
            n_races = 0

            for race_id, df, top3, odds_map in races:
                r = eval_race_with_id(race_id, df, top3, odds_map, cfg)
                if r is None:
                    continue
                _, um_cost, um_ret, tp_cost, tp_ret = r
                um_cost_t += um_cost; um_ret_t  += um_ret
                tp_cost_t += tp_cost; tp_ret_t  += tp_ret
                n_races += 1

            cost_t = um_cost_t + tp_cost_t
            ret_t  = um_ret_t  + tp_ret_t
            if cost_t == 0:
                continue

            roi    = ret_t  / cost_t  * 100
            um_roi = um_ret_t / um_cost_t * 100 if um_cost_t else 0
            tp_roi = tp_ret_t / tp_cost_t * 100 if tp_cost_t else 0
            avg_pt = (um_cost_t + tp_cost_t) / n_races / 100 if n_races else 0

            results.append({
                "cfg": cfg,
                "n_races": n_races,
                "roi": roi,
                "um_roi": um_roi,
                "tp_roi": tp_roi,
                "avg_pt": avg_pt,
                "cost": cost_t,
                "ret": ret_t,
            })

        if not results:
            print(f"  [{sname}] 結果なし\n")
            continue

        results.sort(key=lambda x: -x["roi"])

        # TOP 20 表示
        print(f"\n  ── {sname} TOP 20 (ROI降順, n_races>={max(5, len(races)//5)}) ─────")
        min_r = max(5, len(races) // 5)
        shown = 0
        print(f"  {'#':>3}  {'ROI':>7}  {'馬連ROI':>7}  {'3複ROI':>7}  "
              f"{'R数':>5}  {'点/R':>5}  "
              f"  um_prtn  um1jiku um_box  tp_prtn tp1jiku tp_box")
        print(f"  {'-'*105}")
        for row in results:
            if row["n_races"] < min_r:
                continue
            if shown >= 20:
                break
            c = row["cfg"]
            shown += 1
            print(
                f"  {shown:>3}  {row['roi']:>7.1f}%  {row['um_roi']:>7.1f}%  "
                f"{row['tp_roi']:>7.1f}%  {row['n_races']:>5}R {row['avg_pt']:>5.1f}点  "
                f"  {c['um_partner_min']:.2f}  {c['um_1jiku_max']:>7.0f}  "
                f"{c['um_box_max']:>6.0f}  {c['tp_partner_min']:.2f}  "
                f"{c['tp_1jiku_max']:>7.0f}  {c['tp_box_max']:>6.0f}"
            )

        # ベースライン（SB_B_PARAMSと同等）の順位を表示
        baseline = dict(**FIXED,
                        um_partner_min=0.05, um_1jiku_max=25, um_box_max=15,
                        tp_partner_min=0.04, tp_1jiku_max=50, tp_box_max=20)
        for rank, row in enumerate(results, 1):
            c = row["cfg"]
            if (abs(c["um_partner_min"] - baseline["um_partner_min"]) < 0.001 and
                abs(c["um_1jiku_max"]   - baseline["um_1jiku_max"])   < 0.1   and
                abs(c["um_box_max"]     - baseline["um_box_max"])     < 0.1   and
                abs(c["tp_partner_min"] - baseline["tp_partner_min"]) < 0.001 and
                abs(c["tp_1jiku_max"]   - baseline["tp_1jiku_max"])   < 0.1   and
                abs(c["tp_box_max"]     - baseline["tp_box_max"])     < 0.1):
                print(f"\n  [ベースライン] 順位: {rank}/{len(results)} "
                      f"ROI={row['roi']:.1f}% ({row['n_races']}R)")
                break
        print()

    print("  グリッドサーチ完了\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", default="2026")
    args = parser.parse_args()
    run(args.year)
