"""馬券検討モデルのバックテスト

2026年の実績データを使い、recommend_ticketsが提案した買い目が
実際に当たれば何倍になったかを集計する。

使い方:
    python scripts/backtest_ticket_advisor.py [--year 2026]
"""

import os
import sys
import argparse
from pathlib import Path
from collections import defaultdict

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.logic.betting.ticket_advisor import (
    recommend_tickets, recommend_best_tickets, recommend_hitrate_tickets,
    recommend_hitrate_v2,
    BASIC_MAX, LONGSHOT_MAX,
)
from src.managers import race_info_dataset_manager, race_result_dataset_manager
from src.config import paths


# --- データ読み込み ---

def load_race_card(date_str: str, race_id: str) -> pd.DataFrame:
    path = os.path.join(paths.RACE_CARD_DATA_PATH, date_str, f"{race_id}.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path, index_col=0, dtype=str)


def get_top3(race_id: str):
    """レース結果から (1着馬番, 2着馬番, 3着馬番) を返す。取得失敗時は None。"""
    df = race_result_dataset_manager.get_race_id_result(race_id)
    if df.empty or "着順" not in df.columns or "馬番" not in df.columns:
        return None
    df = df.copy()
    df["着順"] = pd.to_numeric(df["着順"], errors="coerce")
    df["馬番"] = pd.to_numeric(df["馬番"], errors="coerce")
    df = df.dropna(subset=["着順", "馬番"]).sort_values("着順")
    top3 = df[df["着順"] <= 3]["馬番"].astype(int).tolist()
    return tuple(top3[:3]) if len(top3) >= 3 else None


def get_payout(race_id: str, bet_type: str, combo: tuple):
    """払戻額（100円あたり）を取得。外れ or データなしは 0。"""
    df = race_info_dataset_manager.get_race_return_csv_for_race(race_id)
    if df.empty:
        return 0

    type_map = {"馬連": "馬連", "3連複": "三連複", "3連単": "三連単"}
    col_name = type_map.get(bet_type)
    rows = df[df["式別"] == col_name]
    if rows.empty:
        return 0

    if bet_type == "馬連":
        target = f"{min(combo)}-{max(combo)}"
    elif bet_type == "3連複":
        target = "-".join(map(str, sorted(combo)))
    elif bet_type == "3連単":
        target = "→".join(map(str, combo))
    else:
        return 0

    match = rows[rows["馬番"].astype(str) == target]
    if match.empty:
        return 0
    return int(match.iloc[0]["配当"])


# --- 的中チェック ---

def is_hit(bet_type: str, combo: tuple, top3: tuple) -> bool:
    if bet_type == "馬連":
        return set(combo) == set(top3[:2])
    elif bet_type == "3連複":
        return set(combo) == set(top3)
    elif bet_type == "3連単":
        return combo == top3
    return False


# --- バックテスト本体 ---

def run_backtest(year: str):
    """従来モード: 全券種を並行して提案"""
    race_card_root = paths.RACE_CARD_DATA_PATH
    all_dates = sorted([
        d for d in os.listdir(race_card_root)
        if d.startswith(year) and os.path.isdir(os.path.join(race_card_root, d))
    ])

    stats = defaultdict(lambda: {
        "races": 0, "bets": 0, "hits": 0, "hit_races": 0,
        "cost": 0, "return": 0,
        "方式別": defaultdict(lambda: {"bets": 0, "hits": 0, "hit_races": 0, "cost": 0, "return": 0}),
    })

    total_races = 0
    skipped = 0

    for date_str in all_dates:
        day_dir = os.path.join(race_card_root, date_str)
        csv_files = [f for f in os.listdir(day_dir) if f.endswith(".csv")]

        for fname in csv_files:
            race_id = fname.replace(".csv", "")
            df = load_race_card(date_str, race_id)
            if df.empty:
                continue
            needed = ["score", "score_hitrate", "score_value", "馬番"]
            if not all(c in df.columns for c in needed):
                skipped += 1
                continue
            if pd.to_numeric(df["score"], errors="coerce").isna().all():
                skipped += 1
                continue
            top3 = get_top3(race_id)
            if top3 is None:
                skipped += 1
                continue

            total_races += 1
            recs = recommend_tickets(df)

            for bet_type in ["馬連", "3連複", "3連単"]:
                info = recs.get(bet_type)
                if not info:
                    continue
                method = info["選択方式"]
                s = stats[bet_type]
                s["races"] += 1

                race_hit = False
                for tier in ["基本", "穴狙い"]:
                    tickets = info.get(tier, [])
                    sm = s["方式別"][f"{method}/{tier}"]
                    tier_hit = False
                    for t in tickets:
                        combo = t["組合せ"]
                        hit = is_hit(bet_type, combo, top3)
                        payout = get_payout(race_id, bet_type, combo) if hit else 0
                        s["bets"] += 1
                        s["cost"] += 100
                        sm["bets"] += 1
                        sm["cost"] += 100
                        if hit:
                            s["hits"] += 1
                            s["return"] += payout
                            sm["hits"] += 1
                            sm["return"] += payout
                            race_hit = True
                            tier_hit = True
                    if tier_hit:
                        sm["hit_races"] += 1
                if race_hit:
                    s["hit_races"] += 1

    _print_stats(stats, total_races, skipped, year, mode="全券種モード")


def run_best_backtest(year: str):
    """最良券種モード: レースごとに1券種だけ提案"""
    race_card_root = paths.RACE_CARD_DATA_PATH
    all_dates = sorted([
        d for d in os.listdir(race_card_root)
        if d.startswith(year) and os.path.isdir(os.path.join(race_card_root, d))
    ])

    # key = "bet_type/選択方式"
    stats = defaultdict(lambda: {
        "races": 0, "bets": 0, "hits": 0, "hit_races": 0,
        "cost": 0, "return": 0,
    })

    total_races = 0
    skipped = 0

    for date_str in all_dates:
        day_dir = os.path.join(race_card_root, date_str)
        for fname in sorted(f for f in os.listdir(day_dir) if f.endswith(".csv")):
            race_id = fname.replace(".csv", "")
            df = load_race_card(date_str, race_id)
            if df.empty:
                continue
            needed = ["score", "score_hitrate", "score_value", "馬番"]
            if not all(c in df.columns for c in needed):
                skipped += 1
                continue
            if pd.to_numeric(df["score"], errors="coerce").isna().all():
                skipped += 1
                continue
            top3 = get_top3(race_id)
            if top3 is None:
                skipped += 1
                continue

            total_races += 1
            rec = recommend_best_tickets(df)
            if not rec or not rec.get("tickets"):
                continue

            bet_type = rec["bet_type"]
            method   = rec["選択方式"]
            key      = f"{bet_type}/{method}"
            s        = stats[key]
            s["races"] += 1

            race_hit = False
            for t in rec["tickets"]:
                combo = t["組合せ"]
                hit = is_hit(bet_type, combo, top3)
                payout = get_payout(race_id, bet_type, combo) if hit else 0
                s["bets"] += 1
                s["cost"] += 100
                if hit:
                    s["hits"] += 1
                    s["return"] += payout
                    race_hit = True
            if race_hit:
                s["hit_races"] += 1

    # 集計表示
    W = 62
    print(f"\n{'='*W}")
    print(f"  Backtest {year} [最良券種モード]  "
          f"races:{total_races}  skipped:{skipped}")
    print(f"{'='*W}")
    print(f"  {'券種/方式':<18} {'提案R':>6} {'提案%':>6} "
          f"{'点/R':>5} {'的中R':>6} {'的中R%':>7} "
          f"{'ROI':>7}")
    print(f"  {'-'*(W-2)}")

    total_cost = total_ret = 0
    all_hit_races = 0
    for key in ["3連単/1着固定", "馬連/1頭軸", "馬連/BOX"]:
        s = stats.get(key)
        if not s or s["races"] == 0:
            continue
        races    = s["races"]
        pct      = races / total_races * 100
        avg_bets = s["bets"] / races
        hit_r    = s["hit_races"]
        hit_rpct = hit_r / races * 100
        roi      = s["return"] / s["cost"] * 100 if s["cost"] else 0
        print(f"  {key:<18} {races:>6} {pct:>5.1f}% "
              f"{avg_bets:>5.1f} {hit_r:>6} {hit_rpct:>6.1f}% "
              f"{roi:>6.1f}%")
        total_cost += s["cost"]
        total_ret  += s["return"]
        all_hit_races += s["hit_races"]

    print(f"  {'-'*(W-2)}")
    overall_roi = total_ret / total_cost * 100 if total_cost else 0
    all_hit_pct = all_hit_races / total_races * 100 if total_races else 0
    print(f"  {'TOTAL':<18} {total_races:>6} {'100%':>6} "
          f"{'':>5} {all_hit_races:>6} {all_hit_pct:>6.1f}% "
          f"{overall_roi:>6.1f}%")
    print(f"  cost:{total_cost:,}  return:{total_ret:,}")
    print(f"{'='*W}\n")


def _print_stats(stats, total_races, skipped, year, mode=""):
    print(f"\n{'='*60}")
    print(f"  Backtest {year} [{mode}]  races:{total_races}  skipped:{skipped}")
    print(f"{'='*60}")
    for bet_type, s in stats.items():
        cost = s["cost"]
        ret  = s["return"]
        bets = s["bets"]
        hits = s["hits"]
        races = s["races"]
        hit_races = s["hit_races"]
        roi = ret / cost * 100 if cost > 0 else 0
        hit_rate = hits / bets * 100 if bets > 0 else 0
        race_hit_rate = hit_races / races * 100 if races > 0 else 0
        avg_bets = bets / races if races > 0 else 0
        print(f"\n[{bet_type}]  races:{races}  hit_races:{hit_races}({race_hit_rate:.1f}%)  "
              f"avg_bets/R:{avg_bets:.1f}")
        print(f"  bets:{bets}  hits:{hits}  hitrate:{hit_rate:.1f}%")
        print(f"  cost:{cost:,}  return:{ret:,}  ROI:{roi:.1f}%")
        for method, sm in s["方式別"].items():
            if sm["bets"] == 0:
                continue
            sm_roi = sm["return"] / sm["cost"] * 100 if sm["cost"] > 0 else 0
            sm_hit = sm["hits"] / sm["bets"] * 100 if sm["bets"] > 0 else 0
            sm_races_hit_rate = sm["hit_races"] / races * 100 if races > 0 else 0
            print(f"  [{method}] hit_races:{sm['hit_races']}({sm_races_hit_rate:.1f}%)  "
                  f"bets:{sm['bets']}  hitrate:{sm_hit:.1f}%  ROI:{sm_roi:.1f}%")
    print(f"\n{'='*60}")
    total_cost = sum(s["cost"] for s in stats.values())
    total_ret  = sum(s["return"] for s in stats.values())
    overall_roi = total_ret / total_cost * 100 if total_cost > 0 else 0
    print(f"  TOTAL  cost:{total_cost:,}  return:{total_ret:,}  ROI:{overall_roi:.1f}%")
    print(f"{'='*60}\n")


def run_hitrate_backtest(year: str):
    """的中率重視モード: 3連複流し + フォールバック4パターン比較"""
    race_card_root = paths.RACE_CARD_DATA_PATH
    all_dates = sorted([
        d for d in os.listdir(race_card_root)
        if d.startswith(year) and os.path.isdir(os.path.join(race_card_root, d))
    ])

    total_races = 0
    skipped     = 0

    # レースデータをまとめて収集（全パターン共通）
    race_records = []
    for date_str in all_dates:
        day_dir = os.path.join(race_card_root, date_str)
        for fname in sorted(f for f in os.listdir(day_dir) if f.endswith(".csv")):
            race_id = fname.replace(".csv", "")
            df      = load_race_card(date_str, race_id)
            if df.empty:
                continue
            needed = ["score", "score_hitrate", "score_value", "馬番"]
            if not all(c in df.columns for c in needed):
                skipped += 1
                continue
            if pd.to_numeric(df["score"], errors="coerce").isna().all():
                skipped += 1
                continue
            top3 = get_top3(race_id)
            if top3 is None:
                skipped += 1
                continue
            total_races += 1
            race_records.append((race_id, df, top3))

    # ── 比較パターン (label, kwargs) ──────────────────────────
    # axis_gap: 1頭軸 vs 2頭軸 の gap12 閾値
    # fallback: 3連複が0点の場合の券種
    # cands: 候補馬数 (1jiku / 2jiku)
    patterns = [
        ("①標準(gap>0.05)",      dict(axis_gap=0.05, trifplace_cands_1jiku=6, trifplace_cands_2jiku=7, fallback=None)),
        ("②1頭軸拡大(gap>0.03)", dict(axis_gap=0.03, trifplace_cands_1jiku=6, trifplace_cands_2jiku=7, fallback=None)),
        ("③全1頭軸",             dict(axis_gap=0.00, trifplace_cands_1jiku=6, trifplace_cands_2jiku=7, fallback=None)),
        ("④全1頭軸+候補拡大",    dict(axis_gap=0.00, trifplace_cands_1jiku=7, trifplace_cands_2jiku=7, fallback=None)),
        ("⑤①+馬連BOX4頭FB",     dict(axis_gap=0.05, trifplace_cands_1jiku=6, trifplace_cands_2jiku=7, fallback="umaren_box4")),
        ("⑥②+馬連BOX5頭FB",     dict(axis_gap=0.03, trifplace_cands_1jiku=6, trifplace_cands_2jiku=7, fallback="umaren_box5")),
    ]

    # 各パターン集計
    W = 68
    print(f"\n{'='*W}")
    print(f"  Backtest {year} [的中率重視モード 比較]  "
          f"races:{total_races}  skipped:{skipped}")
    print(f"{'='*W}")
    print(f"  {'パターン':<22} {'券種/方式':<12} {'提案R':>6} {'点/R':>5} "
          f"{'的中R':>6} {'的中R%':>7} {'ROI':>7}")
    print(f"  {'-'*(W-2)}")

    for label, kwargs in patterns:
        stats = defaultdict(lambda: {
            "races": 0, "bets": 0, "hits": 0, "hit_races": 0, "cost": 0, "return": 0,
        })
        for race_id, df, top3 in race_records:
            rec = recommend_hitrate_tickets(df, **kwargs)
            if not rec or not rec.get("tickets"):
                continue
            bet_type = rec["bet_type"]
            method   = rec["選択方式"]
            key      = f"{bet_type}/{method}"
            s        = stats[key]
            s["races"] += 1
            race_hit = False
            for t in rec["tickets"]:
                combo  = t["組合せ"]
                hit    = is_hit(bet_type, combo, top3)
                payout = get_payout(race_id, bet_type, combo) if hit else 0
                s["bets"]  += 1
                s["cost"]  += 100
                if hit:
                    s["hits"]   += 1
                    s["return"] += payout
                    race_hit     = True
            if race_hit:
                s["hit_races"] += 1

        total_cost = total_ret = all_hit = 0
        rows = []
        for key in ["3連複/1頭軸", "3連複/2頭軸", "馬連/BOX", "馬連/1頭軸"]:
            s = stats.get(key)
            if not s or s["races"] == 0:
                continue
            r        = s["races"]
            avg_bets = s["bets"] / r
            hit_r    = s["hit_races"]
            hit_rpct = hit_r / r * 100
            roi      = s["return"] / s["cost"] * 100 if s["cost"] else 0
            total_cost += s["cost"]
            total_ret  += s["return"]
            all_hit    += hit_r
            rows.append((key, r, avg_bets, hit_r, hit_rpct, roi))

        overall_roi = total_ret / total_cost * 100 if total_cost else 0
        all_hit_pct = all_hit / total_races * 100
        print(f"  {label:<22} {'[合計]':<12} {total_races:>6} {'':>5} "
              f"{all_hit:>6} {all_hit_pct:>6.1f}% {overall_roi:>6.1f}%")
        for key, r, avg_bets, hit_r, hit_rpct, roi in rows:
            print(f"  {'':22} {key:<12} {r:>6} {avg_bets:>5.1f} "
                  f"{hit_r:>6} {hit_rpct:>6.1f}% {roi:>6.1f}%")
        print(f"  {'-'*(W-2)}")

    print(f"{'='*W}\n")


def run_combo_backtest(year: str):
    """的中率重視モード v2: 馬連メイン + 3連複サブ 同時推奨のバックテスト"""
    race_card_root = paths.RACE_CARD_DATA_PATH
    all_dates = sorted([
        d for d in os.listdir(race_card_root)
        if d.startswith(year) and os.path.isdir(os.path.join(race_card_root, d))
    ])

    total_races = 0
    skipped = 0
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
                skipped += 1
                continue
            if pd.to_numeric(df["score"], errors="coerce").isna().all():
                skipped += 1
                continue
            top3 = get_top3(race_id)
            if top3 is None:
                skipped += 1
                continue
            total_races += 1
            race_records.append((race_id, df, top3))

    # 券種ごとの統計
    um = {"bets": 0, "hits": 0, "hit_races": 0, "cost": 0, "ret": 0, "proposed": 0}
    tp = {"bets": 0, "hits": 0, "hit_races": 0, "cost": 0, "ret": 0, "proposed": 0}
    combined_hit_races = 0

    for race_id, df, top3 in race_records:
        rec = recommend_hitrate_v2(df)
        if not rec:
            continue

        um_tickets = rec.get("馬連", {}).get("tickets", [])
        tp_tickets = rec.get("3連複", {}).get("tickets", [])

        if not um_tickets and not tp_tickets:
            continue

        race_um_hit = False
        race_tp_hit = False

        for t in um_tickets:
            combo = t["組合せ"]
            hit = is_hit("馬連", combo, top3)
            payout = get_payout(race_id, "馬連", combo) if hit else 0
            um["bets"] += 1
            um["cost"] += 100
            if hit:
                um["hits"] += 1
                um["ret"] += payout
                race_um_hit = True
        if um_tickets:
            um["proposed"] += 1
        if race_um_hit:
            um["hit_races"] += 1

        for t in tp_tickets:
            combo = t["組合せ"]
            hit = is_hit("3連複", combo, top3)
            payout = get_payout(race_id, "3連複", combo) if hit else 0
            tp["bets"] += 1
            tp["cost"] += 100
            if hit:
                tp["hits"] += 1
                tp["ret"] += payout
                race_tp_hit = True
        if tp_tickets:
            tp["proposed"] += 1
        if race_tp_hit:
            tp["hit_races"] += 1

        if race_um_hit or race_tp_hit:
            combined_hit_races += 1

    W = 70
    print(f"\n{'='*W}")
    print(f"  Backtest {year} [的中率重視 v2: 馬連メイン + 3連複サブ]  "
          f"races:{total_races}  skipped:{skipped}")
    print(f"{'='*W}")
    print(f"  {'券種':<10} {'提案R':>6} {'提案%':>6} {'点/R':>5} "
          f"{'的中R':>6} {'的中R%':>7} {'ROI':>7}")
    print(f"  {'-'*(W-2)}")

    for label, s in [("馬連(メイン)", um), ("3連複(サブ)", tp)]:
        prop  = s["proposed"]
        ppct  = prop / total_races * 100
        bpr   = s["bets"] / prop if prop else 0
        hr    = s["hit_races"]
        hrpct = hr / total_races * 100
        roi   = s["ret"] / s["cost"] * 100 if s["cost"] else 0
        print(f"  {label:<10} {prop:>6} {ppct:>5.1f}% {bpr:>5.1f} "
              f"{hr:>6} {hrpct:>6.1f}% {roi:>6.1f}%")

    print(f"  {'-'*(W-2)}")
    total_cost = um["cost"] + tp["cost"]
    total_ret  = um["ret"]  + tp["ret"]
    overall_roi = total_ret / total_cost * 100 if total_cost else 0
    combined_pct = combined_hit_races / total_races * 100
    avg_bets = (um["bets"] + tp["bets"]) / total_races if total_races else 0
    print(f"  {'合計(いずれか的中)':<10} {total_races:>6} {'100%':>6} {avg_bets:>5.1f} "
          f"{combined_hit_races:>6} {combined_pct:>6.1f}% {overall_roi:>6.1f}%")
    print(f"  cost:{total_cost:,}  return:{total_ret:,}")
    print(f"{'='*W}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", default="2026")
    parser.add_argument("--mode",
                        choices=["all", "best", "hitrate", "combo", "both"],
                        default="combo",
                        help="all=全券種, best=回収率重視, hitrate=的中率重視比較, "
                             "combo=馬連+3連複, both=best+combo")
    args = parser.parse_args()
    if args.mode in ("all",):
        run_backtest(args.year)
    if args.mode in ("best", "both"):
        run_best_backtest(args.year)
    if args.mode in ("hitrate",):
        run_hitrate_backtest(args.year)
    if args.mode in ("combo", "both"):
        run_combo_backtest(args.year)
