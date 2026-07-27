"""候補馬数を変えたときの的中率 vs ROI のトレードオフを分析するスクリプト"""

import os
import sys
from pathlib import Path
from collections import defaultdict
from itertools import permutations, combinations

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.logic.betting.ticket_advisor import (
    blend_scores, scores_to_probs,
    quinella_prob, trifecta_place_prob, trifecta_prob,
    exp_odds, PAYOUT_RATE,
)
from src.managers import race_info_dataset_manager, race_result_dataset_manager
from src.config import paths


def load_race_card(date_str, race_id):
    path = os.path.join(paths.RACE_CARD_DATA_PATH, date_str, f"{race_id}.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path, index_col=0, dtype=str)


def get_top3(race_id):
    df = race_result_dataset_manager.get_race_id_result(race_id)
    if df.empty or "着順" not in df.columns or "馬番" not in df.columns:
        return None
    df = df.copy()
    df["着順"] = pd.to_numeric(df["着順"], errors="coerce")
    df["馬番"] = pd.to_numeric(df["馬番"], errors="coerce")
    df = df.dropna(subset=["着順", "馬番"]).sort_values("着順")
    top3 = df[df["着順"] <= 3]["馬番"].astype(int).tolist()
    return tuple(top3[:3]) if len(top3) >= 3 else None


def get_payout(race_id, bet_type, combo):
    df = race_info_dataset_manager.get_race_return_csv_for_race(race_id)
    if df.empty:
        return 0
    type_map = {"umaren": "馬連", "trifplace": "三連複", "trifecta": "三連単"}
    col_name = type_map.get(bet_type)
    rows = df[df["式別"] == col_name]
    if rows.empty:
        return 0
    if bet_type == "umaren":
        target = f"{min(combo)}-{max(combo)}"
    elif bet_type == "trifplace":
        target = "-".join(map(str, sorted(combo)))
    elif bet_type == "trifecta":
        target = "→".join(map(str, combo))
    else:
        return 0
    match = rows[rows["馬番"].astype(str) == target]
    return int(match.iloc[0]["配当"]) if not match.empty else 0


def is_hit(bet_type, combo, top3):
    if bet_type == "umaren":
        return set(combo) == set(top3[:2])
    elif bet_type == "trifplace":
        return set(combo) == set(top3)
    elif bet_type == "trifecta":
        return combo == top3
    return False


def build_tickets(df, uma, probs, rank_order, n,
                  umaren_n, trifplace_n, trifecta_n, max_odds):
    """指定した候補馬数で買い目を生成"""
    ax1 = rank_order[0]
    ax2 = rank_order[1]
    gap12 = float(probs[rank_order[0]] - probs[rank_order[1]]) if n >= 2 else 1.0
    gap23 = float(probs[rank_order[1]] - probs[rank_order[2]]) if n >= 3 else 1.0

    result = {}

    # 馬連
    opps = [rank_order[i] for i in range(1, min(umaren_n, n))]
    if gap12 > 0.05:
        tickets = []
        for opp in opps:
            p = quinella_prob(probs, ax1, opp)
            eo = exp_odds(p, "馬連")
            if eo <= max_odds["umaren"]:
                tickets.append((tuple(sorted([uma[ax1], uma[opp]])), eo))
        result["umaren"] = tickets
    else:
        pool = [rank_order[i] for i in range(min(umaren_n, n))]
        tickets = []
        for i, j in combinations(pool, 2):
            p = quinella_prob(probs, i, j)
            eo = exp_odds(p, "馬連")
            if eo <= max_odds["umaren"]:
                tickets.append((tuple(sorted([uma[i], uma[j]])), eo))
        result["umaren"] = tickets

    # 3連複
    if gap12 > 0.05 and gap23 > 0.03:
        opps2 = [rank_order[i] for i in range(1, min(trifplace_n, n))]
        tickets = []
        for i, j in combinations(opps2, 2):
            p = trifecta_place_prob(probs, ax1, i, j)
            eo = exp_odds(p, "3連複")
            if eo <= max_odds["trifplace"]:
                tickets.append((tuple(sorted([uma[ax1], uma[i], uma[j]])), eo))
        result["trifplace"] = tickets
    else:
        opps2 = [rank_order[i] for i in range(2, min(trifplace_n, n))]
        tickets = []
        for opp in opps2:
            p = trifecta_place_prob(probs, ax1, ax2, opp)
            eo = exp_odds(p, "3連複")
            if eo <= max_odds["trifplace"]:
                tickets.append((tuple(sorted([uma[ax1], uma[ax2], uma[opp]])), eo))
        result["trifplace"] = tickets

    # 3連単
    cands = [rank_order[i] for i in range(1, min(trifecta_n, n))]
    tickets = []
    for j, k in permutations(cands, 2):
        p = trifecta_prob(probs, ax1, j, k)
        eo = exp_odds(p, "3連単")
        if eo <= max_odds["trifecta"]:
            tickets.append(((uma[ax1], uma[j], uma[k]), eo))
    result["trifecta"] = tickets

    return result


def run_comparison(year="2026"):
    race_card_root = paths.RACE_CARD_DATA_PATH
    all_dates = sorted([
        d for d in os.listdir(race_card_root)
        if d.startswith(year) and os.path.isdir(os.path.join(race_card_root, d))
    ])

    # (umaren_n, trifplace_n, trifecta_n, max_odds) のパターン
    # umaren_n = 1頭軸のとき軸+n-1頭、BOXのときn頭
    patterns = [
        ("current",  5, 6, 5, {"umaren": 80, "trifplace": 200, "trifecta": 120}),
        ("trifecta+1", 5, 6, 6, {"umaren": 80, "trifplace": 200, "trifecta": 120}),
        ("trifecta+2", 5, 6, 7, {"umaren": 80, "trifplace": 200, "trifecta": 120}),
        ("odds+", 5, 6, 6, {"umaren": 80, "trifplace": 200, "trifecta": 200}),
    ]

    # データ収集
    race_data = []
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

            blended = blend_scores(df)
            probs   = scores_to_probs(blended)
            n       = len(probs)
            if n < 3:
                continue
            uma = pd.to_numeric(df["馬番"], errors="coerce").astype(int).tolist()
            rank_order = np.argsort(probs)[::-1]

            race_data.append((race_id, uma, probs, rank_order, n, top3))

    print(f"\nLoaded {len(race_data)} races  (skipped:{skipped})")
    print(f"\n{'Pattern':<15} {'umaren_hit%':>11} {'tp_hit%':>9} {'tfc_hit%':>9} "
          f"{'u_bets/R':>8} {'tp_bets/R':>9} {'tf_bets/R':>9} "
          f"{'u_ROI':>7} {'tp_ROI':>7} {'tf_ROI':>7}")
    print("-" * 110)

    for label, umaren_n, trifplace_n, trifecta_n, max_odds in patterns:
        stats = {bt: {"hit_races": 0, "bets": 0, "hits": 0, "cost": 0, "ret": 0}
                 for bt in ["umaren", "trifplace", "trifecta"]}

        for race_id, uma, probs, rank_order, n, top3 in race_data:
            tickets = build_tickets(df=None, uma=uma, probs=probs,
                                    rank_order=rank_order, n=n,
                                    umaren_n=umaren_n, trifplace_n=trifplace_n,
                                    trifecta_n=trifecta_n, max_odds=max_odds)
            for bt, tlist in tickets.items():
                s = stats[bt]
                race_hit = False
                for combo, eo in tlist:
                    hit = is_hit(bt, combo, top3)
                    payout = get_payout(race_id, bt, combo) if hit else 0
                    s["bets"] += 1
                    s["cost"] += 100
                    if hit:
                        s["hits"] += 1
                        s["ret"] += payout
                        race_hit = True
                if race_hit:
                    s["hit_races"] += 1

        total = len(race_data)
        def fmt(bt):
            s = stats[bt]
            hr = s["hit_races"] / total * 100 if total else 0
            bpr = s["bets"] / total if total else 0
            roi = s["ret"] / s["cost"] * 100 if s["cost"] else 0
            return hr, bpr, roi

        u_hr, u_bpr, u_roi = fmt("umaren")
        tp_hr, tp_bpr, tp_roi = fmt("trifplace")
        tf_hr, tf_bpr, tf_roi = fmt("trifecta")

        print(f"{label:<15} {u_hr:>10.1f}% {tp_hr:>8.1f}% {tf_hr:>8.1f}% "
              f"{u_bpr:>8.1f} {tp_bpr:>9.1f} {tf_bpr:>9.1f} "
              f"{u_roi:>6.1f}% {tp_roi:>6.1f}% {tf_roi:>6.1f}%")

    print()


if __name__ == "__main__":
    run_comparison()
