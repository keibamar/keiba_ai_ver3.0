"""券種横断 的中率 vs ROI 比較スクリプト

複勝 / ワイド / 馬連 / 3連複 / 3連単 の各戦略を一括比較する。
各レースで1戦略のみ選択（単券種モード）。
"""

import os
import sys
from pathlib import Path
from collections import defaultdict
from itertools import combinations, permutations

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.logic.betting.ticket_advisor import (
    blend_scores, scores_to_probs, PAYOUT_RATE,
    place_prob, wide_prob, quinella_prob,
    trifecta_place_prob, trifecta_prob,
)
from src.managers import race_info_dataset_manager, race_result_dataset_manager
from src.config import paths


# ── データ取得 ─────────────────────────────────────────────────

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
    type_map = {
        "複勝":  "複勝",
        "ワイド": "ワイド",
        "馬連":  "馬連",
        "3連複": "三連複",
        "3連単": "三連単",
    }
    col_name = type_map.get(bet_type)
    rows = df[df["式別"] == col_name]
    if rows.empty:
        return 0

    if bet_type == "複勝":
        target = str(combo[0])
    elif bet_type in ("ワイド", "馬連"):
        target = f"{min(combo)}-{max(combo)}"
    elif bet_type == "3連複":
        target = "-".join(map(str, sorted(combo)))
    elif bet_type == "3連単":
        target = "→".join(map(str, combo))
    else:
        return 0

    match = rows[rows["馬番"].astype(str) == target]
    return int(match.iloc[0]["配当"]) if not match.empty else 0


# ── 的中チェック ───────────────────────────────────────────────

def is_hit(bet_type, combo, top3):
    if bet_type == "複勝":
        return combo[0] in top3
    elif bet_type == "ワイド":
        return combo[0] in top3 and combo[1] in top3
    elif bet_type == "馬連":
        return set(combo) == set(top3[:2])
    elif bet_type == "3連複":
        return set(combo) == set(top3)
    elif bet_type == "3連単":
        return combo == top3
    return False


# ── チケット生成 ───────────────────────────────────────────────

def mk(combo, p, bt, max_odds):
    """期待オッズが上限以内なら ticket dict を返す。超えたら None。"""
    if p <= 1e-10:
        return None
    eo = round(PAYOUT_RATE[bt] / p, 1)
    if eo > max_odds:
        return None
    return {"組合せ": combo, "期待オッズ": eo, "確率%": round(p * 100, 2)}


def build_tickets_for_strategy(strategy, uma, probs, rank_order, n):
    """strategy dict に従い買い目リストを生成する。"""
    bt     = strategy["bet_type"]
    method = strategy["method"]
    mx     = strategy["max_odds"]
    nc     = strategy.get("cands", 5)  # 候補馬数

    ax1 = rank_order[0]
    ax2 = rank_order[1]

    tickets = []

    if bt == "複勝":
        # rank1 ~ rank(nc) の複勝
        for idx in range(min(nc, n)):
            i = rank_order[idx]
            p = place_prob(probs, i)
            t = mk((uma[i],), p, bt, mx)
            if t:
                tickets.append(t)

    elif bt == "ワイド" and method == "1頭軸":
        opps = [rank_order[i] for i in range(1, min(nc, n))]
        for opp in opps:
            p = wide_prob(probs, ax1, opp)
            t = mk(tuple(sorted([uma[ax1], uma[opp]])), p, bt, mx)
            if t:
                tickets.append(t)

    elif bt == "ワイド" and method == "BOX":
        pool = [rank_order[i] for i in range(min(nc, n))]
        for i, j in combinations(pool, 2):
            p = wide_prob(probs, i, j)
            t = mk(tuple(sorted([uma[i], uma[j]])), p, bt, mx)
            if t:
                tickets.append(t)

    elif bt == "馬連" and method == "1頭軸":
        opps = [rank_order[i] for i in range(1, min(nc, n))]
        for opp in opps:
            p = quinella_prob(probs, ax1, opp)
            t = mk(tuple(sorted([uma[ax1], uma[opp]])), p, bt, mx)
            if t:
                tickets.append(t)

    elif bt == "馬連" and method == "BOX":
        pool = [rank_order[i] for i in range(min(nc, n))]
        for i, j in combinations(pool, 2):
            p = quinella_prob(probs, i, j)
            t = mk(tuple(sorted([uma[i], uma[j]])), p, bt, mx)
            if t:
                tickets.append(t)

    elif bt == "3連複" and method == "1頭軸":
        opps = [rank_order[i] for i in range(1, min(nc, n))]
        for i, j in combinations(opps, 2):
            p = trifecta_place_prob(probs, ax1, i, j)
            t = mk(tuple(sorted([uma[ax1], uma[i], uma[j]])), p, bt, mx)
            if t:
                tickets.append(t)

    elif bt == "3連複" and method == "BOX":
        pool = [rank_order[i] for i in range(min(nc, n))]
        for i, j, k in combinations(pool, 3):
            p = trifecta_place_prob(probs, i, j, k)
            t = mk(tuple(sorted([uma[i], uma[j], uma[k]])), p, bt, mx)
            if t:
                tickets.append(t)

    elif bt == "3連単" and method == "1着固定":
        cands = [rank_order[i] for i in range(1, min(nc, n))]
        for j, k in permutations(cands, 2):
            p = trifecta_prob(probs, ax1, j, k)
            t = mk((uma[ax1], uma[j], uma[k]), p, bt, mx)
            if t:
                tickets.append(t)
        tickets = tickets[:10]  # 上限 10 点

    tickets.sort(key=lambda t: t["期待オッズ"])
    return tickets


# ── 比較パターン定義 ───────────────────────────────────────────

STRATEGIES = [
    # ── 馬連 ──────────────────────────────────────────────────────
    # BOX: max_odds を絞って低期待値を除外
    {"label": "馬連BOX 3頭 max20",    "bet_type": "馬連", "method": "BOX",   "cands": 3, "max_odds": 20.0},
    {"label": "馬連BOX 3頭 max30",    "bet_type": "馬連", "method": "BOX",   "cands": 3, "max_odds": 30.0},
    {"label": "馬連BOX 4頭 max20",    "bet_type": "馬連", "method": "BOX",   "cands": 4, "max_odds": 20.0},
    {"label": "馬連BOX 4頭 max30",    "bet_type": "馬連", "method": "BOX",   "cands": 4, "max_odds": 30.0},
    {"label": "馬連BOX 4頭 max40",    "bet_type": "馬連", "method": "BOX",   "cands": 4, "max_odds": 40.0},
    # 1頭軸: max_odds を絞って低期待値を除外
    {"label": "馬連1軸 r2-5 max25",   "bet_type": "馬連", "method": "1頭軸", "cands": 5, "max_odds": 25.0},
    {"label": "馬連1軸 r2-5 max35",   "bet_type": "馬連", "method": "1頭軸", "cands": 5, "max_odds": 35.0},
    {"label": "馬連1軸 r2-6 max80",   "bet_type": "馬連", "method": "1頭軸", "cands": 6, "max_odds": 80.0},  # 従来
    # ── 3連複 ─────────────────────────────────────────────────────
    # BOX: C(4,3)=4, C(5,3)=10
    {"label": "3連複BOX 4頭 max50",   "bet_type": "3連複", "method": "BOX",  "cands": 4, "max_odds":  50.0},
    {"label": "3連複BOX 4頭 max80",   "bet_type": "3連複", "method": "BOX",  "cands": 4, "max_odds":  80.0},
    # 1頭軸: max_odds を絞る
    {"label": "3連複1軸 r2-5 max50",  "bet_type": "3連複", "method": "1頭軸","cands": 5, "max_odds":  50.0},
    {"label": "3連複1軸 r2-5 max80",  "bet_type": "3連複", "method": "1頭軸","cands": 5, "max_odds":  80.0},
    {"label": "3連複1軸 r2-6 max60",  "bet_type": "3連複", "method": "1頭軸","cands": 6, "max_odds":  60.0},
    {"label": "3連複1軸 r2-6 max80",  "bet_type": "3連複", "method": "1頭軸","cands": 6, "max_odds":  80.0},
    {"label": "3連複1軸 r2-6 max150", "bet_type": "3連複", "method": "1頭軸","cands": 6, "max_odds": 150.0},  # 従来
    # ── 3連単 (参考) ───────────────────────────────────────────────
    {"label": "3連単1着固定 r2-6",    "bet_type": "3連単", "method": "1着固定","cands": 6,"max_odds": 200.0},
]


# ── バックテスト本体 ────────────────────────────────────────────

def run(year="2026"):
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

            blended    = blend_scores(df)
            probs      = scores_to_probs(blended)
            n          = len(probs)
            if n < 3:
                continue
            uma        = pd.to_numeric(df["馬番"], errors="coerce").astype(int).tolist()
            rank_order = np.argsort(probs)[::-1]
            total_races += 1
            race_records.append((race_id, uma, probs, rank_order, n, top3))

    print(f"\nLoaded {total_races} races (skipped:{skipped})\n")

    W = 76
    print(f"{'='*W}")
    print(f"  Backtest {year} [券種横断 的中率比較]")
    print(f"{'='*W}")
    print(f"  {'戦略':<24} {'点/R':>5} {'的中R':>6} {'的中R%':>7} "
          f"{'ROI':>7}  (参考: 点数×的中R%)")
    print(f"  {'-'*(W-2)}")

    for st in STRATEGIES:
        s = {"races": 0, "bets": 0, "hits": 0, "hit_races": 0, "cost": 0, "ret": 0}
        for race_id, uma, probs, rank_order, n, top3 in race_records:
            tickets = build_tickets_for_strategy(st, uma, probs, rank_order, n)
            if not tickets:
                continue
            s["races"] += 1
            race_hit = False
            for t in tickets:
                combo = t["組合せ"]
                hit   = is_hit(st["bet_type"], combo, top3)
                payout = get_payout(race_id, st["bet_type"], combo) if hit else 0
                s["bets"] += 1
                s["cost"] += 100
                if hit:
                    s["hits"] += 1
                    s["ret"]  += payout
                    race_hit   = True
            if race_hit:
                s["hit_races"] += 1

        races    = s["races"]
        bpr      = s["bets"]  / races if races else 0
        hit_rpct = s["hit_races"] / total_races * 100
        roi      = s["ret"] / s["cost"] * 100 if s["cost"] else 0
        score    = bpr * hit_rpct  # 点数と的中率の積（高いほど「使いやすい」）
        print(f"  {st['label']:<24} {bpr:>5.1f} {s['hit_races']:>6} {hit_rpct:>6.1f}% "
              f"{roi:>6.1f}%  ({score:.1f})")

    print(f"{'='*W}\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", default="2026")
    args = parser.parse_args()
    run(args.year)
