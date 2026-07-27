"""的中率重視 v2 からの2方向探索スクリプト

Direction A: 的中率を上げる（点数は節制）
Direction B: 回収率を上げる（100%超を狙う）

使い方:
    python scripts/explore_hitrate_v2.py [--year 2026]
"""

import os
import sys
from pathlib import Path
from itertools import permutations

import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.logic.betting.ticket_advisor import (
    blend_scores, scores_to_probs,
    recommend_hitrate_v2, PAYOUT_RATE,
    quinella_prob, trifecta_place_prob, trifecta_prob,
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
    type_map = {"馬連": "馬連", "3連複": "三連複", "3連単": "三連単"}
    rows = df[df["式別"] == type_map.get(bet_type, "")]
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
    m = rows[rows["馬番"].astype(str) == target]
    return int(m.iloc[0]["配当"]) if not m.empty else 0


def is_hit(bet_type, combo, top3):
    if bet_type == "馬連":
        return set(combo) == set(top3[:2])
    if bet_type == "3連複":
        return set(combo) == set(top3)
    if bet_type == "3連単":
        return combo == top3
    return False


# ── 設定グリッド ───────────────────────────────────────────────
#
# umaren_cands   : 馬連 1頭軸の相手候補人数 (含む axis=rank1 で index 1..cands-1)
# umaren_max     : 馬連 期待オッズ上限
# tp_cands       : 3連複 1頭軸の相手候補人数 (C(cands-1, 2) 通り)
# tp_max         : 3連複 期待オッズ上限
# min_axis_prob  : 推奨する最低軸確率（これ未満のレースは馬連も3連複も提案しない）
# max_um_tickets : 馬連 最大買い目点数 (None=制限なし)
# max_tp_tickets : 3連複 最大買い目点数 (None=制限なし)
# add_trifecta   : axis_prob≥0.15 かつ gap12≥0.06 のレースに3連単を追加する
# tfc_max        : 3連単 期待オッズ上限 (add_trifecta=True のとき使用)
# tfc_top        : 3連単 最大買い目点数 (add_trifecta=True のとき使用)

CONFIGS = [
    # ── ベースライン ──────────────────────────────────────────────
    {"label": "★ベースライン",
     "umaren_cands": 5, "umaren_max": 25, "tp_cands": 5, "tp_max": 50,
     "min_axis_prob": 0.0, "max_um": None, "max_tp": None,
     "add_trifecta": False},

    # ── 方向A: 的中率を上げる ────────────────────────────────────
    # A1: 馬連の相手を rank2-6 まで広げる（+1頭）
    {"label": "A1:馬連r2-6",
     "umaren_cands": 6, "umaren_max": 25, "tp_cands": 5, "tp_max": 50,
     "min_axis_prob": 0.0, "max_um": None, "max_tp": None,
     "add_trifecta": False},

    # A2: 馬連オッズ上限を緩める (25→35)
    {"label": "A2:馬連max35",
     "umaren_cands": 5, "umaren_max": 35, "tp_cands": 5, "tp_max": 50,
     "min_axis_prob": 0.0, "max_um": None, "max_tp": None,
     "add_trifecta": False},

    # A3: 3連複の相手を rank2-6 まで広げる (+C(5,2)-C(4,2)=4 点増加)
    {"label": "A3:3連複r2-6",
     "umaren_cands": 5, "umaren_max": 25, "tp_cands": 6, "tp_max": 50,
     "min_axis_prob": 0.0, "max_um": None, "max_tp": None,
     "add_trifecta": False},

    # A4: 3連複オッズ上限を緩める (50→80)
    {"label": "A4:3連複max80",
     "umaren_cands": 5, "umaren_max": 25, "tp_cands": 5, "tp_max": 80,
     "min_axis_prob": 0.0, "max_um": None, "max_tp": None,
     "add_trifecta": False},

    # A5: 馬連・3連複 両方緩める（バランス拡大）
    {"label": "A5:両拡大r2-6/max35/60",
     "umaren_cands": 6, "umaren_max": 35, "tp_cands": 6, "tp_max": 60,
     "min_axis_prob": 0.0, "max_um": None, "max_tp": None,
     "add_trifecta": False},

    # ── 方向B: 回収率を上げる ────────────────────────────────────
    # B1: 軸確率 ≥ 20% のレースだけ推奨
    {"label": "B1:軸確率>20%",
     "umaren_cands": 5, "umaren_max": 25, "tp_cands": 5, "tp_max": 50,
     "min_axis_prob": 0.20, "max_um": None, "max_tp": None,
     "add_trifecta": False},

    # B2: 軸確率 ≥ 25%（さらに絞る）
    {"label": "B2:軸確率>25%",
     "umaren_cands": 5, "umaren_max": 25, "tp_cands": 5, "tp_max": 50,
     "min_axis_prob": 0.25, "max_um": None, "max_tp": None,
     "add_trifecta": False},

    # B3: 各券種 上位2票のみ（確率最上位 = 期待オッズ最小）
    {"label": "B3:上位2票のみ",
     "umaren_cands": 5, "umaren_max": 25, "tp_cands": 5, "tp_max": 50,
     "min_axis_prob": 0.0, "max_um": 2, "max_tp": 2,
     "add_trifecta": False},

    # B4: 軸確率 ≥ 20% + 上位2票のみ
    {"label": "B4:軸>20%+上位2票",
     "umaren_cands": 5, "umaren_max": 25, "tp_cands": 5, "tp_max": 50,
     "min_axis_prob": 0.20, "max_um": 2, "max_tp": 2,
     "add_trifecta": False},

    # B5: 強軸レース (axis_strong) に 3連単をサブとして追加
    {"label": "B5:強軸→3連単追加",
     "umaren_cands": 5, "umaren_max": 25, "tp_cands": 5, "tp_max": 50,
     "min_axis_prob": 0.0, "max_um": None, "max_tp": None,
     "add_trifecta": True, "tfc_max": 200.0, "tfc_top": 5},

    # B6: 軸確率 ≥ 20% + 3連単追加 (B1 + B5 の合体)
    {"label": "B6:軸>20%+3連単",
     "umaren_cands": 5, "umaren_max": 25, "tp_cands": 5, "tp_max": 50,
     "min_axis_prob": 0.20, "max_um": None, "max_tp": None,
     "add_trifecta": True, "tfc_max": 200.0, "tfc_top": 5},
]


# ── バックテスト本体 ────────────────────────────────────────────

def run(year="2026"):
    race_card_root = paths.RACE_CARD_DATA_PATH
    all_dates = sorted([
        d for d in os.listdir(race_card_root)
        if d.startswith(year) and os.path.isdir(os.path.join(race_card_root, d))
    ])

    print(f"\n  データ読み込み中...", end="", flush=True)
    total_races = 0
    skipped = 0
    race_records = []  # (race_id, df, probs, rank_order, uma, top3)

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

    print(f" {total_races}R (skip:{skipped})\n")

    W = 90
    print("=" * W)
    print(f"  {year} 年 探索結果 [ベースライン比較]   total:{total_races}R")
    print("=" * W)
    print(f"  {'設定':<26} {'点/R':>5} {'馬連的中R%':>9} {'3複的中R%':>9} "
          f"{'3単的中R%':>9} {'合計的中R%':>10} {'ROI':>7}")
    print(f"  {'-'*(W-2)}")

    for cfg in CONFIGS:
        um   = {"bets": 0, "hits": 0, "hit_races": 0, "cost": 0, "ret": 0, "prop": 0}
        tp   = {"bets": 0, "hits": 0, "hit_races": 0, "cost": 0, "ret": 0, "prop": 0}
        tfc  = {"bets": 0, "hits": 0, "hit_races": 0, "cost": 0, "ret": 0, "prop": 0}
        combined_hit = 0

        min_ap      = cfg.get("min_axis_prob", 0.0)
        max_um      = cfg.get("max_um")
        max_tp      = cfg.get("max_tp")
        add_tfc     = cfg.get("add_trifecta", False)
        tfc_max_eo  = cfg.get("tfc_max", 200.0)
        tfc_top     = cfg.get("tfc_top", 5)

        for race_id, df, top3 in race_records:
            rec = recommend_hitrate_v2(
                df,
                umaren_cands=cfg["umaren_cands"],
                umaren_max=cfg["umaren_max"],
                trifplace_cands=cfg["tp_cands"],
                trifplace_max=cfg["tp_max"],
            )
            if not rec:
                continue

            meta = rec.get("_meta", {})
            axis_prob   = meta.get("axis_prob", 0)
            axis_strong = meta.get("axis_strong", False)

            # 軸確率フィルター
            if axis_prob < min_ap:
                continue

            um_tickets = rec.get("馬連",  {}).get("tickets", [])
            tp_tickets = rec.get("3連複", {}).get("tickets", [])

            # 上位N票に絞る
            if max_um is not None:
                um_tickets = um_tickets[:max_um]
            if max_tp is not None:
                tp_tickets = tp_tickets[:max_tp]

            # 3連単（強軸レースのみ）
            tfc_tickets = []
            if add_tfc and axis_strong:
                blended    = blend_scores(df)
                probs      = scores_to_probs(blended)
                n          = len(probs)
                uma        = pd.to_numeric(df["馬番"], errors="coerce").astype(int).tolist()
                rank_order = np.argsort(probs)[::-1]
                ax1        = rank_order[0]
                cands      = [rank_order[i] for i in range(1, min(6, n))]
                for j, k in permutations(cands, 2):
                    p = trifecta_prob(probs, ax1, j, k)
                    if p <= 1e-10:
                        continue
                    eo = round(PAYOUT_RATE["3連単"] / p, 1)
                    if eo <= tfc_max_eo:
                        tfc_tickets.append({"組合せ": (uma[ax1], uma[j], uma[k]),
                                            "期待オッズ": eo})
                tfc_tickets.sort(key=lambda t: t["期待オッズ"])
                tfc_tickets = tfc_tickets[:tfc_top]

            if not um_tickets and not tp_tickets and not tfc_tickets:
                continue

            race_um_hit = race_tp_hit = race_tfc_hit = False

            for t in um_tickets:
                hit = is_hit("馬連", t["組合せ"], top3)
                pay = get_payout(race_id, "馬連", t["組合せ"]) if hit else 0
                um["bets"] += 1
                um["cost"] += 100
                if hit:
                    um["hits"] += 1
                    um["ret"]  += pay
                    race_um_hit = True
            if um_tickets:
                um["prop"] += 1
            if race_um_hit:
                um["hit_races"] += 1

            for t in tp_tickets:
                hit = is_hit("3連複", t["組合せ"], top3)
                pay = get_payout(race_id, "3連複", t["組合せ"]) if hit else 0
                tp["bets"] += 1
                tp["cost"] += 100
                if hit:
                    tp["hits"] += 1
                    tp["ret"]  += pay
                    race_tp_hit = True
            if tp_tickets:
                tp["prop"] += 1
            if race_tp_hit:
                tp["hit_races"] += 1

            for t in tfc_tickets:
                hit = is_hit("3連単", t["組合せ"], top3)
                pay = get_payout(race_id, "3連単", t["組合せ"]) if hit else 0
                tfc["bets"] += 1
                tfc["cost"] += 100
                if hit:
                    tfc["hits"] += 1
                    tfc["ret"]  += pay
                    race_tfc_hit = True
            if tfc_tickets:
                tfc["prop"] += 1
            if race_tfc_hit:
                tfc["hit_races"] += 1

            if race_um_hit or race_tp_hit or race_tfc_hit:
                combined_hit += 1

        total_bets = um["bets"] + tp["bets"] + tfc["bets"]
        total_cost = um["cost"] + tp["cost"] + tfc["cost"]
        total_ret  = um["ret"]  + tp["ret"]  + tfc["ret"]
        avg_bets   = total_bets / total_races if total_races else 0
        um_hrpct   = um["hit_races"]  / total_races * 100
        tp_hrpct   = tp["hit_races"]  / total_races * 100
        tfc_hrpct  = tfc["hit_races"] / total_races * 100
        comb_pct   = combined_hit / total_races * 100
        roi        = total_ret / total_cost * 100 if total_cost else 0

        # ベースラインを示すマーカー
        label = cfg["label"]
        print(f"  {label:<26} {avg_bets:>5.1f} "
              f"{um_hrpct:>8.1f}% {tp_hrpct:>8.1f}% "
              f"{tfc_hrpct:>8.1f}% {comb_pct:>9.1f}% {roi:>6.1f}%")

    print("=" * W)
    print("  ※ 合計的中R% = レース単位でいずれか1つでも当たった割合")
    print("  ※ 点/R = 全レース分母（提案なしのレースは0点として計算）\n")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", default="2026")
    args = parser.parse_args()
    run(args.year)
