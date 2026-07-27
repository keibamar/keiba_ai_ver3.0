"""今日の馬券提案リストを表示する

使い方:
    python scripts/show_today_tickets.py [--date YYYYMMDD] [--mode score|tp|best|all]

モード:
    score (default) : 指数ベース 馬連+3連複 (SB-B 案1: 的中12.3% ROI 90.9%)
    tp              : 指数ベース 3連複のみ (ROI 99.5% 的中3.7%)
    best            : 回収率重視 — 3連単1着固定 (ROI 232%)
    all             : 従来全券種モード
"""

import os
import sys
import argparse
from datetime import date
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.logic.betting.ticket_advisor import (
    recommend_tickets, recommend_best_tickets, recommend_hitrate_v2,
    recommend_score_based, SB_B_PARAMS,
    BASIC_MAX, LONGSHOT_MAX,
    _HIT2_UMAREN_MAX, _HIT2_TP_MAX,
)
from src.managers import race_card_dataset_manager
from src.config import constants, paths

PLACE_NAME = {str(i+1).zfill(2): name for i, name in enumerate(constants.NAME_LIST)}


def get_race_label(race_id: str) -> str:
    place_id = race_id[4:6]
    race_no  = int(race_id[10:12])
    venue    = PLACE_NAME.get(place_id, f"場{place_id}")
    return f"{venue} R{race_no}"


def get_race_meta(race_id: str) -> str:
    info = race_card_dataset_manager.get_race_info_csv(race_id)
    if info.empty:
        return ""
    row = info.iloc[0]
    parts = []
    for col in ("race_type", "course_len", "ground_state", "class"):
        if col in row:
            val = str(row[col])
            parts.append(f"{val}m" if col == "course_len" else val)
    return " / ".join(parts)


def fmt_combo(bet_type: str, combo: tuple) -> str:
    if bet_type == "馬連":
        return f"{combo[0]}-{combo[1]}"
    elif bet_type == "3連複":
        return "-".join(map(str, combo))
    elif bet_type == "3連単":
        return "->".join(map(str, combo))
    return str(combo)


def _build_uma_map(df: pd.DataFrame) -> dict:
    uma_map = {}
    for _, row in df.iterrows():
        try:
            no = int(pd.to_numeric(row["馬番"], errors="coerce"))
            uma_map[no] = str(row["馬名"])
        except Exception:
            pass
    return uma_map


def _print_tickets(bet_type: str, tickets: list, method: str) -> None:
    if not tickets:
        return
    print(f"\n  {bet_type}  [{method}]")
    print(f"  {'組合せ':<14} {'期待オッズ':>9}  {'確率':>7}")
    print(f"  {'-'*34}")
    for t in tickets:
        print(f"  {fmt_combo(bet_type, t['組合せ']):<14} "
              f"{t['期待オッズ']:>8.1f}x  {t['確率%']:>6.2f}%")


# ── 指数ベース 馬連+3連複 表示 ───────────────────────────

def show_hitrate_race(race_id: str, df: pd.DataFrame):
    rec = recommend_score_based(df, **SB_B_PARAMS)
    if not rec:
        return

    um_tickets = rec.get("馬連",  {}).get("tickets", [])
    tp_tickets = rec.get("3連複", {}).get("tickets", [])
    if not um_tickets and not tp_tickets:
        return

    label     = get_race_label(race_id)
    meta      = get_race_meta(race_id)
    meta_info = rec.get("_meta", {})
    axis_prob = meta_info.get("axis_prob", 0)
    um_method = rec.get("馬連",  {}).get("選択方式", "?")
    tp_method = rec.get("3連複", {}).get("選択方式", "?")
    ax_list   = (rec.get("馬連") or rec.get("3連複", {})).get("軸", [])
    ax_uma    = ax_list[0] if ax_list else None

    print(f"\n{'='*52}")
    print(f"  {label}  {meta}  axis:{axis_prob*100:.0f}%")
    print(f"{'='*52}")

    if ax_uma:
        uma_map = _build_uma_map(df)
        print(f"  [axis] #{ax_uma} {uma_map.get(ax_uma, '?')}")

    _print_tickets("馬連",  um_tickets, um_method)
    _print_tickets("3連複", tp_tickets, tp_method)


# ── 指数ベース 3連複のみ 表示 ────────────────────────────

def show_tp_race(race_id: str, df: pd.DataFrame):
    rec = recommend_score_based(df, **SB_B_PARAMS)
    if not rec:
        return

    tp_tickets = rec.get("3連複", {}).get("tickets", [])
    if not tp_tickets:
        return

    label     = get_race_label(race_id)
    meta      = get_race_meta(race_id)
    meta_info = rec.get("_meta", {})
    axis_prob = meta_info.get("axis_prob", 0)
    tp_method = rec.get("3連複", {}).get("選択方式", "?")
    tp_axis   = rec.get("3連複", {}).get("軸", [])

    print(f"\n{'='*52}")
    print(f"  {label}  {meta}  axis:{axis_prob*100:.0f}%")
    print(f"{'='*52}")

    if tp_axis:
        uma_map = _build_uma_map(df)
        ax_str = "+".join(f"#{n}{uma_map.get(n, '?')}" for n in tp_axis)
        print(f"  [axis] {ax_str}")

    _print_tickets("3連複", tp_tickets, tp_method)


# ── 回収率重視 表示 ─────────────────────────────────

def show_best_race(race_id: str, df: pd.DataFrame):
    rec = recommend_best_tickets(df)
    if not rec or not rec.get("tickets"):
        return

    label = get_race_label(race_id)
    meta  = get_race_meta(race_id)
    meta_info   = rec.get("_meta", {})
    axis_prob   = meta_info.get("axis_prob", 0)
    axis_strong = meta_info.get("axis_strong", False)
    bet_type    = rec["bet_type"]
    method      = rec["選択方式"]
    ax_list     = rec.get("軸", [])
    ax_uma      = ax_list[0] if ax_list else None

    uma_map = _build_uma_map(df)

    print(f"\n{'='*52}")
    strong_tag = " [軸強]" if axis_strong else ""
    print(f"  {label}  {meta}  axis:{axis_prob*100:.0f}%{strong_tag}")
    print(f"{'='*52}")
    if ax_uma:
        print(f"  [axis] #{ax_uma} {uma_map.get(ax_uma, '?')}")

    _print_tickets(bet_type, rec["tickets"], method)


# ── 従来全券種 表示 ─────────────────────────────────

def show_all_race(race_id: str, df: pd.DataFrame):
    recs = recommend_tickets(df)
    if not recs:
        return

    label       = get_race_label(race_id)
    meta        = get_race_meta(race_id)
    meta_info   = recs.get("_meta", {})
    axis_prob   = meta_info.get("axis_prob", 0)
    axis_strong = meta_info.get("axis_strong", False)

    uma_map = _build_uma_map(df)

    print(f"\n{'='*52}")
    strong_tag = " [軸強]" if axis_strong else ""
    print(f"  {label}  {meta}  axis:{axis_prob*100:.0f}%{strong_tag}")
    print(f"{'='*52}")

    ax_no = recs.get("3連単", {}).get("軸", [None])[0]
    if ax_no:
        print(f"  [axis] #{ax_no} {uma_map.get(ax_no, '?')}")

    for bet_type in ["馬連", "3連複", "3連単"]:
        info = recs.get(bet_type)
        if not info:
            continue
        method   = info["選択方式"]
        basic    = info.get("基本",    [])
        longshot = info.get("穴狙い",  [])
        if not basic and not longshot:
            continue

        print(f"\n  {bet_type} [{method}]")
        if basic:
            print(f"    -- standard (≤{BASIC_MAX[bet_type]:.0f}x) --")
            print(f"    {'組合せ':<14} {'期待オッズ':>8}  {'確率':>7}")
            print(f"    {'-'*34}")
            for t in basic:
                print(f"    {fmt_combo(bet_type, t['組合せ']):<14} "
                      f"{t['期待オッズ']:>7.1f}x  {t['確率%']:>6.2f}%")
        if longshot:
            print(f"    -- longshot ({BASIC_MAX[bet_type]:.0f}x-{LONGSHOT_MAX[bet_type]:.0f}x) --")
            print(f"    {'組合せ':<14} {'期待オッズ':>8}  {'確率':>7}")
            print(f"    {'-'*34}")
            for t in longshot:
                print(f"    {fmt_combo(bet_type, t['組合せ']):<14} "
                      f"{t['期待オッズ']:>7.1f}x  {t['確率%']:>6.2f}%")


# ── メイン ─────────────────────────────────────────

def main(date_str: str, mode: str):
    race_card_root = paths.RACE_CARD_DATA_PATH
    day_dir = os.path.join(race_card_root, date_str)

    if not os.path.isdir(day_dir):
        print(f"レースカードが見つかりません: {date_str}")
        return

    needed = ["score", "score_hitrate", "score_value", "馬番"]

    by_place = {}
    for fname in sorted(f for f in os.listdir(day_dir) if f.endswith(".csv")):
        race_id  = fname.replace(".csv", "")
        place_id = race_id[4:6]
        by_place.setdefault(place_id, []).append(race_id)

    mode_labels = {
        "score":   "指数ベース 馬連+3連複  (ROI 90.9%  的中 12.3%)",
        "hitrate": "指数ベース 馬連+3連複  (ROI 90.9%  的中 12.3%)",
        "tp":      "指数ベース 3連複のみ   (ROI 99.5%  的中  3.7%)",
        "best":    "回収率重視 3連単1着固定 (ROI 232%   的中  1.9%)",
        "all":     "全券種モード",
    }
    print(f"\n  Ticket Suggestions  "
          f"{date_str[:4]}/{date_str[4:6]}/{date_str[6:8]}  "
          f"[{mode_labels.get(mode, mode)}]")

    for place_id in sorted(by_place):
        venue = PLACE_NAME.get(place_id, f"場{place_id}")
        print(f"\n{'#'*52}")
        print(f"  {venue}")
        print(f"{'#'*52}")

        for race_id in by_place[place_id]:
            path = os.path.join(day_dir, f"{race_id}.csv")
            df   = pd.read_csv(path, index_col=0, dtype=str)
            if not all(c in df.columns for c in needed):
                continue
            if pd.to_numeric(df["score"], errors="coerce").isna().all():
                continue

            if mode in ("score", "hitrate"):
                show_hitrate_race(race_id, df)
            elif mode == "tp":
                show_tp_race(race_id, df)
            elif mode == "best":
                show_best_race(race_id, df)
            else:
                show_all_race(race_id, df)

    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default=date.today().strftime("%Y%m%d"))
    parser.add_argument("--mode",
                        choices=["score", "tp", "best", "all", "hitrate"],
                        default="score")
    args = parser.parse_args()
    main(args.date, args.mode)
