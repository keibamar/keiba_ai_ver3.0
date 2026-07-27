"""戦略別 推奨結果レポート

send_ticket_mail.py と同じ戦略ロジックを適用し、
推奨チケットの的中・配当結果を表示する。

使い方:
    python scripts/show_strategy_results.py --date 20260725 20260726
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import constants, paths
from src.logic.betting.ticket_advisor import (
    recommend_score_based, SB_B_PARAMS, SB_GOOD_PARAMS, SB_MUDDY_PARAMS,
)
from src.managers import race_card_dataset_manager
from scripts.explore_score_based import get_top3_and_odds, get_payout, is_hit

PLACE_NAME = {str(i+1).zfill(2): name for i, name in enumerate(constants.NAME_LIST)}
WEEKDAY_JP = ["月", "火", "水", "木", "金", "土", "日"]

STRATEGIES = {
    "ダート良":  dict(rtype="ダート", muddy=False, ap_lo=0.18, ap_hi=0.20, roi_ref="455%"),
    "芝良":      dict(rtype="芝",     muddy=False, ap_lo=0.25, ap_hi=1.01, roi_ref="174%"),
    "道悪":      dict(rtype=None,     muddy=True,  ap_lo=0.00, ap_hi=1.01, roi_ref=" 83%"),
}
MUDDY = {"稍重", "重", "不良"}


def _get_strategy(rtype, ground):
    if ground == "良":
        return "ダート良" if rtype == "ダート" else "芝良"
    if ground in MUDDY:
        return "道悪"
    return None


def _axis_passes(ap, sname):
    s = STRATEGIES[sname]
    return s["ap_lo"] <= ap < s["ap_hi"]


def _fmt_combo(bet_type, combo):
    if bet_type == "馬連":
        return f"{combo[0]}-{combo[1]}"
    return "-".join(map(str, combo))


def run_date(date_str):
    day_dir = os.path.join(paths.RACE_CARD_DATA_PATH, date_str)
    if not os.path.isdir(day_dir):
        print(f"[SKIP] {date_str}: race_card なし")
        return None

    dt      = datetime.strptime(date_str, "%Y%m%d")
    wday    = WEEKDAY_JP[dt.weekday()]
    date_label = f"{dt.year}/{dt.month:02d}/{dt.day:02d}({wday})"

    # 発走時刻マップ
    time_id_df = race_card_dataset_manager.get_race_time_id_list_df(dt.date())
    time_map = {}
    for _, row in time_id_df.iterrows():
        rid = str(row.get("race_id", ""))
        time_map[rid] = {
            "race_time": str(row.get("race_time", "")),
            "race_name": str(row.get("race_name", "")),
        }

    totals = {s: {"cost": 0, "ret": 0, "races": 0, "hits": 0} for s in STRATEGIES}
    race_blocks = {s: [] for s in STRATEGIES}

    for fname in sorted(f for f in os.listdir(day_dir) if f.endswith(".csv")):
        race_id = fname.replace(".csv", "")
        df = pd.read_csv(os.path.join(day_dir, fname), index_col=0, dtype=str)

        if not all(c in df.columns for c in ["score_hitrate", "馬番"]):
            continue
        if pd.to_numeric(df["score_hitrate"], errors="coerce").isna().all():
            continue

        top3, odds_map = get_top3_and_odds(race_id)
        if top3 is None:
            continue

        # race_info
        info = race_card_dataset_manager.get_race_info_csv(race_id)
        if info.empty:
            continue
        row0   = info.iloc[0]
        rtype  = str(row0.get("race_type",    "?")).strip()
        ground = str(row0.get("ground_state", "?")).strip()
        clen   = str(row0.get("course_len",   "")).strip()

        sname = _get_strategy(rtype, ground)
        if sname is None:
            continue

        # axis_prob チェック（SB_B_PARAMS ベース）
        rec_base = recommend_score_based(df, win_odds_col=None, **SB_B_PARAMS)
        if not rec_base:
            continue
        ap = rec_base.get("_meta", {}).get("axis_prob", 0)
        if not _axis_passes(ap, sname):
            continue

        # 戦略別パラメータで推奨
        params = SB_MUDDY_PARAMS if sname == "道悪" else SB_GOOD_PARAMS
        rec = recommend_score_based(df, win_odds_col=None, **params)
        if not rec:
            continue
        rec.setdefault("_meta", {})["axis_prob"] = ap

        um_tickets = rec.get("馬連",  {}).get("tickets", [])
        tp_tickets = rec.get("3連複", {}).get("tickets", [])
        if not um_tickets and not tp_tickets:
            continue

        # ── レース情報 ───────────────────────────────────────────
        place_id   = race_id[4:6]
        race_no    = int(race_id[10:12])
        venue      = PLACE_NAME.get(place_id, f"場{place_id}")
        tm_info    = time_map.get(race_id, {})
        race_name  = tm_info.get("race_name", "")
        race_time_r = tm_info.get("race_time", "")
        start_time = f"{race_time_r[:2]}:{race_time_r[2:]}" if len(race_time_r) == 4 else ""

        mark = "◎" if sname != "道悪" else "△"
        header = (f"{mark} {venue} {race_no}R  {race_name}  {rtype}{clen}m {ground}"
                  f"  {start_time}  [axis={ap*100:.0f}% ★{sname}]")

        lines = [header]
        lines.append(f"  結果: 1着#{top3[0]} 2着#{top3[1]} 3着#{top3[2]}")

        race_cost = race_ret = 0
        race_hit  = False

        # 馬連
        if um_tickets:
            um_method = rec.get("馬連", {}).get("選択方式", "")
            lines.append(f"  馬連({um_method}):")
            for t in um_tickets:
                combo = t["組合せ"]
                hit   = is_hit("馬連", combo, top3)
                pay   = get_payout(race_id, "馬連", combo) if hit else 0
                mark2 = f"HIT! {pay}円" if hit else "miss"
                lines.append(f"    {_fmt_combo('馬連', combo):<10} → {mark2}")
                race_cost += 100; race_ret += pay
                if hit:
                    race_hit = True

        # 3連複
        if tp_tickets:
            tp_method = rec.get("3連複", {}).get("選択方式", "")
            lines.append(f"  3連複({tp_method}):")
            for t in tp_tickets:
                combo = t["組合せ"]
                hit   = is_hit("3連複", combo, top3)
                pay   = get_payout(race_id, "3連複", combo) if hit else 0
                mark2 = f"HIT! {pay}円" if hit else "miss"
                lines.append(f"    {_fmt_combo('3連複', combo):<12} → {mark2}")
                race_cost += 100; race_ret += pay
                if hit:
                    race_hit = True

        roi_r = race_ret / race_cost * 100 if race_cost else 0
        pnl   = race_ret - race_cost
        lines.append(f"  支出{race_cost}円 / 回収{race_ret}円 / PL{'+' if pnl>=0 else ''}{pnl}円 / ROI{roi_r:.0f}%")
        lines.append("")

        totals[sname]["cost"]  += race_cost
        totals[sname]["ret"]   += race_ret
        totals[sname]["races"] += 1
        if race_hit:
            totals[sname]["hits"] += 1

        race_blocks[sname].append("\n".join(lines))

    # ── 出力 ──────────────────────────────────────────────────────
    W = 60
    print(f"\n{'='*W}")
    print(f"  {date_label} 推奨結果")
    print(f"{'='*W}")

    for sname, blocks in race_blocks.items():
        sdef = STRATEGIES[sname]
        if not blocks:
            continue
        t = totals[sname]
        roi = t["ret"] / t["cost"] * 100 if t["cost"] else 0
        print(f"\n── {sname}  (実績参考ROI {sdef['roi_ref']}) ─────────────")
        for b in blocks:
            print(b)

    # ── サマリ ──────────────────────────────────────────────────
    print(f"\n{'─'*W}")
    print(f"  【{date_label} まとめ】")
    print(f"{'─'*W}")
    grand_cost = grand_ret = 0
    for sname, t in totals.items():
        if t["races"] == 0:
            continue
        roi  = t["ret"] / t["cost"] * 100 if t["cost"] else 0
        pnl  = t["ret"] - t["cost"]
        mark = "★" if sname != "道悪" else "⚠"
        print(f"  {mark}{sname}  {t['races']}R {t['hits']}的中  "
              f"支出{t['cost']:,}円 回収{t['ret']:,}円 "
              f"PL{'+' if pnl>=0 else ''}{pnl:,}円  ROI{roi:.0f}%")
        grand_cost += t["cost"]
        grand_ret  += t["ret"]

    grand_roi = grand_ret / grand_cost * 100 if grand_cost else 0
    grand_pnl = grand_ret - grand_cost
    print(f"{'─'*40}")
    print(f"  合計  支出{grand_cost:,}円 回収{grand_ret:,}円 "
          f"PL{'+' if grand_pnl>=0 else ''}{grand_pnl:,}円  ROI{grand_roi:.0f}%")
    print()

    return {s: totals[s] for s in STRATEGIES}


def run(dates):
    all_totals = {s: {"cost": 0, "ret": 0, "races": 0, "hits": 0} for s in STRATEGIES}

    for d in dates:
        day_totals = run_date(d)
        if day_totals:
            for s in STRATEGIES:
                for k in ("cost", "ret", "races", "hits"):
                    all_totals[s][k] += day_totals[s][k]

    if len(dates) > 1:
        W = 60
        print(f"\n{'='*W}")
        print(f"  【期間合計: {dates[0]}〜{dates[-1]}】")
        print(f"{'='*W}")
        grand_cost = grand_ret = 0
        for sname, t in all_totals.items():
            if t["races"] == 0:
                continue
            roi  = t["ret"] / t["cost"] * 100 if t["cost"] else 0
            pnl  = t["ret"] - t["cost"]
            mark = "★" if sname != "道悪" else "⚠"
            print(f"  {mark}{sname}  {t['races']}R {t['hits']}的中  "
                  f"支出{t['cost']:,}円 回収{t['ret']:,}円 "
                  f"PL{'+' if pnl>=0 else ''}{pnl:,}円  ROI{roi:.0f}%")
            grand_cost += t["cost"]
            grand_ret  += t["ret"]
        grand_roi = grand_ret / grand_cost * 100 if grand_cost else 0
        grand_pnl = grand_ret - grand_cost
        print(f"{'─'*40}")
        print(f"  合計  支出{grand_cost:,}円 回収{grand_ret:,}円 "
              f"PL{'+' if grand_pnl>=0 else ''}{grand_pnl:,}円  ROI{grand_roi:.0f}%")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", nargs="+",
                        default=["20260725", "20260726"])
    args = parser.parse_args()
    run(args.date)
