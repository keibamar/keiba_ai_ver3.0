"""馬券提案をメール送信するスクリプト

戦略:
  ダート × 良馬場 : axis_prob [0.18, 0.20)  実績ROI 455%
  芝   × 良馬場 : axis_prob >= 0.25        実績ROI 174%
  道悪（稍重/重/不良）: 全体提案 + 実績低調を注記  実績ROI 67%

使い方:
    python scripts/send_ticket_mail.py [--date YYYYMMDD [YYYYMMDD ...]] [--dry-run]

    --dry-run: メール送信せず本文を標準出力に表示するだけ
"""

import argparse
import os
import sys
from datetime import date, datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import constants, paths
from src.logic.betting.ticket_advisor import (
    recommend_score_based, SB_B_PARAMS, SB_GOOD_PARAMS, SB_MUDDY_PARAMS,
)
from src.managers import race_card_dataset_manager
from src.output.prediction_publisher import make_mime_text, send_gmail

# ── 場所名マップ ─────────────────────────────────────────────────
PLACE_NAME = {str(i+1).zfill(2): name for i, name in enumerate(constants.NAME_LIST)}

# ── 戦略別フィルター定義 ─────────────────────────────────────────
# (lo, hi) : hi=1.0 は上限なし扱い
STRATEGY = {
    "ダート良":  (0.18, 0.20),
    "芝良":      (0.25, 1.01),
    "道悪":      (0.00, 1.01),
}

STRATEGY_ROI = {
    "ダート良": "455%",
    "芝良":     "174%",
    "道悪":      "83%",
}

WEEKDAY_JP = ["月", "火", "水", "木", "金", "土", "日"]


def _get_strategy(rtype: str, ground: str) -> str:
    if ground == "良":
        return "ダート良" if rtype == "ダート" else "芝良"
    return "道悪"


def _axis_passes(ap: float, strategy: str) -> bool:
    lo, hi = STRATEGY[strategy]
    return lo <= ap < hi


def _fmt_tickets(bet_type: str, tickets: list) -> list[str]:
    lines = []
    for t in tickets:
        combo = t["組合せ"]
        if bet_type == "馬連":
            c = f"{combo[0]}-{combo[1]}"
        elif bet_type == "3連複":
            c = "-".join(map(str, combo))
        else:
            c = str(combo)
        lines.append(c)
    return lines


def build_race_block(race_id: str, date_str: str, time_map: dict, strategy: str,
                     rec: dict, df: pd.DataFrame) -> tuple[str, int]:
    """1レース分のテキストブロックを返す。(text, 支出額)"""

    place_id = race_id[4:6]
    race_no  = int(race_id[10:12])
    venue    = PLACE_NAME.get(place_id, f"場{place_id}")
    race_info_map = time_map.get(race_id, {})
    race_name = race_info_map.get("race_name", "")
    race_time_raw = race_info_map.get("race_time", "")
    start_time = f"{race_time_raw[:2]}:{race_time_raw[2:]}" if len(race_time_raw) == 4 else ""

    meta      = rec.get("_meta", {})
    axis_prob = meta.get("axis_prob", 0)
    um_rec    = rec.get("馬連",  {})
    tp_rec    = rec.get("3連複", {})
    um_tickets = um_rec.get("tickets", [])
    tp_tickets = tp_rec.get("tickets", [])
    um_method  = um_rec.get("選択方式", "")
    tp_method  = tp_rec.get("選択方式", "")

    ax_list = um_rec.get("軸") or tp_rec.get("軸") or []

    # 軸馬名
    uma_map = {}
    for _, row in df.iterrows():
        try:
            no = int(pd.to_numeric(row["馬番"], errors="coerce"))
            uma_map[no] = str(row["馬名"])
        except Exception:
            pass

    header_mark  = "◎" if strategy != "道悪" else "△"
    strategy_tag = f"★{strategy}" if strategy != "道悪" else "道悪"
    cost = (len(um_tickets) + len(tp_tickets)) * 100

    lines = []
    lines.append(
        f"{header_mark} {venue} {race_no}R  {race_name}  {start_time}"
        f"  [axis={axis_prob*100:.0f}% {strategy_tag}]"
    )

    if ax_list:
        ax_str = " + ".join(f"#{n} {uma_map.get(n, '?')}" for n in ax_list)
        lines.append(f"  軸 {ax_str}")

    if um_tickets:
        combos = _fmt_tickets("馬連", um_tickets)
        lines.append(f"  馬連({um_method}): {', '.join(combos)}")

    if tp_tickets:
        combos = _fmt_tickets("3連複", tp_tickets)
        lines.append(f"  3連複({tp_method}): {', '.join(combos)}")

    lines.append(f"  支出 {cost}円  ({len(um_tickets)}点馬連 / {len(tp_tickets)}点3連複)")

    return "\n".join(lines), cost


def build_mail_body(date_str: str, all_races: list[dict]) -> str:
    """
    all_races: list of dicts with keys:
      venue, place_id, race_id, strategy, block_text, cost
    """
    dt = datetime.strptime(date_str, "%Y%m%d")
    weekday = WEEKDAY_JP[dt.weekday()]
    date_label = f"{dt.year}/{dt.month:02d}/{dt.day:02d}({weekday})"

    # カテゴリ別集計
    stats: dict[str, dict] = {s: {"races": 0, "cost": 0} for s in STRATEGY}
    for r in all_races:
        s = r["strategy"]
        stats[s]["races"] += 1
        stats[s]["cost"]  += r["cost"]

    total_races = len(all_races)
    total_cost  = sum(r["cost"] for r in all_races)

    lines = []
    lines.append(f"競馬AI 馬券提案 — {date_label}")
    lines.append("=" * 52)
    lines.append("【戦略】ダート良: axis 18-20% (実績499%)  芝良: axis 30%+ (実績236%)")
    lines.append("【道悪】稍重/重/不良は全提案（実績ROI 83% 参考のみ — 買い控え推奨）")
    lines.append("")

    # 良馬場分
    good_races = [r for r in all_races if r["strategy"] != "道悪"]
    muddy_races = [r for r in all_races if r["strategy"] == "道悪"]

    if good_races:
        lines.append("◆◆ 良馬場提案 ◆◆")
        lines.append("=" * 52)
        # 場所でグループ化
        venues_seen = []
        by_venue: dict[str, list] = {}
        for r in good_races:
            v = r["venue"]
            if v not in by_venue:
                venues_seen.append(v)
                by_venue[v] = []
            by_venue[v].append(r)

        for v in venues_seen:
            lines.append(f"\n■ {v} ■")
            for r in by_venue[v]:
                lines.append(r["block_text"])
                lines.append("")

    if muddy_races:
        lines.append("")
        lines.append("⚠ 道悪開催（実績ROI 83%: 参考提案 — 買い控え推奨）")
        lines.append("=" * 52)
        venues_seen = []
        by_venue = {}
        for r in muddy_races:
            v = r["venue"]
            if v not in by_venue:
                venues_seen.append(v)
                by_venue[v] = []
            by_venue[v].append(r)

        for v in venues_seen:
            lines.append(f"\n■ {v} ■")
            for r in by_venue[v]:
                lines.append(r["block_text"])
                lines.append("")

    # サマリ
    lines.append("")
    lines.append("【まとめ】")
    lines.append("-" * 40)
    for s, roi in STRATEGY_ROI.items():
        st = stats[s]
        if st["races"] == 0:
            continue
        mark = "⚠" if s == "道悪" else "★"
        lines.append(
            f"  {mark}{s}  (実績ROI {roi}): "
            f"{st['races']}R  {st['cost']:,}円"
        )
    lines.append("-" * 40)
    lines.append(f"  合計: {total_races}R  {total_cost:,}円")
    lines.append("=" * 52)

    return "\n".join(lines)


def run(date_str: str, dry_run: bool = False):
    day_dir = os.path.join(paths.RACE_CARD_DATA_PATH, date_str)
    if not os.path.isdir(day_dir):
        print(f"[SKIP] レースカードなし: {date_str}")
        return

    # 発走時刻・レース名マップを構築
    dt = datetime.strptime(date_str, "%Y%m%d").date()
    time_id_df = race_card_dataset_manager.get_race_time_id_list_df(dt)
    time_map: dict[str, dict] = {}
    for _, row in time_id_df.iterrows():
        rid = str(row.get("race_id", ""))
        time_map[rid] = {
            "race_time": str(row.get("race_time", "")),
            "race_name": str(row.get("race_name", "")),
        }

    needed = ["score_hitrate", "馬番"]
    all_races: list[dict] = []

    race_files = sorted(f for f in os.listdir(day_dir) if f.endswith(".csv"))
    for fname in race_files:
        race_id = fname.replace(".csv", "")
        path    = os.path.join(day_dir, fname)
        df      = pd.read_csv(path, index_col=0, dtype=str)

        if not all(c in df.columns for c in needed):
            continue
        if pd.to_numeric(df["score_hitrate"], errors="coerce").isna().all():
            continue

        # race_info から馬場種別・状態を取得
        info = race_card_dataset_manager.get_race_info_csv(race_id)
        if info.empty:
            continue
        row0     = info.iloc[0]
        rtype    = str(row0.get("race_type",    "?")).strip()
        ground   = str(row0.get("ground_state", "?")).strip()
        course_len = str(row0.get("course_len",  "")).strip()
        cls      = str(row0.get("class",        "")).strip()

        strategy = _get_strategy(rtype, ground)

        # axis_prob 確認のため標準パラメータで1回実行
        rec_base = recommend_score_based(df, win_odds_col=None, **SB_B_PARAMS)
        if not rec_base:
            continue

        axis_prob = rec_base.get("_meta", {}).get("axis_prob", 0)

        # axis_prob フィルター適用
        if not _axis_passes(axis_prob, strategy):
            continue

        # 戦略別パラメータで本番実行
        params = SB_MUDDY_PARAMS if strategy == "道悪" else SB_GOOD_PARAMS
        rec = recommend_score_based(df, win_odds_col=None, **params)
        if not rec:
            continue
        # axis_prob はスコア分布から決まるため base と同値
        rec.setdefault("_meta", {})["axis_prob"] = axis_prob

        # チケットが空の場合はスキップ
        um_t = rec.get("馬連",  {}).get("tickets", [])
        tp_t = rec.get("3連複", {}).get("tickets", [])
        if not um_t and not tp_t:
            continue

        um_tickets = rec.get("馬連",  {}).get("tickets", [])
        tp_tickets = rec.get("3連複", {}).get("tickets", [])
        if not um_tickets and not tp_tickets:
            continue

        # race_info のコース情報をレース名行に付加
        place_id = race_id[4:6]
        venue    = PLACE_NAME.get(place_id, f"場{place_id}")
        course_label = f"{rtype}{course_len}m {ground}"
        # time_map のrace_nameにコース情報を追加（表示用）
        if race_id in time_map:
            base_name = time_map[race_id]["race_name"]
            time_map[race_id]["race_name"] = f"{base_name}  {course_label}"
        else:
            time_map[race_id] = {"race_time": "", "race_name": course_label}

        block_text, cost = build_race_block(race_id, date_str, time_map, strategy, rec, df)

        all_races.append({
            "venue":      venue,
            "place_id":   place_id,
            "race_id":    race_id,
            "strategy":   strategy,
            "block_text": block_text,
            "cost":       cost,
        })

    if not all_races:
        print(f"[{date_str}] 提案レースなし（条件を満たすレースがありません）")
        return

    # 場所・レース番号順でソート
    all_races.sort(key=lambda r: (r["place_id"], r["race_id"][-2:]))

    body = build_mail_body(date_str, all_races)

    dt   = datetime.strptime(date_str, "%Y%m%d")
    wday = WEEKDAY_JP[dt.weekday()]
    good_count  = sum(1 for r in all_races if r["strategy"] != "道悪")
    muddy_count = sum(1 for r in all_races if r["strategy"] == "道悪")
    subject = (
        f"【競馬AI】{dt.month}/{dt.day}({wday}) "
        f"良馬場{good_count}R"
        + (f" 道悪{muddy_count}R" if muddy_count else "")
    )

    if dry_run:
        print(f"件名: {subject}")
        print("=" * 60)
        print(body)
    else:
        msg = make_mime_text(
            mail_to=os.environ.get("GMAIL_SEND_TO", ""),
            subject=subject,
            body=body,
        )
        send_gmail(msg)
        print(f"送信完了: {subject}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", nargs="+",
                        default=[date.today().strftime("%Y%m%d")])
    parser.add_argument("--dry-run", action="store_true",
                        help="メール送信せず内容を標準出力に表示")
    args = parser.parse_args()

    for d in args.date:
        print(f"\n{'='*60}")
        print(f"  処理日: {d}")
        print(f"{'='*60}")
        run(d, dry_run=args.dry_run)
