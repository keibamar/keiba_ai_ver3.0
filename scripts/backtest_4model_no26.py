"""backtest_4model_no26.py

7/18・7/19の各レースで4モデル戦略を検証する。

戦略:
  ①③ 回収率重視: 単複・3連複ともに v12_nodds（前日可）
  ②   的中率重視: 単複 v11α0.6 / 3連複 v15α0.5（直前・オッズあり）
  ④   バランス  : 単複 v11_nodds / 3連複 v12α0.4（単複前日可・3連複直前）

入力:
  data/race_card/{日付}/{race_id}.csv
  data/race_result/{会場}/{年}/{race_id}.csv
  data/race_info/race_returns/{会場}/{年}/{race_id}.csv
  logs/race_records_unified_2026.pkl
  logs/race_records_nodds_2026.pkl

実行: python scripts/backtest_4model_no26.py
"""

import glob
import itertools
import os
import sys
import pickle

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.config.constants import PLACE_LIST

TARGET_DAYS = ["20260718", "20260719"]
BET_UNIT    = 100
EVAL_YEAR   = 2026

# ── キャッシュ読み込み ──
with open(os.path.join(PROJECT_ROOT, "logs", f"race_records_unified_{EVAL_YEAR}.pkl"), "rb") as f:
    unified_list = pickle.load(f)
with open(os.path.join(PROJECT_ROOT, "logs", f"race_records_nodds_{EVAL_YEAR}.pkl"), "rb") as f:
    nodds_list = pickle.load(f)

unified_map = {r["race_id"]: r for r in unified_list}
nodds_map   = {r["race_id"]: r for r in nodds_list}


# ── ユーティリティ ──
def _norm(arr):
    arr = np.array(arr, dtype=float)
    mn, mx = np.nanmin(arr), np.nanmax(arr)
    return (arr - mn) / (mx - mn + 1e-12)


def blend_score(ur, nr, tan_mk, tan_a, san_mk, san_a):
    """単複本命スコアと3連複スコアを返す（正規化済み）"""
    def _get(mk, a, r_unified, r_nodds):
        if mk in ("v11n", "v12n", "v15n"):
            s = r_nodds.get(f"s_{mk}") if r_nodds else None
            return _norm(s) if s is not None else None
        sv7 = r_unified.get("s_v7") if r_unified else None
        if mk == "v7":
            return _norm(sv7) if sv7 is not None else None
        svx = r_unified.get(f"s_{mk}") if r_unified else None
        if svx is None or sv7 is None:
            return None
        return (1 - a) * _norm(svx) + a * _norm(sv7)

    s_tan = _get(tan_mk, tan_a, ur, nr)
    s_san = _get(san_mk, san_a, ur, nr)
    return s_tan, s_san


def get_honmei_san5(s_tan, s_san, umabans):
    """本命馬番と3連複5頭リストを返す"""
    if s_tan is None:
        return None, None
    honmei = umabans[int(np.argmax(s_tan))]
    if s_san is None:
        s_san = s_tan
    order = np.argsort(-s_san)
    san5 = [honmei]
    for i in order:
        u = umabans[i]
        if u != honmei:
            san5.append(u)
        if len(san5) == 5:
            break
    return honmei, san5


def san_hit(san5, san_winner):
    if san_winner is None or san5 is None:
        return False
    combs = [{a, b, c} for a, b, c in itertools.combinations(san5, 3)]
    return any(c == san_winner for c in combs)


PLACE_NAME_MAP = {str(i + 1).zfill(2): PLACE_LIST[i] for i in range(len(PLACE_LIST))}

RACE_TYPE_MAP = {
    "芝": "芝",
    "ダ": "ダ",
}


# ── 4戦略定義 ──
STRATEGIES = {
    "①③回収率": ("v12n", 0.0, "v12n", 0.0),
    "②的中率  ": ("v11",  0.6, "v15",  0.5),
    "④バランス": ("v11n", 0.0, "v12",  0.4),
}

# ── 集計用 ──
totals = {key: {"n":0,"tan_hit":0,"tan_pay":0,"tan_bet":0,
                "fuku_hit":0,"fuku_pay":0,"fuku_bet":0,
                "san_hit":0,"san_pay":0,"san_bet":0} for key in STRATEGIES}


def _get_race_info(race_id):
    place_id = int(str(race_id)[4:6])
    place_name = PLACE_LIST[place_id - 1]
    paths = glob.glob(
        os.path.join(PROJECT_ROOT, "data", "race_info",
                     place_name, str(EVAL_YEAR), f"{race_id}.csv")
    )
    if not paths:
        return None
    df = pd.read_csv(paths[0], index_col=0)
    return df


# ── メイン処理 ──
for day in TARGET_DAYS:
    race_card_paths = sorted(glob.glob(
        os.path.join(PROJECT_ROOT, "data", "race_card", day, "*.csv")
    ))
    if not race_card_paths:
        print(f"\n{day}: race_card なし")
        continue

    print(f"\n{'='*80}")
    print(f"  {day[:4]}/{day[4:6]}/{day[6:]}  ({len(race_card_paths)}R)")
    print(f"{'='*80}")

    for rc_path in race_card_paths:
        race_id = os.path.basename(rc_path).replace(".csv", "")

        ur = unified_map.get(race_id)
        nr = nodds_map.get(race_id)
        if ur is None:
            print(f"\n  [{race_id}] キャッシュなし（スキップ）")
            continue

        # race_card 読み込み
        try:
            rc_df = pd.read_csv(rc_path, index_col=0).reset_index(drop=True)
        except Exception:
            continue
        if "馬名" not in rc_df.columns or "馬番" not in rc_df.columns:
            continue

        # レース情報
        ri = _get_race_info(race_id)
        if ri is not None and not ri.empty:
            race_label = f"{ri.at[0,'race_type']}{ri.at[0,'course_len']}m {ri.at[0,'class']}"
        else:
            place_id = int(str(race_id)[4:6])
            race_label = PLACE_LIST[place_id - 1]

        # 馬名マップ（馬番→馬名）
        umabans = ur["umabans"]
        umaban_to_name = {}
        for _, row in rc_df.iterrows():
            try:
                ub = str(int(float(row["馬番"])))
                umaban_to_name[ub] = row["馬名"]
            except Exception:
                pass

        # 実際の結果
        winner_set  = ur["winner_set"]
        top3_set    = ur["top3_set"]
        san_winner  = ur["san_winner"]
        san_odds_v  = ur["san_odds"]
        tan_ret     = ur["tan_ret"]
        fuku_rets   = ur["fuku_rets"]

        # ── ヘッダー行 ──
        place_id = int(str(race_id)[4:6])
        kai  = str(race_id)[6:8]
        nichi = str(race_id)[8:10]
        rnum = str(race_id)[10:12]
        print(f"\n  [{race_label}]  {PLACE_LIST[place_id-1]} {int(kai)}回{int(nichi)}日目 {int(rnum)}R")

        # 全馬の人気・オッズ表示（参考）
        try:
            pop_map = {str(int(float(r["馬番"]))): f"{r['馬名']}({r['人気']}番人気/{r['オッズ']}倍)"
                       for _, r in rc_df.iterrows()
                       if pd.notna(r.get("人気")) and pd.notna(r.get("オッズ"))}
        except Exception:
            pop_map = {}

        # ── 各戦略の予想 ──
        pred = {}
        for label, (tm, ta, sm, sa) in STRATEGIES.items():
            s_tan, s_san = blend_score(ur, nr, tm, ta, sm, sa)
            honmei, san5 = get_honmei_san5(s_tan, s_san, umabans)
            pred[label] = (honmei, san5, s_tan, s_san)

        # ── 結果 ──
        winner_name = ", ".join(umaban_to_name.get(u, u) for u in winner_set)
        top3_names  = " - ".join(
            f"{umaban_to_name.get(u,u)}" for u in sorted(top3_set,
            key=lambda x: int(x) if x.isdigit() else 99)
        )
        san5_names  = f"{'-'.join(sorted(san_winner, key=lambda x: int(x) if x.isdigit() else 99))}（3連複{san_odds_v}円）" if san_winner else "---"

        print(f"    【結果】 1着: {winner_name}  /  3着内: {top3_names}  /  3連複: {san5_names}")

        # ── 各戦略の予想と結果判定 ──
        for label, (tm, ta, sm, sa) in STRATEGIES.items():
            honmei, san5, s_tan, s_san = pred[label]
            if honmei is None:
                print(f"    {label}: スコアなし")
                continue

            honmei_name = umaban_to_name.get(honmei, honmei)
            san5_names_pred = " ".join(f"{umaban_to_name.get(u,u)}" for u in san5)

            tan_ok   = "◎" if honmei in winner_set else "✗"
            fuku_ok  = "○" if honmei in top3_set  else "✗"
            san_ok   = "◎" if san_winner and san_hit(san5, san_winner) else "✗"

            tan_pay_val  = tan_ret if honmei in winner_set and tan_ret else 0
            fuku_pay_val = (fuku_rets.get(honmei) or 0) if honmei in top3_set else 0
            san_pay_val  = san_odds_v if (san_winner and san_hit(san5, san_winner) and san_odds_v) else 0

            print(f"    {label}: 本命={honmei_name}")
            print(f"      単{tan_ok}{tan_pay_val:>6}円  複{fuku_ok}{fuku_pay_val:>6}円  "
                  f"3連複{san_ok}{san_pay_val:>7}円  3連複BOX=[{san5_names_pred}]")

            # 集計
            t = totals[label]
            t["n"]        += 1
            t["tan_bet"]  += BET_UNIT
            t["fuku_bet"] += BET_UNIT
            t["san_bet"]  += BET_UNIT * 10
            if honmei in winner_set and tan_ret:
                t["tan_hit"] += 1; t["tan_pay"] += tan_ret
            if honmei in top3_set:
                t["fuku_hit"] += 1
                fv = fuku_rets.get(honmei)
                if fv: t["fuku_pay"] += fv
            if san_winner and san_hit(san5, san_winner) and san_odds_v:
                t["san_hit"] += 1; t["san_pay"] += san_odds_v


# ── 集計結果 ──
print(f"\n{'='*80}")
print(f"  集計結果（{'/'.join(TARGET_DAYS)}）")
print(f"{'='*80}")
print(f"  {'戦略':<12}  {'単勝的中':>8} {'単勝回収':>8}  {'複勝的中':>8} {'複勝回収':>8}  {'3連複的中':>9} {'3連複回収':>9}")
for label, t in totals.items():
    if t["n"] == 0:
        continue
    n = t["n"]
    print(f"  {label}  "
          f"{t['tan_hit']:3}/{n:3}={100*t['tan_hit']/n:5.1f}% "
          f"{100*t['tan_pay']/t['tan_bet']:6.1f}%  "
          f"{t['fuku_hit']:3}/{n:3}={100*t['fuku_hit']/n:5.1f}% "
          f"{100*t['fuku_pay']/t['fuku_bet']:6.1f}%  "
          f"{t['san_hit']:3}/{n:3}={100*t['san_hit']/n:5.1f}% "
          f"{100*t['san_pay']/t['san_bet']:6.1f}%")
