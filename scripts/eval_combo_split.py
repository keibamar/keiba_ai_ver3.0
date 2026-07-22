"""
単複モデル × 3連複モデル 分離最適化スクリプト

【戦略】
  - 単勝・複勝: 「単複モデル」の指数1位馬のみで投票
  - 3連複: 「単複モデルの1位」+「3連複モデルの1〜4位(重複時は5位まで繰り上げ)」5頭BOX
            C(5,3)=10通り × 100円

【ブレンド定義】
  blend(α) = (1-α)*norm(s_v15) + α*norm(s_v7)
  α=0.0 → v15pace 100%  /  α=1.0 → v7odds 100%

実行: python scripts/eval_combo_split.py
"""

import os, sys, pickle, itertools
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

BET_UNIT = 100
CACHE    = os.path.join(PROJECT_ROOT, "logs", "race_records_v15eval_2026.pkl")

print(f"キャッシュ読み込み: {CACHE}")
with open(CACHE, "rb") as f:
    all_records = pickle.load(f)

# v7・v15 両方あるレコードのみ
records = [r for r in all_records if r.get("s_v7") is not None and r.get("s_v15") is not None]
print(f"有効レコード: {len(records)}R（全{len(all_records)}R中）")


def _norm(arr):
    mn, mx = arr.min(), arr.max()
    return (arr - mn) / (mx - mn + 1e-12)


def blend(r, alpha):
    return (1 - alpha) * _norm(r["s_v15"]) + alpha * _norm(r["s_v7"])


def top_n(scores, umabans, n):
    order = np.argsort(-np.array(scores))
    return [umabans[i] for i in order[:n]]


def evaluate(records, a_tan, a_san):
    n = tan_hit = tan_pay = tan_bet = 0
    fuku_hit = fuku_pay = fuku_bet = 0
    san_hit = san_pay = san_bet = 0

    for r in records:
        s_tan_arr = blend(r, a_tan)
        s_san_arr = blend(r, a_san)
        umabans   = r["umabans"]

        # 単複モデルの本命（1位）
        honmei = top_n(s_tan_arr, umabans, 1)[0]

        # 3連複用5頭を構成
        san_pool = top_n(s_san_arr, umabans, 5)
        san5 = [honmei]
        for u in san_pool:
            if u != honmei:
                san5.append(u)
            if len(san5) == 5:
                break

        # 単勝
        tan_bet += BET_UNIT
        if honmei in r["winner_set"]:
            tan_hit += 1
            if r["tan_ret"]:
                tan_pay += r["tan_ret"]

        # 複勝
        fuku_bet += BET_UNIT
        if honmei in r["top3_set"]:
            fuku_hit += 1
            ret = r["fuku_rets"].get(honmei)
            if ret:
                fuku_pay += ret

        # 3連複BOX（C(5,3)=10通り）
        combs = list(itertools.combinations(san5, 3))
        san_bet += BET_UNIT * len(combs)
        if r["san_winner"] is not None:
            if any({a, b, c} == r["san_winner"] for a, b, c in combs):
                san_hit += 1
                if r["san_odds"]:
                    san_pay += r["san_odds"]

        n += 1

    tan_pct  = 100 * tan_hit  / n         if n          else 0
    tan_rec  = 100 * tan_pay  / tan_bet   if tan_bet    else 0
    fuku_pct = 100 * fuku_hit / n         if n          else 0
    fuku_rec = 100 * fuku_pay / fuku_bet  if fuku_bet   else 0
    san_pct  = 100 * san_hit  / n         if n          else 0
    san_rec  = 100 * san_pay  / san_bet   if san_bet    else 0
    return dict(n=n,
                tan_pct=tan_pct,  tan_rec=tan_rec,
                fuku_pct=fuku_pct, fuku_rec=fuku_rec,
                san_pct=san_pct,  san_rec=san_rec)


# ── 全121通りスイープ ──
alphas = [round(a * 0.1, 1) for a in range(0, 11)]

print("\nスイープ中 (121通り)...")
results = []
for a_tan in alphas:
    for a_san in alphas:
        res = evaluate(records, a_tan, a_san)
        results.append((a_tan, a_san, res))

# ── ベースライン（v7同一・従来方式）──
base = evaluate(records, 1.0, 1.0)
SEP = "=" * 110
print(f"\n{SEP}")
print(f"  ベースライン(v7同一・単複1位/3連複同一5頭BOX): "
      f"n={base['n']}  "
      f"単勝 {base['tan_pct']:.1f}%/{base['tan_rec']:.1f}%  "
      f"複勝 {base['fuku_pct']:.1f}%/{base['fuku_rec']:.1f}%  "
      f"3連複 {base['san_pct']:.1f}%/{base['san_rec']:.1f}%  (10点×100円)")
print(SEP)

# ── 各指標の最良 ──
metrics = [
    ("単勝的中率",   lambda x: x[2]["tan_pct"]),
    ("単勝回収率",   lambda x: x[2]["tan_rec"]),
    ("複勝的中率",   lambda x: x[2]["fuku_pct"]),
    ("複勝回収率",   lambda x: x[2]["fuku_rec"]),
    ("3連複的中率",  lambda x: x[2]["san_pct"]),
    ("3連複回収率",  lambda x: x[2]["san_rec"]),
]

print("\n▼ 各指標最良（単複α × 3連複α）\n")
print(f"  {'指標':<12}  {'単複α':>5}  {'3連複α':>6}   {'単勝的中/回収':>14}  {'複勝的中/回収':>14}  {'3連複的中/回収':>14}")
print("  " + "-" * 100)
for label, key_fn in metrics:
    a_tan, a_san, res = max(results, key=key_fn)
    print(f"  {label:<12}  {a_tan:>5.1f}  {a_san:>6.1f}   "
          f"{res['tan_pct']:>6.1f}%/{res['tan_rec']:>6.1f}%   "
          f"{res['fuku_pct']:>6.1f}%/{res['fuku_rec']:>6.1f}%   "
          f"{res['san_pct']:>6.1f}%/{res['san_rec']:>6.1f}%")

# ── 全121通りの一覧（3連複的中率降順）──
print(f"\n\n▼ 全121通り（3連複的中率降順）\n")
print(f"  {'単複α':>5}  {'3連複α':>6}   {'単勝的中/回収':>14}  {'複勝的中/回収':>14}  {'3連複的中/回収':>14}")
print("  " + "-" * 90)
for a_tan, a_san, res in sorted(results, key=lambda x: -x[2]["san_pct"])[:30]:
    marker = "  ◀" if a_tan != a_san else ""
    print(f"  {a_tan:>5.1f}  {a_san:>6.1f}   "
          f"{res['tan_pct']:>6.1f}%/{res['tan_rec']:>6.1f}%   "
          f"{res['fuku_pct']:>6.1f}%/{res['fuku_rec']:>6.1f}%   "
          f"{res['san_pct']:>6.1f}%/{res['san_rec']:>6.1f}%{marker}")

print(f"\n{'=' * 110}")
print("完了")
