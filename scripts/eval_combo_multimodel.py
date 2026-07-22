"""
eval_combo_multimodel.py

各モデル系列で「単複モデル × 3連複モデル」α分離スイープを実施し最良値を比較。

  v11eval (1777R): s_v7 / s_v9 / s_v11
    → blend_tan = (1-a)*norm(s_v11) + a*norm(s_v7)
    → blend_san = (1-b)*norm(s_v11) + b*norm(s_v7)
    → 11×11=121通り
  v12eval (1777R): s_v7 / s_v12
    → 同様 121通り
  v15eval (1834R): s_v7 / s_v15
    → 同様 121通り（=前回 eval_combo_split の再集計）

各系列の最良値を並べて「v11 vs v12 vs v15」を比較する。
"""

import os, sys, pickle, itertools
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

BET_UNIT = 100
CACHE_DIR = os.path.join(PROJECT_ROOT, "logs")


# ─── 共通ユーティリティ ────────────────────────────────────────────────────────

def _norm(arr):
    mn, mx = arr.min(), arr.max()
    return (arr - mn) / (mx - mn + 1e-12)


def evaluate_combo(records, key_new, alpha_tan, alpha_san):
    """
    blend_tan = (1-alpha_tan)*norm(key_new) + alpha_tan*norm(s_v7)  → 単複本命1頭
    blend_san = (1-alpha_san)*norm(key_new) + alpha_san*norm(s_v7)  → 3連複5頭プール
    """
    n = tan_hit = tan_pay = tan_bet = 0
    fuku_hit = fuku_pay = fuku_bet = 0
    san_hit = san_pay = san_bet = 0

    for r in records:
        s_new = r.get(key_new)
        s_v7  = r.get("s_v7")
        if s_new is None or s_v7 is None:
            continue

        s_new_n = _norm(np.array(s_new, dtype=float))
        s_v7_n  = _norm(np.array(s_v7,  dtype=float))

        s_tan = (1 - alpha_tan) * s_new_n + alpha_tan * s_v7_n
        s_san = (1 - alpha_san) * s_new_n + alpha_san * s_v7_n

        umabans   = r["umabans"]
        honmei    = umabans[np.argmax(s_tan)]

        order_san = np.argsort(-s_san)
        san5 = [honmei]
        for i in order_san:
            u = umabans[i]
            if u != honmei:
                san5.append(u)
            if len(san5) == 5:
                break

        tan_bet += BET_UNIT
        if honmei in r["winner_set"]:
            tan_hit += 1
            if r["tan_ret"]: tan_pay += r["tan_ret"]

        fuku_bet += BET_UNIT
        if honmei in r["top3_set"]:
            fuku_hit += 1
            ret = r["fuku_rets"].get(honmei)
            if ret: fuku_pay += ret

        combs = list(itertools.combinations(san5, 3))
        san_bet += BET_UNIT * len(combs)
        if r["san_winner"] is not None:
            if any({a, b, c} == r["san_winner"] for a, b, c in combs):
                san_hit += 1
                if r["san_odds"]: san_pay += r["san_odds"]

        n += 1

    if n == 0:
        return None
    return dict(
        n       = n,
        tan_pct = 100 * tan_hit  / n,
        tan_rec = 100 * tan_pay  / tan_bet  if tan_bet  else 0,
        fuku_pct= 100 * fuku_hit / n,
        fuku_rec= 100 * fuku_pay / fuku_bet if fuku_bet else 0,
        san_pct = 100 * san_hit  / n,
        san_rec = 100 * san_pay  / san_bet  if san_bet  else 0,
    )


def sweep(records, key_new, label):
    """α 11×11=121通りをスイープし、各指標の最良を返す"""
    alphas  = [round(a * 0.1, 1) for a in range(0, 11)]
    results = []
    for a_tan in alphas:
        for a_san in alphas:
            res = evaluate_combo(records, key_new, a_tan, a_san)
            if res:
                results.append((a_tan, a_san, res))

    if not results:
        return

    # ベースライン (v7同一)
    base = evaluate_combo(records, key_new, 1.0, 1.0)

    print(f"\n{'='*115}")
    print(f"  【{label}】  n={base['n']}R  ブレンド: (1-α)*{key_new} + α*v7")
    print(f"{'='*115}")
    print(f"  ベースライン (v7同一):"
          f"  単勝 {base['tan_pct']:.1f}%/{base['tan_rec']:.1f}%"
          f"  複勝 {base['fuku_pct']:.1f}%/{base['fuku_rec']:.1f}%"
          f"  3連複 {base['san_pct']:.1f}%/{base['san_rec']:.1f}%")
    print()

    metrics = [
        ("単勝的中率最良",  lambda x: x[2]["tan_pct"],  "tan_pct"),
        ("単勝回収率最良",  lambda x: x[2]["tan_rec"],  "tan_rec"),
        ("複勝的中率最良",  lambda x: x[2]["fuku_pct"], "fuku_pct"),
        ("3連複的中率最良", lambda x: x[2]["san_pct"],  "san_pct"),
        ("3連複回収率最良", lambda x: x[2]["san_rec"],  "san_rec"),
    ]
    print(f"  {'目的':<14}  {'単複α':>5}  {'3連複α':>6}   "
          f"{'単勝的中/回収':>14}  {'複勝的中/回収':>14}  {'3連複的中/回収':>14}  分離?")
    print("  " + "-" * 100)
    for desc, fn, _ in metrics:
        a_tan, a_san, res = max(results, key=fn)
        diff = "◀" if a_tan != a_san else ""
        print(f"  {desc:<14}  {a_tan:>5.1f}  {a_san:>6.1f}   "
              f"{res['tan_pct']:>6.1f}%/{res['tan_rec']:>6.1f}%   "
              f"{res['fuku_pct']:>6.1f}%/{res['fuku_rec']:>6.1f}%   "
              f"{res['san_pct']:>6.1f}%/{res['san_rec']:>6.1f}%  {diff}")

    # 総合バランス最良 (単勝回収85%以上 AND 3連複回収85%以上 で的中率合計最大)
    balanced = [(a, b, r) for a, b, r in results
                if r["tan_rec"] >= 85 and r["san_rec"] >= 85]
    if balanced:
        a_tan, a_san, res = max(balanced,
                                key=lambda x: x[2]["tan_pct"] + x[2]["fuku_pct"] + x[2]["san_pct"])
        diff = "◀" if a_tan != a_san else ""
        print(f"\n  {'バランス最良':<14}  {a_tan:>5.1f}  {a_san:>6.1f}   "
              f"{res['tan_pct']:>6.1f}%/{res['tan_rec']:>6.1f}%   "
              f"{res['fuku_pct']:>6.1f}%/{res['fuku_rec']:>6.1f}%   "
              f"{res['san_pct']:>6.1f}%/{res['san_rec']:>6.1f}%  {diff}")
        print("  ※ 単勝回収85%以上 AND 3連複回収85%以上 の条件下で的中率合計最大")

    return results


# ─── 各キャッシュを読み込んでスイープ ─────────────────────────────────────────────

print("=" * 115)
print("  モデル比較: v11feat / v12feat / v15pace  (単複 × 3連複 分離最適化)")
print("=" * 115)

# ── v11eval: s_v11 ──
print("\n▼ v11eval キャッシュ読み込み...")
with open(os.path.join(CACHE_DIR, "race_records_v11eval_2026.pkl"), "rb") as f:
    v11_records = pickle.load(f)
v11_valid = [r for r in v11_records if r.get("s_v11") is not None and r.get("s_v7") is not None]
print(f"  有効: {len(v11_valid)}R")

sweep(v11_valid, "s_v11", "v11feat × v7ブレンド")

# ── v12eval: s_v12 ──
print("\n▼ v12eval キャッシュ読み込み...")
with open(os.path.join(CACHE_DIR, "race_records_v12eval_2026.pkl"), "rb") as f:
    v12_records = pickle.load(f)
v12_valid = [r for r in v12_records if r.get("s_v12") is not None and r.get("s_v7") is not None]
print(f"  有効: {len(v12_valid)}R")

sweep(v12_valid, "s_v12", "v12feat × v7ブレンド")

# ── v15eval: s_v15 ──
print("\n▼ v15eval キャッシュ読み込み...")
with open(os.path.join(CACHE_DIR, "race_records_v15eval_2026.pkl"), "rb") as f:
    v15_records = pickle.load(f)
v15_valid = [r for r in v15_records if r.get("s_v15") is not None and r.get("s_v7") is not None]
print(f"  有効: {len(v15_valid)}R")

sweep(v15_valid, "s_v15", "v15pace × v7ブレンド")

# ── v11eval: v9 も評価（おまけ）──
v9_valid = [r for r in v11_records if r.get("s_v9") is not None and r.get("s_v7") is not None]
if v9_valid:
    sweep(v9_valid, "s_v9", "v9feat × v7ブレンド（参考）")

print(f"\n{'='*115}")
print("完了")
