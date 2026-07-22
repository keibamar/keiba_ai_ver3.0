"""eval_cross_model_all_no26.py

オッズあり（v7/v11/v12/v15 × α）＋ オッズなし（v11n/v12n/v15n）の
全モデルを統合して「単複モデル × 3連複モデル」のクロス評価を行う。

モデル候補（37通り）:
  オッズあり: v7(1) + v11/v12/v15 × α0.0〜0.9 (30) = 31通り
  オッズなし: v11_nodds / v12_nodds / v15_nodds = 3通り
  合計 34通り（重複排除）

入力:
  logs/race_records_unified_2026.pkl  ... race_id, s_v7/v11/v12/v15
  logs/race_records_nodds_2026.pkl    ... race_id, s_v11n/v12n/v15n

出力:
  logs/eval_cross_model_all_log.txt  （stdout をリダイレクト）
"""

import os, sys, pickle, itertools, time
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

BET_UNIT = 100
EVAL_YEAR = 2026

UNIFIED_CACHE = os.path.join(PROJECT_ROOT, "logs", f"race_records_unified_{EVAL_YEAR}.pkl")
NODDS_CACHE   = os.path.join(PROJECT_ROOT, "logs", f"race_records_nodds_{EVAL_YEAR}.pkl")

# ── キャッシュ読み込み & マージ ──
print("キャッシュ読み込み...")
with open(UNIFIED_CACHE, "rb") as f:
    unified = pickle.load(f)
with open(NODDS_CACHE, "rb") as f:
    nodds_list = pickle.load(f)

nodds_map = {r["race_id"]: r for r in nodds_list}

records = []
for r in unified:
    nd = nodds_map.get(r["race_id"])
    if nd is None:
        continue
    merged = dict(r)
    merged["s_v11n"] = nd.get("s_v11n")
    merged["s_v12n"] = nd.get("s_v12n")
    merged["s_v15n"] = nd.get("s_v15n")
    records.append(merged)

print(f"マージ完了: {len(records)}R")


# ── ブレンドスコア ──
def _norm(arr):
    mn, mx = arr.min(), arr.max()
    return (arr - mn) / (mx - mn + 1e-12)


def _blend_score(r, model_key, alpha):
    """
    model_key: 'v7' | 'v11'|'v12'|'v15' (オッズあり, alpha 適用)
             | 'v11n'|'v12n'|'v15n'      (オッズなし, alpha 無視)
    alpha: v7 との線形補間係数 (オッズありのみ有効)
    """
    if model_key in ("v11n", "v12n", "v15n"):
        key = f"s_{model_key}"
        s = r.get(key)
        if s is None:
            return None
        return _norm(s)

    sv7 = r.get("s_v7")
    if model_key == "v7":
        if sv7 is None:
            return None
        return _norm(sv7)

    svx = r.get(f"s_{model_key}")
    if svx is None or sv7 is None:
        return None
    return (1 - alpha) * _norm(svx) + alpha * _norm(sv7)


# ── モデルキー列挙 ──
ALPHAS = [round(a * 0.1, 1) for a in range(10)]  # 0.0〜0.9

# オッズあり（v7 は alpha 不要、v11/v12/v15 × 10段階）
ODDS_MODELS = [("v7", 0.0)]
for mk in ("v11", "v12", "v15"):
    for a in ALPHAS:
        ODDS_MODELS.append((mk, a))
# 重複 (v7 相当 alpha=1.0) は既にないので34通り

# オッズなし
NODDS_MODELS = [("v11n", 0.0), ("v12n", 0.0), ("v15n", 0.0)]

ALL_MODELS = ODDS_MODELS + NODDS_MODELS  # 34通り

def model_label(mk, alpha):
    if mk in ("v7", "v11n", "v12n", "v15n"):
        return mk
    return f"{mk}α{alpha:.1f}"


# ── 評価 ──
def evaluate(records, tan_mk, tan_a, san_mk, san_a):
    n = tan_hit = tan_pay = tan_bet = 0
    fuku_hit = fuku_pay = fuku_bet = 0
    san_hit  = san_pay  = san_bet  = 0

    for r in records:
        s_tan = _blend_score(r, tan_mk, tan_a)
        s_san = _blend_score(r, san_mk, san_a)
        if s_tan is None or s_san is None:
            continue

        umabans = r["umabans"]
        honmei  = umabans[np.argmax(s_tan)]

        order = np.argsort(-s_san)
        san5  = [honmei]
        for i in order:
            u = umabans[i]
            if u != honmei:
                san5.append(u)
            if len(san5) == 5:
                break

        tan_bet += BET_UNIT
        if honmei in r["winner_set"]:
            tan_hit += 1
            if r["tan_ret"]:
                tan_pay += r["tan_ret"]

        fuku_bet += BET_UNIT
        if honmei in r["top3_set"]:
            fuku_hit += 1
            ret = r["fuku_rets"].get(honmei)
            if ret:
                fuku_pay += ret

        combs    = list(itertools.combinations(san5, 3))
        san_bet += BET_UNIT * len(combs)
        if r["san_winner"] is not None:
            if any({a, b, c} == r["san_winner"] for a, b, c in combs):
                san_hit += 1
                if r["san_odds"]:
                    san_pay += r["san_odds"]

        n += 1

    if n == 0:
        return None
    return {
        "n":        n,
        "tan_hit":  100 * tan_hit  / n,
        "tan_ret":  100 * tan_pay  / tan_bet,
        "fuku_hit": 100 * fuku_hit / n,
        "fuku_ret": 100 * fuku_pay / fuku_bet,
        "san_hit":  100 * san_hit  / n,
        "san_ret":  100 * san_pay  / san_bet,
    }


# ── 全スイープ ──
t0 = time.time()
print(f"\n全 {len(ALL_MODELS)}×{len(ALL_MODELS)} = {len(ALL_MODELS)**2} 通り評価中...")

results = []
for (tan_mk, tan_a) in ALL_MODELS:
    for (san_mk, san_a) in ALL_MODELS:
        m = evaluate(records, tan_mk, tan_a, san_mk, san_a)
        if m is None:
            continue
        m["tan_label"] = model_label(tan_mk, tan_a)
        m["san_label"] = model_label(san_mk, san_a)
        m["tan_mk"]    = tan_mk
        m["san_mk"]    = san_mk
        results.append(m)

print(f"完了: {time.time()-t0:.0f}秒  {len(results)}通り")

HDR = f"  {'単複モデル':<14} {'3連複モデル':<14}   {'単勝的中/回収':>16}   {'複勝的中/回収':>16}   {'3連複的中/回収':>18}"
ROW = "  {:<14} × {:<14}   {:5.1f}%/{:6.1f}%   {:5.1f}%/{:6.1f}%   {:5.1f}%/{:7.1f}%{}"


def prow(r, mark=""):
    return ROW.format(
        r["tan_label"], r["san_label"],
        r["tan_hit"], r["tan_ret"],
        r["fuku_hit"], r["fuku_ret"],
        r["san_hit"], r["san_ret"],
        mark,
    )


# ── ベースライン ──
print(f"\n{'='*90}")
print("【ベースライン (v7同一)】")
print(HDR)
bl = next((r for r in results if r["tan_label"] == "v7" and r["san_label"] == "v7"), None)
if bl:
    print(prow(bl))

# ── 各指標最良 ──
print(f"\n【各指標 最良】")
print(HDR)
for metric, label in [
    ("tan_hit",  "単勝的中率"),
    ("tan_ret",  "単勝回収率"),
    ("fuku_hit", "複勝的中率"),
    ("san_hit",  "3連複的中率"),
    ("san_ret",  "3連複回収率"),
]:
    best = max(results, key=lambda r: r[metric])
    print(f"  {label:<12}: {prow(best, ' ◀')}")

# ── バランス: 単勝回収85%+ AND 3連複回収85%+ ──
print(f"\n【バランス最良A（単勝回収85%以上 AND 3連複回収85%以上）】")
print(HDR)
cands = [r for r in results if r["tan_ret"] >= 85.0 and r["san_ret"] >= 85.0]
if cands:
    best = max(cands, key=lambda r: r["tan_hit"] + r["san_hit"])
    print(f"  {prow(best, ' ◀')}")
    top = sorted(cands, key=lambda r: r["tan_hit"] + r["san_hit"], reverse=True)[:5]
    for r in top[1:]:
        print(f"  {prow(r)}")
else:
    print("  （条件を満たす組み合わせなし）")

# ── バランス: 単勝回収100%+ AND 3連複回収100%+ ──
print(f"\n【バランス最良B（単勝回収100%以上 AND 3連複回収100%以上）】")
print(HDR)
cands100 = [r for r in results if r["tan_ret"] >= 100.0 and r["san_ret"] >= 100.0]
if cands100:
    best = max(cands100, key=lambda r: r["tan_hit"] + r["san_hit"])
    print(f"  {prow(best, ' ◀')}")
    top = sorted(cands100, key=lambda r: r["tan_hit"] + r["san_hit"], reverse=True)[:10]
    for r in top[1:]:
        print(f"  {prow(r)}")
else:
    print("  （条件を満たす組み合わせなし）")

# ── バランス: 単勝回収115%+ AND 3連複回収130%+ ──
print(f"\n【バランス最良C（単勝回収115%以上 AND 3連複回収130%以上）】")
print(HDR)
cands_high = [r for r in results if r["tan_ret"] >= 115.0 and r["san_ret"] >= 130.0]
if cands_high:
    best = max(cands_high, key=lambda r: r["tan_hit"] + r["san_hit"])
    print(f"  {prow(best, ' ◀')}")
    top = sorted(cands_high, key=lambda r: r["tan_hit"] + r["san_hit"], reverse=True)[:10]
    for r in top[1:]:
        print(f"  {prow(r)}")
else:
    print("  （条件を満たす組み合わせなし）")

# ── 単勝回収率 上位20 ──
print(f"\n【単勝回収率 上位20】")
print(HDR)
for r in sorted(results, key=lambda x: x["tan_ret"], reverse=True)[:20]:
    cross = " ◀" if r["tan_mk"] != r["san_mk"] else ""
    print(prow(r, cross))

# ── 3連複回収率 上位20 ──
print(f"\n【3連複回収率 上位20】")
print(HDR)
for r in sorted(results, key=lambda x: x["san_ret"], reverse=True)[:20]:
    cross = " ◀" if r["tan_mk"] != r["san_mk"] else ""
    print(prow(r, cross))

# ── 総合スコア（単勝回収 + 3連複回収）上位20 ──
print(f"\n【総合スコア（単勝回収 + 3連複回収）上位20】")
print(HDR)
for r in sorted(results, key=lambda x: x["tan_ret"] + x["san_ret"], reverse=True)[:20]:
    cross = " ◀" if r["tan_mk"] != r["san_mk"] else ""
    total = r["tan_ret"] + r["san_ret"]
    print(prow(r, f"{cross}  合計={total:.1f}%"))

print(f"\n{'='*90}")
print(f"評価完了  {time.time()-t0:.0f}秒")
