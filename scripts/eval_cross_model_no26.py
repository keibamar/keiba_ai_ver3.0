"""eval_cross_model_no26.py

統合キャッシュ（race_records_unified_2026.pkl）を使って、
「単複モデル × 3連複モデル」のクロスモデル全組み合わせを評価する。

モデル選択肢:
  v7  : s_v7  (オッズ特化ベースライン)
  v11 : s_v11 (特徴量拡張)
  v12 : s_v12 (コーナー追走/上がりトレンド追加)
  v15 : s_v15 (ペース適性追加)

ブレンド: blend(α) = (1-α)*norm(s_vXX) + α*norm(s_v7)
  α=0.0 → vXX 100%  /  α=1.0 → v7 100%
  ※ v7 を選択した場合 α は意味なし (常に s_v7 = blend で α=1.0 相当)

評価戦略:
  単複 → blend_tan の指数1位馬のみ投票
  3連複 → 単複1位(本命) + blend_san 上位から5頭 BOX (C(5,3)=10点)

スイープ範囲:
  単複: (tan_model, a_tan) ∈ [v7/v11/v12/v15] × [0.0..1.0 step 0.1]
  3連複: (san_model, a_san) ∈ [v7/v11/v12/v15] × [0.0..1.0 step 0.1]
  → 実効組み合わせ数: 33 × 33 = 1089通り
     (v7はα=1.0と等価なので重複排除すると (3×11+1) × (3×11+1) = 34×34=1156通り)

実行:
    python scripts/eval_cross_model_no26.py
"""

import os, sys, pickle, itertools, time
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

BET_UNIT     = 100
UNIFIED_CACHE = os.path.join(PROJECT_ROOT, "logs", "race_records_unified_2026.pkl")
LOG_FILE      = os.path.join(PROJECT_ROOT, "logs", "eval_cross_model_log.txt")

print(f"統合キャッシュ読み込み: {UNIFIED_CACHE}")
with open(UNIFIED_CACHE, "rb") as f:
    all_records = pickle.load(f)
print(f"  全{len(all_records)}R")

records = [r for r in all_records if r.get("s_v7") is not None]
print(f"  s_v7有効: {len(records)}R")


def _norm(arr):
    mn, mx = arr.min(), arr.max()
    return (arr - mn) / (mx - mn + 1e-12)


def _blend_score(r, model_key, alpha):
    """
    blend = (1-alpha)*norm(vXX) + alpha*norm(v7)
    alpha=1.0 または model_key="s_v7" なら v7 のみ。
    """
    s_v7 = r.get("s_v7")
    if model_key == "s_v7" or alpha >= 1.0:
        return _norm(np.array(s_v7, dtype=float))
    s_new = r.get(model_key)
    if s_new is None:
        return _norm(np.array(s_v7, dtype=float))
    v7n  = _norm(np.array(s_v7,  dtype=float))
    newn = _norm(np.array(s_new, dtype=float))
    return (1 - alpha) * newn + alpha * v7n


def evaluate(records, tan_model, a_tan, san_model, a_san):
    n = tan_hit = tan_pay = tan_bet = 0
    fuku_hit = fuku_pay = fuku_bet = 0
    san_hit  = san_pay  = san_bet  = 0

    for r in records:
        s_tan = _blend_score(r, tan_model, a_tan)
        s_san = _blend_score(r, san_model, a_san)
        umabans = r["umabans"]

        honmei = umabans[np.argmax(s_tan)]

        order_san = np.argsort(-s_san)
        san5 = [honmei]
        for i in order_san:
            u = umabans[i]
            if u != honmei:
                san5.append(u)
            if len(san5) == 5:
                break

        tan_bet  += BET_UNIT
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
        n        = n,
        tan_pct  = 100 * tan_hit  / n,
        tan_rec  = 100 * tan_pay  / tan_bet  if tan_bet  else 0,
        fuku_pct = 100 * fuku_hit / n,
        fuku_rec = 100 * fuku_pay / fuku_bet if fuku_bet else 0,
        san_pct  = 100 * san_hit  / n,
        san_rec  = 100 * san_pay  / san_bet  if san_bet  else 0,
    )


# ── スイープ設定 ──
models = ["s_v7", "s_v11", "s_v12", "s_v15"]
alphas = [round(a * 0.1, 1) for a in range(0, 11)]

# v7 は常に α=1.0 と等価なので tan/san × α の候補を作る
# (model, alpha) のリストを生成。v7 は alpha=1.0 のみ（重複排除）
def model_alpha_candidates(model_keys, alphas):
    cands = []
    for mk in model_keys:
        if mk == "s_v7":
            cands.append(("s_v7", 1.0))
        else:
            for a in alphas:
                cands.append((mk, a))
    return cands

tan_cands = model_alpha_candidates(models, alphas)
san_cands = model_alpha_candidates(models, alphas)
print(f"\nスイープ: {len(tan_cands)} × {len(san_cands)} = {len(tan_cands)*len(san_cands)} 通り")

# ── v11/v12/v15 の有効レコード数確認 ──
for mk in ["s_v11", "s_v12", "s_v15"]:
    n = sum(1 for r in records if r.get(mk) is not None)
    print(f"  {mk} 有効: {n}R")

t0 = time.time()
results = []
done = 0
total = len(tan_cands) * len(san_cands)

for tan_mk, a_tan in tan_cands:
    for san_mk, a_san in san_cands:
        # 対象レコード: tan/san 両モデルのスコアがある R
        if tan_mk == "s_v7":
            tan_valid = records
        else:
            tan_valid = [r for r in records if r.get(tan_mk) is not None]
        if san_mk == "s_v7":
            recs = tan_valid
        else:
            recs = [r for r in tan_valid if r.get(san_mk) is not None]

        res = evaluate(recs, tan_mk, a_tan, san_mk, a_san)
        if res:
            results.append((tan_mk, a_tan, san_mk, a_san, res))
        done += 1
        if done % 500 == 0:
            print(f"  {done}/{total} 完了 ({time.time()-t0:.0f}秒)")

print(f"\nスイープ完了: {len(results)} 条件 ({time.time()-t0:.0f}秒)")

# ── 出力ヘルパー ──
MODEL_LABEL = {
    "s_v7":  "v7",
    "s_v11": "v11",
    "s_v12": "v12",
    "s_v15": "v15",
}

def fmt(tm, at, sm, ab, res):
    tl = f"{MODEL_LABEL[tm]}α{at:.1f}" if tm != "s_v7" else "v7"
    sl = f"{MODEL_LABEL[sm]}α{ab:.1f}" if sm != "s_v7" else "v7"
    cross = "◀" if tm != sm else ""
    return (f"  {tl:<11} × {sl:<11}  "
            f"{res['tan_pct']:>5.1f}%/{res['tan_rec']:>6.1f}%  "
            f"{res['fuku_pct']:>5.1f}%/{res['fuku_rec']:>6.1f}%  "
            f"{res['san_pct']:>5.1f}%/{res['san_rec']:>6.1f}%  {cross}")

HDR = f"  {'単複モデル':<12}  {'3連複モデル':<12}  {'単勝的中/回収':>13}  {'複勝的中/回収':>13}  {'3連複的中/回収':>14}"
SEP = "  " + "-" * 100

lines = []
lines.append("\n" + "=" * 110)
lines.append(f"  クロスモデル最適化評価（単複 × 3連複 分離）  n≒{results[0][4]['n']}R")
lines.append("=" * 110)

# ① ベースライン
base = next((r for r in results if r[0]=="s_v7" and r[2]=="s_v7"), None)
if base:
    lines.append(f"\n【ベースライン (v7同一)】")
    lines.append(HDR); lines.append(SEP)
    lines.append(fmt(*base))

# ② 各指標の全体最良
metrics = [
    ("単勝的中率",   lambda x: x[4]["tan_pct"]),
    ("単勝回収率",   lambda x: x[4]["tan_rec"]),
    ("複勝的中率",   lambda x: x[4]["fuku_pct"]),
    ("3連複的中率",  lambda x: x[4]["san_pct"]),
    ("3連複回収率",  lambda x: x[4]["san_rec"]),
]
lines.append(f"\n【各指標 全体最良】")
lines.append(HDR); lines.append(SEP)
for label, fn in metrics:
    best = max(results, key=fn)
    lines.append(f"  {label:<12}: " + fmt(*best))

# ③ バランス最良（単勝回収85%以上 AND 3連複回収85%以上 で的中率合計最大）
balanced = [(tm, at, sm, ab, res) for tm, at, sm, ab, res in results
            if res["tan_rec"] >= 85 and res["san_rec"] >= 85]
if balanced:
    best_bal = max(balanced,
                   key=lambda x: x[4]["tan_pct"] + x[4]["fuku_pct"] + x[4]["san_pct"])
    lines.append(f"\n【バランス最良（単勝回収85%+ AND 3連複回収85%+）】")
    lines.append(HDR); lines.append(SEP)
    lines.append(fmt(*best_bal))
    lines.append("  ※ 単勝回収85%以上 AND 3連複回収85%以上 の条件下で的中率合計最大")

# ④ クロスモデル（tan_model ≠ san_model）の最良
cross = [(tm, at, sm, ab, res) for tm, at, sm, ab, res in results if tm != sm]
if cross:
    lines.append(f"\n【クロスモデル最良（単複モデル ≠ 3連複モデル）】")
    lines.append(HDR); lines.append(SEP)
    for label, fn in metrics:
        best = max(cross, key=fn)
        lines.append(f"  {label:<12}: " + fmt(*best))
    # クロス + バランス条件
    cross_bal = [(tm, at, sm, ab, res) for tm, at, sm, ab, res in cross
                 if res["tan_rec"] >= 85 and res["san_rec"] >= 85]
    if cross_bal:
        best_cbal = max(cross_bal,
                        key=lambda x: x[4]["tan_pct"] + x[4]["fuku_pct"] + x[4]["san_pct"])
        lines.append(f"\n  バランス最良（クロス+回収条件）: " + fmt(*best_cbal))

# ⑤ 各モデルペアごとの最良比較（同一モデル+クロス）
lines.append(f"\n【モデルペア別 バランス最良（回収85%+条件）上位20】")
lines.append(HDR); lines.append(SEP)
bal_sorted = sorted(balanced,
                    key=lambda x: x[4]["tan_pct"] + x[4]["fuku_pct"] + x[4]["san_pct"],
                    reverse=True)[:20]
for item in bal_sorted:
    lines.append(fmt(*item))

# ⑥ 3連複的中率 上位20
lines.append(f"\n【3連複的中率 上位20】")
lines.append(HDR); lines.append(SEP)
top_san = sorted(results, key=lambda x: -x[4]["san_pct"])[:20]
for item in top_san:
    lines.append(fmt(*item))

# ⑦ 単勝的中率 上位20
lines.append(f"\n【単勝的中率 上位20】")
lines.append(HDR); lines.append(SEP)
top_tan = sorted(results, key=lambda x: -x[4]["tan_pct"])[:20]
for item in top_tan:
    lines.append(fmt(*item))

output = "\n".join(lines)
print(output)

with open(LOG_FILE, "w", encoding="utf-8") as f:
    f.write(output)
print(f"\n結果を保存: {LOG_FILE}")
