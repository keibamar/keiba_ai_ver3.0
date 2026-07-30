"""eval_3way_blend_no26.py

既存の v9feat キャッシュを使って、v7 × v15 × v18listwise の
3-way blend グリッドサーチを実行する。

再学習なし・即実行可能。

実行: python scripts/eval_3way_blend_no26.py
"""

import sys, os, warnings, pickle, itertools
warnings.simplefilter("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np

EVAL_YEAR = 2026
V9FEAT_CACHE = os.path.join(PROJECT_ROOT, "logs", f"race_records_v9feat_{EVAL_YEAR}.pkl")
BET_UNIT = 100

# ---- キャッシュ読み込み ----
print(f"v9featキャッシュ読み込み: {V9FEAT_CACHE}")
with open(V9FEAT_CACHE, "rb") as f:
    race_records = pickle.load(f)
print(f"  {len(race_records)}R読み込み完了")

# スコアキーの存在確認
keys_present = {k: sum(1 for r in race_records if r.get(k) is not None)
                for k in ["s_v7", "s_v15", "s_v16", "s_v17", "s_v18"]}
for k, cnt in keys_present.items():
    print(f"  {k}: {cnt}R")


def _norm(arr):
    mn, mx = arr.min(), arr.max()
    return (arr - mn) / (mx - mn + 1e-12)


class Stats:
    def __init__(self):
        self.n = self.tan_hit = self.tan_pay = self.tan_bet = 0
        self.san_hit = self.san_pay = self.san_bet = 0

    def add(self, tan_h, tan_ret, san_h, san_ret, san_bets):
        self.n += 1
        self.tan_bet += BET_UNIT
        self.san_bet += BET_UNIT * san_bets
        if tan_h and tan_ret:
            self.tan_hit += 1
            self.tan_pay += tan_ret
        if san_h and san_ret:
            self.san_hit += 1
            self.san_pay += san_ret

    def tan_pct(self):  return 100 * self.tan_hit / self.n if self.n else 0
    def san_pct(self):  return 100 * self.san_hit / self.n if self.n else 0
    def tan_rec(self):  return 100 * self.tan_pay / self.tan_bet if self.tan_bet else 0
    def san_rec(self):  return 100 * self.san_pay / self.san_bet if self.san_bet else 0


def evaluate_quiet(records, score_fn):
    """ベスト探索用: Stats を返すだけ（printしない）"""
    st = Stats()
    for r in records:
        scores = score_fn(r)
        if scores is None:
            continue
        order = np.argsort(-scores)
        top5 = [r["umabans"][i] for i in order[:5]]
        honmei = top5[0]
        san_combs = list(itertools.combinations(top5, 3))
        san_h = (r["san_winner"] is not None) and any(
            {a, b, c} == r["san_winner"] for a, b, c in san_combs
        )
        st.add(
            honmei in r["winner_set"], r["tan_ret"],
            san_h, r["san_odds"], len(san_combs),
        )
    return st


def evaluate(label, records, score_fn, width=65):
    st = evaluate_quiet(records, score_fn)
    print(f"  {label:<{width}}  n={st.n:>4}  "
          f"単勝 {st.tan_pct():>5.1f}%/{st.tan_rec():>6.1f}%  "
          f"3連複 {st.san_pct():>5.1f}%/{st.san_rec():>6.1f}%")
    return st


SEP = "=" * 130

# ---- 参照: 2-way ベースライン ----
print(f"\n{SEP}")
print("【参照: 2-way ベースライン】")
print(SEP)
recs_v7v15 = [r for r in race_records if r.get("s_v7") is not None]
for alpha in [0.3, 0.4, 0.5, 0.6]:
    evaluate(
        f"v15×v7  v15:{1-alpha:.1f} v7:{alpha:.1f}",
        recs_v7v15,
        lambda r, a=alpha: (1 - a) * _norm(r["s_v15"]) + a * _norm(r["s_v7"]),
    )
recs_v7v18 = [r for r in race_records if r.get("s_v7") is not None and r.get("s_v18") is not None]
for alpha in [0.2, 0.3, 0.4]:
    evaluate(
        f"v18×v7  v18:{alpha:.1f} v7:{1-alpha:.1f}",
        recs_v7v18,
        lambda r, a=alpha: a * _norm(r["s_v18"]) + (1 - a) * _norm(r["s_v7"]),
    )

# ---- 3-way grid sweep ----
def run_3way_sweep(label, recs, keys, step=0.1):
    """weights sum to 1.0, grid over steps."""
    print(f"\n{SEP}")
    print(f"【3-way blend: {label}】  n={len(recs)}R")
    print(f"  {'組み合わせ':<62}  {'件数':>5}  {'単勝%/回収%':>15}  {'3連複%/回収%':>16}")
    print(SEP)

    steps = round(1.0 / step)
    best_san = None
    best_info = None
    best_tan = None
    best_tan_info = None
    results = []

    for i in range(steps + 1):
        for j in range(steps + 1 - i):
            k = steps - i - j
            w0 = i / steps
            w1 = j / steps
            w2 = k / steps
            combo = (w0, w1, w2)

            def sfn(r, c=combo, ks=keys):
                s = [r.get(ks[0]), r.get(ks[1]), r.get(ks[2])]
                if any(x is None for x in s):
                    return None
                return c[0] * _norm(s[0]) + c[1] * _norm(s[1]) + c[2] * _norm(s[2])

            st = evaluate_quiet(recs, sfn)
            results.append((combo, st))

            if best_san is None or st.san_pct() > best_san:
                best_san = st.san_pct()
                best_info = (combo, st)
            if best_tan is None or st.tan_pct() > best_tan:
                best_tan = st.tan_pct()
                best_tan_info = (combo, st)

    # 上位10件を3連複的中率でソートして表示
    results_sorted = sorted(results, key=lambda x: -x[1].san_pct())[:20]
    k0, k1, k2 = keys
    short = [k.replace("s_", "") for k in keys]
    for combo, st in results_sorted:
        label_str = f"{short[0]}:{combo[0]:.1f} {short[1]}:{combo[1]:.1f} {short[2]}:{combo[2]:.1f}"
        evaluate(label_str, recs, lambda r, c=combo, ks=keys: (
            c[0] * _norm(r[ks[0]]) + c[1] * _norm(r[ks[1]]) + c[2] * _norm(r[ks[2]])
            if all(r.get(k) is not None for k in ks) else None
        ))

    print(f"\n  ★ 3連複的中率ベスト: {short[0]}:{best_info[0][0]:.1f} {short[1]}:{best_info[0][1]:.1f} {short[2]}:{best_info[0][2]:.1f}")
    print(f"     単勝 {best_info[1].tan_pct():.1f}%/{best_info[1].tan_rec():.1f}%  3連複 {best_info[1].san_pct():.1f}%/{best_info[1].san_rec():.1f}%")
    print(f"  ★ 単勝的中率ベスト: {short[0]}:{best_tan_info[0][0]:.1f} {short[1]}:{best_tan_info[0][1]:.1f} {short[2]}:{best_tan_info[0][2]:.1f}")
    print(f"     単勝 {best_tan_info[1].tan_pct():.1f}%/{best_tan_info[1].tan_rec():.1f}%  3連複 {best_tan_info[1].san_pct():.1f}%/{best_tan_info[1].san_rec():.1f}%")


# v7 × v15 × v18
recs_v7v15v18 = [r for r in race_records
                 if r.get("s_v7") is not None and r.get("s_v18") is not None]
run_3way_sweep("v7 × v15 × v18listwise", recs_v7v15v18, ["s_v7", "s_v15", "s_v18"])

# v7 × v15 × v16
recs_v7v15v16 = [r for r in race_records
                 if r.get("s_v7") is not None and r.get("s_v16") is not None]
run_3way_sweep("v7 × v15 × v16profit", recs_v7v15v16, ["s_v7", "s_v15", "s_v16"])

# v7 × v15 × v17
recs_v7v15v17 = [r for r in race_records
                 if r.get("s_v7") is not None and r.get("s_v17") is not None]
run_3way_sweep("v7 × v15 × v17focal", recs_v7v15v17, ["s_v7", "s_v15", "s_v17"])

print(f"\n{SEP}")
print("完了")
