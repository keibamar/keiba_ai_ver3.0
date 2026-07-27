"""馬券検討モデル

MAR / MAR-hit / MAR-val の3スコアから各馬の勝率推定値を計算し、
Harville公式で組合せ的中確率と期待オッズを算出して買い目を提案する。

recommend_tickets       : 3券種すべてを出力（従来モード）
recommend_best_tickets  : レースごとに最も期待値の高い1券種だけ出力（新モード）
"""

import numpy as np
import pandas as pd
from itertools import combinations, permutations


# ── JRA 払戻率（券種別）
PAYOUT_RATE = {
    "複勝":  0.800,
    "ワイド": 0.750,
    "馬連":  0.775,
    "3連複": 0.750,
    "3連単": 0.725,
}

# ── 基本 ティアの期待オッズ上限
BASIC_MAX = {
    "馬連":  20.0,
    "3連複": 60.0,
    "3連単": 200.0,  # 120 -> 200 に拡張（rank6候補追加に合わせて）
}

# ── 穴狙い ティアの期待オッズ上限
LONGSHOT_MAX = {
    "馬連":  80.0,
    "3連複": 200.0,
    "3連単": 500.0,
}

# ── 穴狙いを出す条件（軸馬への信頼度）
#   軸馬の推定勝率がこの値以上 かつ 2位との差がこの値以上の場合のみ穴狙いを追加
LONGSHOT_AXIS_PROB  = 0.15   # 軸馬の推定勝率 ≥ 15%
LONGSHOT_AXIS_GAP   = 0.06   # 軸馬と2位の確率差 ≥ 6%

# ── 3モデルのブレンド重み
BLEND_WEIGHTS = {
    "score_value":   0.4,   # MAR-val（回収率重視）
    "score":         0.4,   # MAR main
    "score_hitrate": 0.2,   # MAR-hit（的中率）
}


# ───────────────────────────────────────────
#  確率計算
# ───────────────────────────────────────────

def blend_scores(df: pd.DataFrame) -> np.ndarray:
    """3スコアを加重平均してブレンドスコアを返す"""
    total = np.zeros(len(df))
    weight_sum = 0.0
    for col, w in BLEND_WEIGHTS.items():
        if col in df.columns:
            vals = pd.to_numeric(df[col], errors="coerce").values
            mask = ~np.isnan(vals)
            if mask.sum() > 0:
                total += np.where(mask, vals * w, 0.0)
                weight_sum += w
    if weight_sum > 0:
        total /= weight_sum
    return total


def scores_to_probs(scores: np.ndarray) -> np.ndarray:
    """softmax で勝率推定値（合計=1）に変換する"""
    scores = np.array(scores, dtype=float)
    valid = ~np.isnan(scores)
    probs = np.zeros(len(scores))
    if valid.sum() == 0:
        probs[:] = 1.0 / max(len(scores), 1)
        return probs
    s = scores[valid]
    e = np.exp(s - s.max())
    probs[valid] = e / e.sum()
    return probs


def _harville(probs: np.ndarray, order: list) -> float:
    """Harville公式: order[i]が(i+1)着になる同時確率"""
    remaining = float(probs.sum())
    p = 1.0
    for idx in order:
        if remaining <= 1e-10:
            return 0.0
        p *= probs[idx] / remaining
        remaining -= probs[idx]
    return p


def place_prob(probs: np.ndarray, i: int) -> float:
    """複勝: horse i が 3着以内に入る確率 (Harville)"""
    n = len(probs)
    s = float(probs.sum())
    # 1着
    p = probs[i] / s
    # 2着
    for j in range(n):
        if j == i:
            continue
        p += probs[j] / s * probs[i] / (s - probs[j])
    # 3着
    for j in range(n):
        if j == i:
            continue
        rem_j = s - probs[j]
        for k in range(n):
            if k == i or k == j:
                continue
            p += probs[j] / s * probs[k] / rem_j * probs[i] / (rem_j - probs[k])
    return p


def wide_prob(probs: np.ndarray, i: int, j: int) -> float:
    """ワイド: horse i と j が共に 3着以内に入る確率 (Harville)"""
    total = 0.0
    n = len(probs)
    for k in range(n):
        if k == i or k == j:
            continue
        for perm in permutations([i, j, k]):
            total += _harville(probs, list(perm))
    return total


def quinella_prob(probs, i, j):
    """馬連: i,j が 1・2着（順不同）に入る確率"""
    return _harville(probs, [i, j]) + _harville(probs, [j, i])


def trifecta_place_prob(probs, i, j, k):
    """3連複: i,j,k が 3着以内（順不同）に入る確率"""
    return sum(_harville(probs, list(perm)) for perm in permutations([i, j, k]))


def trifecta_prob(probs, i, j, k):
    """3連単: i→j→k の順で 1・2・3着に入る確率"""
    return _harville(probs, [i, j, k])


def exp_odds(prob: float, bet_type: str) -> float:
    if prob <= 1e-10:
        return float("inf")
    return PAYOUT_RATE[bet_type] / prob


# ───────────────────────────────────────────
#  買い目ビルダー
# ───────────────────────────────────────────

def _ticket(combo, prob, bet_type):
    eo = round(exp_odds(prob, bet_type), 1)
    return {"組合せ": combo, "期待オッズ": eo, "確率%": round(prob * 100, 2)}


def _split(tickets, basic_max, longshot_max):
    """期待オッズで 基本 / 穴狙い に分類する"""
    basic     = [t for t in tickets if t["期待オッズ"] <= basic_max]
    longshot  = [t for t in tickets if basic_max < t["期待オッズ"] <= longshot_max]
    return (sorted(basic,    key=lambda t: t["期待オッズ"]),
            sorted(longshot, key=lambda t: t["期待オッズ"]))


# ───────────────────────────────────────────
#  メイン推奨関数
# ───────────────────────────────────────────

def recommend_tickets(df: pd.DataFrame) -> dict:
    """出馬表DataFrameから推奨買い目を返す

    Returns:
        {
          "馬連": {
            "選択方式": str,
            "軸": list[int],
            "基本": list[dict],       # 期待オッズ ≤ BASIC_MAX
            "穴狙い": list[dict],     # BASIC_MAX < 期待オッズ ≤ LONGSHOT_MAX
                                      # 軸馬が強い場合のみ
          },
          "3連複": { ... },
          "3連単": { ... },
          "_meta": {                  # 内部情報（表示用）
            "axis_prob": float,       # 軸馬の推定勝率
            "gap12": float,           # 1位と2位の確率差
            "axis_strong": bool,      # 穴狙い条件を満たすか
          }
        }
    """
    df = df.copy().reset_index(drop=True)
    blended = blend_scores(df)
    probs   = scores_to_probs(blended)
    n       = len(probs)

    if n < 3:
        return {}

    uma        = pd.to_numeric(df["馬番"], errors="coerce").astype(int).tolist()
    rank_order = np.argsort(probs)[::-1]

    gap12 = float(probs[rank_order[0]] - probs[rank_order[1]]) if n >= 2 else 1.0
    gap23 = float(probs[rank_order[1]] - probs[rank_order[2]]) if n >= 3 else 1.0

    ax1 = rank_order[0]
    ax2 = rank_order[1]

    axis_prob    = float(probs[ax1])
    axis_strong  = axis_prob >= LONGSHOT_AXIS_PROB and gap12 >= LONGSHOT_AXIS_GAP

    result = {"_meta": {"axis_prob": axis_prob, "gap12": gap12, "axis_strong": axis_strong}}

    # ── 馬連 ──────────────────────────────
    # 相手候補: rank2-6（基本は上位4頭、穴狙いはrank5-6まで広げる）
    if gap12 > 0.05:
        method    = "1頭軸"
        base_n    = min(5, n)    # 基本: rank2-5（4点）
        long_n    = min(7, n)    # 穴狙い対象: rank2-7（6点、基本を含む）
        base_opps = [rank_order[i] for i in range(1, base_n)]
        long_opps = [rank_order[i] for i in range(1, long_n)]
        all_base = []
        for opp in base_opps:
            p = quinella_prob(probs, ax1, opp)
            all_base.append(_ticket(tuple(sorted([uma[ax1], uma[opp]])), p, "馬連"))
        all_long = []
        for opp in long_opps:
            p = quinella_prob(probs, ax1, opp)
            all_long.append(_ticket(tuple(sorted([uma[ax1], uma[opp]])), p, "馬連"))
    else:
        method    = "BOX"
        base_n    = min(4, n)
        long_n    = min(5, n)
        base_pool = [rank_order[i] for i in range(base_n)]
        long_pool = [rank_order[i] for i in range(long_n)]
        all_base = []
        for i, j in combinations(base_pool, 2):
            p = quinella_prob(probs, i, j)
            all_base.append(_ticket(tuple(sorted([uma[i], uma[j]])), p, "馬連"))
        all_long = []
        for i, j in combinations(long_pool, 2):
            p = quinella_prob(probs, i, j)
            all_long.append(_ticket(tuple(sorted([uma[i], uma[j]])), p, "馬連"))

    basic, _ = _split(all_base, BASIC_MAX["馬連"], LONGSHOT_MAX["馬連"])
    _, longshot = _split(all_long, BASIC_MAX["馬連"], LONGSHOT_MAX["馬連"])
    result["馬連"] = {
        "選択方式": method,
        "軸": [uma[ax1]] if method == "1頭軸" else [],
        "基本": basic,
        "穴狙い": longshot if axis_strong else [],
    }

    # ── 3連複 ─────────────────────────────
    # 基本は上位5-6頭の組合せ、穴狙いはrank7まで広げる（同じ軸）
    if gap12 > 0.05 and gap23 > 0.03:
        method = "1頭軸"
        base_opps = [rank_order[i] for i in range(1, min(6, n))]
        long_opps = [rank_order[i] for i in range(1, min(8, n))]
        all_base = []
        for i, j in combinations(base_opps, 2):
            p = trifecta_place_prob(probs, ax1, i, j)
            all_base.append(_ticket(tuple(sorted([uma[ax1], uma[i], uma[j]])), p, "3連複"))
        all_long = []
        for i, j in combinations(long_opps, 2):
            p = trifecta_place_prob(probs, ax1, i, j)
            all_long.append(_ticket(tuple(sorted([uma[ax1], uma[i], uma[j]])), p, "3連複"))
    else:
        method = "2頭軸"
        # 2頭軸は穴狙いなし（ROI悪化のため）。基本のみ rank6まで。
        base_opps = [rank_order[i] for i in range(2, min(7, n))]
        all_base = []
        for opp in base_opps:
            p = trifecta_place_prob(probs, ax1, ax2, opp)
            all_base.append(_ticket(tuple(sorted([uma[ax1], uma[ax2], uma[opp]])), p, "3連複"))
        all_long = []

    basic, _ = _split(all_base, BASIC_MAX["3連複"], LONGSHOT_MAX["3連複"])
    _, longshot = _split(all_long, BASIC_MAX["3連複"], LONGSHOT_MAX["3連複"])
    result["3連複"] = {
        "選択方式": method,
        "軸": [uma[ax1]] if method == "1頭軸" else [uma[ax1], uma[ax2]],
        "基本": basic,
        "穴狙い": longshot if (axis_strong and method == "1頭軸") else [],
    }

    # ── 3連単（1着固定）────────────────────
    # 2/3着候補: rank2-6（上位5頭の順列, 期待オッズ200倍以内）
    base_cands = [rank_order[i] for i in range(1, min(6, n))]

    all_base = []
    for j, k in permutations(base_cands, 2):
        p = trifecta_prob(probs, ax1, j, k)
        all_base.append(_ticket((uma[ax1], uma[j], uma[k]), p, "3連単"))

    basic, _ = _split(all_base, BASIC_MAX["3連単"], LONGSHOT_MAX["3連単"])
    result["3連単"] = {
        "選択方式": "1着固定",
        "軸": [uma[ax1]],
        "基本": basic[:12],
        "穴狙い": [],
    }

    return result


# ───────────────────────────────────────────
#  最良券種モード (recommend_best_tickets)
# ───────────────────────────────────────────
#
# 選択ルール:
#   1. 全レース → 3連単 (1着固定, rank2-6, オッズ上限可変)
#        axis_strong (prob≥15%, gap12≥6%): 上限 120倍
#        otherwise                        : 上限 200倍
#      → 最大 _MAX_TFC_TICKETS 点に制限（期待オッズ低い順）
#   2. 3連単が 0 点になったレース → 馬連 BOX (rank1-5, 80倍)

_BEST_AXIS_PROB   = 0.15
_BEST_AXIS_GAP12  = 0.06

_TFC_CANDS        = 6      # 3連単 2/3着候補 rank2〜6 (5頭) → P(5,2)=20
_TFC_MAX_STRONG   = 120.0  # axis_strong の 3連単 上限オッズ
_TFC_MAX_NORMAL   = 200.0  # それ以外の 3連単 上限オッズ
_MAX_TFC_TICKETS  = 10     # 1レース最大点数

_BOX_CANDS    = 4          # 馬連BOX: rank1〜4 (4頭) → C(4,2)=6
_UMAREN_MAX   = 80.0       # 馬連上限オッズ


def recommend_best_tickets(df: pd.DataFrame) -> dict:
    """レースごとに最も期待値の高い券種を1つ選んで買い目を返す。

    Returns:
        {
          "bet_type": str,       # "3連単" | "馬連"
          "選択方式": str,        # "1着固定" | "BOX"
          "軸": list[int],
          "tickets": list[dict], # 期待オッズ昇順
          "_meta": dict,
        }
    """
    df = df.copy().reset_index(drop=True)
    blended = blend_scores(df)
    probs   = scores_to_probs(blended)
    n       = len(probs)
    if n < 3:
        return {}

    uma        = pd.to_numeric(df["馬番"], errors="coerce").astype(int).tolist()
    rank_order = np.argsort(probs)[::-1]
    ax1        = rank_order[0]
    gap12      = float(probs[ax1] - probs[rank_order[1]]) if n >= 2 else 1.0
    axis_prob  = float(probs[ax1])
    axis_strong = axis_prob >= _BEST_AXIS_PROB and gap12 >= _BEST_AXIS_GAP12

    meta = {"axis_prob": axis_prob, "gap12": gap12, "axis_strong": axis_strong}

    def mk(combo, p, bt):
        eo = round(PAYOUT_RATE[bt] / p, 1) if p > 1e-10 else float("inf")
        return {"組合せ": combo, "期待オッズ": eo, "確率%": round(p * 100, 2)}

    # ── 3連単（全レース優先）────────────────
    tfc_max = _TFC_MAX_STRONG if axis_strong else _TFC_MAX_NORMAL
    cands   = [rank_order[i] for i in range(1, min(_TFC_CANDS, n))]
    tickets = []
    for j, k in permutations(cands, 2):
        p = trifecta_prob(probs, ax1, j, k)
        t = mk((uma[ax1], uma[j], uma[k]), p, "3連単")
        if t["期待オッズ"] <= tfc_max:
            tickets.append(t)
    tickets.sort(key=lambda t: t["期待オッズ"])
    tickets = tickets[:_MAX_TFC_TICKETS]

    if tickets:
        return {"bet_type": "3連単", "選択方式": "1着固定",
                "軸": [uma[ax1]], "tickets": tickets, "_meta": meta}

    # ── 馬連 BOX（3連単が 0 点のとき）──────
    pool = [rank_order[i] for i in range(min(_BOX_CANDS, n))]
    tickets = []
    for i, j in combinations(pool, 2):
        p = quinella_prob(probs, i, j)
        t = mk(tuple(sorted([uma[i], uma[j]])), p, "馬連")
        if t["期待オッズ"] <= _UMAREN_MAX:
            tickets.append(t)
    tickets.sort(key=lambda t: t["期待オッズ"])
    return {"bet_type": "馬連", "選択方式": "BOX",
            "軸": [], "tickets": tickets, "_meta": meta}


# ───────────────────────────────────────────
#  的中率重視モード (recommend_hitrate_tickets)
# ───────────────────────────────────────────
#
# 選択ルール:
#   gap12 > _HIT_AXIS_GAP → 3連複 1頭軸 (rank1軸, rank2-N の C(N-1,2) 通り)
#   それ以外              → 3連複 2頭軸 (rank1,rank2軸, rank3-N の N-2 通り)
#   3連複が0点            → fallback (引数で指定)
#
# fallback 選択肢:
#   None           : 提案なし
#   "umaren_box4"  : 馬連BOX 4頭 (C(4,2)=6 ≤ 80倍)
#   "umaren_box5"  : 馬連BOX 5頭 (C(5,2)=10 ≤ 80倍)
#   "umaren_axis"  : 馬連1頭軸 (rank1軸, rank2-6 ≤ 80倍)

_HIT_AXIS_GAP        = 0.05   # 1頭軸 vs 2頭軸の閾値
_HIT_1JIKU_CANDS     = 6      # 1頭軸相手: rank2〜6 (5頭) → C(5,2)=10
_HIT_2JIKU_CANDS     = 7      # 2頭軸相手: rank3〜7 (5頭) → 5点
_HIT_TRIFPLACE_MAX   = 150.0  # 3連複 期待オッズ上限
_HIT_UMAREN_MAX      = 80.0   # 馬連 期待オッズ上限


def recommend_hitrate_tickets(df: pd.DataFrame, fallback=None,
                              axis_gap=_HIT_AXIS_GAP,
                              trifplace_cands_1jiku=_HIT_1JIKU_CANDS,
                              trifplace_cands_2jiku=_HIT_2JIKU_CANDS) -> dict:
    """的中率重視モード: 3連複流しを軸に、fallbackで馬連。

    Args:
        fallback: None | "umaren_box4" | "umaren_box5" | "umaren_axis"

    Returns:
        {
          "bet_type": str,       # "3連複" | "馬連"
          "選択方式": str,        # "1頭軸" | "2頭軸" | "1頭軸" | "BOX"
          "軸": list[int],
          "tickets": list[dict],
          "_meta": dict,
        }
    """
    df = df.copy().reset_index(drop=True)
    blended = blend_scores(df)
    probs   = scores_to_probs(blended)
    n       = len(probs)
    if n < 3:
        return {}

    uma        = pd.to_numeric(df["馬番"], errors="coerce").astype(int).tolist()
    rank_order = np.argsort(probs)[::-1]
    ax1        = rank_order[0]
    ax2        = rank_order[1]
    gap12      = float(probs[ax1] - probs[ax2]) if n >= 2 else 1.0
    axis_prob  = float(probs[ax1])
    axis_strong = axis_prob >= _BEST_AXIS_PROB and gap12 >= _BEST_AXIS_GAP12

    meta = {"axis_prob": axis_prob, "gap12": gap12, "axis_strong": axis_strong}

    def mk(combo, p, bt):
        eo = round(PAYOUT_RATE[bt] / p, 1) if p > 1e-10 else float("inf")
        return {"組合せ": combo, "期待オッズ": eo, "確率%": round(p * 100, 2)}

    # ── 3連複 ─────────────────────────────
    if gap12 > axis_gap:
        # 1頭軸: rank1 固定、相手から2頭選ぶ
        method = "1頭軸"
        opps   = [rank_order[i] for i in range(1, min(trifplace_cands_1jiku, n))]
        axis_uma = [uma[ax1]]
        tickets  = []
        for i, j in combinations(opps, 2):
            p = trifecta_place_prob(probs, ax1, i, j)
            t = mk(tuple(sorted([uma[ax1], uma[i], uma[j]])), p, "3連複")
            if t["期待オッズ"] <= _HIT_TRIFPLACE_MAX:
                tickets.append(t)
    else:
        # 2頭軸: rank1,rank2 固定、rank3以降から1頭選ぶ
        method   = "2頭軸"
        thirds   = [rank_order[i] for i in range(2, min(trifplace_cands_2jiku, n))]
        axis_uma = [uma[ax1], uma[ax2]]
        tickets  = []
        for t3 in thirds:
            p = trifecta_place_prob(probs, ax1, ax2, t3)
            t = mk(tuple(sorted([uma[ax1], uma[ax2], uma[t3]])), p, "3連複")
            if t["期待オッズ"] <= _HIT_TRIFPLACE_MAX:
                tickets.append(t)

    tickets.sort(key=lambda t: t["期待オッズ"])

    if tickets:
        return {"bet_type": "3連複", "選択方式": method,
                "軸": axis_uma, "tickets": tickets, "_meta": meta}

    # ── fallback ───────────────────────────
    if fallback is None:
        return {"bet_type": None, "選択方式": None,
                "軸": [], "tickets": [], "_meta": meta}

    if fallback in ("umaren_box4", "umaren_box5"):
        size = 4 if fallback == "umaren_box4" else 5
        pool = [rank_order[i] for i in range(min(size, n))]
        fb_tickets = []
        for i, j in combinations(pool, 2):
            p = quinella_prob(probs, i, j)
            t = mk(tuple(sorted([uma[i], uma[j]])), p, "馬連")
            if t["期待オッズ"] <= _HIT_UMAREN_MAX:
                fb_tickets.append(t)
        fb_tickets.sort(key=lambda t: t["期待オッズ"])
        return {"bet_type": "馬連", "選択方式": "BOX",
                "軸": [], "tickets": fb_tickets, "_meta": meta}

    if fallback == "umaren_axis":
        opps = [rank_order[i] for i in range(1, min(7, n))]
        fb_tickets = []
        for opp in opps:
            p = quinella_prob(probs, ax1, opp)
            t = mk(tuple(sorted([uma[ax1], uma[opp]])), p, "馬連")
            if t["期待オッズ"] <= _HIT_UMAREN_MAX:
                fb_tickets.append(t)
        fb_tickets.sort(key=lambda t: t["期待オッズ"])
        return {"bet_type": "馬連", "選択方式": "1頭軸",
                "軸": [uma[ax1]], "tickets": fb_tickets, "_meta": meta}

    return {"bet_type": None, "選択方式": None, "軸": [], "tickets": [], "_meta": meta}


# ───────────────────────────────────────────
#  的中率重視モード v2 (recommend_hitrate_v2)
# ───────────────────────────────────────────
#
# 馬連 (メイン) : 1頭軸, rank2-5 相手, 期待オッズ ≤ 25倍
# 3連複 (サブ) : 1頭軸, rank2-5 相手, 期待オッズ ≤ 50倍
# どちらも0点なら空を返す

_HIT2_UMAREN_CANDS = 5     # rank1軸, rank2-5 → 最大4点
_HIT2_UMAREN_MAX   = 25.0
_HIT2_TP_CANDS     = 5     # rank1軸, rank2-5 → 最大 C(4,2)=6点
_HIT2_TP_MAX       = 50.0


def recommend_hitrate_v2(df: pd.DataFrame,
                         umaren_cands: int   = _HIT2_UMAREN_CANDS,
                         umaren_max: float   = _HIT2_UMAREN_MAX,
                         trifplace_cands: int  = _HIT2_TP_CANDS,
                         trifplace_max: float  = _HIT2_TP_MAX) -> dict:
    """的中率重視モード v2: 馬連メイン + 3連複サブ。

    Returns:
        {
          "馬連":  {"選択方式": "1頭軸", "軸": list[int], "tickets": list[dict]},
          "3連複": {"選択方式": "1頭軸", "軸": list[int], "tickets": list[dict]},
          "_meta": dict,
        }
        tickets が空の券種は提案なし。両方空なら何も買わない。
    """
    df = df.copy().reset_index(drop=True)
    blended = blend_scores(df)
    probs   = scores_to_probs(blended)
    n       = len(probs)
    if n < 3:
        return {}

    uma        = pd.to_numeric(df["馬番"], errors="coerce").astype(int).tolist()
    rank_order = np.argsort(probs)[::-1]
    ax1        = rank_order[0]
    gap12      = float(probs[ax1] - probs[rank_order[1]]) if n >= 2 else 1.0
    axis_prob  = float(probs[ax1])

    meta = {
        "axis_prob":   axis_prob,
        "gap12":       gap12,
        "axis_strong": axis_prob >= LONGSHOT_AXIS_PROB and gap12 >= LONGSHOT_AXIS_GAP,
    }

    def mk(combo, p, bt):
        eo = round(PAYOUT_RATE[bt] / p, 1) if p > 1e-10 else float("inf")
        return {"組合せ": combo, "期待オッズ": eo, "確率%": round(p * 100, 2)}

    # ── 馬連 (メイン) ─────────────────────────
    opps = [rank_order[i] for i in range(1, min(umaren_cands, n))]
    um_tickets = []
    for opp in opps:
        p = quinella_prob(probs, ax1, opp)
        t = mk(tuple(sorted([uma[ax1], uma[opp]])), p, "馬連")
        if t["期待オッズ"] <= umaren_max:
            um_tickets.append(t)
    um_tickets.sort(key=lambda t: t["期待オッズ"])

    # ── 3連複 (サブ) ──────────────────────────
    tp_opps = [rank_order[i] for i in range(1, min(trifplace_cands, n))]
    tp_tickets = []
    for i, j in combinations(tp_opps, 2):
        p = trifecta_place_prob(probs, ax1, i, j)
        t = mk(tuple(sorted([uma[ax1], uma[i], uma[j]])), p, "3連複")
        if t["期待オッズ"] <= trifplace_max:
            tp_tickets.append(t)
    tp_tickets.sort(key=lambda t: t["期待オッズ"])

    return {
        "馬連":  {"選択方式": "1頭軸", "軸": [uma[ax1]], "tickets": um_tickets},
        "3連複": {"選択方式": "1頭軸", "軸": [uma[ax1]], "tickets": tp_tickets},
        "_meta": meta,
    }


# ───────────────────────────────────────────
#  指数ベースモード (recommend_score_based)
# ───────────────────────────────────────────
#
# rankの固定カットではなく「確率値」で動的に以下を決定する：
#
#  馬連
#   ・prob[rank1] < um_axis_min           → 推奨なし（軸になれる馬がいない）
#   ・prob[rank1] >= um_axis_min
#       gap12 < um_box_gap                → BOX（上位が接近 → 軸を固定しない）
#       gap12 >= um_box_gap               → 1頭軸（rank1 が明確に抜けている）
#   相手候補: prob >= um_partner_min の馬すべて（rankではなく指数で選ぶ）
#
#  3連複
#   ・prob[rank1] >= tp_axis_min AND prob[rank2] >= tp_dual_min
#       gap23 >= tp_box_gap               → 2頭軸（rank1+rank2 固定, rank3以降から相手）
#       gap23 < tp_box_gap                → BOX（上位3頭以上が接近）
#   ・prob[rank1] >= tp_axis_min AND prob[rank2] < tp_dual_min
#       gap12 >= tp_box_gap               → 1頭軸
#       gap12 < tp_box_gap                → BOX（rank1と rank2が接近）
#   ・prob[rank1] < tp_axis_min           → 推奨なし
#   相手候補: prob >= tp_partner_min の馬すべて（rankではなく指数で選ぶ）

SB_B_PARAMS = {
    "um_axis_min":    0.12,
    "um_box_gap":     0.04,
    "um_partner_min": 0.05,
    "tp_axis_min":    0.12,
    "tp_dual_min":    0.09,
    "tp_box_gap":     0.04,
    "tp_partner_min": 0.04,
    "um_1jiku_max":   25.0,
    "um_box_max":     15.0,
    "tp_1jiku_max":   50.0,
    "tp_2jiku_max":   30.0,
    "tp_box_max":     20.0,
}

# 良馬場専用パラメータ（ダート良×[0.18,0.20) ROI 499% / 芝良×>=0.30 ROI 236%）
# 相手閾値を上げて絞り込む戦略。axis_prob フィルターと組み合わせて使う。
SB_GOOD_PARAMS = {
    "um_axis_min":    0.12,
    "um_box_gap":     0.04,
    "um_partner_min": 0.10,   # 0.05 → 0.10
    "tp_axis_min":    0.12,
    "tp_dual_min":    0.09,
    "tp_box_gap":     0.04,
    "tp_partner_min": 0.07,   # 0.04 → 0.07
    "um_1jiku_max":   20.0,
    "um_box_max":     15.0,
    "tp_1jiku_max":   30.0,
    "tp_2jiku_max":   30.0,
    "tp_box_max":     15.0,   # 20 → 15
}

# 道悪用パラメータ（最良でもROI 82%: 参考提案のみ）
SB_MUDDY_PARAMS = {
    "um_axis_min":    0.12,
    "um_box_gap":     0.04,
    "um_partner_min": 0.10,
    "tp_axis_min":    0.12,
    "tp_dual_min":    0.09,
    "tp_box_gap":     0.04,
    "tp_partner_min": 0.05,
    "um_1jiku_max":   20.0,
    "um_box_max":     20.0,
    "tp_1jiku_max":   30.0,
    "tp_2jiku_max":   30.0,
    "tp_box_max":     20.0,
}


def recommend_score_based(
    df: pd.DataFrame,
    # ── 軸・相手・BOX 判定パラメータ ──
    um_axis_min:    float = 0.12,  # 馬連: rank1 がこれ以上なら軸にする
    um_box_gap:     float = 0.04,  # 馬連: gap12 がこれ未満 → BOX
    um_partner_min: float = 0.05,  # 馬連: この確率以上の馬を相手に含める
    tp_axis_min:    float = 0.12,  # 3連複: rank1 の最低確率（軸条件）
    tp_dual_min:    float = 0.09,  # 3連複: rank2 がこれ以上 → 2頭軸を検討
    tp_box_gap:     float = 0.04,  # 3連複: gap がこれ未満 → BOX に切替
    tp_partner_min: float = 0.04,  # 3連複: この確率以上の馬を相手に含める
    # ── 方式別 期待オッズ上限（BOXは組合せが多いので1軸より厳しくする）──
    um_1jiku_max:   float = 25.0,  # 馬連 1頭軸
    um_box_max:     float = 25.0,  # 馬連 BOX（案1では 15 程度に絞る）
    tp_1jiku_max:   float = 50.0,  # 3連複 1頭軸
    tp_2jiku_max:   float = 50.0,  # 3連複 2頭軸（案1では 30 程度に絞る）
    tp_box_max:     float = 50.0,  # 3連複 BOX（案1では 20 程度に絞る）
    # ── 市場オッズ対比フィルター（券種別に独立）──
    win_odds_col:       str   = "オッズ",  # 単勝オッズ列名
    um_value_ratio_min: float = 1.0,       # 馬連: 軸馬の「モデル確率/市場implied確率」の最低値
    tp_value_ratio_min: float = 1.0,       # 3連複: 同上（馬連と独立して設定可能）
    # ── 期待オッズ下限（低配当チケット排除）──
    um_min_exp_odds: float = 1.0,          # 馬連 期待オッズ下限（1.0=無効）
    tp_min_exp_odds: float = 1.0,          # 3連複 期待オッズ下限
) -> dict:
    """指数（確率）ベースで軸・相手・BOX を動的に決定する。
    方式ごとに期待オッズ上限を変えられる（BOXは票数が増えるので上限を厳しくする）。

    Returns:
        {
          "馬連":  {"選択方式": str, "軸": list[int], "tickets": list[dict]},
          "3連複": {"選択方式": str, "軸": list[int], "tickets": list[dict]},
          "_meta": dict,   # axis_prob, gap12, gap23, um_method, tp_method
        }
    """
    df = df.copy().reset_index(drop=True)
    blended = blend_scores(df)
    probs   = scores_to_probs(blended)
    n       = len(probs)
    if n < 3:
        return {}

    uma        = pd.to_numeric(df["馬番"], errors="coerce").astype(int).tolist()
    rank_order = np.argsort(probs)[::-1]
    ax1 = rank_order[0]
    ax2 = rank_order[1]
    ax3 = rank_order[2]

    p1 = float(probs[ax1])
    p2 = float(probs[ax2])
    p3 = float(probs[ax3])
    gap12 = p1 - p2
    gap23 = p2 - p3

    # ── 市場オッズ対比フィルター（券種別）─────────────────────
    _TANSHO_RATE = 0.80
    value_ratio = None
    _need_vr = (um_value_ratio_min > 1.0 or tp_value_ratio_min > 1.0)
    if _need_vr and win_odds_col in df.columns:
        raw_odds = pd.to_numeric(df[win_odds_col], errors="coerce").values
        odds_ax1 = float(raw_odds[ax1]) if not np.isnan(raw_odds[ax1]) else 0.0
        if odds_ax1 > 0:
            value_ratio = p1 / (_TANSHO_RATE / odds_ax1)

    _um_value_ok = (value_ratio is not None and value_ratio >= um_value_ratio_min) \
                   if um_value_ratio_min > 1.0 else True
    _tp_value_ok = (value_ratio is not None and value_ratio >= tp_value_ratio_min) \
                   if tp_value_ratio_min > 1.0 else True

    def mk(combo, p, bt):
        eo = round(PAYOUT_RATE[bt] / p, 1) if p > 1e-10 else float("inf")
        return {"組合せ": combo, "期待オッズ": eo, "確率%": round(p * 100, 2)}

    # ── 馬連 ──────────────────────────────────
    um_method  = None
    um_axis    = []
    um_tickets = []

    def _in_range_um(t, max_o):
        return um_min_exp_odds <= t["期待オッズ"] <= max_o

    def _in_range_tp(t, max_o):
        return tp_min_exp_odds <= t["期待オッズ"] <= max_o

    if _um_value_ok and p1 >= um_axis_min:
        if gap12 < um_box_gap:
            # 上位が接近 → BOX（rank1 含む全候補の組み合わせ）
            um_method = "BOX"
            pool = [i for i in range(n) if probs[i] >= um_partner_min]
            for i, j in combinations(pool, 2):
                p = quinella_prob(probs, i, j)
                t = mk(tuple(sorted([uma[i], uma[j]])), p, "馬連")
                if _in_range_um(t, um_box_max):
                    um_tickets.append(t)
        else:
            # rank1 が明確に抜けている → 1頭軸
            um_method = "1頭軸"
            um_axis   = [uma[ax1]]
            partners  = [i for i in range(n) if i != ax1 and probs[i] >= um_partner_min]
            for opp in partners:
                p = quinella_prob(probs, ax1, opp)
                t = mk(tuple(sorted([uma[ax1], uma[opp]])), p, "馬連")
                if _in_range_um(t, um_1jiku_max):
                    um_tickets.append(t)

    um_tickets.sort(key=lambda t: t["期待オッズ"])

    # ── 3連複 ─────────────────────────────────
    tp_method  = None
    tp_axis    = []
    tp_tickets = []

    if _tp_value_ok and p1 >= tp_axis_min:
        if p2 >= tp_dual_min:
            # rank1 と rank2 がともに強い
            if gap23 >= tp_box_gap:
                # rank3 以降との差が大きい → 2頭軸（rank1+rank2 固定）
                tp_method = "2頭軸"
                tp_axis   = [uma[ax1], uma[ax2]]
                partners  = [i for i in range(n)
                             if i not in (ax1, ax2) and probs[i] >= tp_partner_min]
                for opp in partners:
                    p = trifecta_place_prob(probs, ax1, ax2, opp)
                    t = mk(tuple(sorted([uma[ax1], uma[ax2], uma[opp]])), p, "3連複")
                    if _in_range_tp(t, tp_2jiku_max):
                        tp_tickets.append(t)
            else:
                # rank2 と rank3 も接近 → BOX
                tp_method = "BOX"
                pool = [i for i in range(n) if probs[i] >= tp_partner_min]
                for i, j, k in combinations(pool, 3):
                    p = trifecta_place_prob(probs, i, j, k)
                    t = mk(tuple(sorted([uma[i], uma[j], uma[k]])), p, "3連複")
                    if _in_range_tp(t, tp_box_max):
                        tp_tickets.append(t)
        else:
            # rank1 のみ強い
            if gap12 >= tp_box_gap:
                # rank2 との差も大きい → 1頭軸
                tp_method = "1頭軸"
                tp_axis   = [uma[ax1]]
                partners  = [i for i in range(n) if i != ax1 and probs[i] >= tp_partner_min]
                for i, j in combinations(partners, 2):
                    p = trifecta_place_prob(probs, ax1, i, j)
                    t = mk(tuple(sorted([uma[ax1], uma[i], uma[j]])), p, "3連複")
                    if _in_range_tp(t, tp_1jiku_max):
                        tp_tickets.append(t)
            else:
                # rank1 と rank2 が接近 → BOX
                tp_method = "BOX"
                pool = [i for i in range(n) if probs[i] >= tp_partner_min]
                for i, j, k in combinations(pool, 3):
                    p = trifecta_place_prob(probs, i, j, k)
                    t = mk(tuple(sorted([uma[i], uma[j], uma[k]])), p, "3連複")
                    if _in_range_tp(t, tp_box_max):
                        tp_tickets.append(t)

    tp_tickets.sort(key=lambda t: t["期待オッズ"])

    meta = {
        "axis_prob":   p1,
        "gap12":       gap12,
        "gap23":       gap23,
        "axis_strong": p1 >= LONGSHOT_AXIS_PROB and gap12 >= LONGSHOT_AXIS_GAP,
        "um_method":   um_method,
        "tp_method":   tp_method,
        "value_ratio": value_ratio,
    }

    return {
        "馬連":  {"選択方式": um_method or "なし", "軸": um_axis, "tickets": um_tickets},
        "3連複": {"選択方式": tp_method or "なし", "軸": tp_axis, "tickets": tp_tickets},
        "_meta": meta,
    }
