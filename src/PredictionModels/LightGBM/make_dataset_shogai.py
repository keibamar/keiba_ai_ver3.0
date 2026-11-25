"""障害戦専用モデルのデータセット作成

障害戦は年間 約120レースしかなく、コース別モデル（芝/ダート用）を作れないため、
全場共通の1モデルにする（新馬戦専用モデル shinba と同じ考え方）。

特徴量は「その馬の過去成績（障害戦 / 平地戦を分けて集計）」「今回の条件」
「騎手の障害戦成績」が中心。過去成績はすべて対象レースの日付より前のものだけを使う
（学習時・予測時で同じ関数 horse_history_features を通すため、未来情報は混ざらない）。

- 学習用:   make_training_dataset()
- 予測用:   make_rows_for_prediction()
"""

import glob
import os
import re
from datetime import datetime

import numpy as np
import pandas as pd

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
RACE_RESULT_DIR = os.path.join(_PROJECT_ROOT, "data", "race_result")
PAST_PERF_DIR = os.path.join(_PROJECT_ROOT, "data", "horse", "past_performance")

PLACE_NAMES = ["札幌", "函館", "福島", "新潟", "東京", "中山", "中京", "京都", "阪神", "小倉"]
_GROUND = {"良": 0, "稍重": 1, "稍": 1, "重": 2, "不良": 3, "不": 3}

# 馬の過去成績・今回条件・騎手（オッズを使わない）: 回収率重視(val)モデル用
BASE_FEATURES = [
    # 今回の条件
    "place_id", "course_len", "ground", "is_open", "field_size",
    "waku", "umaban", "kinryo", "age", "sex", "horse_weight", "weight_diff",
    # 障害戦の過去成績
    "j_n", "j_win_rate", "j_top3_rate", "j_mean_rank_ratio", "j_best_rank",
    "j_last_rank_ratio", "j_last_margin", "j_mean_margin3", "j_last_agari",
    "j_dnf_rate", "j_days_since", "j_last_pop", "j_same_course_n",
    "j_same_course_top3_rate", "j_last_corner1_ratio",
    # 平地戦の過去成績
    "f_n", "f_win_n", "f_best_rank", "f_mean_margin3",
    # 全体
    "all_n", "days_since", "last_rank_ratio", "last_margin",
    # 騎手の障害戦成績（対象日より前の累積）
    "jk_n", "jk_win_rate", "jk_top3_rate",
]
# 的中率重視(hit)モデル用: 上記 + 当日オッズ
ODDS_FEATURES = ["log_odds", "popularity"]
HIT_FEATURES = BASE_FEATURES + ODDS_FEATURES
VAL_FEATURES = BASE_FEATURES


# ---------- 小さなパーサ ----------

def is_shogai(race_type) -> bool:
    return "障" in str(race_type)


def _num(v):
    try:
        return float(re.sub(r"[^\d.\-]", "", str(v)))
    except Exception:
        return np.nan


def _rank_num(v):
    """着順を数値に。中止・失格・除外・取消は NaN"""
    s = str(v).strip().split("(")[0]
    if not re.fullmatch(r"\d+(\.0)?", s):
        return np.nan
    return float(s)


def _is_dnf(v) -> bool:
    s = str(v)
    return ("中" in s) or ("失" in s)


def _is_scratch(v) -> bool:
    s = str(v)
    return ("取" in s) or ("除" in s)


def _parse_date(s):
    s = str(s)
    for fmt in ("%Y/%m/%d", "%Y年%m月%d日", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None


def _sex_age(v):
    s = str(v)
    sex = 1.0 if s.startswith("牝") else (2.0 if s.startswith("セ") else 0.0)
    return sex, _num(s[1:])


def _weight(v):
    m = re.match(r"\s*(\d+)", str(v))
    return float(m.group(1)) if m else np.nan


def _corner1_ratio(passing, heads):
    try:
        first = float(str(passing).split("-")[0])
        return first / float(heads)
    except Exception:
        return np.nan


# ---------- 馬の過去成績 ----------

_past_cache: dict = {}


def load_past_performance(horse_id, loader=None):
    """過去成績CSVを読み、日付列 _dt を付けて新しい順に並べて返す"""
    hid = str(horse_id)
    if hid in _past_cache:
        return _past_cache[hid]
    df = pd.DataFrame()
    try:
        if loader is not None:
            df = loader(hid)
        else:
            p = os.path.join(PAST_PERF_DIR, f"{hid}.csv")
            if os.path.isfile(p):
                df = pd.read_csv(p, index_col=0)
    except Exception:
        df = pd.DataFrame()
    if not df.empty and "日付" in df.columns:
        df = df.copy()
        df["_dt"] = df["日付"].map(_parse_date)
        df = df[df["_dt"].notna()].sort_values("_dt", ascending=False).reset_index(drop=True)
    _past_cache[hid] = df
    return df


def clear_cache():
    _past_cache.clear()


def horse_history_features(past_df, race_date, place_id, course_len):
    """race_date より前の過去成績から特徴量 dict を作る"""
    f = {}
    if past_df is None or past_df.empty:
        prev = pd.DataFrame()
    else:
        prev = past_df[past_df["_dt"] < race_date]
        prev = prev[~prev["着順"].map(_is_scratch)]

    def days(d):
        return float((race_date - d).days) if d is not None else np.nan

    f["all_n"] = float(len(prev))
    if len(prev):
        last = prev.iloc[0]
        f["days_since"] = days(last["_dt"])
        f["last_rank_ratio"] = _rank_num(last["着順"]) / _num(last.get("頭数"))
        f["last_margin"] = _num(last.get("着差"))
    else:
        f["days_since"] = f["last_rank_ratio"] = f["last_margin"] = np.nan

    j = prev[prev["race_type"].map(is_shogai)] if len(prev) else prev
    fl = prev[~prev["race_type"].map(is_shogai)] if len(prev) else prev

    # 障害戦
    f["j_n"] = float(len(j))
    if len(j):
        ranks = j["着順"].map(_rank_num)
        heads = j["頭数"].map(_num)
        f["j_win_rate"] = float((ranks == 1).sum()) / len(j)
        f["j_top3_rate"] = float((ranks <= 3).sum()) / len(j)
        f["j_mean_rank_ratio"] = float(np.nanmean(ranks / heads)) if ranks.notna().any() else np.nan
        f["j_best_rank"] = float(np.nanmin(ranks)) if ranks.notna().any() else np.nan
        f["j_last_rank_ratio"] = ranks.iloc[0] / heads.iloc[0] if heads.iloc[0] else np.nan
        margins = j["着差"].map(_num)
        f["j_last_margin"] = margins.iloc[0]
        f["j_mean_margin3"] = float(np.nanmean(margins.iloc[:3])) if margins.iloc[:3].notna().any() else np.nan
        f["j_last_agari"] = _num(j.iloc[0].get("上り"))
        f["j_dnf_rate"] = float(j["着順"].map(_is_dnf).sum()) / len(j)
        f["j_days_since"] = days(j.iloc[0]["_dt"])
        f["j_last_pop"] = _num(j.iloc[0].get("人気"))
        same = j[(j["開催"].astype(str) == PLACE_NAMES[place_id - 1])
                 & (j["course_len"].map(_num) == float(course_len))]
        f["j_same_course_n"] = float(len(same))
        f["j_same_course_top3_rate"] = (
            float((same["着順"].map(_rank_num) <= 3).sum()) / len(same) if len(same) else np.nan
        )
        f["j_last_corner1_ratio"] = _corner1_ratio(j.iloc[0].get("通過"), j.iloc[0].get("頭数"))
    else:
        for k in ["j_win_rate", "j_top3_rate", "j_mean_rank_ratio", "j_best_rank",
                  "j_last_rank_ratio", "j_last_margin", "j_mean_margin3", "j_last_agari",
                  "j_dnf_rate", "j_days_since", "j_last_pop", "j_same_course_top3_rate",
                  "j_last_corner1_ratio"]:
            f[k] = np.nan
        f["j_same_course_n"] = 0.0

    # 平地戦
    f["f_n"] = float(len(fl))
    if len(fl):
        franks = fl["着順"].map(_rank_num)
        f["f_win_n"] = float((franks == 1).sum())
        f["f_best_rank"] = float(np.nanmin(franks)) if franks.notna().any() else np.nan
        fm = fl["着差"].map(_num).iloc[:3]
        f["f_mean_margin3"] = float(np.nanmean(fm)) if fm.notna().any() else np.nan
    else:
        f["f_win_n"] = 0.0
        f["f_best_rank"] = f["f_mean_margin3"] = np.nan

    # 前走馬体重（weight_diff 用）
    f["_last_weight"] = _weight(prev.iloc[0].get("馬体重")) if len(prev) else np.nan
    return f


# ---------- 障害戦の全結果（学習データ・騎手成績の元） ----------

_results_cache = {}


def load_all_shogai_results():
    """data/race_result/*/YYYY_race_results.csv から障害戦だけを集める"""
    if "df" in _results_cache:
        return _results_cache["df"]
    frames = []
    for p in sorted(glob.glob(os.path.join(RACE_RESULT_DIR, "*", "20*_race_results.csv"))):
        try:
            d = pd.read_csv(p, index_col=0, low_memory=False)
        except Exception:
            continue
        d = d[d["race_type"].map(is_shogai)]
        if d.empty:
            continue
        d = d.copy()
        d["race_id"] = d.index.astype(str)
        frames.append(d.reset_index(drop=True))
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not df.empty:
        df["_dt"] = df["date"].map(_parse_date)
        df = df[df["_dt"].notna()]
        df["_rank"] = df["着順"].map(_rank_num)
        df = df.drop_duplicates(subset=["race_id", "horse_id"])
    _results_cache["df"] = df
    return df


class JockeyStats:
    """騎手の障害戦成績を「指定日より前」の累積で返す"""

    def __init__(self, results_df):
        d = results_df[~results_df["着順"].map(_is_scratch)][["jockey_id", "_dt", "_rank"]].copy()
        d["jockey_id"] = d["jockey_id"].astype(str)
        d["win"] = (d["_rank"] == 1).astype(float)
        d["top3"] = (d["_rank"] <= 3).astype(float)
        d = d.sort_values("_dt")
        self._by = {k: g.reset_index(drop=True) for k, g in d.groupby("jockey_id")}

    def get(self, jockey_id, before):
        g = self._by.get(str(jockey_id))
        if g is None:
            return 0.0, np.nan, np.nan
        n = int(np.searchsorted(g["_dt"].values, np.datetime64(before), side="left"))
        if n == 0:
            return 0.0, np.nan, np.nan
        sub = g.iloc[:n]
        return float(n), float(sub["win"].mean()), float(sub["top3"].mean())


def _row_features(hist, race_date, place_id, course_len, ground, cls, field_size,
                  waku, umaban, kinryo, seirei, weight_str, jockey_stats, jockey_id,
                  odds, popularity):
    sex, age = _sex_age(seirei)
    hw = _weight(weight_str)
    row = {
        "place_id": float(place_id),
        "course_len": float(course_len),
        "ground": float(_GROUND.get(str(ground), np.nan)),
        "is_open": 0.0 if "未勝利" in str(cls) else 1.0,
        "field_size": float(field_size),
        "waku": _num(waku), "umaban": _num(umaban), "kinryo": _num(kinryo),
        "age": age, "sex": sex, "horse_weight": hw,
        "weight_diff": hw - hist["_last_weight"] if not np.isnan(hist["_last_weight"]) else np.nan,
    }
    row.update({k: v for k, v in hist.items() if not k.startswith("_")})
    n, w, t3 = jockey_stats.get(jockey_id, race_date) if jockey_stats else (0.0, np.nan, np.nan)
    row["jk_n"], row["jk_win_rate"], row["jk_top3_rate"] = n, w, t3
    o = _num(odds)
    row["log_odds"] = float(np.log(o)) if o and o > 0 else np.nan
    row["popularity"] = _num(popularity)
    return row


def relevance(rank):
    if rank == 1:
        return 3
    if rank == 2:
        return 2
    if rank == 3:
        return 1
    return 0


# ---------- 学習データ ----------

def make_training_dataset(progress=True):
    """障害戦の全結果から学習データを作る

    Returns:
        pd.DataFrame: race_id, date, horse_id, rank, label, tan_odds + HIT_FEATURES
                      （日付→race_id 順にソート済み。取消・除外は除く）
    """
    res = load_all_shogai_results()
    res = res[~res["着順"].map(_is_scratch)].copy()
    res = res.sort_values(["_dt", "race_id"]).reset_index(drop=True)
    js = JockeyStats(load_all_shogai_results())
    field = res.groupby("race_id")["horse_id"].transform("count")

    rows = []
    it = range(len(res))
    if progress:
        try:
            from tqdm import tqdm
            it = tqdm(it, desc="shogai dataset")
        except ImportError:
            pass
    for i in it:
        r = res.iloc[i]
        race_id = str(r["race_id"])
        place_id = int(race_id[4:6])
        hist = horse_history_features(load_past_performance(r["horse_id"]), r["_dt"],
                                      place_id, r["course_len"])
        row = _row_features(hist, r["_dt"], place_id, r["course_len"], r["ground_state"],
                            r["class"], field.iloc[i], r["枠番"], r["馬番"], r["斤量"],
                            r["性齢"], r["馬体重"], js, r["jockey_id"], r["単勝"], r["人気"])
        rk = r["_rank"]
        row.update({
            "race_id": race_id, "date": r["_dt"], "horse_id": str(r["horse_id"]),
            "rank": rk, "label": relevance(rk), "tan_odds": _num(r["単勝"]),
        })
        rows.append(row)
    return pd.DataFrame(rows)


# ---------- 予測用 ----------

_js_cache = {}


def _jockey_stats_live():
    if "js" not in _js_cache:
        res = load_all_shogai_results()
        _js_cache["js"] = JockeyStats(res) if not res.empty else None
    return _js_cache["js"]


def make_rows_for_prediction(race_id, race_date, horse_ids, race_info, card,
                             past_loader=None):
    """出馬表から予測用の特徴量 DataFrame（HIT_FEATURES 列）を作る

    Args:
        race_id: race_id
        race_date (datetime): 開催日
        horse_ids (list): horse_id（出馬表の並び順）
        race_info (dict): course_len, ground_state, class
        card (dict[str, list]): 枠, 馬番, 斤量, 性齢, 馬体重, jockey_id, オッズ, 人気
                                （各リストは horse_ids と同じ長さ。無い列は省略可）
        past_loader: horse_id -> 過去成績DataFrame（None ならCSVを直接読む）
    """
    place_id = int(str(race_id)[4:6])
    js = _jockey_stats_live()
    n = len(horse_ids)

    def col(name, i):
        v = card.get(name)
        return v[i] if v is not None and i < len(v) else np.nan

    rows = []
    for i, hid in enumerate(horse_ids):
        past = load_past_performance(hid, loader=past_loader)
        hist = horse_history_features(past, race_date, place_id, race_info["course_len"])
        rows.append(_row_features(
            hist, race_date, place_id, race_info["course_len"], race_info.get("ground_state"),
            race_info.get("class"), n, col("枠", i), col("馬番", i), col("斤量", i),
            col("性齢", i), col("馬体重", i), js, col("jockey_id", i),
            col("オッズ", i), col("人気", i),
        ))
    return pd.DataFrame(rows)[HIT_FEATURES]
