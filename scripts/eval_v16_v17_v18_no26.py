"""eval_v16_v17_v18_no26.py

v16profit / v17focal / v18listwise を v7odds × v15pace ベースラインと比較評価。

特徴量の構築コストが高いため、v9特徴量（104列）と v7用特徴量（60列）を含む
キャッシュ（race_records_v9feat_2026.pkl）を1回構築し、追加学習後は
同キャッシュから即再スコアリングできる設計。

実行: python scripts/eval_v16_v17_v18_no26.py
"""

import sys, warnings, glob, os, time, pickle, itertools, re
warnings.simplefilter("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
sys.path.append(r"C:\keiba_ai\keiba_ai_ver2.0\libs")
sys.path.append(r"C:\keiba_ai\keiba_ai_ver2.0\src\PredictionModels\LightGBM")

import numpy as np
import pandas as pd
import lightgbm as lgb

from src.config import paths
from src.config.constants import PLACE_LIST
from src.logic.prediction.race_prediction_engine import (
    make_dataset_for_lightgbm_v3,
    _build_jockey_course_stats,
    prediction_error,
)
from src.managers import race_card_dataset_manager
from src.PredictionModels.LightGBM.make_dataset_v5 import build_pedigree_vocab, get_pedigree_cats
from src.PredictionModels.LightGBM.make_dataset_v6 import get_extra_past_race_features_v6
from src.PredictionModels.LightGBM.make_dataset_v2 import _parse_weight
from src.PredictionModels.LightGBM.make_dataset_v7 import _parse_kinryo
from src.PredictionModels.LightGBM.make_dataset_v8 import get_extra_past_race_features_v8
from src.PredictionModels.LightGBM.make_dataset_v9 import (
    get_extra_past_race_features_v9, index_v9,
)

import past_performance

BET_UNIT  = 100
EVAL_YEAR = 2026

# feat_rows を含むキャッシュ（v16/v17/v18 共通）
V9FEAT_CACHE = os.path.join(PROJECT_ROOT, "logs", f"race_records_v9feat_{EVAL_YEAR}.pkl")

_FEAT_COLS = [c for c in index_v9 if c != "race_id"]

print("血統vocab読み込み中...")
_VOCAB = build_pedigree_vocab()
_VOCAB_SIZE = len(_VOCAB)
print(f"  {_VOCAB_SIZE}種類")

_lgb_cache = {}


def _safe_float(v, default=np.nan):
    try: return float(v)
    except: return default


def _get_lgb_model(place_id, race_type, length, suffix):
    key = (place_id, race_type, length, suffix)
    if key in _lgb_cache:
        return _lgb_cache[key]
    type_str = "turf" if race_type == "芝" else "dirt"
    mp = os.path.join(paths.PREDICTION_MODEL_PATH, PLACE_LIST[place_id - 1],
                      f"{type_str}{length}_lambdarank_model{suffix}.txt")
    if not os.path.isfile(mp):
        raise FileNotFoundError(mp)
    m = lgb.Booster(model_file=mp)
    _lgb_cache[key] = m
    return m


def _get_current_date_from_path(race_card_path):
    try:
        return pd.to_datetime(os.path.basename(os.path.dirname(race_card_path)), format="%Y%m%d")
    except:
        return pd.NaT


def build_v9feat_row(race_id, rc_df, race_info_df, race_card_path):
    """v9特徴量（104列）と v7用 v3特徴量（60列）を構築して返す。"""
    try:
        place_id   = int(str(race_id)[4:6])
        race_type  = race_info_df.at[0, "race_type"]
        course_len = str(race_info_df.at[0, "course_len"])
        course_info = [place_id, race_type, course_len,
                       race_info_df.at[0, "ground_state"], race_info_df.at[0, "class"]]
        horse_ids  = rc_df["horse_id"].tolist()
        jockey_ids = rc_df["jockey_id"].tolist()
        waku_df    = rc_df[["枠", "馬番"]].reset_index(drop=True)

        kinryo_col     = rc_df.columns[4]
        n_horses_today = float(len(rc_df))
        current_dt     = _get_current_date_from_path(race_card_path)
        js             = _build_jockey_course_stats(place_id)

        feat_rows, v3_rows = [], []

        for idx, hid in enumerate(horse_ids):
            try:
                row_v3 = make_dataset_for_lightgbm_v3(race_id, course_info, hid)
                v3_vals = row_v3.values[0].tolist() if not row_v3.empty else [np.nan] * 60
            except:
                v3_vals = [np.nan] * 60

            f_cat, mf_cat, pgf_cat = get_pedigree_cats(hid, _VOCAB)

            try:
                race_info_5 = past_performance.get_past_race_info(hid, race_id, race_num=5)
            except:
                race_info_5 = pd.DataFrame()

            if not race_info_5.empty:
                extra_v6 = get_extra_past_race_features_v6(race_info_5)
                extra_v8 = get_extra_past_race_features_v8(race_info_5)
                extra_v9 = get_extra_past_race_features_v9(race_info_5)
            else:
                extra_v6 = [np.nan] * 16
                extra_v8 = [np.nan] * 7
                extra_v9 = [np.nan] * 7

            jid    = str(jockey_ids[idx])
            wr, pr = js.get((jid, race_type, course_len), (np.nan, np.nan))
            kinryo = _parse_kinryo(rc_df[kinryo_col].iloc[idx])

            days_since = n_horses_1 = hw_abs_1 = np.nan
            if not race_info_5.empty:
                if not pd.isna(current_dt):
                    try:
                        prev_dt = pd.to_datetime(
                            str(race_info_5.iloc[0].get("日付", "")).strip().replace("/", "-"),
                            format="%Y-%m-%d",
                        )
                        days_since = float((current_dt - prev_dt).days)
                    except:
                        pass
                n_horses_1 = _safe_float(race_info_5.iloc[0].get("頭数", ""))
                hw_abs_1   = _parse_weight(race_info_5.iloc[0].get("馬体重", ""))

            waku   = _safe_float(waku_df.at[idx, "枠"])
            umaban = _safe_float(waku_df.at[idx, "馬番"])

            row = (
                v3_vals[0:60]
                + [wr, pr]
                + [waku, umaban]
                + [np.nan, np.nan]             # current_odds / current_popularity（後で fill）
                + [f_cat, mf_cat, pgf_cat]
                + list(extra_v6)
                + [kinryo, days_since, n_horses_today, n_horses_1, hw_abs_1]
                + list(extra_v8[:5])
                + [extra_v8[5] if len(extra_v8) > 5 else np.nan,
                   extra_v8[6] if len(extra_v8) > 6 else np.nan]
                + list(extra_v9[:5])
                + [extra_v9[5] if len(extra_v9) > 5 else np.nan,
                   extra_v9[6] if len(extra_v9) > 6 else np.nan]
            )
            assert len(row) == len(_FEAT_COLS), f"列数不一致: {len(row)} != {len(_FEAT_COLS)}"
            feat_rows.append(row)
            v3_rows.append(v3_vals)

        return feat_rows, v3_rows, place_id, race_type, course_len, horse_ids, jockey_ids, js, waku_df, True
    except Exception as e:
        prediction_error(e)
        return None, None, None, None, None, None, None, None, None, False


def _fill_odds_pop(feat_rows, final_odds, final_pop):
    odds_arr = np.where(np.isfinite(final_odds.astype(float).values), final_odds.astype(float).values, -1)
    pop_arr  = np.where(np.isfinite(final_pop.astype(float).values),  final_pop.astype(float).values,  -1)
    odds_idx = _FEAT_COLS.index("current_odds")
    pop_idx  = _FEAT_COLS.index("current_popularity")
    rows = []
    for i, row in enumerate(feat_rows):
        r = list(row)
        r[odds_idx] = float(odds_arr[i])
        r[pop_idx]  = float(pop_arr[i])
        rows.append(r)
    return pd.DataFrame(rows, columns=_FEAT_COLS).fillna(-1)


def score_v9model(feat_rows, final_odds, final_pop, place_id, race_type, course_len, suffix):
    try:
        df = _fill_odds_pop(feat_rows, final_odds, final_pop)
        model = _get_lgb_model(place_id, race_type, course_len, suffix)
        return model.predict(df, num_iteration=model.best_iteration)
    except Exception as e:
        prediction_error(e)
        return None


def score_v7(v3_rows, place_id, race_type, course_len, final_odds, final_pop, jockey_stats, jid_list, waku_df):
    try:
        odds_arr = final_odds.astype(float).values
        pop_arr  = final_pop.astype(float).values
        rows_66 = []
        for i, v3 in enumerate(v3_rows):
            wr, pr = jockey_stats.get((str(jid_list[i]), race_type, course_len), (np.nan, np.nan))
            waku   = _safe_float(waku_df.at[i, "枠"])
            umaban = _safe_float(waku_df.at[i, "馬番"])
            od = odds_arr[i] if np.isfinite(odds_arr[i]) else -1
            pp = pop_arr[i]  if np.isfinite(pop_arr[i])  else -1
            rows_66.append(v3[:60] + [wr, pr, waku, umaban, od, pp])
        df66 = pd.DataFrame(rows_66).fillna(-1)
        model = _get_lgb_model(place_id, race_type, course_len, "_v7odds_binary_no26")
        return model.predict(df66, num_iteration=model.best_iteration)
    except Exception as e:
        prediction_error(e)
        return None


def get_return(ret_df, shikibetsu, umaban_str=None):
    rows = ret_df[ret_df["式別"] == shikibetsu]
    if rows.empty: return None
    if umaban_str is not None:
        rows = rows[rows["馬番"].astype(str) == umaban_str]
        if rows.empty: return None
    return int(rows["配当"].iloc[0])


def _norm(arr):
    mn, mx = arr.min(), arr.max()
    return (arr - mn) / (mx - mn + 1e-12)


# ---------- データ収集（feat_rows をキャッシュに含める） ----------

if os.path.exists(V9FEAT_CACHE):
    print(f"\nv9featキャッシュ読み込み: {V9FEAT_CACHE}")
    with open(V9FEAT_CACHE, "rb") as f:
        race_records = pickle.load(f)
    print(f"  {len(race_records)}R")
else:
    active_places = []
    for pid in range(1, 11):
        pfx = f"{EVAL_YEAR}{pid:02d}"
        cnt = sum(len(glob.glob(f"{d}/{pfx}*.csv")) for d in glob.glob(f"data/race_card/{EVAL_YEAR}*"))
        if cnt > 0:
            active_places.append(pid)
    print(f"対象場: {[PLACE_LIST[p-1] for p in active_places]}")

    race_records = []
    t0 = time.time()
    processed = skipped = 0

    for place_id in active_places:
        prefix = f"{EVAL_YEAR}{place_id:02d}"
        race_paths = []
        for dd in sorted(glob.glob(f"data/race_card/{EVAL_YEAR}*")):
            race_paths += sorted(glob.glob(f"{dd}/{prefix}*.csv"))

        for cp in race_paths:
            race_id = os.path.basename(cp).replace(".csv", "")
            res_paths = glob.glob(f"data/race_result/**/{race_id}.csv", recursive=True)
            if not res_paths: skipped += 1; continue
            res_df = pd.read_csv(res_paths[0], index_col=0)
            if "着順" not in res_df.columns: skipped += 1; continue
            ret_paths = glob.glob(f"data/race_info/race_returns/**/{race_id}.csv", recursive=True)
            if not ret_paths: skipped += 1; continue
            ret_df = pd.read_csv(ret_paths[0], index_col=0)
            rc_df  = pd.read_csv(cp, index_col=0).reset_index(drop=True)
            if "枠" not in rc_df.columns or "horse_id" not in rc_df.columns: skipped += 1; continue
            race_info_df = race_card_dataset_manager.get_race_info_csv(race_id)
            if race_info_df.empty: skipped += 1; continue

            nums       = pd.to_numeric(res_df["着順"], errors="coerce")
            winner_set = set(res_df[nums == 1]["馬番"].astype(str))
            top3_set   = set(res_df[nums <= 3]["馬番"].astype(str))
            san_row    = ret_df[ret_df["式別"] == "三連複"]
            san_winner = set(san_row["馬番"].iloc[0].split("-")) if not san_row.empty else None
            san_odds   = int(san_row["配当"].iloc[0]) if not san_row.empty else None

            key_series = rc_df["馬番"].astype(str).str.strip()
            odds_map   = dict(zip(res_df["馬番"].astype(str), pd.to_numeric(res_df["単勝"], errors="coerce")))
            pop_map    = dict(zip(res_df["馬番"].astype(str), pd.to_numeric(res_df["人気"],  errors="coerce")))
            final_odds = key_series.map(odds_map)
            final_pop  = key_series.map(pop_map)

            result = build_v9feat_row(race_id, rc_df, race_info_df, cp)
            feat_rows, v3_rows, pid, rtype, clen, hids, jids, js, waku_df_l, ok = result
            if not ok: skipped += 1; continue

            s_v7  = score_v7(v3_rows, pid, rtype, clen, final_odds, final_pop, js, jids, waku_df_l)
            s_v15 = score_v9model(feat_rows, final_odds, final_pop, pid, rtype, clen, "_v15pace_no26")
            if s_v15 is None: skipped += 1; continue

            umabans   = [str(int(float(u))) for u in rc_df["馬番"].tolist()]
            tan_ret   = get_return(ret_df, "単勝")
            fuku_rets = {u: get_return(ret_df, "複勝", u) for u in umabans}

            race_records.append({
                "winner_set": winner_set, "top3_set": top3_set,
                "san_winner": san_winner, "san_odds": san_odds,
                "tan_ret": tan_ret, "fuku_rets": fuku_rets, "umabans": umabans,
                "s_v7":  np.array(s_v7,  dtype=float) if s_v7 is not None else None,
                "s_v15": np.array(s_v15, dtype=float),
                # 追加スコアリング用（feat_rows をキャッシュに保持）
                "feat_rows":   feat_rows,
                "v3_rows":     v3_rows,
                "final_odds":  final_odds.values.tolist(),
                "final_pop":   final_pop.values.tolist(),
                "place_id":    pid,
                "race_type":   rtype,
                "course_len":  clen,
                "jid_list":    jids,
                "js":          js,
                "waku_df":     waku_df_l.to_dict("list"),
            })
            processed += 1
            if processed % 200 == 0:
                print(f"  {processed}R収集済み ({time.time()-t0:.0f}秒)")
                with open(V9FEAT_CACHE, "wb") as _f:
                    pickle.dump(race_records, _f)

    print(f"\n収集完了: {processed}R ({time.time()-t0:.0f}秒), スキップ:{skipped}R")
    with open(V9FEAT_CACHE, "wb") as f:
        pickle.dump(race_records, f)
    print(f"キャッシュ保存: {V9FEAT_CACHE}")


# ---------- 新モデルのスコアを追加（追学習後に再実行するだけで即評価可能） ----------

NEW_MODELS = {
    "s_v16": "_v16profit_no26",
    "s_v17": "_v17focal_no26",
    "s_v18": "_v18listwise_no26",
}

updated = False
for score_key, suffix in NEW_MODELS.items():
    if all(score_key in r for r in race_records):
        n_scored = sum(1 for r in race_records if r.get(score_key) is not None)
        print(f"{score_key}: キャッシュ済み ({n_scored}R)")
        continue

    print(f"\n{score_key}（{suffix}）スコアリング中...")
    ok_cnt = skip_cnt = 0
    for r in race_records:
        if score_key in r:
            ok_cnt += 1
            continue
        fo = pd.Series(r["final_odds"])
        fp = pd.Series(r["final_pop"])
        s = score_v9model(r["feat_rows"], fo, fp, r["place_id"], r["race_type"], r["course_len"], suffix)
        if s is not None:
            r[score_key] = np.array(s, dtype=float)
            ok_cnt += 1
        else:
            skip_cnt += 1
    print(f"  完了: {ok_cnt}R / スキップ: {skip_cnt}R")
    updated = True

if updated:
    with open(V9FEAT_CACHE, "wb") as f:
        pickle.dump(race_records, f)
    print("キャッシュ更新完了")


# ---------- 評価 ----------

class Stats:
    def __init__(self):
        self.n = self.tan_hit = self.tan_pay = self.tan_bet = 0
        self.fuku_hit = self.fuku_pay = self.fuku_bet = 0
        self.san_hit  = self.san_pay  = self.san_bet  = 0

    def add(self, tan_h, tan_ret, fuku_h, fuku_ret, san_h, san_ret, san_bets):
        self.n += 1
        self.tan_bet += BET_UNIT; self.fuku_bet += BET_UNIT
        self.san_bet += BET_UNIT * san_bets
        if tan_h:
            self.tan_hit += 1
            if tan_ret: self.tan_pay += tan_ret
        if fuku_h:
            self.fuku_hit += 1
            if fuku_ret: self.fuku_pay += fuku_ret
        if san_h:
            self.san_hit += 1
            if san_ret: self.san_pay += san_ret

    def tan_pct(self):  return 100 * self.tan_hit  / self.n if self.n else 0
    def fuku_pct(self): return 100 * self.fuku_hit / self.n if self.n else 0
    def san_pct(self):  return 100 * self.san_hit  / self.n if self.n else 0
    def tan_rec(self):  return 100 * self.tan_pay  / self.tan_bet  if self.tan_bet  else 0
    def fuku_rec(self): return 100 * self.fuku_pay / self.fuku_bet if self.fuku_bet else 0
    def san_rec(self):  return 100 * self.san_pay  / self.san_bet  if self.san_bet  else 0


def evaluate(label, records, score_fn, width=62):
    st = Stats()
    for r in records:
        scores = score_fn(r)
        if scores is None: continue
        order  = np.argsort(-scores)
        top5   = [r["umabans"][i] for i in order[:5]]
        honmei = top5[0]
        san_combs = list(itertools.combinations(top5, 3))
        san_h = (r["san_winner"] is not None) and any(
            {a, b, c} == r["san_winner"] for a, b, c in san_combs
        )
        st.add(
            honmei in r["winner_set"], r["tan_ret"],
            honmei in r["top3_set"],  r["fuku_rets"].get(honmei),
            san_h, r["san_odds"], len(san_combs),
        )
    print(f"  {label:<{width}}  n={st.n:>4}  "
          f"単勝 {st.tan_pct():>5.1f}%/{st.tan_rec():>6.1f}%  "
          f"複勝 {st.fuku_pct():>5.1f}%/{st.fuku_rec():>6.1f}%  "
          f"3連複 {st.san_pct():>5.1f}%/{st.san_rec():>6.1f}%")
    return st


W = 62
SEP = "=" * 130
print(f"\n{SEP}")
print(f"  {'モデル':<{W}}  {'件数':>6}  {'単勝的中/回収':>15}  {'複勝的中/回収':>15}  {'3連複的中/回収':>16}")
print(SEP)

# ベースライン
evaluate("v7odds（ベースライン）",
         [r for r in race_records if r.get("s_v7") is not None],
         lambda r: r["s_v7"])

evaluate("v15pace（ペース適性 v9 LightGBM）",
         race_records,
         lambda r: r.get("s_v15"))

# 各新モデル 単独
for sk, label in [("s_v16", "v16profit（ROI直接最適化 × v9）"),
                  ("s_v17", "v17focal（Focal Loss × v9）"),
                  ("s_v18", "v18listwise（Listwise Softmax ROI × v9）")]:
    recs = [r for r in race_records if sk in r]
    if recs:
        evaluate(label, recs, lambda r, k=sk: r.get(k))
    else:
        print(f"  {label:<{W}}: 未学習（スキップ）")

# ブレンドスイープ
for sk, label_pfx in [("s_v16", "v16profit"),
                       ("s_v17", "v17focal"),
                       ("s_v18", "v18listwise")]:
    recs = [r for r in race_records if sk in r and r.get("s_v7") is not None]
    if not recs:
        continue
    print(f"\n--- {label_pfx} × v7odds ブレンド ---")
    for alpha in [round(a * 0.1, 1) for a in range(0, 11)]:
        evaluate(
            f"{label_pfx}×v7 α={alpha:.1f}  ({label_pfx}:{alpha:.1f} v7:{1-alpha:.1f})",
            recs,
            lambda r, a=alpha, k=sk: a * _norm(r[k]) + (1 - a) * _norm(r["s_v7"]),
            width=56,
        )

# v15 × v7 ブレンド（比較用）
print(f"\n--- v15pace × v7odds ブレンド（比較用）---")
for alpha in [0.3, 0.4, 0.5, 0.6]:
    recs = [r for r in race_records if r.get("s_v7") is not None]
    evaluate(
        f"v15×v7 α={alpha:.1f}",
        recs,
        lambda r, a=alpha: (1 - a) * _norm(r["s_v15"]) + a * _norm(r["s_v7"]),
        width=56,
    )

print(SEP)
