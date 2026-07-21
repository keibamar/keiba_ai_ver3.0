"""eval_v15pace_no26.py

v15pace_no26（ペース適性特徴量 v9 データセット、105列 LightGBM）を v7odds と比較評価。

実行: python scripts/eval_v15pace_no26.py
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
from src.PredictionModels.LightGBM.make_dataset_v2 import (
    _parse_agari, _parse_margin, _parse_corner_ratio, _get_time_info, _parse_weight,
)
from src.PredictionModels.LightGBM.make_dataset_v7 import _parse_kinryo
from src.PredictionModels.LightGBM.make_dataset_v8 import get_extra_past_race_features_v8
from src.PredictionModels.LightGBM.make_dataset_v9 import (
    get_extra_past_race_features_v9, index_v9,
)

import past_performance

BET_UNIT  = 100
EVAL_YEAR = 2026
V15_CACHE = os.path.join(PROJECT_ROOT, "logs", f"race_records_v15eval_{EVAL_YEAR}.pkl")

# v9 の列順（race_id を除く 104列）
_FEAT_COLS = [c for c in index_v9 if c != "race_id"]

print("血統vocab読み込み中...")
_VOCAB = build_pedigree_vocab()
_VOCAB_SIZE = len(_VOCAB)
print(f"  {_VOCAB_SIZE}種類")

_lgb_cache = {}


def _safe_float(v, default=np.nan):
    try: return float(v)
    except: return default


def _parse_result(val):
    s = str(val)
    if any(c in s for c in ("除", "取", "中", "失")):
        return np.nan
    try:
        digits = re.sub(r"[^\d]", "", s)
        return float(digits) if digits else np.nan
    except:
        return np.nan


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


def build_v15_row(race_id, rc_df, race_info_df, race_card_path):
    """v15pace用: 各馬の 104 特徴量リストを構築する。

    Returns:
        feat_rows : list of 104-element lists (per horse, in index_v9 order excluding race_id)
        v3_rows   : list of 60-element lists (for v7 scoring)
        place_id, race_type, course_len, horse_ids, ok
    """
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

        js = _build_jockey_course_stats(place_id)

        feat_rows = []
        v3_rows   = []

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

            try:
                extra_v6 = get_extra_past_race_features_v6(race_info_5)
            except:
                extra_v6 = [np.nan] * 16

            try:
                extra_v8 = get_extra_past_race_features_v8(race_info_5)
            except:
                extra_v8 = [np.nan] * 7

            try:
                extra_v9 = get_extra_past_race_features_v9(race_info_5)
            except:
                extra_v9 = [np.nan] * 7

            jid = str(jockey_ids[idx])
            wr, pr = js.get((jid, race_type, course_len), (np.nan, np.nan))
            kinryo = _parse_kinryo(rc_df[kinryo_col].iloc[idx])

            days_since = np.nan
            n_horses_1 = np.nan
            hw_abs_1   = np.nan
            if not race_info_5.empty:
                if not pd.isna(current_dt):
                    try:
                        prev_dt = pd.to_datetime(
                            str(race_info_5.iloc[0].get("日付", "")).strip().replace("/", "-"),
                            format="%Y-%m-%d"
                        )
                        days_since = float((current_dt - prev_dt).days)
                    except:
                        pass
                n_horses_1 = _safe_float(race_info_5.iloc[0].get("頭数", ""))
                hw_abs_1   = _parse_weight(race_info_5.iloc[0].get("馬体重", ""))

            waku   = _safe_float(waku_df.at[idx, "枠"])
            umaban = _safe_float(waku_df.at[idx, "馬番"])

            # 104特徴量を index_v9 順に組み立て
            # (race_id は除く、current_odds/pop は後で fill → とりあえず NaN)
            row = (
                v3_vals[0:60]                    # blood(36) + past3(21) + trends(3)
                + [wr, pr]                        # jockey (2)
                + [waku, umaban]                  # waku/umaban (2)
                + [np.nan, np.nan]               # current_odds, current_popularity (filled later)
                + [f_cat, mf_cat, pgf_cat]        # blood ids (3)
                + list(extra_v6)                  # v6 extras (16)
                + [kinryo, days_since, n_horses_today, n_horses_1, hw_abs_1]  # v7 extras (5)
                + list(extra_v8[:5])              # corner_chase_1〜5 (5)
                + [extra_v8[5] if len(extra_v8) > 5 else np.nan,   # agari_trend_5
                   extra_v8[6] if len(extra_v8) > 6 else np.nan]   # time_diff_trend_5
                + list(extra_v9[:5])              # agari_df_course_1〜5 (5)
                + [extra_v9[5] if len(extra_v9) > 5 else np.nan,   # corner_ratio_std5
                   extra_v9[6] if len(extra_v9) > 6 else np.nan]   # agari_std5
            )
            assert len(row) == len(_FEAT_COLS), f"行長さ不一致: {len(row)} vs {len(_FEAT_COLS)}"
            feat_rows.append(row)
            v3_rows.append(v3_vals)

        return feat_rows, v3_rows, place_id, race_type, course_len, horse_ids, True
    except Exception as e:
        prediction_error(e)
        return None, None, None, None, None, None, False


def score_v15pace(feat_rows, final_odds, final_pop, place_id, race_type, course_len):
    try:
        odds_arr = np.where(np.isfinite(final_odds.astype(float).values), final_odds.astype(float).values, -1)
        pop_arr  = np.where(np.isfinite(final_pop.astype(float).values),  final_pop.astype(float).values,  -1)

        rows = []
        for i, row in enumerate(feat_rows):
            r = list(row)
            # current_odds / current_popularity の位置を更新（index 62, 63 = _FEAT_COLS[62:64]）
            odds_idx = _FEAT_COLS.index("current_odds")
            pop_idx  = _FEAT_COLS.index("current_popularity")
            r[odds_idx] = float(odds_arr[i])
            r[pop_idx]  = float(pop_arr[i])
            rows.append(r)

        df = pd.DataFrame(rows, columns=_FEAT_COLS).fillna(-1)
        model = _get_lgb_model(place_id, race_type, course_len, "_v15pace_no26")
        return model.predict(df, num_iteration=model.best_iteration)
    except Exception as e:
        prediction_error(e)
        return None


def score_v7(v3_rows, place_id, race_type, course_len, final_odds, final_pop, jockey_stats, rc_df, waku_df):
    try:
        odds_arr = final_odds.astype(float).values
        pop_arr  = final_pop.astype(float).values
        rows_66 = []
        for i, v3 in enumerate(v3_rows):
            jid = str(rc_df["jockey_id"].iloc[i])
            wr, pr = jockey_stats.get((jid, race_type, course_len), (np.nan, np.nan))
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


# ---------- データ収集 ----------

if os.path.exists(V15_CACHE):
    print(f"\nv15evalキャッシュ読み込み: {V15_CACHE}")
    with open(V15_CACHE, "rb") as f:
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

            waku_df_local = rc_df[["枠", "馬番"]].reset_index(drop=True)
            js = _build_jockey_course_stats(int(str(race_id)[4:6]))

            result = build_v15_row(race_id, rc_df, race_info_df, cp)
            feat_rows, v3_rows, pid, rtype, clen, hids, ok = result
            if not ok: skipped += 1; continue

            s_v7 = score_v7(v3_rows, pid, rtype, clen, final_odds, final_pop,
                            js, rc_df, waku_df_local)
            s_v15 = score_v15pace(feat_rows, final_odds, final_pop, pid, rtype, clen)
            if s_v15 is None: skipped += 1; continue

            umabans   = [str(int(float(u))) for u in rc_df["馬番"].tolist()]
            tan_ret   = get_return(ret_df, "単勝")
            fuku_rets = {u: get_return(ret_df, "複勝", u) for u in umabans}

            race_records.append({
                "winner_set": winner_set, "top3_set": top3_set,
                "san_winner": san_winner, "san_odds": san_odds,
                "tan_ret": tan_ret, "fuku_rets": fuku_rets, "umabans": umabans,
                "s_v7":  np.array(s_v7,  dtype=float) if s_v7  is not None else None,
                "s_v15": np.array(s_v15, dtype=float),
            })
            processed += 1
            if processed % 200 == 0:
                print(f"  {processed}R収集済み ({time.time()-t0:.0f}秒)")
                with open(V15_CACHE, "wb") as _f:
                    pickle.dump(race_records, _f)

    print(f"\n収集完了: {processed}R ({time.time()-t0:.0f}秒), スキップ:{skipped}R")
    with open(V15_CACHE, "wb") as f:
        pickle.dump(race_records, f)
    print(f"キャッシュ保存: {V15_CACHE}")


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

evaluate("v7odds_binary_no26（ベースライン）",
         [r for r in race_records if r["s_v7"] is not None],
         lambda r: r["s_v7"])

evaluate("v15pace_no26（ペース適性 v9 LightGBM）",
         race_records,
         lambda r: r["s_v15"])

print(f"\n--- v15pace × v7odds ブレンド αスイープ ---")
for alpha in [round(a * 0.1, 1) for a in range(0, 11)]:
    records_blend = [r for r in race_records if r["s_v7"] is not None]
    evaluate(
        f"alpha={alpha:.1f} (v15:{1-alpha:.1f} v7:{alpha:.1f})",
        records_blend,
        lambda r, a=alpha: (1 - a) * _norm(r["s_v15"]) + a * _norm(r["s_v7"]),
        width=40,
    )
