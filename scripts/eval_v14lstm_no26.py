"""eval_v14lstm_no26.py

v14lstm_no26（Sequence LSTM）を v7odds と比較評価。

実行: python scripts/eval_v14lstm_no26.py
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
import torch

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
from src.PredictionModels.LightGBM.make_dataset_v4 import _parse_popularity
from src.PredictionModels.LightGBM.make_dataset_v8 import (
    get_extra_past_race_features_v8, index_v8, _parse_corner_chase,
)
from src.PredictionModels.LSTM.sequence_lstm import SequenceLSTM, SEQ_LEN, SEQ_FEATURES

import past_performance
import lightgbm as lgb

BET_UNIT  = 100
EVAL_YEAR = 2026
DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")

V14_CACHE = os.path.join(PROJECT_ROOT, "logs", f"race_records_v14eval_{EVAL_YEAR}.pkl")

EMBED_COLS = ["father_cat", "mother_father_cat", "paternal_gf_cat"]

# 時系列列名（5走前→1走前: oldest first）
SEQ_COLS_FLAT = []
for _i in range(5, 0, -1):
    SEQ_COLS_FLAT += [
        f"time_df_course{_i}", f"time_df_class{_i}",
        f"ninki_{_i}", f"result_{_i}", f"agari_{_i}", f"margin_{_i}", f"corner_ratio_{_i}",
        f"corner_chase_{_i}",
    ]

# 現在レース特徴列（race_id・embed・sequence を除く、54列）
CURRENT_COLS = [c for c in index_v8 if c != "race_id" and c not in EMBED_COLS and c not in SEQ_COLS_FLAT]
NUM_CURRENT  = len(CURRENT_COLS)
assert NUM_CURRENT == 54, f"NUM_CURRENT={NUM_CURRENT}"

print("血統vocab読み込み中...")
_VOCAB = build_pedigree_vocab()
_VOCAB_SIZE = len(_VOCAB)
print(f"  {_VOCAB_SIZE}種類")

_lstm_cache = {}
_lgb_cache  = {}


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


def _parse_kinryo(val):
    try: return float(str(val).strip())
    except: return np.nan


def _safe_time_info(row):
    """_get_time_info の結果を安全に取得（[time_df_course, time_df_class] の順）。"""
    try:
        td = _get_time_info(row)
        v0 = float(td[0]) if td and len(td) > 0 else np.nan
        v1 = float(td[1]) if td and len(td) > 1 else np.nan
        return (v0 if not np.isnan(v0) else np.nan,
                v1 if not np.isnan(v1) else np.nan)
    except:
        return np.nan, np.nan


def _build_seq(race_info_5):
    """過去5走の時系列 (5, 8) を返す（5走前=oldest が先頭）。"""
    rows = []
    for i in range(SEQ_LEN - 1, -1, -1):  # 4, 3, 2, 1, 0 (oldest first)
        if i < len(race_info_5):
            row = race_info_5.iloc[i]
            tdc, tdcl = _safe_time_info(row)
            vals = [
                tdc,
                tdcl,
                _safe_float(row.get("人気", "")),
                _parse_result(row.get("着順", "")),
                _parse_agari(row.get("上り", "")),
                _parse_margin(row.get("着差", "")),
                _parse_corner_ratio(row.get("通過", "")),
                _parse_corner_chase(
                    row.get("通過", ""), row.get("頭数", row.get("頭 数", ""))
                ),
            ]
            rows.append(vals)
        else:
            rows.append([np.nan] * SEQ_FEATURES)
    return np.array(rows, dtype=np.float32)  # (5, 8)


def _get_lstm_model(place_id, race_type, length):
    key = (place_id, race_type, length)
    if key in _lstm_cache:
        return _lstm_cache[key]
    type_str = "turf" if race_type == "芝" else "dirt"
    mp = os.path.join(paths.PREDICTION_MODEL_PATH, PLACE_LIST[place_id - 1],
                      f"{type_str}{length}_lambdarank_model_v14lstm_no26.pt")
    if not os.path.isfile(mp):
        raise FileNotFoundError(mp)
    ckpt = torch.load(mp, map_location=DEVICE)
    model = SequenceLSTM(
        vocab_size=ckpt["vocab_size"],
        seq_features=ckpt.get("seq_features", SEQ_FEATURES),
        num_current=ckpt["num_current"],
        embed_dim=ckpt["params"].get("embed_dim", 16),
        lstm_hidden=ckpt["params"].get("lstm_hidden", 64),
        lstm_layers=ckpt["params"].get("lstm_layers", 1),
        hidden=ckpt["params"].get("hidden", [128, 64]),
        dropout=ckpt["params"].get("dropout", 0.2),
    ).to(DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    _lstm_cache[key] = model
    return model


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


def build_v14_row(race_id, rc_df, race_info_df, race_card_path):
    """v14用: シーケンス・現在特徴・血統IDを馬ごとに構築する。

    Returns:
        x_cur_rows : list of 54-element lists (per horse, current features)
        seq_rows   : list of (5, 8) arrays (per horse, sequence)
        horse_ids  : list
        ok         : bool
        (+ v7 scoring inputs)
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

        x_cur_rows = []
        seq_rows   = []
        father_ids_list, mf_ids_list, pgf_ids_list = [], [], []
        v3_rows = []  # 66列用（v7スコアリング向け）

        for idx, hid in enumerate(horse_ids):
            # v3 60列（blood 36 + past3+extras 24）
            try:
                row_v3 = make_dataset_for_lightgbm_v3(race_id, course_info, hid)
                v3_vals = row_v3.values[0].tolist() if not row_v3.empty else [np.nan] * 60
            except:
                v3_vals = [np.nan] * 60

            # 血統 ID
            f_cat, mf_cat, pgf_cat = get_pedigree_cats(hid, _VOCAB)
            father_ids_list.append(f_cat)
            mf_ids_list.append(mf_cat)
            pgf_ids_list.append(pgf_cat)

            # 過去5走データ
            try:
                race_info_5 = past_performance.get_past_race_info(hid, race_id, race_num=5)
            except:
                race_info_5 = pd.DataFrame()

            # シーケンス (5, 8)
            seq_rows.append(_build_seq(race_info_5))

            # v6 extra（rank_trend_5 / win_rate_recent5 など 16列）
            try:
                extra_v6 = get_extra_past_race_features_v6(race_info_5)
            except:
                extra_v6 = [np.nan] * 16

            # v8 extra（agari_trend_5 / time_diff_trend_5 など 7列）
            try:
                extra_v8 = get_extra_past_race_features_v8(race_info_5)
            except:
                extra_v8 = [np.nan] * 7

            # 騎手成績
            jid = str(jockey_ids[idx])
            wr, pr = js.get((jid, race_type, course_len), (np.nan, np.nan))

            # 斤量
            kinryo = _parse_kinryo(rc_df[kinryo_col].iloc[idx])

            # 前走からの日数・前走頭数・前走馬体重
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

            waku    = _safe_float(waku_df.at[idx, "枠"])
            umaban  = _safe_float(waku_df.at[idx, "馬番"])

            # current feature vector: CURRENT_COLS 順
            # blood(36) + rank_trend,weight_change,dist_change(3) + jockey(2)
            # + waku,umaban(2) + odds(2: filled later) + rank_trend_5,win_rate_recent5(2)
            # + kinryo,days_since,n_horses_today,n_horses_1,hw_abs_1(5)
            # + agari_trend_5,time_diff_trend_5(2)
            # = 54  (current_oddsとcurrent_popularityは後で fill)
            x_cur_rows.append({
                "blood":           v3_vals[0:36],
                "rank_trend":      v3_vals[57] if len(v3_vals) > 57 else np.nan,
                "weight_change":   v3_vals[58] if len(v3_vals) > 58 else np.nan,
                "dist_change":     v3_vals[59] if len(v3_vals) > 59 else np.nan,
                "jockey_win_rate": wr,
                "jockey_place_rate": pr,
                "waku":            waku,
                "umaban":          umaban,
                # current_odds / current_popularity → filled outside
                "extra_v6_14":     extra_v6[14] if len(extra_v6) > 14 else np.nan,  # rank_trend_5
                "extra_v6_15":     extra_v6[15] if len(extra_v6) > 15 else np.nan,  # win_rate_recent5
                "kinryo":          kinryo,
                "days_since":      days_since,
                "n_horses_today":  n_horses_today,
                "n_horses_1":      n_horses_1,
                "hw_abs_1":        hw_abs_1,
                "agari_trend_5":   extra_v8[5] if len(extra_v8) > 5 else np.nan,
                "time_diff_trend_5": extra_v8[6] if len(extra_v8) > 6 else np.nan,
            })

            # v7 scoring: 66列 (v3_60 + jockey2 + waku2 + odds2 added outside)
            v3_rows.append(v3_vals)

        return (x_cur_rows, seq_rows, father_ids_list, mf_ids_list, pgf_ids_list,
                v3_rows, place_id, race_type, course_len, horse_ids, True)
    except Exception as e:
        prediction_error(e)
        return None, None, None, None, None, None, None, None, None, None, False


def score_v14lstm(x_cur_rows, seq_rows, f_ids_list, mf_ids_list, pgf_ids_list,
                  final_odds, final_pop, place_id, race_type, course_len):
    try:
        odds_arr = np.where(np.isfinite(final_odds.astype(float).values), final_odds.astype(float).values, -1)
        pop_arr  = np.where(np.isfinite(final_pop.astype(float).values),  final_pop.astype(float).values,  -1)

        x_cur_list = []
        for i, d in enumerate(x_cur_rows):
            row = (
                d["blood"]
                + [d["rank_trend"], d["weight_change"], d["dist_change"]]
                + [d["jockey_win_rate"], d["jockey_place_rate"]]
                + [d["waku"], d["umaban"]]
                + [float(odds_arr[i]), float(pop_arr[i])]
                + [d["extra_v6_14"], d["extra_v6_15"]]
                + [d["kinryo"], d["days_since"], d["n_horses_today"], d["n_horses_1"], d["hw_abs_1"]]
                + [d["agari_trend_5"], d["time_diff_trend_5"]]
            )
            x_cur_list.append(row)

        x_cur_arr  = np.array(x_cur_list, dtype=np.float32)
        np.nan_to_num(x_cur_arr, nan=-1, copy=False)
        seq_arr = np.stack(seq_rows, axis=0)  # (n_horses, 5, 8)
        np.nan_to_num(seq_arr, nan=-1, copy=False)

        def _safe_id(arr):
            arr = np.array(arr, dtype=np.int64)
            arr = np.where(arr < 0, 0, arr)
            arr = np.where(arr > _VOCAB_SIZE, _VOCAB_SIZE, arr)
            return torch.tensor(arr, dtype=torch.long).to(DEVICE)

        seq_t  = torch.tensor(seq_arr, dtype=torch.float32).to(DEVICE)
        x_cur_t = torch.tensor(x_cur_arr, dtype=torch.float32).to(DEVICE)
        f_t    = _safe_id(f_ids_list)
        mf_t   = _safe_id(mf_ids_list)
        pgf_t  = _safe_id(pgf_ids_list)

        model = _get_lstm_model(place_id, race_type, course_len)
        with torch.no_grad():
            logits = model(seq_t, f_t, mf_t, pgf_t, x_cur_t)
        return logits.cpu().numpy()
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

if os.path.exists(V14_CACHE):
    print(f"\nv14evalキャッシュ読み込み: {V14_CACHE}")
    with open(V14_CACHE, "rb") as f:
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
            pop_map    = dict(zip(res_df["馬番"].astype(str), pd.to_numeric(res_df["人気"], errors="coerce")))
            final_odds = key_series.map(odds_map)
            final_pop  = key_series.map(pop_map)

            waku_df_local = rc_df[["枠", "馬番"]].reset_index(drop=True)
            js = _build_jockey_course_stats(int(str(race_id)[4:6]))

            result = build_v14_row(race_id, rc_df, race_info_df, cp)
            (x_cur_rows, seq_rows, f_ids_list, mf_ids_list, pgf_ids_list,
             v3_rows, pid, rtype, clen, hids, ok) = result
            if not ok: skipped += 1; continue

            s_v7 = score_v7(v3_rows, pid, rtype, clen, final_odds, final_pop,
                            js, rc_df, waku_df_local)
            s_v14 = score_v14lstm(x_cur_rows, seq_rows, f_ids_list, mf_ids_list, pgf_ids_list,
                                  final_odds, final_pop, pid, rtype, clen)
            if s_v14 is None: skipped += 1; continue

            umabans   = [str(int(float(u))) for u in rc_df["馬番"].tolist()]
            tan_ret   = get_return(ret_df, "単勝")
            fuku_rets = {u: get_return(ret_df, "複勝", u) for u in umabans}

            race_records.append({
                "winner_set": winner_set, "top3_set": top3_set,
                "san_winner": san_winner, "san_odds": san_odds,
                "tan_ret": tan_ret, "fuku_rets": fuku_rets, "umabans": umabans,
                "s_v7":  np.array(s_v7,  dtype=float) if s_v7  is not None else None,
                "s_v14": np.array(s_v14, dtype=float),
            })
            processed += 1
            if processed % 200 == 0:
                print(f"  {processed}R収集済み ({time.time()-t0:.0f}秒)")
                with open(V14_CACHE, "wb") as _f:
                    pickle.dump(race_records, _f)

    print(f"\n収集完了: {processed}R ({time.time()-t0:.0f}秒), スキップ:{skipped}R")
    with open(V14_CACHE, "wb") as f:
        pickle.dump(race_records, f)
    print(f"キャッシュ保存: {V14_CACHE}")


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

evaluate("v14lstm_no26（Sequence LSTM）",
         race_records,
         lambda r: r["s_v14"])

print(f"\n--- v14lstm × v7odds ブレンド αスイープ ---")
for alpha in [round(a * 0.1, 1) for a in range(0, 11)]:
    recs = [r for r in race_records if r["s_v7"] is not None]
    evaluate(f"v14×v7 α={alpha:.1f}", recs,
             lambda r, a=alpha: a * _norm(r["s_v14"]) + (1 - a) * _norm(r["s_v7"]))

print(SEP)
