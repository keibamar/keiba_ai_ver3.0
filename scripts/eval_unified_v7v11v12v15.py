"""eval_unified_v7v11v12v15.py

1 回のパスで v7 / v11feat / v12feat / v15pace の全スコアを同時計算し、
race_id 付きの統合キャッシュ（race_records_unified_2026.pkl）を生成する。

特徴量は計算コストが最大の v15（105列）を構築する過程で
v7/v11/v12 の必要部分もすべて取得できるため、追加コストは推論のみ。

キャッシュ構造:
  {
    "race_id":   str,
    "winner_set": set, "top3_set": set,
    "san_winner": set|None, "san_odds": int|None,
    "tan_ret": int|None, "fuku_rets": {umaban: int},
    "umabans":  [str],
    "s_v7":  np.ndarray,
    "s_v11": np.ndarray,
    "s_v12": np.ndarray,
    "s_v15": np.ndarray,
  }

実行:
    python scripts/eval_unified_v7v11v12v15.py
"""

import os, sys, glob, time, pickle, warnings, re
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

BET_UNIT    = 100
EVAL_YEAR   = 2026
UNIFIED_CACHE = os.path.join(PROJECT_ROOT, "logs", f"race_records_unified_{EVAL_YEAR}.pkl")

_FEAT_COLS_V15 = [c for c in index_v9 if c != "race_id"]  # 104列

print("血統vocab読み込み中...")
_VOCAB = build_pedigree_vocab()
print(f"  {len(_VOCAB)}種類")

_lgb_cache = {}


def _get_lgb(place_id, race_type, length, suffix):
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


def _safe_float(v):
    try: return float(v)
    except: return np.nan


def _parse_result(val):
    s = str(val)
    if any(c in s for c in ("除", "取", "中", "失")):
        return np.nan
    try:
        digits = re.sub(r"[^\d]", "", s)
        return float(digits) if digits else np.nan
    except:
        return np.nan


def _current_date(race_card_path):
    try:
        return pd.to_datetime(os.path.basename(os.path.dirname(race_card_path)), format="%Y%m%d")
    except:
        return pd.NaT


def build_features(race_id, rc_df, race_info_df, race_card_path):
    """全モデル共通の特徴量を一括構築。

    Returns (full_df_64, ped_cats, v6_df, v7_df, v8_df, feat_rows_v15,
             place_id, race_type, course_len, horse_ids, ok)

      full_df_64     : DataFrame (n_horses × 64)  [v3_base(60)+jockey(2)+waku(2)]
      ped_cats       : list of (f,mf,pgf)
      v6_df          : DataFrame (n_horses × 16)
      v7_df          : DataFrame (n_horses × 5)
      v8_df          : DataFrame (n_horses × 7)
      feat_rows_v15  : list of list (n_horses × 104)  ← v15直接入力用
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
        kinryo_col = rc_df.columns[4]
        n_horses   = float(len(rc_df))
        current_dt = _current_date(race_card_path)

        js = _build_jockey_course_stats(place_id)

        dataset_v3 = pd.DataFrame()
        extra_v6_rows, extra_v8_rows, extra_v9_rows = [], [], []
        ped_cats = []
        days_since_list, n_horses_1_list, hw_abs_1_list = [], [], []
        feat_rows_v15 = []

        for idx, hid in enumerate(horse_ids):
            # v3 base (60)
            try:
                row_v3 = make_dataset_for_lightgbm_v3(race_id, course_info, hid)
                v3_vals = row_v3.values[0].tolist()
            except:
                v3_vals = [np.nan] * 60
            dataset_v3 = pd.concat([dataset_v3.reset_index(drop=True),
                                    pd.DataFrame([v3_vals])])

            # pedigree cats
            f_cat, mf_cat, pgf_cat = get_pedigree_cats(hid, _VOCAB)
            ped_cats.append((f_cat, mf_cat, pgf_cat))

            # past 5 races
            try:
                r5 = past_performance.get_past_race_info(hid, race_id, race_num=5)
            except:
                r5 = pd.DataFrame()

            extra_v6 = get_extra_past_race_features_v6(r5) if not r5.empty else [np.nan] * 16
            extra_v8 = get_extra_past_race_features_v8(r5) if not r5.empty else [np.nan] * 7
            extra_v9 = get_extra_past_race_features_v9(r5) if not r5.empty else [np.nan] * 7
            extra_v6_rows.append(extra_v6)
            extra_v8_rows.append(extra_v8)
            extra_v9_rows.append(extra_v9)

            # v7 extras
            days_since = n_horses_1 = hw_abs_1 = np.nan
            if not r5.empty:
                if not pd.isna(current_dt):
                    try:
                        prev_dt = pd.to_datetime(
                            str(r5.iloc[0].get("日付", "")).strip().replace("/", "-"),
                            format="%Y-%m-%d"
                        )
                        days_since = float((current_dt - prev_dt).days)
                    except: pass
                n_horses_1 = _safe_float(r5.iloc[0].get("頭数", ""))
                hw_abs_1   = _parse_weight(r5.iloc[0].get("馬体重", ""))
            days_since_list.append(days_since)
            n_horses_1_list.append(n_horses_1)
            hw_abs_1_list.append(hw_abs_1)

            # v15 feat row (104列、race_id 除く)
            jid = str(jockey_ids[idx])
            wr, pr = js.get((jid, race_type, course_len), (np.nan, np.nan))
            kinryo  = _parse_kinryo(rc_df[kinryo_col].iloc[idx])
            waku    = _safe_float(waku_df.at[idx, "枠"])
            umaban  = _safe_float(waku_df.at[idx, "馬番"])

            row_v15 = (
                v3_vals[0:60]
                + [wr, pr]
                + [waku, umaban]
                + [np.nan, np.nan]              # current_odds / current_popularity (後で fill)
                + [f_cat, mf_cat, pgf_cat]
                + list(extra_v6)
                + [kinryo, days_since, n_horses, n_horses_1, hw_abs_1]
                + list(extra_v8[:5])
                + [extra_v8[5] if len(extra_v8) > 5 else np.nan,
                   extra_v8[6] if len(extra_v8) > 6 else np.nan]
                + list(extra_v9[:5])
                + [extra_v9[5] if len(extra_v9) > 5 else np.nan,
                   extra_v9[6] if len(extra_v9) > 6 else np.nan]
            )
            feat_rows_v15.append(row_v15)

        # DataFrameに変換
        dataset_v3 = dataset_v3.reset_index(drop=True)
        v6_df = pd.DataFrame(extra_v6_rows)
        v8_df = pd.DataFrame(extra_v8_rows, columns=[
            "corner_chase_1","corner_chase_2","corner_chase_3",
            "corner_chase_4","corner_chase_5","agari_trend_5","time_diff_trend_5",
        ])
        v7_df = pd.DataFrame({
            "kinryo":              [_parse_kinryo(rc_df[kinryo_col].iloc[i]) for i in range(len(horse_ids))],
            "days_since_last_race": days_since_list,
            "n_horses_today":       [n_horses] * len(horse_ids),
            "n_horses_1":           n_horses_1_list,
            "horse_weight_abs_1":   hw_abs_1_list,
        })

        wr_list = [js.get((str(j), race_type, course_len), (np.nan, np.nan))[0] for j in jockey_ids]
        pr_list = [js.get((str(j), race_type, course_len), (np.nan, np.nan))[1] for j in jockey_ids]

        full_df_64 = pd.concat([
            dataset_v3,
            pd.DataFrame({"jockey_win_rate": wr_list, "jockey_place_rate": pr_list}),
            waku_df,
        ], axis=1).fillna(-1)

        return (full_df_64, ped_cats, v6_df, v7_df, v8_df, feat_rows_v15,
                place_id, race_type, course_len, horse_ids, True)
    except Exception as e:
        prediction_error(e)
        return None, None, None, None, None, None, None, None, None, None, False


def score_v7(full_df_64, pid, rtype, clen, odds_arr, pop_arr):
    try:
        extra = pd.DataFrame({
            "current_odds":       np.where(np.isfinite(odds_arr), odds_arr, -1),
            "current_popularity": np.where(np.isfinite(pop_arr),  pop_arr,  -1),
        })
        df = pd.concat([full_df_64.reset_index(drop=True), extra], axis=1)
        m = _get_lgb(pid, rtype, clen, "_v7odds_binary_no26")
        return m.predict(df, num_iteration=m.best_iteration)
    except Exception as e:
        prediction_error(e); return None


def score_v11(full_df_64, ped_cats, v6_df, v7_df,
              pid, rtype, clen, odds_arr, pop_arr):
    try:
        f_ids, mf_ids, pgf_ids = zip(*ped_cats)
        extra = pd.DataFrame({
            "current_odds":       np.where(np.isfinite(odds_arr), odds_arr, -1),
            "current_popularity": np.where(np.isfinite(pop_arr),  pop_arr,  -1),
            "father_cat":         list(f_ids),
            "mother_father_cat":  list(mf_ids),
            "paternal_gf_cat":    list(pgf_ids),
        })
        df = pd.concat([
            full_df_64.reset_index(drop=True),
            extra,
            v6_df.fillna(-1).reset_index(drop=True),
            v7_df.fillna(-1).reset_index(drop=True),
        ], axis=1)
        m = _get_lgb(pid, rtype, clen, "_v11feat_no26")
        return m.predict(df, num_iteration=m.best_iteration)
    except Exception as e:
        prediction_error(e); return None


def score_v12(full_df_64, ped_cats, v6_df, v7_df, v8_df,
              pid, rtype, clen, odds_arr, pop_arr):
    try:
        f_ids, mf_ids, pgf_ids = zip(*ped_cats)
        extra = pd.DataFrame({
            "current_odds":       np.where(np.isfinite(odds_arr), odds_arr, -1),
            "current_popularity": np.where(np.isfinite(pop_arr),  pop_arr,  -1),
            "father_cat":         list(f_ids),
            "mother_father_cat":  list(mf_ids),
            "paternal_gf_cat":    list(pgf_ids),
        })
        df = pd.concat([
            full_df_64.reset_index(drop=True),
            extra,
            v6_df.fillna(-1).reset_index(drop=True),
            v7_df.fillna(-1).reset_index(drop=True),
            v8_df.fillna(-1).reset_index(drop=True),
        ], axis=1)
        m = _get_lgb(pid, rtype, clen, "_v12feat_no26")
        return m.predict(df, num_iteration=m.best_iteration)
    except Exception as e:
        prediction_error(e); return None


def score_v15(feat_rows, pid, rtype, clen, odds_arr, pop_arr):
    try:
        odds_idx = _FEAT_COLS_V15.index("current_odds")
        pop_idx  = _FEAT_COLS_V15.index("current_popularity")
        rows = []
        for i, row in enumerate(feat_rows):
            r = list(row)
            r[odds_idx] = float(odds_arr[i]) if np.isfinite(odds_arr[i]) else -1
            r[pop_idx]  = float(pop_arr[i])  if np.isfinite(pop_arr[i])  else -1
            rows.append(r)
        df = pd.DataFrame(rows, columns=_FEAT_COLS_V15).fillna(-1)
        m  = _get_lgb(pid, rtype, clen, "_v15pace_no26")
        return m.predict(df, num_iteration=m.best_iteration)
    except Exception as e:
        prediction_error(e); return None


def get_return(ret_df, shikibetsu, umaban_str=None):
    rows = ret_df[ret_df["式別"] == shikibetsu]
    if rows.empty: return None
    if umaban_str is not None:
        rows = rows[rows["馬番"].astype(str) == umaban_str]
        if rows.empty: return None
    return int(rows["配当"].iloc[0])


# ── キャッシュ確認 ──
if os.path.exists(UNIFIED_CACHE):
    print(f"\n統合キャッシュ読み込み: {UNIFIED_CACHE}")
    with open(UNIFIED_CACHE, "rb") as f:
        race_records = pickle.load(f)
    print(f"  {len(race_records)}R (s_keys: {[k for k in race_records[0] if k.startswith('s_')]})")
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
            odds_arr   = final_odds.astype(float).values
            pop_arr    = final_pop.astype(float).values

            umabans    = [str(int(float(u))) for u in rc_df["馬番"].tolist()]
            tan_ret    = get_return(ret_df, "単勝")
            fuku_rets  = {u: get_return(ret_df, "複勝", u) for u in umabans}

            result = build_features(race_id, rc_df, race_info_df, cp)
            (full_df_64, ped_cats, v6_df, v7_df, v8_df, feat_rows_v15,
             pid, rtype, clen, horse_ids, ok) = result
            if not ok: skipped += 1; continue

            s_v7  = score_v7(full_df_64, pid, rtype, clen, odds_arr, pop_arr)
            s_v11 = score_v11(full_df_64, ped_cats, v6_df, v7_df, pid, rtype, clen, odds_arr, pop_arr)
            s_v12 = score_v12(full_df_64, ped_cats, v6_df, v7_df, v8_df, pid, rtype, clen, odds_arr, pop_arr)
            s_v15 = score_v15(feat_rows_v15, pid, rtype, clen, odds_arr, pop_arr)

            if s_v7 is None: skipped += 1; continue  # v7 は必須

            race_records.append({
                "race_id":    race_id,
                "winner_set": winner_set,
                "top3_set":   top3_set,
                "san_winner": san_winner,
                "san_odds":   san_odds,
                "tan_ret":    tan_ret,
                "fuku_rets":  fuku_rets,
                "umabans":    umabans,
                "s_v7":  np.array(s_v7,  dtype=float),
                "s_v11": np.array(s_v11, dtype=float) if s_v11 is not None else None,
                "s_v12": np.array(s_v12, dtype=float) if s_v12 is not None else None,
                "s_v15": np.array(s_v15, dtype=float) if s_v15 is not None else None,
            })
            processed += 1
            if processed % 200 == 0:
                print(f"  {processed}R収集済み ({time.time()-t0:.0f}秒)")
                with open(UNIFIED_CACHE, "wb") as _f:
                    pickle.dump(race_records, _f)

    print(f"\n収集完了: {processed}R ({time.time()-t0:.0f}秒), スキップ:{skipped}R")
    with open(UNIFIED_CACHE, "wb") as f:
        pickle.dump(race_records, f)
    print(f"統合キャッシュ保存: {UNIFIED_CACHE}")

print(f"\n合計: {len(race_records)}R")
print("完了 — eval_combo_multimodel.py に unified キャッシュを使うバージョンを次に実行してください")
