"""eval_v12feat_no26.py

v12feat_no26（98列: v7 + コーナー追走効率×5 + 上がりトレンド + タイム差トレンド）を
既存モデルと比較評価。

v11eval キャッシュ (race_records_v11eval_2026.pkl) を流用し、
v12feat スコアを追加した新キャッシュを作成する。

実行:
    python scripts/eval_v12feat_no26.py
"""
import sys, warnings, glob, os, time, pickle, itertools
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
from src.PredictionModels.LightGBM.make_dataset_v7 import get_v7_extra_features
from src.PredictionModels.LightGBM.make_dataset_v8 import get_extra_past_race_features_v8

import past_performance

BET_UNIT  = 100
EVAL_YEAR = 2026

V11_CACHE = os.path.join(PROJECT_ROOT, "logs", f"race_records_v11eval_{EVAL_YEAR}.pkl")
V12_CACHE = os.path.join(PROJECT_ROOT, "logs", f"race_records_v12eval_{EVAL_YEAR}.pkl")

print("血統vocab読み込み中...")
_VOCAB = build_pedigree_vocab()
print(f"  {len(_VOCAB)}種類")


def _get_model(place_id, race_type, length, suffix):
    type_str = "turf" if race_type == "芝" else "dirt"
    mp = os.path.join(paths.PREDICTION_MODEL_PATH, PLACE_LIST[place_id - 1],
                      f"{type_str}{length}_lambdarank_model{suffix}.txt")
    if not os.path.isfile(mp):
        raise FileNotFoundError(mp)
    return lgb.Booster(model_file=mp)


def _parse_kinryo(val):
    try:
        return float(str(val).strip())
    except Exception:
        return np.nan


def _get_current_date_from_path(race_card_path):
    try:
        dir_name = os.path.basename(os.path.dirname(race_card_path))
        return pd.to_datetime(dir_name, format="%Y%m%d")
    except Exception:
        return pd.NaT


def build_v8_features(race_id, rc_df, race_info_df, race_card_path):
    """v8特徴量（97列）を構築する。

    Returns:
        full_df_64   : v3_base(60) + jockey(2) + waku/umaban(2) = 64列
        v6_extra_df  : 16列
        v7_extra_df  : 5列
        v8_extra_df  : 7列（corner_chase×5 / agari_trend / time_diff_trend）
        place_id, race_type, course_len, ok
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
        kinryo_list    = [_parse_kinryo(v) for v in rc_df[kinryo_col].tolist()]
        n_horses_today = float(len(rc_df))
        current_dt     = _get_current_date_from_path(race_card_path)

        dataset_v3 = pd.DataFrame()
        extra_v6_rows = []
        days_since_list, n_horses_1_list, hw_abs_1_list = [], [], []
        extra_v8_rows = []

        for hid in horse_ids:
            try:
                row_v3 = make_dataset_for_lightgbm_v3(race_id, course_info, hid)
            except Exception:
                row_v3 = pd.DataFrame([[np.nan] * 60])
            dataset_v3 = pd.concat([dataset_v3.reset_index(drop=True), row_v3.reset_index(drop=True)])

            try:
                race_info_5 = past_performance.get_past_race_info(hid, race_id, race_num=5)
                extra_v6 = get_extra_past_race_features_v6(race_info_5)

                days_since = np.nan
                n_horses_1 = np.nan
                hw_abs_1   = np.nan
                if not race_info_5.empty:
                    if not pd.isna(current_dt):
                        prev_date_str = race_info_5.iloc[0].get("日付", "")
                        try:
                            prev_dt = pd.to_datetime(
                                str(prev_date_str).strip().replace("/", "-"),
                                format="%Y-%m-%d"
                            )
                            days_since = float((current_dt - prev_dt).days)
                        except Exception:
                            pass
                    n_horses_1 = _parse_kinryo(race_info_5.iloc[0].get("頭数", ""))
                    hw_abs_1   = _parse_weight(race_info_5.iloc[0].get("馬体重", ""))

                extra_v8 = get_extra_past_race_features_v8(race_info_5)
            except Exception:
                extra_v6   = [np.nan] * 16
                days_since = n_horses_1 = hw_abs_1 = np.nan
                extra_v8   = [np.nan] * 7

            extra_v6_rows.append(extra_v6)
            days_since_list.append(days_since)
            n_horses_1_list.append(n_horses_1)
            hw_abs_1_list.append(hw_abs_1)
            extra_v8_rows.append(extra_v8)

        dataset_v6_extra = pd.DataFrame(extra_v6_rows)
        dataset_v8_extra = pd.DataFrame(extra_v8_rows, columns=[
            "corner_chase_1", "corner_chase_2", "corner_chase_3",
            "corner_chase_4", "corner_chase_5",
            "agari_trend_5", "time_diff_trend_5",
        ])

        js = _build_jockey_course_stats(place_id)
        wr_list, pr_list = [], []
        for jid in jockey_ids:
            wr, pr = js.get((str(jid), race_type, course_len), (np.nan, np.nan))
            wr_list.append(wr); pr_list.append(pr)

        full_df_64 = pd.concat([
            dataset_v3.reset_index(drop=True),
            pd.DataFrame({"jockey_win_rate": wr_list, "jockey_place_rate": pr_list}),
            waku_df.reset_index(drop=True),
        ], axis=1).fillna(-1)

        v7_extra_df = pd.DataFrame({
            "kinryo":               kinryo_list,
            "days_since_last_race": days_since_list,
            "n_horses_today":       [n_horses_today] * len(horse_ids),
            "n_horses_1":           n_horses_1_list,
            "horse_weight_abs_1":   hw_abs_1_list,
        })

        return full_df_64, dataset_v6_extra, v7_extra_df, dataset_v8_extra, \
               place_id, race_type, course_len, True
    except Exception as e:
        prediction_error(e)
        return None, None, None, None, None, None, None, False


def score_v12feat(full_df_64, v6_extra_df, v7_extra_df, v8_extra_df,
                  place_id, race_type, course_len, final_odds, final_pop, horse_ids):
    """v12feat_no26: v8特徴量(97列)でスコアリング。
    列順: v3_base(60)+jockey(2)+waku/umaban(2)+odds(1)+pop(1)+ped_cat(3)+v6(16)+v7(5)+v8(7)
    """
    try:
        odds_arr = final_odds.astype(float).values
        pop_arr  = final_pop.astype(float).values
        f_ids, mf_ids, pgf_ids = [], [], []
        for hid in horse_ids:
            f, mf, pgf = get_pedigree_cats(hid, _VOCAB)
            f_ids.append(f); mf_ids.append(mf); pgf_ids.append(pgf)

        df_full = pd.concat([
            full_df_64.reset_index(drop=True),
            pd.DataFrame({
                "current_odds":       np.where(np.isfinite(odds_arr), odds_arr, -1),
                "current_popularity": np.where(np.isfinite(pop_arr),  pop_arr,  -1),
                "father_cat":         f_ids,
                "mother_father_cat":  mf_ids,
                "paternal_gf_cat":    pgf_ids,
            }),
            v6_extra_df.fillna(-1).reset_index(drop=True),
            v7_extra_df.fillna(-1).reset_index(drop=True),
            v8_extra_df.fillna(-1).reset_index(drop=True),
        ], axis=1)

        model = _get_model(place_id, race_type, course_len, "_v12feat_no26")
        return model.predict(df_full, num_iteration=model.best_iteration)
    except Exception as e:
        prediction_error(e)
        return None


def score_v4model(full_df_64, place_id, race_type, course_len, final_odds, final_pop, suffix):
    try:
        odds_arr = final_odds.astype(float).values
        pop_arr  = final_pop.astype(float).values
        extra = pd.DataFrame({
            "current_odds":       np.where(np.isfinite(odds_arr), odds_arr, -1),
            "current_popularity": np.where(np.isfinite(pop_arr),  pop_arr,  -1),
        })
        df66 = pd.concat([full_df_64.reset_index(drop=True), extra.reset_index(drop=True)], axis=1)
        model = _get_model(place_id, race_type, course_len, suffix)
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

if os.path.exists(V12_CACHE):
    print(f"\nv12evalキャッシュ読み込み: {V12_CACHE}")
    with open(V12_CACHE, "rb") as f:
        race_records = pickle.load(f)
    print(f"  {len(race_records)}R")
elif os.path.exists(V11_CACHE):
    print(f"\nv11evalキャッシュ読み込み: {V11_CACHE}")
    with open(V11_CACHE, "rb") as f:
        v11_records = pickle.load(f)
    print(f"  {len(v11_records)}R → v12featスコアを追加します")

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
            ret_df  = pd.read_csv(ret_paths[0], index_col=0)
            rc_df   = pd.read_csv(cp, index_col=0).reset_index(drop=True)
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

            full_df_64, v6_extra_df, v7_extra_df, v8_extra_df, pid, rtype, clen, ok = \
                build_v8_features(race_id, rc_df, race_info_df, cp)
            if not ok: skipped += 1; continue

            s_v7  = score_v4model(full_df_64, pid, rtype, clen, final_odds, final_pop, "_v7odds_binary_no26")
            s_v11 = None
            try:
                # v11feat スコアは v8_extra_df なしで再構築
                from src.PredictionModels.LightGBM.make_dataset_v5 import get_pedigree_cats as gpc
                from src.PredictionModels.LightGBM.make_dataset_v8 import _get_model as _gm
                s_v11 = score_v4model(full_df_64, pid, rtype, clen, final_odds, final_pop, "_v11feat_no26") if False else None
            except Exception:
                pass

            s_v12 = score_v12feat(full_df_64, v6_extra_df, v7_extra_df, v8_extra_df,
                                  pid, rtype, clen, final_odds, final_pop,
                                  rc_df["horse_id"].tolist())
            if s_v12 is None: skipped += 1; continue

            umabans   = [str(int(float(u))) for u in rc_df["馬番"].tolist()]
            tan_ret   = get_return(ret_df, "単勝")
            fuku_rets = {u: get_return(ret_df, "複勝", u) for u in umabans}

            race_records.append({
                "winner_set": winner_set, "top3_set": top3_set,
                "san_winner": san_winner, "san_odds": san_odds,
                "tan_ret": tan_ret, "fuku_rets": fuku_rets, "umabans": umabans,
                "s_v7":  np.array(s_v7,  dtype=float) if s_v7  is not None else None,
                "s_v12": np.array(s_v12, dtype=float),
            })
            processed += 1
            if processed % 200 == 0:
                print(f"  {processed}R収集済み ({time.time()-t0:.0f}秒)")
                with open(V12_CACHE, "wb") as _f:
                    pickle.dump(race_records, _f)

    print(f"\n収集完了: {processed}R ({time.time()-t0:.0f}秒), スキップ:{skipped}R")
    with open(V12_CACHE, "wb") as f:
        pickle.dump(race_records, f)
    print(f"キャッシュ保存: {V12_CACHE}")

else:
    print("ERROR: v11eval キャッシュが見つかりません。先に eval_v11feat_no26.py を実行してください。")
    sys.exit(1)


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


def evaluate(label, records, score_fn, width=58):
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


W = 58
SEP = "=" * 126
print(f"\n{SEP}")
print(f"  {'モデル':<{W}}  {'件数':>6}  {'単勝的中/回収':>15}  {'複勝的中/回収':>15}  {'3連複的中/回収':>16}")
print(SEP)

evaluate("v7odds_binary_no26（ベースライン）",
         [r for r in race_records if r["s_v7"] is not None],
         lambda r: r["s_v7"])

evaluate("v12feat_no26（+コーナー追走効率/上がりトレンド/タイムトレンド）",
         race_records,
         lambda r: r["s_v12"])

print(f"\n--- v12feat × v7odds ブレンド αスイープ ---")
for alpha in [round(a * 0.1, 1) for a in range(0, 11)]:
    recs = [r for r in race_records if r["s_v7"] is not None]
    evaluate(f"v12×v7 α={alpha:.1f}", recs,
             lambda r, a=alpha: a * _norm(r["s_v12"]) + (1 - a) * _norm(r["s_v7"]))

print(SEP)
