"""eval_v7odds_binary_no26.py

v7odds_binary_no26（オッズ重み付き二値分類）の2026年評価スクリプト。
no26ブレンドモデル（α=0.3）と並べて比較する。

評価指標: 単勝・複勝・3連複 の 的中率 / 回収率
収集データはキャッシュに保存し、2回目以降は即座に開始する。

実行:
    python scripts/eval_v7odds_binary_no26.py
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
from src.PredictionModels.LightGBM.make_dataset_v5 import (
    build_pedigree_vocab, get_pedigree_cats,
)

BET_UNIT   = 100
EVAL_YEAR  = 2026
ALPHA_NO26 = 0.3  # no26ブレンドの最良α
CACHE_PATH = os.path.join(PROJECT_ROOT, "logs", f"race_records_v7eval_{EVAL_YEAR}.pkl")

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


def build_v3_features(race_id, rc_df, race_info_df):
    try:
        place_id   = int(str(race_id)[4:6])
        race_type  = race_info_df.at[0, "race_type"]
        course_len = str(race_info_df.at[0, "course_len"])
        course_info = [place_id, race_type, course_len,
                       race_info_df.at[0, "ground_state"], race_info_df.at[0, "class"]]
        horse_ids  = rc_df["horse_id"].tolist()
        jockey_ids = rc_df["jockey_id"].tolist()
        waku_df    = rc_df[["枠", "馬番"]].reset_index(drop=True)
        dataset = pd.DataFrame()
        for hid in horse_ids:
            row = make_dataset_for_lightgbm_v3(race_id, course_info, hid)
            dataset = pd.concat([dataset.reset_index(drop=True), row.reset_index(drop=True)])
        js = _build_jockey_course_stats(place_id)
        wr_list, pr_list = [], []
        for jid in jockey_ids:
            wr, pr = js.get((str(jid), race_type, course_len), (np.nan, np.nan))
            wr_list.append(wr); pr_list.append(pr)
        full_df_64 = pd.concat([
            dataset.reset_index(drop=True),
            pd.DataFrame({"jockey_win_rate": wr_list, "jockey_place_rate": pr_list}),
            waku_df.reset_index(drop=True),
        ], axis=1).fillna(-1)
        return full_df_64, place_id, race_type, course_len, True
    except Exception as e:
        prediction_error(e)
        return None, None, None, None, False


def score_v5prime(full_df_64, place_id, race_type, course_len, horse_ids):
    try:
        f_ids, mf_ids, pgf_ids = [], [], []
        for hid in horse_ids:
            f, mf, pgf = get_pedigree_cats(hid, _VOCAB)
            f_ids.append(f); mf_ids.append(mf); pgf_ids.append(pgf)
        extra = pd.DataFrame({
            "father_cat": f_ids, "mother_father_cat": mf_ids, "paternal_gf_cat": pgf_ids,
        })
        df67 = pd.concat([full_df_64.reset_index(drop=True), extra.reset_index(drop=True)], axis=1)
        model = _get_model(place_id, race_type, course_len, "_v5prime_no26")
        return model.predict(df67, num_iteration=model.best_iteration)
    except Exception as e:
        prediction_error(e)
        return None


def score_v4model(full_df_64, place_id, race_type, course_len, final_odds, final_pop, suffix):
    """v4フィーチャー（current_odds, current_popularity を追加）でスコアリング。
    v6hit_extreme_no26 / v7odds_binary_no26 どちらにも使用。"""
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


# ---------- データ収集（キャッシュ機能付き）----------

if os.path.exists(CACHE_PATH):
    print(f"\nキャッシュ読み込み: {CACHE_PATH}")
    with open(CACHE_PATH, "rb") as f:
        race_records = pickle.load(f)
    print(f"  {len(race_records)}R 読み込み完了")
else:
    active_places = []
    for pid in range(1, 11):
        pfx = f"{EVAL_YEAR}{pid:02d}"
        cnt = sum(len(glob.glob(f"{d}/{pfx}*.csv")) for d in glob.glob(f"data/race_card/{EVAL_YEAR}*"))
        if cnt > 0:
            active_places.append(pid)

    print(f"\n評価年: {EVAL_YEAR}")
    print(f"対象場: {[PLACE_LIST[p-1] for p in active_places]}")
    print("レースデータ収集中（初回のみ時間がかかります）...")

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
            rc_df = pd.read_csv(cp, index_col=0).reset_index(drop=True)
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

            full_df_64, pid, rtype, clen, ok = build_v3_features(race_id, rc_df, race_info_df)
            if not ok: skipped += 1; continue

            # no26ブレンド用スコア
            s_v5p = score_v5prime(full_df_64, pid, rtype, clen, rc_df["horse_id"].tolist())
            s_ext = score_v4model(full_df_64, pid, rtype, clen, final_odds, final_pop, "_v6hit_extreme_no26")
            if s_v5p is None or s_ext is None: skipped += 1; continue

            # v7odds スコア（binary分類の予測確率）
            s_v7 = score_v4model(full_df_64, pid, rtype, clen, final_odds, final_pop, "_v7odds_binary_no26")

            umabans   = [str(int(float(u))) for u in rc_df["馬番"].tolist()]
            tan_ret   = get_return(ret_df, "単勝")
            fuku_rets = {u: get_return(ret_df, "複勝", u) for u in umabans}

            race_records.append({
                "winner_set": winner_set, "top3_set": top3_set,
                "san_winner": san_winner, "san_odds": san_odds,
                "tan_ret": tan_ret, "fuku_rets": fuku_rets, "umabans": umabans,
                "s_v5p": np.array(s_v5p, dtype=float),
                "s_ext": np.array(s_ext, dtype=float),
                "s_v7":  np.array(s_v7,  dtype=float) if s_v7 is not None else None,
            })
            processed += 1
            if processed % 200 == 0:
                print(f"  {processed}R収集済み ({time.time()-t0:.0f}秒)")

    print(f"\n収集完了: {processed}R ({time.time()-t0:.0f}秒), スキップ:{skipped}R")
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(race_records, f)
    print(f"キャッシュ保存: {CACHE_PATH}")


# ---------- 評価 ----------

class Stats:
    def __init__(self):
        self.n = self.tan_hit = self.tan_pay = self.tan_bet = 0
        self.fuku_hit = self.fuku_pay = self.fuku_bet = 0
        self.san_hit = self.san_pay = self.san_bet = 0

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


def evaluate(label, records, score_fn):
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
    print(f"  {label:<42}  "
          f"n={st.n:>4}  "
          f"単勝 {st.tan_pct():>5.1f}%/{st.tan_rec():>6.1f}%  "
          f"複勝 {st.fuku_pct():>5.1f}%/{st.fuku_rec():>6.1f}%  "
          f"3連複 {st.san_pct():>5.1f}%/{st.san_rec():>6.1f}%")


print(f"\n{'='*115}")
print(f"  {'モデル':<42}  {'件数':>6}  {'単勝的中率/回収率':>18}  {'複勝的中率/回収率':>18}  {'3連複的中率/回収率':>19}")
print(f"{'='*115}")

# no26ブレンド（α=0.3 固定）
no26_records = [r for r in race_records]
evaluate(
    f"no26ブレンド α={ALPHA_NO26}（v5prime×v6hit_extreme）",
    no26_records,
    lambda r: ALPHA_NO26 * _norm(r["s_ext"]) + (1 - ALPHA_NO26) * _norm(r["s_v5p"]),
)

# v7odds_binary 単独
v7_records = [r for r in race_records if r["s_v7"] is not None]
evaluate(
    "v7odds_binary_no26（オッズ重み付き二値分類）",
    v7_records,
    lambda r: r["s_v7"],
)

# v7odds × v5prime のブレンド（αスイープ）
print(f"\n--- v7odds_binary × v5prime_no26 ブレンド αスイープ ---")
for alpha in [round(a * 0.1, 1) for a in range(0, 11)]:
    records = [r for r in race_records if r["s_v7"] is not None]
    st = Stats()
    for r in records:
        blended = alpha * _norm(r["s_v7"]) + (1 - alpha) * _norm(r["s_v5p"])
        order   = np.argsort(-blended)
        top5    = [r["umabans"][i] for i in order[:5]]
        honmei  = top5[0]
        san_combs = list(itertools.combinations(top5, 3))
        san_h = (r["san_winner"] is not None) and any(
            {a, b, c} == r["san_winner"] for a, b, c in san_combs
        )
        st.add(
            honmei in r["winner_set"], r["tan_ret"],
            honmei in r["top3_set"],  r["fuku_rets"].get(honmei),
            san_h, r["san_odds"], len(san_combs),
        )
    print(f"  α={alpha:.1f}  単勝 {st.tan_pct():>5.1f}%/{st.tan_rec():>6.1f}%  "
          f"複勝 {st.fuku_pct():>5.1f}%/{st.fuku_rec():>6.1f}%  "
          f"3連複 {st.san_pct():>5.1f}%/{st.san_rec():>6.1f}%")

print(f"\n{'='*115}")
