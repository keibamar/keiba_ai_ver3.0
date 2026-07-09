"""
v6hit（的中率重視）× v5prime（回収率重視）のブレンド比率αスイープ

score = α × v6hit_score + (1-α) × v5prime_score

α を 0.0〜1.0 で 0.1刻みにスイープし、各αで
  単勝的中率・回収率 / 複勝的中率・回収率 / 3連複5頭的中率・回収率
をバックテストして一覧表示する。

実行:
    python scripts/alpha_sweep_v6hit_v5prime.py
"""
import sys, warnings, glob, itertools, os, time
warnings.simplefilter("ignore")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
sys.path.append(r"C:\keiba_ai\keiba_ai_ver2.0\libs")

import numpy as np
import pandas as pd
import lightgbm as lgb

from src.config import paths
from src.config.constants import PLACE_LIST
from src.logic.prediction.race_prediction_engine import (
    make_dataset_for_lightgbm_v3,
    _build_jockey_course_stats,
    get_lightgbm_v3_model,
    rank_index,
    prediction_error,
)
from src.managers import race_card_dataset_manager
from src.PredictionModels.LightGBM.make_dataset_v5 import (
    build_pedigree_vocab, get_pedigree_cats, CAT_COLS,
)

BET_UNIT = 100
ALPHAS   = [round(a * 0.1, 1) for a in range(0, 11)]  # 0.0〜1.0

print("血統vocab読み込み中...")
_VOCAB = build_pedigree_vocab()
print(f"  {len(_VOCAB)}種類")


# ---------- モデルロード ----------

def _get_model(place_id, race_type, length, suffix):
    type_str = "turf" if race_type == "芝" else "dirt"
    mp = os.path.join(paths.PREDICTION_MODEL_PATH, PLACE_LIST[place_id - 1],
                      f"{type_str}{length}_lambdarank_model{suffix}.txt")
    if not os.path.isfile(mp):
        raise FileNotFoundError(mp)
    return lgb.Booster(model_file=mp)


# ---------- 特徴量ビルド ----------

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


def get_pedigree_extra(horse_ids):
    father_ids, mf_ids, pgf_ids = [], [], []
    for hid in horse_ids:
        f, mf, pgf = get_pedigree_cats(hid, _VOCAB)
        father_ids.append(f); mf_ids.append(mf); pgf_ids.append(pgf)
    return father_ids, mf_ids, pgf_ids


# ---------- 各モデルのスコア計算 ----------

def score_v6hit(full_df_64, place_id, race_type, course_len,
                final_odds, final_pop):
    """v6hit: v3_64列 + オッズ + 人気（66列）"""
    try:
        odds_arr = final_odds.astype(float).values
        pop_arr  = final_pop.astype(float).values
        extra = pd.DataFrame({
            "current_odds":       np.where(np.isfinite(odds_arr), odds_arr, -1),
            "current_popularity": np.where(np.isfinite(pop_arr),  pop_arr,  -1),
        })
        df66  = pd.concat([full_df_64.reset_index(drop=True), extra.reset_index(drop=True)], axis=1)
        model = _get_model(place_id, race_type, course_len, "_v6hit")
        return model.predict(df66, num_iteration=model.best_iteration)
    except Exception as e:
        prediction_error(e)
        return None


def score_v5prime(full_df_64, place_id, race_type, course_len, horse_ids):
    """v5prime: v3_64列 + 血統3列（67列）"""
    try:
        father_ids, mf_ids, pgf_ids = get_pedigree_extra(horse_ids)
        extra = pd.DataFrame({
            "father_cat":        father_ids,
            "mother_father_cat": mf_ids,
            "paternal_gf_cat":   pgf_ids,
        })
        df67  = pd.concat([full_df_64.reset_index(drop=True), extra.reset_index(drop=True)], axis=1)
        model = _get_model(place_id, race_type, course_len, "_v5prime")
        return model.predict(df67, num_iteration=model.best_iteration)
    except Exception as e:
        prediction_error(e)
        return None


# ---------- 集計 ----------

class Stats:
    def __init__(self):
        self.n = self.tan_hit = self.tan_pay = self.tan_bet = 0
        self.fuku_hit = self.fuku_pay = self.fuku_bet = 0
        self.san_hit = self.san_pay = self.san_bet = 0

    def add(self, tan_h, tan_ret, fuku_h, fuku_ret, san_h, san_ret, san_bets):
        self.n += 1
        self.tan_bet  += BET_UNIT
        self.fuku_bet += BET_UNIT
        self.san_bet  += BET_UNIT * san_bets
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


def get_return(ret_df, shikibetsu, umaban_str=None):
    rows = ret_df[ret_df["式別"] == shikibetsu]
    if rows.empty: return None
    if umaban_str is not None:
        rows = rows[rows["馬番"].astype(str) == umaban_str]
        if rows.empty: return None
    return int(rows["配当"].iloc[0])


# ---------- メイン ----------

active_places = []
for pid in range(1, 11):
    pfx = f"2026{pid:02d}"
    cnt = sum(len(glob.glob(f"{d}/{pfx}*.csv")) for d in glob.glob("data/race_card/2026*"))
    if cnt > 0:
        active_places.append(pid)

print(f"\n対象場: {[PLACE_LIST[p-1] for p in active_places]}")
print("レースデータを収集中...")

# 全レースのスコアをキャッシュしてからαスイープ
race_records = []  # list of (tan_winner_set, top3_set, san_winner, san_odds, v6hit_scores, v5prime_scores, umabans)

t0 = time.time()
processed = skipped = 0

for place_id in active_places:
    prefix = f"2026{place_id:02d}"
    race_paths = []
    for dd in sorted(glob.glob("data/race_card/2026*")):
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
        if "枠" not in rc_df.columns or "horse_id" not in rc_df.columns:
            skipped += 1; continue

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

        s_v6   = score_v6hit(full_df_64, pid, rtype, clen, final_odds, final_pop)
        s_v5p  = score_v5prime(full_df_64, pid, rtype, clen, rc_df["horse_id"].tolist())

        if s_v6 is None or s_v5p is None:
            skipped += 1; continue

        umabans = rc_df["馬番"].tolist()
        tan_ret  = get_return(ret_df, "単勝")
        fuku_rets = {str(int(float(u))): get_return(ret_df, "複勝", str(int(float(u)))) for u in umabans}

        race_records.append({
            "winner_set": winner_set,
            "top3_set":   top3_set,
            "san_winner": san_winner,
            "san_odds":   san_odds,
            "tan_ret":    tan_ret,
            "fuku_rets":  fuku_rets,
            "umabans":    [str(int(float(u))) for u in umabans],
            "s_v6":       np.array(s_v6, dtype=float),
            "s_v5p":      np.array(s_v5p, dtype=float),
        })
        processed += 1

        if processed % 200 == 0:
            print(f"  {processed}R収集済み ({time.time()-t0:.0f}秒)")

print(f"\n収集完了: {processed}R ({time.time()-t0:.0f}秒), スキップ:{skipped}R")

# ===== αスイープ =====
print(f"\n{'='*95}")
print(f"αスイープ結果  (α=1.0がv6hit単独, α=0.0がv5prime単独)")
print(f"{'='*95}")
print(f"{'α':>5}  {'単勝的中率':>10} {'単勝回収率':>10}  {'複勝的中率':>10} {'複勝回収率':>10}  {'3連複的中率':>11} {'3連複回収率':>11}  {'R数':>5}")
print("-" * 95)

results = []
for alpha in ALPHAS:
    st = Stats()
    for r in race_records:
        # スコアをnormalizeしてブレンド
        v6  = r["s_v6"]
        v5p = r["s_v5p"]

        def _norm(arr):
            mn, mx = arr.min(), arr.max()
            return (arr - mn) / (mx - mn + 1e-12)

        blended = alpha * _norm(v6) + (1 - alpha) * _norm(v5p)
        order   = np.argsort(-blended)
        top5    = [r["umabans"][i] for i in order[:5]]

        honmei   = top5[0]
        tan_h    = honmei in r["winner_set"]
        fuku_h   = honmei in r["top3_set"]
        fuku_ret = r["fuku_rets"].get(honmei)
        san_combs = list(itertools.combinations(top5, 3))
        san_h = (r["san_winner"] is not None) and any(
            {a, b, c} == r["san_winner"] for a, b, c in san_combs
        )
        st.add(tan_h, r["tan_ret"], fuku_h, fuku_ret, san_h, r["san_odds"], len(san_combs))

    print(f"  {alpha:>3}  "
          f"{st.tan_pct():>9.1f}% {st.tan_rec():>9.1f}%  "
          f"{st.fuku_pct():>9.1f}% {st.fuku_rec():>9.1f}%  "
          f"{st.san_pct():>10.1f}% {st.san_rec():>10.1f}%  "
          f"{st.n:>5}")
    results.append({
        "alpha": alpha,
        "tan_pct": st.tan_pct(), "tan_rec": st.tan_rec(),
        "fuku_pct": st.fuku_pct(), "fuku_rec": st.fuku_rec(),
        "san_pct": st.san_pct(), "san_rec": st.san_rec(),
        "n": st.n,
    })

print("=" * 95)

# knee point: 単勝回収率が最大になるα
best_tan  = max(results, key=lambda x: x["tan_rec"])
best_fuku = max(results, key=lambda x: x["fuku_rec"])
best_san  = max(results, key=lambda x: x["san_rec"])
best_bal  = max(results, key=lambda x: x["tan_rec"] + x["fuku_rec"] + x["san_rec"])

print(f"\n【参考: 各指標最大のα】")
print(f"  単勝回収率最大  : α={best_tan['alpha']}  ({best_tan['tan_pct']:.1f}%, {best_tan['tan_rec']:.1f}%)")
print(f"  複勝回収率最大  : α={best_fuku['alpha']}  ({best_fuku['fuku_pct']:.1f}%, {best_fuku['fuku_rec']:.1f}%)")
print(f"  3連複回収率最大 : α={best_san['alpha']}  ({best_san['san_pct']:.1f}%, {best_san['san_rec']:.1f}%)")
print(f"  3指標合計最大   : α={best_bal['alpha']}  "
      f"単{best_bal['tan_pct']:.1f}%/{best_bal['tan_rec']:.1f}%  "
      f"複{best_bal['fuku_pct']:.1f}%/{best_bal['fuku_rec']:.1f}%  "
      f"3連{best_bal['san_pct']:.1f}%/{best_bal['san_rec']:.1f}%")
